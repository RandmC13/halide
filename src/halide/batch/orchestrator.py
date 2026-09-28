"""Batch processing: discover files in a directory and process them in parallel via a process
pool, reusing halide.processing.process_scan (full inversion) or halide.processing.
export_delivery_image (delivery-format export) per file.

This module has no terminal/UI concerns — it reports progress via a plain callback so the CLI
layer (or a future GUI) can render it however it likes, or not at all. It also has no calibration
*policy* — the caller resolves a single shared DensityProfile (or None, meaning "each worker
computes its own per-frame automatic profile") before calling run_batch.

The memory-aware worker-pool sizing (estimate_worker_memory_bytes/default_worker_count/
memory_budget_warning) is shared machinery parameterized by a per-workload (baseline, multiplier)
pair, since a single worker's peak RSS depends on what it's actually doing per file — the full
inversion pipeline and a plain delivery-format export have very different memory profiles (see
each workload's own calibrated constants below). run_batch/run_export_batch each pass their own
measured constants through the same underlying estimator rather than duplicating the model.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import shutil
import sys
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from halide.core.types import DensityProfile, Stage, ToneCurveParams
from halide.device import ComputeDevice
from halide.interrupts import cancel_on_hangup_and_term, children_ignore_terminal_signals, ignore_terminal_signals
from halide.io.contact_sheet_defaults import DEFAULT_FRAME_WIDTH
from halide.io.roll import TIFF_SUFFIXES, list_scans  # noqa: F401 -- TIFF_SUFFIXES re-exported

# tifffile and halide.processing (numpy, Pillow, colour-science) are imported inside the functions
# that use them: every CLI command imports this module, and `halide --help` shouldn't pay for them.
# Workers don't pay either — the forkserver preloads halide.processing (_FORKSERVER_PRELOAD).
# halide.gpu_service and halide.shared_frames likewise (the latter imports numpy).


# Peak worker memory fits `baseline + K * decoded_pixel_bytes`. Measured on real worker processes
# (`--workers 1` batches over the four real full-res scans, 3276x4849 = ~182 MiB decoded, peak
# sampled from /proc/<pid>/status's VmHWM): worst case `--auto-density` 509 MiB RSS (481 PSS),
# `--output flat` 487, print output 365, `halide print` 386 — K ~= 2.2 over a ~106 MiB process.
# K = 3 with the 150 MiB baseline (695 MiB for such a scan) leaves ~37% margin over the worst case.
# exiftool (run after every output) has no term of its own but does use part of that margin: on
# Linux each worker keeps one exiftool running between frames (halide.io.exiftool), a separate
# process that streams the file (measured 67 MiB peak on a 130 MiB output). Worst worker plus a
# whole exiftool peak is 509 + 67 = 576 MiB, still ~17% under the 695 MiB estimate; process_scan
# frees the frame before exiftool writes, so its peak doesn't even coincide with the worker's.
#
# History, so K isn't "tuned" back up: it was ~23, then 11 after fixing a float64 upcast and
# per-line temporaries in core/ (see git history), then 3 after moving every full-resolution path
# to band-by-band processing into one owned buffer (halide/banding.py) — the frame itself, the
# print fit's 1/3-frame luminance, and at most one extra frame for flat output's percentile or auto
# calibration's statistics, where it used to be ~9 frames of scratch.
_PEAK_RSS_MULTIPLIER = 3
_BASELINE_PROCESS_OVERHEAD_BYTES = 150 * 1024 * 1024
_FALLBACK_PER_WORKER_BYTES = 5 * 1024**3  # used only if a file's header can't be read at all

# Export (and `halide contact`'s thumbnail pool, which does the same read + convert shape of work),
# measured the same way: export 379 MiB RSS (340 PSS), contact 378 — K ~= 1.5. K = 2 with the
# 100 MiB baseline (464 MiB) leaves ~22% margin over RSS. Was 15 before export converted to 8-bit
# sRGB band by band: colour.RGB_to_RGB computes in float64 internally, so converting the whole
# frame at once held several float64 copies of it.
_EXPORT_PEAK_RSS_MULTIPLIER = 2
_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES = 100 * 1024 * 1024

# GPU workers (docs/plans/gpu-acceleration.md §3.5, §7): peak *device* memory per worker, modelled
# like RAM as `context + K * decoded_pixel_bytes`. Fitted from the user's RTX 3070 benchmark
# (docs/plans/gpu-acceleration-bench.py, 2026-09-27), one 181 MiB frame developed on the GPU:
#   - CuPy's pool peaked at 821 MiB with a given profile and 878 MiB with --auto-density (4.54 and
#     4.86 x the frame) — the resident frame, the 64 MiB device bands' temporaries (banding.
#     DEVICE_BAND_BYTES) and the whole-frame statistics together, so they need no terms of their own;
#   - nvidia-smi showed 1046 MiB for that process: ~168 MiB of CUDA context and library overhead.
# Rounded up (too high costs a worker; too low costs frames redone on the CPU after running out of
# VRAM): 256 MiB + 5 x frame = 1161 MiB for a real scan, 11% over the measured 1046. The same
# benchmark's 4-worker batch ran with no CPU fallbacks.
# Previously PROVISIONAL (768 MiB + 4 x frame = 1492 MiB, from estimates before any real card ran).
_CUDA_CONTEXT_BYTES = 256 * 1024 * 1024
_DEVICE_CONTEXT_BYTES = _CUDA_CONTEXT_BYTES
_DEVICE_FRAME_MULTIPLIER = 5

# A GPU worker's *host* memory is much larger than a CPU worker's: the same benchmark measured the
# largest GPU worker at 1212-1235 MiB RSS against 392-403 MiB for a CPU worker on the same frames
# (and a single GPU `invert` 1209 vs 369 MiB) — CUDA's and CuPy's own host-side libraries, ~840 MiB.
# Added to the RAM estimate for GPU pools (rounded up to 1 GiB), so a machine with a big card and
# little RAM isn't given more workers than its RAM holds. On the user's machine (7.6 GiB free) this,
# not the card, is what limits per-worker GPU mode (the fallback when the shared service below
# can't be used) to 4 workers.
_GPU_HOST_OVERHEAD_BYTES = 1024 * 1024 * 1024

# The shared GPU service (docs/plans/gpu-batch-throughput.md Part B, halide/gpu_service.py): on a GPU,
# one service process holds the only CUDA context and develops every worker's frames, and the
# workers are CPU-only processes that hand it frames through shared memory (halide/shared_frames.py).
# The service's own host memory is taken off the RAM budget before sizing the workers:
# `_GPU_SERVICE_HOST_BYTES` (CUDA's and CuPy's host-side libraries — the measured GPU worker overhead
# above) plus one frame's working memory, per batch (estimate_service_host_bytes): 1205 MiB for a
# real 181 MiB frame. Measured on the user's RTX 3070 (Task B4): the service peaked at ~1178 MiB RSS
# (~933 MiB PSS), so this stands as it is.
_GPU_SERVICE_HOST_BYTES = _GPU_HOST_OVERHEAD_BYTES

# Shared-memory frames live in /dev/shm on Linux (tmpfs: RAM, but with its own size limit — 64 MiB
# in a default Docker container, half of RAM on a desktop). Each service-mode worker holds one frame
# there at a time (an export also its 8-bit output). Only this fraction of the free space is planned
# on, so another program's (or another halide's) segments arriving mid-batch don't starve the last
# worker. A worker that finds no room anyway develops that frame on the CPU, with a warning.
_SHM_DIR = "/dev/shm"
_SHM_USABLE_FRACTION = 0.8

# What batch_compute starts: a "gpu" service in normal use. Tests set "cpu" (the numpy path in a
# real separate process) or a child initializer (tests/unit/_fake_device.py's install_as_gpu).
_SERVICE_KIND = "gpu"
_SERVICE_INITIALIZER = None

# A troubleshooting switch: HALIDE_GPU_SERVICE=0 (or "off") makes a GPU batch use per-worker GPU
# mode — every worker its own CUDA context, as before the service existed — instead of the shared
# service. On by default (unset, or any other value). Also how the benchmark
# (docs/plans/gpu-acceleration-bench.py) and tests/gpu time and compare the two modes in one run.
SERVICE_ENV = "HALIDE_GPU_SERVICE"
_SERVICE_OFF = ("0", "off", "false", "no")

# Controller ruling (final whole-branch review): the shared service needs Python 3.13 or newer.
# shared_frames.attach_frame's service-side attach uses `SharedMemory(..., track=False)`, which is
# a Python 3.13 addition (gh-82300) — this project's own floor is 3.11 (pyproject.toml). Below
# 3.13 attach_frame raises SharedMemoryUnavailable and sweep does nothing (a tracked attach would
# corrupt the batch's shared resource tracker — see shared_frames' module docstring), so a GPU batch
# there uses per-worker GPU mode, with the reason on the run sheet. One helper, not
# `sys.version_info` inline, so a test can patch it without patching the interpreter itself.
_MIN_SERVICE_PYTHON = (3, 13)


def _service_python_supported() -> bool:
    return sys.version_info >= _MIN_SERVICE_PYTHON


def gpu_service_enabled() -> bool:
    """False when $HALIDE_GPU_SERVICE turns the shared GPU service off (case and space ignored)."""
    return os.environ.get(SERVICE_ENV, "").strip().lower() not in _SERVICE_OFF


@dataclass(frozen=True)
class BatchJob:
    input_path: Path
    output_path: Path | None  # None = don't keep the full-size output (contact-sheet preview)
    scan_gain: float = 1.0  # see processing.process_scan / --match-scan-exposure
    thumbnail_path: Path | None = None  # contact-sheet thumbnail to write alongside, if any


@dataclass(frozen=True)
class BatchResult:
    job: BatchJob
    error: str | None  # None on success
    warning: str | None = None  # non-fatal, e.g. export's "doesn't look like ACEScg" notice


def discover_jobs(input_dir: str | Path, output_dir: str | Path, suffix: str = "") -> list[BatchJob]:
    """One job per scan in `input_dir` (halide.io.roll.list_scans decides what is a scan)."""
    files, _ = list_scans(input_dir)
    return jobs_for_files(files, output_dir, suffix)


def jobs_for_files(files: list[Path], output_dir: str | Path, suffix: str = "") -> list[BatchJob]:
    output_dir = Path(output_dir)
    jobs = []
    for f in files:
        name = f"{f.stem}{suffix}{f.suffix}" if suffix else f.name
        jobs.append(BatchJob(input_path=f, output_path=output_dir / name))
    return jobs


def _decoded_pixel_bytes(path: Path) -> int | None:
    """Cheaply estimate a TIFF's decoded in-memory size from its header alone (page shape/dtype),
    without reading any pixel data — used to size the worker pool before processing starts. Returns
    None if the header can't be read (caller falls back to a conservative default)."""
    import tifffile

    try:
        with tifffile.TiffFile(path) as tf:
            # series[0], as io/tiff.py's read_tiff/read_tiff_shape: what's actually decoded.
            series = tf.series[0]
            shape = series.shape
            itemsize = series.dtype.itemsize
    except Exception:  # noqa: BLE001 — a bad header here must not abort worker-count sizing
        return None
    size = itemsize
    for dim in shape:
        size *= dim
    return size


def estimate_worker_memory_bytes(
    jobs: list[BatchJob],
    *,
    baseline_bytes: int = _BASELINE_PROCESS_OVERHEAD_BYTES,
    multiplier: int = _PEAK_RSS_MULTIPLIER,
    device: ComputeDevice | None = None,
) -> int:
    """Estimate the peak RSS a single worker needs to process the largest job in this batch (see
    the module-level constants' docstring for how the estimate itself was derived). `baseline_bytes`
    /`multiplier` default to the full-inversion-pipeline fit; pass export's own calibrated constants
    (or use estimate_export_worker_memory_bytes) when sizing a pool of export workers instead. A GPU
    `device` adds the CUDA/CuPy host-side overhead (_GPU_HOST_OVERHEAD_BYTES) every GPU worker carries."""
    sizes = [b for b in (_decoded_pixel_bytes(job.input_path) for job in jobs) if b is not None]
    if not sizes:
        return _FALLBACK_PER_WORKER_BYTES
    gpu_overhead = _GPU_HOST_OVERHEAD_BYTES if device is not None and device.kind == "gpu" else 0
    return baseline_bytes + max(sizes) * multiplier + gpu_overhead


def estimate_worker_device_bytes(jobs: list[BatchJob]) -> int:
    """Estimate the peak GPU memory one worker needs for the largest job in this batch: its own
    CUDA context, plus the frame resident on the device, its band temporaries and the whole-frame
    statistics (see the constants above, fitted on a real RTX 3070). Used for every GPU pool — develop,
    print and export alike: export keeps the same resident frame and less scratch, so this bounds
    it too."""
    sizes = [b for b in (_decoded_pixel_bytes(job.input_path) for job in jobs) if b is not None]
    if not sizes:
        return _FALLBACK_PER_WORKER_BYTES
    return _DEVICE_CONTEXT_BYTES + max(sizes) * _DEVICE_FRAME_MULTIPLIER


def device_worker_cap(jobs: list[BatchJob], device: ComputeDevice | None) -> int | None:
    """How many GPU workers fit in the card's free memory, or None when there's no GPU (or its free
    memory wasn't reported). Uses the free memory the parent's own resolve_device already measured —
    no further CUDA calls here (the parent's own context is already counted out of it)."""
    if device is None or device.kind != "gpu" or not device.memory_free:
        return None
    return max(1, device.memory_free // estimate_worker_device_bytes(jobs))


def device_budget_warning(jobs: list[BatchJob], requested_workers: int, device: ComputeDevice | None) -> str | None:
    """The GPU counterpart of memory_budget_warning: a warning if an explicit --workers N looks
    like more GPU workers than the card's free memory holds (device_worker_cap), else None — also
    None on the CPU or when the card's free memory wasn't reported. Each worker's memory-pool share
    shrinks as N grows (_device_worker_args), and a frame that doesn't fit its share is developed
    on the CPU instead, so too many GPU workers mostly means slow CPU fallbacks, not a crash."""
    safe_workers = device_worker_cap(jobs, device)
    if safe_workers is None or requested_workers <= safe_workers:
        return None
    per_worker = estimate_worker_device_bytes(jobs)
    return (
        f"--workers {requested_workers} may not fit in free GPU memory (~{per_worker / 1024**3:.1f} GB "
        f"estimated per worker vs. ~{device.memory_free / 1024**3:.1f} GB free suggests {safe_workers} "
        f"worker(s)) — frames that don't fit are developed on the CPU instead, more slowly."
    )


def _device_worker_args(device: ComputeDevice | None, workers: int) -> tuple[str, int | None]:
    """What each worker is handed about the device: the resolved kind ("cpu"/"gpu") — never the
    device itself, since each worker makes its own CUDA context (CUDA must not be initialised before
    the fork) — and, on a GPU, its memory-pool limit: an equal share of the free memory, less its
    own CUDA context, so one worker's cached blocks can't starve the rest. A share that works out
    at nothing (far more --workers than fit) becomes 1 byte, not 0: CuPy reads 0 as "no limit".
    Such a worker's frames then run out of GPU memory and are redone on the CPU, with a warning."""
    if device is None or device.kind != "gpu":
        return "cpu", None
    if not device.memory_free:
        return "gpu", None
    return "gpu", max(1, device.memory_free // workers - _CUDA_CONTEXT_BYTES)


def _cpu_cap() -> int:
    """Most workers worth running regardless of memory: one per *physical* core. Each worker is
    numpy- and memory-bandwidth-bound, so a hyperthread sibling adds little speed but a whole extra
    frame of memory. Replaces a fixed `min(cpu_count, 6)` that dated from when memory, not cores,
    was the real limit (see default_worker_count). Falls back to logical cores, then 1."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        physical = None
    return physical or os.cpu_count() or 1


def default_worker_count(
    jobs: list[BatchJob],
    *,
    device: ComputeDevice | None = None,
    baseline_bytes: int = _BASELINE_PROCESS_OVERHEAD_BYTES,
    multiplier: int = _PEAK_RSS_MULTIPLIER,
) -> int:
    """Auto-select a worker count that respects available RAM, not just CPU count: min(physical
    cores, available memory / per-worker estimate, number of jobs). Found via real testing:
    full-resolution scans are memory-heavy (a single frame's pipeline once peaked around 4GB RSS;
    ~0.5 GB now, see the constants above), and a CPU-only worker count was enough to get a worker
    OOM-killed. Falls back to the CPU-only cap if `psutil` can't report available
    memory, or if none of the batch's files' headers could be read at all. `baseline_bytes`/
    `multiplier` default to the full-inversion-pipeline fit — see estimate_worker_memory_bytes.

    On a GPU `device`, a third cap: how many workers' device memory fits in the card's free memory
    (estimate_worker_device_bytes). A CPU device, or None, is exactly the CPU-only count."""
    cap = _cpu_cap()
    if not jobs:
        return cap
    device_cap = device_worker_cap(jobs, device)
    if device_cap is not None:
        cap = min(cap, device_cap)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return cap
    per_worker = estimate_worker_memory_bytes(
        jobs, baseline_bytes=baseline_bytes, multiplier=multiplier, device=device
    )
    memory_cap = max(1, available // per_worker)
    return max(1, min(cap, memory_cap, len(jobs)))


def memory_budget_warning(
    jobs: list[BatchJob],
    requested_workers: int,
    *,
    baseline_bytes: int = _BASELINE_PROCESS_OVERHEAD_BYTES,
    multiplier: int = _PEAK_RSS_MULTIPLIER,
    device: ComputeDevice | None = None,
) -> str | None:
    """Returns a human-readable warning if an explicitly-requested worker count looks likely to
    exceed available memory, or None if it looks safe (or memory couldn't be checked at all). Pure
    computation only, matching this module's no-UI-concerns design (see module docstring) — the
    caller decides whether/how to display it. `baseline_bytes`/`multiplier` default to the
    full-inversion-pipeline fit — see estimate_worker_memory_bytes; a GPU `device` counts each
    worker's CUDA/CuPy host memory too."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return None
    per_worker = estimate_worker_memory_bytes(
        jobs, baseline_bytes=baseline_bytes, multiplier=multiplier, device=device
    )
    safe_workers = max(1, available // per_worker)
    if requested_workers <= safe_workers:
        return None
    return (
        f"--workers {requested_workers} may exceed available memory (~{per_worker / 1024**3:.1f} "
        f"GB estimated per worker vs. ~{available / 1024**3:.1f} GB available suggests {safe_workers} "
        f"worker(s) is safer) — continuing with {requested_workers} since it was explicitly requested."
    )


def estimate_export_worker_memory_bytes(jobs: list[BatchJob]) -> int:
    """Same as estimate_worker_memory_bytes, using export's own calibrated constants."""
    return estimate_worker_memory_bytes(
        jobs, baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES, multiplier=_EXPORT_PEAK_RSS_MULTIPLIER
    )


def default_export_worker_count(jobs: list[BatchJob], *, device: ComputeDevice | None = None) -> int:
    """Same as default_worker_count, using export's own calibrated constants."""
    return default_worker_count(
        jobs, device=device,
        baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES, multiplier=_EXPORT_PEAK_RSS_MULTIPLIER,
    )


def export_memory_budget_warning(
    jobs: list[BatchJob], requested_workers: int, device: ComputeDevice | None = None
) -> str | None:
    """Same as memory_budget_warning, using export's own calibrated constants."""
    return memory_budget_warning(
        jobs,
        requested_workers,
        baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES,
        multiplier=_EXPORT_PEAK_RSS_MULTIPLIER,
        device=device,
    )


# --- The shared GPU service: which mode a GPU batch runs in, and how many workers it gets ---------


def _shared_frame_bytes(jobs: list[BatchJob], workload: str) -> int | None:
    """The shared memory one service-mode worker holds at a time for the largest job: the decoded
    float32 frame (read_tiff always decodes to float32, whatever the file's own sample type), and
    for an export its 8-bit output too. None if no job's header could be read."""
    from halide.io.tiff import read_tiff_shape

    sizes = []
    for job in jobs:
        try:
            shape = read_tiff_shape(job.input_path)
        except Exception:  # noqa: BLE001 — a bad header here must not abort worker-count sizing
            continue
        pixels = 1
        for dim in shape:
            pixels *= dim
        sizes.append(pixels * (4 + 1 if workload == "export" else 4))
    return max(sizes) if sizes else None


def _shared_memory_free() -> int | None:
    """Free bytes where shared-memory frames live, or None where that can't be told (or isn't a
    separate limit): only Linux has a /dev/shm of its own size. On Windows a segment is backed by
    the page file and counts against RAM like anything else, which the RAM cap already covers."""
    if sys.platform != "linux":
        return None
    try:
        return shutil.disk_usage(_SHM_DIR).free
    except OSError:
        return 0  # no /dev/shm at all: no shared frames either


def shared_memory_worker_cap(jobs: list[BatchJob], workload: str = "develop") -> int | None:
    """How many service-mode workers' frames fit in /dev/shm at once (see _SHM_USABLE_FRACTION):
    0 when not even one does, None when there's no separate limit to respect."""
    free = _shared_memory_free()
    per_worker = _shared_frame_bytes(jobs, workload)
    if free is None or not per_worker:
        return None
    return int(free * _SHM_USABLE_FRACTION) // per_worker


def estimate_service_host_bytes(jobs: list[BatchJob]) -> int:
    """The GPU service process's own host memory (see _GPU_SERVICE_HOST_BYTES): its libraries plus
    one frame's working memory for the largest job — the decoded float32 frame it attaches to."""
    return _GPU_SERVICE_HOST_BYTES + (_shared_frame_bytes(jobs, "develop") or 0)


def _workload_constants(workload: str) -> tuple[int, int]:
    if workload == "export":
        return _EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES, _EXPORT_PEAK_RSS_MULTIPLIER
    return _BASELINE_PROCESS_OVERHEAD_BYTES, _PEAK_RSS_MULTIPLIER


def estimate_service_worker_memory_bytes(jobs: list[BatchJob], workload: str = "develop") -> int:
    """One service-mode worker's peak RAM: a CPU worker's (it's a CPU-only process — no CUDA, no
    CuPy) plus its shared frame(s), counted on top even though they're the same working buffer the
    CPU constants already include. Deliberately generous and not refit after Task B4 measured
    ~525 MiB against this ~876 MiB for a real scan: if the service dies mid-batch, every worker
    develops on the CPU while still holding its shared frame — about what this estimates — and the
    margin keeps that from swapping. 6 workers (what it picks on 7.6 GiB) vs 8 measured 0.45 vs
    0.44 s/frame, so little is lost."""
    baseline, multiplier = _workload_constants(workload)
    return estimate_worker_memory_bytes(jobs, baseline_bytes=baseline, multiplier=multiplier) + (
        _shared_frame_bytes(jobs, workload) or 0
    )


@dataclass(frozen=True)
class GpuService:
    """What a service-mode worker is handed: where the batch's GPU service listens (a picklable
    gpu_service.ServiceAddress) and the name prefix for its shared frames (shared_frames.
    batch_prefix, made in the parent so sweep() can clean up after a worker that died)."""

    address: object  # gpu_service.ServiceAddress — typed loosely to keep gpu_service out of this import
    shm_prefix: str


@dataclass(frozen=True)
class BatchCompute:
    """How a batch's workers reach the device — decided once, before the pool starts, by
    batch_compute():
      - "cpu": no GPU; workers get exactly the arguments they always had;
      - "service": one GPU service shared by CPU-only workers (the normal GPU case);
      - "per_worker": each worker makes its own CUDA context, as before Part B — only when the
        service can't be used, and `fallback_reason` says why (/dev/shm too small, or the service
        didn't start)."""

    device: ComputeDevice | None
    workload: str = "develop"  # "develop" (batch, print, the contact sheet window) or "export"
    service: GpuService | None = None
    shm_cap: int | None = None  # service-mode workers whose frames fit in /dev/shm; None = no limit
    fallback_reason: str | None = None

    @property
    def mode(self) -> str:
        if self.service is not None:
            return "service"
        return "per_worker" if self.device is not None and self.device.kind == "gpu" else "cpu"

    @property
    def shm_prefix(self) -> str | None:
        return None if self.service is None else self.service.shm_prefix

    def worker_args(self, workers: int) -> tuple:
        """The device arguments every worker function takes last. Service mode: the worker's own
        device is the CPU (it never touches CUDA; it's also where a frame goes when the service
        can't take it) plus the service — never ("gpu", pool share)."""
        if self.service is not None:
            return ("cpu", None, self.service)
        return _device_worker_args(self.device, workers)


def _start_service(kind: str, initializer, cancel: Callable[[], bool] | None = None):
    """gpu_service.running_service — one indirection, so the test suite can keep a real GPU
    service from ever starting on a developer's machine (tests/conftest.py). `cancel` is forwarded
    unchanged — see running_service's own docstring (bounds a caller's wait while the service is
    still starting, e.g. the picker's contact sheet window being closed early)."""
    from halide.gpu_service import running_service

    return running_service(kind, initializer=initializer, cancel=cancel)


def _size(n: int) -> str:
    return f"{n / 2**20:.0f} MiB" if n >= 2**20 else f"{n / 2**10:.0f} KiB"


@contextlib.contextmanager
def batch_compute(jobs: list[BatchJob], device: ComputeDevice | None, workload: str = "develop",
                  cancel: Callable[[], bool] | None = None) -> Iterator[BatchCompute]:
    """Decide how this batch reaches the device, and on a GPU start the shared service for as long
    as the block runs. Run the worker pool *inside* the block: on the way out the service is
    stopped and then this batch's leftover shared frames are swept — which is only safe once every
    worker is gone (shared_frames.sweep), so the pool must have shut down by then.

    A GPU batch falls back to per-worker GPU mode (fallback_reason says why, for the run sheet) when
    the interpreter is older than Python 3.13 (_service_python_supported), /dev/shm can't hold even
    one frame (a small container), the service doesn't start, or $HALIDE_GPU_SERVICE turns it off.
    It never fails the batch: that mode is what every GPU batch did before the service existed.

    `cancel`: forwarded to gpu_service.running_service — an optional no-argument callable polled
    while the service is still starting, so a caller that wants to give up early (the picker's
    contact sheet window, closed mid-startup) isn't stuck waiting out the service's full startup
    timeout on a hung driver probe. CLI callers don't have anything to cancel on, so they pass
    nothing."""
    if device is None or device.kind != "gpu":
        yield BatchCompute(device=device, workload=workload)
        return
    from halide.shared_frames import sweep_stale

    # F12: frames a batch killed outright left behind (nothing of it survived to sweep them) —
    # first, so the /dev/shm cap below counts the space they held as free.
    sweep_stale()
    if not _service_python_supported():
        yield BatchCompute(device=device, workload=workload,
                           fallback_reason="the shared GPU service needs Python 3.13 or newer")
        return
    if not gpu_service_enabled():
        value = os.environ.get(SERVICE_ENV, "").strip()
        yield BatchCompute(device=device, workload=workload, fallback_reason=f"turned off by {SERVICE_ENV}={value}")
        return
    shm_cap = shared_memory_worker_cap(jobs, workload)
    if shm_cap == 0:
        free = _shared_memory_free() or 0
        needed = _shared_frame_bytes(jobs, workload) or 0
        reason = (f"shared memory ({_SHM_DIR}) has {_size(free)} free, too little for one "
                  f"{_size(needed)} frame")
        yield BatchCompute(device=device, workload=workload, fallback_reason=reason)
        return
    from halide.shared_frames import batch_prefix, sweep

    with contextlib.ExitStack() as stack:
        # F12: while the service runs, closing the terminal (SIGHUP) or SIGTERM cancels like Ctrl-C
        # (KeyboardInterrupt), so the unwinding below still stops the service and sweeps. Entered
        # first, so it is the last thing undone.
        stack.enter_context(cancel_on_hangup_and_term())
        prefix = batch_prefix()
        # Registered before the service is entered, so — since an ExitStack unwinds last-registered
        # first (LIFO) — this callback is the *last* thing to run on the way out: the service (entered
        # next, so it unwinds first) has already stopped by the time sweep runs, which is the whole
        # point (sweep must only ever run once nothing can still be writing into a frame).
        stack.callback(sweep, prefix)
        try:
            address = stack.enter_context(_start_service(_SERVICE_KIND, _SERVICE_INITIALIZER, cancel))
        except Exception as exc:  # noqa: BLE001 — ServiceUnavailable, or the process couldn't even spawn
            stack.close()
            # One line for the run sheet; the sheet adds its own punctuation after it.
            yield BatchCompute(device=device, workload=workload, fallback_reason=str(exc).rstrip("."))
            return
        yield BatchCompute(device=device, workload=workload, service=GpuService(address, prefix), shm_cap=shm_cap)


def service_worker_count(jobs: list[BatchJob], compute: BatchCompute) -> int:
    """default_worker_count for service mode: min(physical cores, RAM cap, /dev/shm cap, jobs).
    The RAM cap uses the CPU worker constants plus each worker's shared frame, after the service's
    own memory (estimate_service_host_bytes). No VRAM cap: however many workers there are, the card
    only ever holds the service's one context and one frame at a time."""
    cap = _cpu_cap()
    if not jobs:
        return cap
    if compute.shm_cap is not None:
        cap = min(cap, compute.shm_cap)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return max(1, min(cap, len(jobs)))
    per_worker = estimate_service_worker_memory_bytes(jobs, compute.workload)
    memory_cap = max(1, (available - estimate_service_host_bytes(jobs)) // per_worker)
    return max(1, min(cap, memory_cap, len(jobs)))


def service_budget_warnings(jobs: list[BatchJob], requested_workers: int, compute: BatchCompute) -> list[str]:
    """For an explicit --workers N in service mode: the RAM warning (as memory_budget_warning, with
    the service mode's own estimates), and one if N workers' frames won't all fit in /dev/shm at
    once — those frames are then developed on the CPU in their workers, not lost."""
    warnings = []
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        available = None
    if available is not None:
        per_worker = estimate_service_worker_memory_bytes(jobs, compute.workload)
        safe = max(1, (available - estimate_service_host_bytes(jobs)) // per_worker)
        if requested_workers > safe:
            warnings.append(
                f"--workers {requested_workers} may exceed available memory (~{per_worker / 1024**3:.1f} GB "
                f"estimated per worker, plus the GPU service, vs. ~{available / 1024**3:.1f} GB available "
                f"suggests {safe} worker(s) is safer) — continuing with {requested_workers} since it was "
                f"explicitly requested."
            )
    if compute.shm_cap is not None and requested_workers > compute.shm_cap:
        warnings.append(
            f"--workers {requested_workers} won't all fit in shared memory ({_SHM_DIR} holds "
            f"{compute.shm_cap} worker(s)' frames) — frames that don't fit are developed on the CPU "
            f"instead, more slowly."
        )
    return warnings


# This worker process's device, resolved on its first job and kept for the rest (see
# _worker_device). Keyed by the pool-limit share so an in-process caller can't get a stale one.
_WORKER_DEVICES: dict[int | None, tuple[ComputeDevice, BaseException | None]] = {}


def _worker_device(device_kind: str, memory_limit: int | None) -> tuple[ComputeDevice, BaseException | None]:
    """The device this worker process runs its frames on, and — if it was asked for a GPU but
    couldn't get one — why not.

    Resolved here, in the worker, on its first GPU job (never in the parent or the forkserver: a
    CUDA context doesn't survive a fork), then cached for the process's lifetime, so the CUDA
    context and CuPy's memory pool stay warm from frame to frame. On a GPU it caps the pool at
    `memory_limit` bytes (this worker's share, see _device_worker_args). A GPU that can't be used
    here — typically no room left on the card for one more CUDA context — never fails the batch:
    the worker develops on the CPU and every frame it does so says why."""
    if device_kind != "gpu":
        return ComputeDevice(kind="cpu"), None
    if memory_limit not in _WORKER_DEVICES:
        from halide import device as halide_device

        try:
            device = halide_device.resolve_device("gpu")
            if memory_limit is not None:
                import cupy  # already imported by resolve_device; see halide.device

                cupy.get_default_memory_pool().set_limit(size=memory_limit)
            _WORKER_DEVICES[memory_limit] = (device, None)
        except Exception as exc:  # noqa: BLE001 — any GPU problem: this worker runs on the CPU
            # resolve_device("gpu") wraps the real error in an install hint meant for the CLI; the
            # frame's warning wants the error itself ("out of GPU memory", a driver error).
            _WORKER_DEVICES[memory_limit] = (ComputeDevice(kind="cpu"), exc.__cause__ or exc)
    return _WORKER_DEVICES[memory_limit]


# This worker process's connection to the batch's GPU service, made on its first frame and kept
# (see _worker_service). A client that found the service gone stays dead: every later frame goes
# straight to the CPU with a warning, without waiting on it again.
_WORKER_CLIENTS: dict = {}


def _worker_service(service: GpuService | None):
    """(ServiceClient or None, shm prefix or None) for `service` — the worker's cached client."""
    if service is None:
        return None, None
    client = _WORKER_CLIENTS.get(service.address)
    if client is None:
        from halide.gpu_service import ServiceClient

        client = _WORKER_CLIENTS[service.address] = ServiceClient(service.address)
    return client, service.shm_prefix


class _Warnings:
    """Collects a frame's warnings for its BatchResult: a worker's output isn't anyone's terminal,
    so processing.py's on_warning messages have to travel back in the result. The frame's own path,
    which those messages start with, is dropped — the CLI prints each warning under the frame's
    name already."""

    def __init__(self, job: BatchJob) -> None:
        self._prefix = f"{job.input_path}: "
        self.messages: list[str] = []

    def __call__(self, message: str | None) -> None:
        if message:
            self.messages.append(message.removeprefix(self._prefix))

    def text(self) -> str | None:
        return "; ".join(self.messages) or None


def _start_on_device(job: BatchJob, device_kind: str, memory_limit: int | None, action: str):
    """(device, warnings) for one frame: the worker's device, with the warning already noted if a
    requested GPU turned out to be unusable in this worker."""
    from halide.processing import _gpu_fallback_message

    warnings = _Warnings(job)
    device, unusable = _worker_device(device_kind, memory_limit)
    if unusable is not None:
        warnings(_gpu_fallback_message(job.input_path, unusable, action=action))
    return device, warnings


def _worker(
    job: BatchJob,
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
    thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH,
    device_kind: str = "cpu",
    device_memory_limit: int | None = None,
    service: GpuService | None = None,
) -> BatchResult:
    from halide.processing import process_scan

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "developed this frame")
        client, shm_prefix = _worker_service(service)
        process_scan(
            job.input_path, job.output_path, stage, density_profile, tone_params, scan_gain=job.scan_gain,
            thumbnail_path=job.thumbnail_path, thumbnail_long_edge=thumbnail_long_edge,
            device=device, on_warning=warnings, service=client, shm_prefix=shm_prefix,
        )
        return BatchResult(job=job, error=None, warning=warnings.text())
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc), warning=warnings.text() if warnings else None)


def _export_worker(
    job: BatchJob, quality: int, device_kind: str = "cpu", device_memory_limit: int | None = None,
    service: GpuService | None = None,
) -> BatchResult:
    from halide.processing import export_delivery_image

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "exported this file")
        client, shm_prefix = _worker_service(service)
        warnings(export_delivery_image(job.input_path, job.output_path, quality=quality,
                                       device=device, on_warning=warnings, service=client, shm_prefix=shm_prefix))
        return BatchResult(job=job, error=None, warning=warnings.text())
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc), warning=warnings.text() if warnings else None)


def _print_worker(
    job: BatchJob, tone_params: ToneCurveParams, device_kind: str = "cpu", device_memory_limit: int | None = None,
    service: GpuService | None = None,
) -> BatchResult:
    from halide.processing import print_scan

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "developed this frame")
        client, shm_prefix = _worker_service(service)
        _, warning = print_scan(job.input_path, job.output_path, tone_params, device=device, on_warning=warnings,
                                service=client, shm_prefix=shm_prefix)
        warnings(warning)
        return BatchResult(job=job, error=None, warning=warnings.text())
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc), warning=warnings.text() if warnings else None)


def _thumbnail_worker(job: BatchJob, thumbnail_long_edge: int) -> BatchResult:
    from halide.processing import thumbnail_existing_output

    try:
        thumbnail_existing_output(job.input_path, job.thumbnail_path, thumbnail_long_edge)
        return BatchResult(job=job, error=None)
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc))


# What the forkserver imports once, before forking workers: its own default (`__main__`) plus the
# worker code. Workers then share those pages copy-on-write instead of each importing numpy,
# colour-science and scipy afresh — measured: 4 idle workers' proportional memory 294 -> 73 MiB.
# colour is listed explicitly because halide imports it lazily (only where it's used), so
# preloading halide.processing alone no longer loads it.
# Never cupy: importing it is harmless, but anything that touched CUDA here would leave every forked
# worker with an unusable CUDA context. Each GPU worker imports it and makes its own context on its
# first job (_worker_device); a test pins that a fresh worker hasn't imported it.
_FORKSERVER_PRELOAD = ["__main__", "halide.processing", "colour"]


def _start_forkserver(context) -> None:
    """Start the pool's forkserver (if it isn't already running) with Ctrl-C and SIGHUP ignored
    from its first instruction (2.4-9): Ctrl-C in the first second of a batch otherwise interrupted
    its preload of colour/scipy and printed pages of tracebacks. It passes the ignored signals on
    to every worker it forks, and the pool's worker initializer ignores them again anyway. Main
    thread only, like any signal handling; the picker's background pools start it as before."""
    if not isinstance(context, multiprocessing.context.ForkServerContext):
        return
    from multiprocessing import forkserver

    with children_ignore_terminal_signals():
        forkserver.ensure_running()


def _pool_context():
    """The multiprocessing context for worker pools: forkserver with halide preloaded where the
    platform has it (Linux; Python 3.14's default there anyway), else the platform default. Only
    changes how worker code gets loaded, never what it computes. Must be used after _run_pool's
    thread-count environment setup, so the forkserver's own numpy import sees it."""
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return None
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(_FORKSERVER_PRELOAD)
    return context


def _run_pool(
    jobs: list[BatchJob],
    worker: Callable[..., BatchResult],
    worker_args: tuple,
    max_workers: int,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    shm_prefix: str | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    """Process every job in a process pool, in parallel, calling `worker(job, *worker_args)` for
    each. Calls on_start(job) when a job is handed to a worker and on_result(result) when it
    completes, if given — purely for progress reporting, this function has no rendering logic of
    its own. A single job's failure is captured in its BatchResult, not raised — the rest of the
    batch keeps running.

    Jobs are submitted in a rolling window of at most `max_workers` in flight at once (rather than
    all of them up front) specifically so on_start reflects jobs actually dispatched to a worker,
    not merely queued — a caller rendering "processing" status for every submitted job would
    otherwise show the whole batch as active immediately even though only `max_workers` can run at
    a time.

    That guarantee covers exceptions raised *inside* a worker (caught by each worker's own
    try/except). A worker process dying outright — OOM-killed, segfault, anything that kills it
    before its own try/except can run — is a different failure mode: found via real testing on
    full-resolution scans (a single frame's pipeline run can peak around 4GB RSS; enough parallel
    workers on a memory-constrained machine gets one OOM-killed). Once that happens,
    ProcessPoolExecutor marks the whole pool broken and every *other* pending/future submission
    raises BrokenProcessPool too — without the handling below, that would propagate straight out of
    this function, discarding every already-completed BatchResult and surfacing nothing but a bare
    traceback that gives no hint what happened or which file was involved. Instead, each job that
    hits BrokenProcessPool (whether already in flight or still queued when the pool broke) is
    recorded as its own failed BatchResult with an actionable message, exactly like a per-file
    processing error — the rest of the batch's real results are preserved either way. Shared by
    run_batch and run_export_batch — only the worker function and its extra arguments differ
    between a full inversion and a delivery-format export.

    `shm_prefix`: the batch's shared-frame prefix in GPU service mode (BatchCompute.shm_prefix).
    Once the pool has fully shut down — on every way out, a crash or Ctrl+C included — whatever a
    dead worker left in /dev/shm under it is swept (shared_frames.sweep); never before, since
    every worker shares the prefix.

    `on_cancel(message)`: called once after a cancel if frames are still in progress, while they
    finish (_finish_in_flight) — so the caller can say why the batch hasn't stopped yet."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")

    # Restrict underlying BLAS/OpenMP threading per worker process to avoid CPU oversubscription
    # when running several worker *processes* in parallel, each of which would otherwise also try
    # to multithread its own numpy/colour-science operations.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    results: list[BatchResult] = []

    def record(result: BatchResult) -> None:
        results.append(result)
        if on_result:
            on_result(result)

    in_flight: dict = {}
    stopped: list = []  # the pool's worker processes, once a cancel stopped it without waiting
    try:
        # F12: closing the terminal (SIGHUP) or SIGTERM cancels exactly like Ctrl-C.
        with cancel_on_hangup_and_term():
            try:
                context = _pool_context()
                _start_forkserver(context)
                # Workers ignore Ctrl-C and SIGHUP (2.4-9): the parent owns cancelling, below.
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context,
                                         initializer=ignore_terminal_signals) as executor:
                    try:
                        pending = list(jobs)

                        def submit_next() -> None:
                            # Loops rather than submitting exactly one job, so that once the pool is
                            # broken every remaining queued job is immediately resolved as a crashed
                            # result instead of being submitted (and raising) one at a time as later
                            # callers happen to invoke this again.
                            while pending:
                                job = pending.pop(0)
                                try:
                                    future = executor.submit(worker, job, *worker_args)
                                except BrokenProcessPool:
                                    record(_crashed_result(job))
                                    continue
                                in_flight[future] = job
                                if on_start:
                                    on_start(job)
                                break

                        for _ in range(min(max_workers, len(jobs))):
                            submit_next()

                        while in_flight:
                            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                            for future in done:
                                record(_finished(future, in_flight.pop(future)))
                                submit_next()
                    except KeyboardInterrupt:
                        # Ctrl+C (or SIGHUP/SIGTERM, above) can land on any bytecode boundary, from
                        # the first submit (starting the workers — 2.4-9, the first second of a
                        # batch) to the last result. Return whatever completed rather than
                        # propagating, so the caller can report "X/N completed, cancelled" (see
                        # batch_cmd.py/export_cmd.py) instead of every already-finished result being
                        # lost to a bare KeyboardInterrupt traceback — the same "don't discard good
                        # results" principle this function already applies to BrokenProcessPool.
                        # Nothing new starts (cancel_futures); the workers ignored the signal, so the
                        # frames they are on finish and are counted (_finish_in_flight).
                        # The processes, first: a non-waiting shutdown forgets them, and the `with`'s
                        # own shutdown(wait=True) then has nothing left to wait for.
                        stopped = list((getattr(executor, "_processes", None) or {}).values())
                        executor.shutdown(wait=False, cancel_futures=True)
                        _finish_in_flight(in_flight, record, stopped, on_cancel)
                        return results
            except KeyboardInterrupt:
                return results  # before the pool existed: nothing started, nothing to stop
    finally:
        # Only once every worker is gone is nothing using a frame, and the sweep safe.
        _wait_for_exit(stopped)
        if shm_prefix is not None:
            from halide.shared_frames import sweep

            sweep(shm_prefix)

    return results


# How long the frames in progress get to finish after a cancel before their workers are
# terminated. The workers ignore Ctrl-C (the parent owns cancelling), so each finishes the frame it
# is on — normally a few seconds; this bounds one that is stuck.
_INTERRUPT_GRACE = 10.0

# What a caller's `on_cancel` is told while those frames finish (the CLI prints it).
FINISHING_NOTICE = "Finishing the frames in progress — Ctrl-C again to stop now"


def _finished(future, job: BatchJob) -> BatchResult:
    try:
        return future.result()
    except BrokenProcessPool:
        return _crashed_result(job)


def _crashed_result(job: BatchJob) -> BatchResult:
    return BatchResult(
        job=job,
        error=(
            "worker process crashed while processing this file or another file in the "
            "same batch (often caused by running out of memory) — try re-running with "
            "a lower --workers value"
        ),
    )


def _finish_in_flight(in_flight: dict, record: Callable[[BatchResult], None], processes: list,
                      on_cancel: Callable[[str], None] | None = None) -> None:
    """After a cancel: record the frames still in progress as they finish, for up to
    _INTERRUPT_GRACE, so "X/N frames processed" counts every frame written. A second Ctrl-C (or
    SIGHUP/SIGTERM) stops waiting: the workers are terminated at once. `on_cancel(FINISHING_NOTICE)`
    is called first when there is anything to wait for, so the wait isn't silent."""
    try:
        if in_flight and on_cancel:
            on_cancel(FINISHING_NOTICE)
        done, not_done = wait(in_flight, timeout=_INTERRUPT_GRACE)
    except KeyboardInterrupt:
        _terminate(processes)
        return
    for future in done:
        if not future.cancelled():
            record(_finished(future, in_flight.pop(future)))
    if not_done:
        _terminate(processes)  # stuck: stop waiting for them


def _terminate(processes: list) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()


def _wait_for_exit(processes: list) -> None:
    """Wait for `processes` to exit — terminating any still there after _INTERRUPT_GRACE, or at
    once on a second Ctrl-C (or SIGHUP/SIGTERM) meanwhile."""
    deadline = time.monotonic() + _INTERRUPT_GRACE
    try:
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    _terminate(processes)
    for process in processes:
        process.join()


@contextlib.contextmanager
def _compute_for(compute: BatchCompute | None, jobs: list[BatchJob], device: ComputeDevice | None,
                 workload: str) -> Iterator[BatchCompute]:
    """The caller's BatchCompute (the CLI starts one before its run sheet closes, so the sheet can
    say how the GPU is used), or one started just for this run."""
    if compute is not None:
        yield compute
    else:
        with batch_compute(jobs, device, workload) as started:
            yield started


def run_batch(
    jobs: list[BatchJob],
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH,
    device: ComputeDevice | None = None,
    compute: BatchCompute | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    """Invert every job in a process pool, in parallel — see _run_pool for the shared failure-
    handling and progress-callback behavior. `max_workers` defaults to default_worker_count(jobs,
    device=device) if not given (service_worker_count in GPU service mode). Jobs with a
    thumbnail_path also get a contact-sheet thumbnail.

    `device` is the run's already-resolved device (None = CPU). On a GPU the frames are developed by
    one shared GPU service (see batch_compute; `compute`, if given, is one the caller already
    started), or — when the service can't be used — by each worker on a CUDA context of its own:
    those workers are told only the device's kind and their share of its memory, and resolve it
    themselves (see _worker_device). A frame a worker had to develop on the CPU instead says so in
    its BatchResult.warning."""
    with _compute_for(compute, jobs, device, "develop") as compute:
        workers = max_workers if max_workers is not None else _default_count(jobs, compute, default_worker_count)
        return _run_pool(
            jobs, _worker, (stage, density_profile, tone_params, thumbnail_long_edge, *compute.worker_args(workers)),
            workers, on_result=on_result, on_start=on_start, shm_prefix=compute.shm_prefix, on_cancel=on_cancel,
        )


def _default_count(jobs: list[BatchJob], compute: BatchCompute, default: Callable[..., int]) -> int:
    """The default worker count for this run's mode: `default(jobs, device=...)` (CPU or per-worker
    GPU), or service_worker_count in service mode."""
    if compute.service is not None:
        return service_worker_count(jobs, compute)
    return default(jobs, device=compute.device)


def run_export_batch(
    jobs: list[BatchJob],
    quality: int = 95,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    device: ComputeDevice | None = None,
    compute: BatchCompute | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    """Export every job (ACEScg TIFF -> delivery PNG/JPEG) in a process pool, in parallel — see
    _run_pool for the shared failure-handling and progress-callback behavior. `max_workers`
    defaults to default_export_worker_count(jobs, device=device) if not given; `device` and
    `compute` as run_batch."""
    with _compute_for(compute, jobs, device, "export") as compute:
        workers = max_workers if max_workers is not None else _default_count(jobs, compute, default_export_worker_count)
        return _run_pool(
            jobs, _export_worker, (quality, *compute.worker_args(workers)), workers,
            on_result=on_result, on_start=on_start, shm_prefix=compute.shm_prefix, on_cancel=on_cancel,
        )


def run_print_batch(
    jobs: list[BatchJob],
    tone_params: ToneCurveParams,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    device: ComputeDevice | None = None,
    compute: BatchCompute | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    """`halide print` every job (flat positive -> print) in a process pool — see _run_pool for the
    shared failure-handling and progress-callback behavior. `max_workers` defaults to
    default_worker_count(jobs, device=device): the full-pipeline memory estimate, a safe upper
    bound for the print stage alone (which is the tail of that same pipeline); `device` and
    `compute` as run_batch."""
    with _compute_for(compute, jobs, device, "develop") as compute:
        workers = max_workers if max_workers is not None else _default_count(jobs, compute, default_worker_count)
        return _run_pool(
            jobs, _print_worker, (tone_params, *compute.worker_args(workers)), workers,
            on_result=on_result, on_start=on_start, shm_prefix=compute.shm_prefix, on_cancel=on_cancel,
        )


def run_thumbnail_batch(
    jobs: list[BatchJob],
    thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    on_cancel: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    """Contact-sheet thumbnails of already-processed files (`halide contact`), in a process pool.
    Sized with export's constants: reading + colour-converting one full-size file is the same
    shape of work as an export, and the thumbnail itself is tiny. Always on the CPU: the frames are
    already developed, and what's left — decode, (usually no) colour conversion, a block-average
    down to thumbnail size — would spend longer uploading the frame than computing on it."""
    workers = max_workers if max_workers is not None else default_export_worker_count(jobs)
    return _run_pool(jobs, _thumbnail_worker, (thumbnail_long_edge,), workers, on_result=on_result, on_start=on_start,
                     on_cancel=on_cancel)
