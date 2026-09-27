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

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from halide.core.types import DensityProfile, Stage, ToneCurveParams
from halide.device import ComputeDevice
from halide.io.contact_sheet_defaults import DEFAULT_FRAME_WIDTH

# tifffile and halide.processing (numpy, Pillow, colour-science) are imported inside the functions
# that use them: every CLI command imports this module, and `halide --help` shouldn't pay for them.
# Workers don't pay either — the forkserver preloads halide.processing (_FORKSERVER_PRELOAD).

TIFF_SUFFIXES = (".tif", ".tiff")

# Peak worker memory fits `baseline + K * decoded_pixel_bytes`. Measured on real worker processes
# (`--workers 1` batches over the four real full-res scans, 3276x4849 = ~182 MiB decoded, peak
# sampled from /proc/<pid>/status's VmHWM): worst case `--auto-density` 509 MiB RSS (481 PSS),
# `--output flat` 487, print output 365, `halide print` 386 — K ~= 2.2 over a ~106 MiB process.
# K = 3 with the 150 MiB baseline (695 MiB for such a scan) leaves ~37% margin over the worst case.
# exiftool (run after every output) needs no term of its own: it streams the file (measured 67 MiB
# peak on a 130 MiB output), and process_scan frees the frame before running it.
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

# GPU workers (docs/plans/gpu-acceleration.md §3.5): peak *device* memory per worker, modelled like
# RAM as `context + K * decoded_pixel_bytes`. PROVISIONAL — not yet measured on a real card; refit
# from `docs/plans/gpu-acceleration-bench.py`'s memory section on the user's RTX 3070. Evidence so
# far, and why each is rounded up (a too-high estimate costs a worker; too low costs frames redone
# on the CPU after running out of VRAM):
#   - a CUDA context is ~300 MB (an estimate, not measured here) -> 384 MiB;
#   - the banded device path (banding.DEVICE_BAND_BYTES = 64 MiB) keeps the frame resident and
#     needs a few bands' temporaries at once — the plan's §3.3 budget is ~6 x 64 MiB -> 384 MiB;
#   - the frame itself (181 MiB for a real scan) plus the print fit's whole-frame luminance and
#     percentile scratch (~1/3 frame each, sorted copies included): ~2 frames. K = 4 doubles that.
#     Upper bound: the probe's *unbanded* whole-frame run held 2.1 GiB of pool on a 181 MiB frame
#     (~12 frames); banded, this estimate (768 MiB + 4 x 181 MiB = ~1.5 GiB) is ~70% of that.
#   - per-frame auto calibration (calibration/auto.py, on the device since Task 8) is the peak for
#     `--auto-density`: the luminance argsort holds its keys (1/3 frame) and int64 order (2/3
#     frame) plus the sort's own scratch, then the candidate mask/boolean index and percentile
#     sorts ~1/2 frame each — ~2-3 frames beside the resident one, not banded (whole-frame
#     statistics). Within K = 4, but the least-margin case; measure it when refitting.
# On the RTX 3070 (~6.8 GiB free on the desktop) that is 4 workers.
_CUDA_CONTEXT_BYTES = 384 * 1024 * 1024
_DEVICE_BAND_SCRATCH_BYTES = 6 * 64 * 1024 * 1024
_DEVICE_CONTEXT_BYTES = _CUDA_CONTEXT_BYTES + _DEVICE_BAND_SCRATCH_BYTES
_DEVICE_FRAME_MULTIPLIER = 4


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
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES
    )
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
            page = tf.pages[0]
            shape = page.shape
            itemsize = page.dtype.itemsize
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
) -> int:
    """Estimate the peak RSS a single worker needs to process the largest job in this batch (see
    the module-level constants' docstring for how the estimate itself was derived). `baseline_bytes`
    /`multiplier` default to the full-inversion-pipeline fit; pass export's own calibrated constants
    (or use estimate_export_worker_memory_bytes) when sizing a pool of export workers instead."""
    sizes = [b for b in (_decoded_pixel_bytes(job.input_path) for job in jobs) if b is not None]
    if not sizes:
        return _FALLBACK_PER_WORKER_BYTES
    return baseline_bytes + max(sizes) * multiplier


def estimate_worker_device_bytes(jobs: list[BatchJob]) -> int:
    """Estimate the peak GPU memory one worker needs for the largest job in this batch: its own
    CUDA context and band scratch, plus the frame resident on the device and the print fit's
    whole-frame statistics (see the PROVISIONAL constants above). Used for every GPU pool — develop,
    print and export alike: export keeps the same resident frame and less scratch, so this bounds
    it too."""
    sizes = [b for b in (_decoded_pixel_bytes(job.input_path) for job in jobs) if b is not None]
    if not sizes:
        return _FALLBACK_PER_WORKER_BYTES
    return _DEVICE_CONTEXT_BYTES + max(sizes) * _DEVICE_FRAME_MULTIPLIER


def _device_cap(jobs: list[BatchJob], device: ComputeDevice | None) -> int | None:
    """How many GPU workers fit in the card's free memory, or None when there's no GPU (or its free
    memory wasn't reported). Uses the free memory the parent's own resolve_device already measured —
    no further CUDA calls here (the parent's own context is already counted out of it)."""
    if device is None or device.kind != "gpu" or not device.memory_free:
        return None
    return max(1, device.memory_free // estimate_worker_device_bytes(jobs))


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
    device_cap = _device_cap(jobs, device)
    if device_cap is not None:
        cap = min(cap, device_cap)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return cap
    per_worker = estimate_worker_memory_bytes(jobs, baseline_bytes=baseline_bytes, multiplier=multiplier)
    memory_cap = max(1, available // per_worker)
    return max(1, min(cap, memory_cap, len(jobs)))


def memory_budget_warning(
    jobs: list[BatchJob],
    requested_workers: int,
    *,
    baseline_bytes: int = _BASELINE_PROCESS_OVERHEAD_BYTES,
    multiplier: int = _PEAK_RSS_MULTIPLIER,
) -> str | None:
    """Returns a human-readable warning if an explicitly-requested worker count looks likely to
    exceed available memory, or None if it looks safe (or memory couldn't be checked at all). Pure
    computation only, matching this module's no-UI-concerns design (see module docstring) — the
    caller decides whether/how to display it. `baseline_bytes`/`multiplier` default to the
    full-inversion-pipeline fit — see estimate_worker_memory_bytes."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return None
    per_worker = estimate_worker_memory_bytes(jobs, baseline_bytes=baseline_bytes, multiplier=multiplier)
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


def export_memory_budget_warning(jobs: list[BatchJob], requested_workers: int) -> str | None:
    """Same as memory_budget_warning, using export's own calibrated constants."""
    return memory_budget_warning(
        jobs,
        requested_workers,
        baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES,
        multiplier=_EXPORT_PEAK_RSS_MULTIPLIER,
    )


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
) -> BatchResult:
    from halide.processing import process_scan

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "developed this frame")
        process_scan(
            job.input_path, job.output_path, stage, density_profile, tone_params, scan_gain=job.scan_gain,
            thumbnail_path=job.thumbnail_path, thumbnail_long_edge=thumbnail_long_edge,
            device=device, on_warning=warnings,
        )
        return BatchResult(job=job, error=None, warning=warnings.text())
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc), warning=warnings.text() if warnings else None)


def _export_worker(
    job: BatchJob, quality: int, device_kind: str = "cpu", device_memory_limit: int | None = None
) -> BatchResult:
    from halide.processing import export_delivery_image

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "exported this file")
        warnings(export_delivery_image(job.input_path, job.output_path, quality=quality,
                                       device=device, on_warning=warnings))
        return BatchResult(job=job, error=None, warning=warnings.text())
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc), warning=warnings.text() if warnings else None)


def _print_worker(
    job: BatchJob, tone_params: ToneCurveParams, device_kind: str = "cpu", device_memory_limit: int | None = None
) -> BatchResult:
    from halide.processing import print_scan

    warnings = None
    try:
        device, warnings = _start_on_device(job, device_kind, device_memory_limit, "developed this frame")
        _, warning = print_scan(job.input_path, job.output_path, tone_params, device=device, on_warning=warnings)
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
    between a full inversion and a delivery-format export."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")

    # Restrict underlying BLAS/OpenMP threading per worker process to avoid CPU oversubscription
    # when running several worker *processes* in parallel, each of which would otherwise also try
    # to multithread its own numpy/colour-science operations.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    results: list[BatchResult] = []

    def _crashed_result(job: BatchJob) -> BatchResult:
        return BatchResult(
            job=job,
            error=(
                "worker process crashed while processing this file or another file in the "
                "same batch (often caused by running out of memory) — try re-running with "
                "a lower --workers value"
            ),
        )

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_pool_context()) as executor:
        pending = list(jobs)
        in_flight: dict = {}

        def submit_next() -> None:
            # Loops rather than submitting exactly one job, so that once the pool is broken every
            # remaining queued job is immediately resolved as a crashed result instead of being
            # submitted (and raising) one at a time as later callers happen to invoke this again.
            while pending:
                job = pending.pop(0)
                try:
                    future = executor.submit(worker, job, *worker_args)
                except BrokenProcessPool:
                    result = _crashed_result(job)
                    results.append(result)
                    if on_result:
                        on_result(result)
                    continue
                in_flight[future] = job
                if on_start:
                    on_start(job)
                break

        for _ in range(min(max_workers, len(jobs))):
            submit_next()

        try:
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    job = in_flight.pop(future)
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        result = _crashed_result(job)
                    results.append(result)
                    if on_result:
                        on_result(result)
                    submit_next()
        except KeyboardInterrupt:
            # Ctrl+C can land on any bytecode boundary in this loop, not just inside wait() — the
            # try wraps the whole loop body rather than just the blocking call. Workers in the same
            # process group receive SIGINT directly too and will exit on their own; cancel_futures
            # drops anything not yet started rather than waiting for a full drain. Return whatever
            # completed rather than propagating, so the caller can report "X/N completed, cancelled"
            # (see batch_cmd.py/export_cmd.py) instead of every already-finished result being lost
            # to a bare KeyboardInterrupt traceback — the same "don't discard good results" principle
            # this function already applies to BrokenProcessPool above.
            executor.shutdown(wait=False, cancel_futures=True)
            return results

    return results


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
) -> list[BatchResult]:
    """Invert every job in a process pool, in parallel — see _run_pool for the shared failure-
    handling and progress-callback behavior. `max_workers` defaults to default_worker_count(jobs,
    device=device) if not given. Jobs with a thumbnail_path also get a contact-sheet thumbnail.

    `device` is the run's already-resolved device (None = CPU). Workers are told only its kind and
    their share of its memory, and resolve it themselves (see _worker_device); a frame a worker had
    to develop on the CPU instead says so in its BatchResult.warning."""
    workers = max_workers if max_workers is not None else default_worker_count(jobs, device=device)
    return _run_pool(
        jobs, _worker,
        (stage, density_profile, tone_params, thumbnail_long_edge, *_device_worker_args(device, workers)),
        workers, on_result=on_result, on_start=on_start,
    )


def run_export_batch(
    jobs: list[BatchJob],
    quality: int = 95,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    device: ComputeDevice | None = None,
) -> list[BatchResult]:
    """Export every job (ACEScg TIFF -> delivery PNG/JPEG) in a process pool, in parallel — see
    _run_pool for the shared failure-handling and progress-callback behavior. `max_workers`
    defaults to default_export_worker_count(jobs, device=device) if not given; `device` as
    run_batch."""
    workers = max_workers if max_workers is not None else default_export_worker_count(jobs, device=device)
    return _run_pool(
        jobs, _export_worker, (quality, *_device_worker_args(device, workers)), workers,
        on_result=on_result, on_start=on_start,
    )


def run_print_batch(
    jobs: list[BatchJob],
    tone_params: ToneCurveParams,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
    device: ComputeDevice | None = None,
) -> list[BatchResult]:
    """`halide print` every job (flat positive -> print) in a process pool — see _run_pool for the
    shared failure-handling and progress-callback behavior. `max_workers` defaults to
    default_worker_count(jobs, device=device): the full-pipeline memory estimate, a safe upper
    bound for the print stage alone (which is the tail of that same pipeline); `device` as
    run_batch."""
    workers = max_workers if max_workers is not None else default_worker_count(jobs, device=device)
    return _run_pool(
        jobs, _print_worker, (tone_params, *_device_worker_args(device, workers)), workers,
        on_result=on_result, on_start=on_start,
    )


def run_thumbnail_batch(
    jobs: list[BatchJob],
    thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
) -> list[BatchResult]:
    """Contact-sheet thumbnails of already-processed files (`halide contact`), in a process pool.
    Sized with export's constants: reading + colour-converting one full-size file is the same
    shape of work as an export, and the thumbnail itself is tiny. Always on the CPU: the frames are
    already developed, and what's left — decode, (usually no) colour conversion, a block-average
    down to thumbnail size — would spend longer uploading the frame than computing on it."""
    workers = max_workers if max_workers is not None else default_export_worker_count(jobs)
    return _run_pool(jobs, _thumbnail_worker, (thumbnail_long_edge,), workers, on_result=on_result, on_start=on_start)
