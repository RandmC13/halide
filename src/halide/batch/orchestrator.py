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

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tifffile

from halide.core.types import DensityProfile, ToneCurveParams
from halide.processing import Stage, export_delivery_image, print_scan, process_scan

TIFF_SUFFIXES = (".tif", ".tiff")

# Empirically measured (three real full-res scans + one small synthetic image, all run through the
# real CLI with peak RSS sampled from /proc/<pid>/status's VmHWM): peak worker RSS fits closely to
# `baseline + K * decoded_pixel_bytes`, K ~= 9.7, baseline ~= 108 MiB, after the memory-usage pass
# that fixed the two real causes of the old K ~= 23 (see git history around this comment and
# CLAUDE.md for the investigation): io/icc.py's working-space conversion was silently upcasting
# every pixel from float32 to float64 for the rest of the pipeline (a straight 2x on its own), and
# core/density.py, core/invert.py, core/tone_render.py, and io/lut.py's chain of elementwise numpy
# ops was allocating a fresh full-size temporary at nearly every line instead of reusing buffers via
# `out=`/in-place ops. Both constants below are rounded up from the fit for safety margin.
_PEAK_RSS_MULTIPLIER = 11
_BASELINE_PROCESS_OVERHEAD_BYTES = 150 * 1024 * 1024
_FALLBACK_PER_WORKER_BYTES = 5 * 1024**3  # used only if a file's header can't be read at all

# Same methodology as above, fit from a real 3276x4849 scan (~182 MiB decoded -> ~2.61 GiB peak)
# and a small 500x500 synthetic image (~2.9 MiB decoded -> ~133 MiB peak) run through
# halide.processing.export_delivery_image specifically: K ~= 13.8, baseline ~= 93 MiB (rounded up
# below for safety margin). Despite doing far less math than the full inversion pipeline, export's
# K is not that much smaller — colour.RGB_to_RGB (ACEScg -> sRGB) computes internally in float64
# regardless of input dtype (same behavior noted for XYZ_to_RGB elsewhere in this codebase), so a
# single delivery conversion still holds several float64-sized temporaries of the full image alive
# at once, not just the small 8-bit output it ultimately writes.
_EXPORT_PEAK_RSS_MULTIPLIER = 15
_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class BatchJob:
    input_path: Path
    output_path: Path
    scan_gain: float = 1.0  # see processing.process_scan / --match-scan-exposure


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


def default_worker_count(
    jobs: list[BatchJob],
    *,
    baseline_bytes: int = _BASELINE_PROCESS_OVERHEAD_BYTES,
    multiplier: int = _PEAK_RSS_MULTIPLIER,
) -> int:
    """Auto-select a worker count that respects available RAM, not just CPU count. Found via real
    testing: full-resolution scans are memory-heavy enough (a single frame's pipeline run can peak
    around 4GB RSS) that on a typical machine, available memory — not CPU thread count — is the
    actual limiting factor; running out of --workers-only-capped processes was still enough to get
    a worker OOM-killed. Falls back to the old CPU-only heuristic if `psutil` can't report available
    memory, or if none of the batch's files' headers could be read at all. `baseline_bytes`/
    `multiplier` default to the full-inversion-pipeline fit — see estimate_worker_memory_bytes."""
    cpu_cap = min(os.cpu_count() or 1, 6)
    if not jobs:
        return cpu_cap
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return cpu_cap
    per_worker = estimate_worker_memory_bytes(jobs, baseline_bytes=baseline_bytes, multiplier=multiplier)
    memory_cap = max(1, available // per_worker)
    return max(1, min(cpu_cap, memory_cap, len(jobs)))


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
        f"Warning: --workers {requested_workers} may exceed available memory (~{per_worker / 1024**3:.1f} "
        f"GB estimated per worker vs. ~{available / 1024**3:.1f} GB available suggests {safe_workers} "
        f"worker(s) is safer) — continuing with {requested_workers} since it was explicitly requested."
    )


def estimate_export_worker_memory_bytes(jobs: list[BatchJob]) -> int:
    """Same as estimate_worker_memory_bytes, using export's own calibrated constants."""
    return estimate_worker_memory_bytes(
        jobs, baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES, multiplier=_EXPORT_PEAK_RSS_MULTIPLIER
    )


def default_export_worker_count(jobs: list[BatchJob]) -> int:
    """Same as default_worker_count, using export's own calibrated constants."""
    return default_worker_count(
        jobs, baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES, multiplier=_EXPORT_PEAK_RSS_MULTIPLIER
    )


def export_memory_budget_warning(jobs: list[BatchJob], requested_workers: int) -> str | None:
    """Same as memory_budget_warning, using export's own calibrated constants."""
    return memory_budget_warning(
        jobs,
        requested_workers,
        baseline_bytes=_EXPORT_BASELINE_PROCESS_OVERHEAD_BYTES,
        multiplier=_EXPORT_PEAK_RSS_MULTIPLIER,
    )


def _worker(
    job: BatchJob,
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
) -> BatchResult:
    try:
        process_scan(job.input_path, job.output_path, stage, density_profile, tone_params, scan_gain=job.scan_gain)
        return BatchResult(job=job, error=None)
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc))


def _export_worker(job: BatchJob, quality: int) -> BatchResult:
    try:
        warning = export_delivery_image(job.input_path, job.output_path, quality=quality)
        return BatchResult(job=job, error=None, warning=warning)
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc))


def _print_worker(job: BatchJob, tone_params: ToneCurveParams) -> BatchResult:
    try:
        _, warning = print_scan(job.input_path, job.output_path, tone_params)
        return BatchResult(job=job, error=None, warning=warning)
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc))


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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
) -> list[BatchResult]:
    """Invert every job in a process pool, in parallel — see _run_pool for the shared failure-
    handling and progress-callback behavior. `max_workers` defaults to default_worker_count(jobs)
    if not given."""
    workers = max_workers if max_workers is not None else default_worker_count(jobs)
    return _run_pool(
        jobs, _worker, (stage, density_profile, tone_params), workers, on_result=on_result, on_start=on_start
    )


def run_export_batch(
    jobs: list[BatchJob],
    quality: int = 95,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
) -> list[BatchResult]:
    """Export every job (ACEScg TIFF -> delivery PNG/JPEG) in a process pool, in parallel — see
    _run_pool for the shared failure-handling and progress-callback behavior. `max_workers`
    defaults to default_export_worker_count(jobs) if not given."""
    workers = max_workers if max_workers is not None else default_export_worker_count(jobs)
    return _run_pool(jobs, _export_worker, (quality,), workers, on_result=on_result, on_start=on_start)


def run_print_batch(
    jobs: list[BatchJob],
    tone_params: ToneCurveParams,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
    on_start: Callable[[BatchJob], None] | None = None,
) -> list[BatchResult]:
    """`halide print` every job (flat positive -> print) in a process pool — see _run_pool for the
    shared failure-handling and progress-callback behavior. `max_workers` defaults to
    default_worker_count(jobs): the full-pipeline memory estimate, a safe upper bound for the print
    stage alone (which is the tail of that same pipeline)."""
    workers = max_workers if max_workers is not None else default_worker_count(jobs)
    return _run_pool(jobs, _print_worker, (tone_params,), workers, on_result=on_result, on_start=on_start)
