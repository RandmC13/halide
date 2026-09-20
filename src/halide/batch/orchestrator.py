"""Batch processing: discover files in a directory and process them in parallel via a process
pool, reusing halide.processing.process_scan per file.

This module has no terminal/UI concerns — it reports progress via a plain callback so the CLI
layer (or a future GUI) can render it however it likes, or not at all. It also has no calibration
*policy* — the caller resolves a single shared DensityProfile (or None, meaning "each worker
computes its own per-frame automatic profile") before calling run_batch.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tifffile

from halide.core.types import DensityProfile, ToneCurveParams
from halide.processing import Stage, process_scan

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


@dataclass(frozen=True)
class BatchJob:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class BatchResult:
    job: BatchJob
    error: str | None  # None on success


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


def estimate_worker_memory_bytes(jobs: list[BatchJob]) -> int:
    """Estimate the peak RSS a single worker needs to process the largest job in this batch (see
    the module-level constants' docstring for how the estimate itself was derived)."""
    sizes = [b for b in (_decoded_pixel_bytes(job.input_path) for job in jobs) if b is not None]
    if not sizes:
        return _FALLBACK_PER_WORKER_BYTES
    return _BASELINE_PROCESS_OVERHEAD_BYTES + max(sizes) * _PEAK_RSS_MULTIPLIER


def default_worker_count(jobs: list[BatchJob]) -> int:
    """Auto-select a worker count that respects available RAM, not just CPU count. Found via real
    testing: full-resolution scans are memory-heavy enough (a single frame's pipeline run can peak
    around 4GB RSS) that on a typical machine, available memory — not CPU thread count — is the
    actual limiting factor; running out of --workers-only-capped processes was still enough to get
    a worker OOM-killed. Falls back to the old CPU-only heuristic if `psutil` can't report available
    memory, or if none of the batch's files' headers could be read at all."""
    cpu_cap = min(os.cpu_count() or 1, 6)
    if not jobs:
        return cpu_cap
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return cpu_cap
    per_worker = estimate_worker_memory_bytes(jobs)
    memory_cap = max(1, available // per_worker)
    return max(1, min(cpu_cap, memory_cap, len(jobs)))


def memory_budget_warning(jobs: list[BatchJob], requested_workers: int) -> str | None:
    """Returns a human-readable warning if an explicitly-requested worker count looks likely to
    exceed available memory, or None if it looks safe (or memory couldn't be checked at all). Pure
    computation only, matching this module's no-UI-concerns design (see module docstring) — the
    caller decides whether/how to display it."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 — an unsupported platform must not break batch processing
        return None
    per_worker = estimate_worker_memory_bytes(jobs)
    safe_workers = max(1, available // per_worker)
    if requested_workers <= safe_workers:
        return None
    return (
        f"Warning: --workers {requested_workers} may exceed available memory (~{per_worker / 1024**3:.1f} "
        f"GB estimated per worker vs. ~{available / 1024**3:.1f} GB available suggests {safe_workers} "
        f"worker(s) is safer) — continuing with {requested_workers} since it was explicitly requested."
    )


def _worker(
    job: BatchJob,
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
) -> BatchResult:
    try:
        process_scan(job.input_path, job.output_path, stage, density_profile, tone_params)
        return BatchResult(job=job, error=None)
    except Exception as exc:  # noqa: BLE001 — one frame's failure must not take down the batch
        return BatchResult(job=job, error=str(exc))


def run_batch(
    jobs: list[BatchJob],
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
    max_workers: int | None = None,
    on_result: Callable[[BatchResult], None] | None = None,
) -> list[BatchResult]:
    """Process every job in a process pool, in parallel. Calls on_result(result) as each job
    completes, if given — purely for progress reporting, this function has no rendering logic
    of its own. A single job's failure is captured in its BatchResult, not raised — the rest of
    the batch keeps running.

    That guarantee covers exceptions raised *inside* a worker (caught by _worker's own try/except).
    A worker process dying outright — OOM-killed, segfault, anything that kills it before its own
    try/except can run — is a different failure mode: found via real testing on full-resolution
    scans (a single frame's pipeline run can peak around 4GB RSS; enough parallel workers on a
    memory-constrained machine gets one OOM-killed). Once that happens, ProcessPoolExecutor marks
    the whole pool broken and every *other* pending future raises BrokenProcessPool too — without
    the handling below, that would propagate straight out of this function, discarding every
    already-completed BatchResult and surfacing nothing but a bare traceback that gives no hint
    what happened or which file was involved. Instead, each future that raises BrokenProcessPool is
    recorded as its own failed BatchResult with an actionable message, exactly like a per-file
    processing error — the rest of the batch's real results are preserved either way."""
    if max_workers is not None and max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")

    # Restrict underlying BLAS/OpenMP threading per worker process to avoid CPU oversubscription
    # when running several worker *processes* in parallel, each of which would otherwise also try
    # to multithread its own numpy/colour-science operations.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    workers = max_workers if max_workers is not None else default_worker_count(jobs)
    results: list[BatchResult] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, job, stage, density_profile, tone_params): job for job in jobs
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except BrokenProcessPool:
                job = futures[future]
                result = BatchResult(
                    job=job,
                    error=(
                        "worker process crashed while processing this file or another file in the "
                        "same batch (often caused by running out of memory) — try re-running with "
                        "a lower --workers value"
                    ),
                )
            results.append(result)
            if on_result:
                on_result(result)

    return results
