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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from halide.core.types import DensityProfile, ToneCurveParams
from halide.processing import Stage, process_scan

TIFF_SUFFIXES = (".tif", ".tiff")


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
    the batch keeps running."""
    # Restrict underlying BLAS/OpenMP threading per worker process to avoid CPU oversubscription
    # when running several worker *processes* in parallel, each of which would otherwise also try
    # to multithread its own numpy/colour-science operations.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    workers = max_workers or min(os.cpu_count() or 1, 6)
    results: list[BatchResult] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, job, stage, density_profile, tone_params): job for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result:
                on_result(result)

    return results
