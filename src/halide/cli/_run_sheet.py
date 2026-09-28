"""Run-sheet rows shared by the multi-frame commands (`batch`, `export`, `print`) — see
console.RunSheet for the block itself."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Callable

from halide.batch.orchestrator import (
    BatchCompute,
    BatchJob,
    batch_compute,
    device_budget_warning,
    device_worker_cap,
    service_budget_warnings,
    service_worker_count,
)
from halide.cli._device_args import device_fallback_warning, device_row, gpu_hint
from halide.cli.console import RunSheet
from halide.device import ComputeDevice


def frame_count(n: int) -> str:
    return f"{n} frame" if n == 1 else f"{n} frames"


def roll_row(sheet: RunSheet, input_dir: Path, n_frames: int, destination: str) -> None:
    # resolve(): `halide batch .` would otherwise name the roll "" (Path(".").name is empty).
    sheet.row("Roll", f"{input_dir.resolve().name}{RunSheet.SEP}{frame_count(n_frames)} → {destination}")


def skipped_row(sheet: RunSheet, skipped) -> None:
    """One row for everything a folder listing left out (halide.io.roll.Skipped) — never a line per file."""
    if skipped:
        sheet.row("Skipped", skipped.describe())


def start_compute(stack: contextlib.ExitStack, sheet: RunSheet, jobs: list[BatchJob], device: ComputeDevice,
                  workload: str = "develop") -> BatchCompute:
    """How the batch reaches the device (batch/orchestrator.py::batch_compute) — on a GPU, the
    shared GPU service, started now so the sheet can say whether it's in use, and kept running on
    `stack` until the caller's pool is done. A spinner row while it starts (a few seconds: a fresh
    Python, CuPy, a CUDA context)."""
    if device.kind != "gpu":
        return stack.enter_context(batch_compute(jobs, device, workload))
    with sheet.working("Compute", "starting the GPU service…"):
        return stack.enter_context(batch_compute(jobs, device, workload))


def compute_row(sheet: RunSheet, device: ComputeDevice, compute: BatchCompute | None = None) -> None:
    """The run sheet's Compute row (see cli/_device_args.py::device_row), plus, when there's
    something to add: how a GPU batch's workers share the card (`compute`), why an `auto` choice
    fell back from GPU to CPU, or (Task 6b) that an NVIDIA card was found but GPU support isn't
    installed."""
    text = device_row(device)
    if compute is not None and compute.mode == "service":
        text = f"{text}{RunSheet.SEP}one GPU process shared by all workers"
    elif compute is not None and compute.mode == "per_worker":
        text = f"{text}{RunSheet.SEP}one GPU context per worker"
    hint = gpu_hint(device)
    if hint:
        text = f"{text} — {hint}"
    sheet.row("Compute", text)
    warning = device_fallback_warning(device)
    if warning:
        sheet.warn("Compute", warning)
    if compute is not None and compute.fallback_reason:
        sheet.warn("Compute", f"GPU service not used: {compute.fallback_reason} — each worker uses the GPU "
                              f"itself instead, which takes more memory per worker")


def choose_workers(
    args: argparse.Namespace,
    jobs: list[BatchJob],
    sheet: RunSheet,
    *,
    default_count: Callable[[list[BatchJob]], int],
    budget_warning: Callable[..., str | None],  # (jobs, workers, device=) -> warning or None
    device: ComputeDevice | None = None,
    compute: BatchCompute | None = None,
) -> int:
    """An explicit --workers N wins (with a warning if it looks too many for free memory, or on a
    GPU for the card's free memory); otherwise the memory-aware default. Either way the choice goes
    on the sheet, naming GPU memory when that's what set the count.

    With the shared GPU service (`compute` in service mode) the count is service_worker_count's —
    RAM, cores and /dev/shm; no GPU memory cap, since the card holds one frame whatever the count."""
    if compute is not None and compute.mode == "service":
        return _choose_service_workers(args, jobs, sheet, compute)
    if args.workers is not None:
        sheet.row("Workers", f"{args.workers} (--workers)")
        for warning in (
            budget_warning(jobs, args.workers, device=device),
            device_budget_warning(jobs, args.workers, device),
        ):
            if warning:
                sheet.warn("Workers", warning)
        return args.workers
    workers = default_count(jobs)
    if device_worker_cap(jobs, device) == workers:
        reason = "auto-selected to fit free GPU memory"
    else:
        reason = "auto-selected from free memory and CPU cores"
    sheet.row("Workers", f"{workers}{RunSheet.SEP}{reason} (--workers N overrides)")
    return workers


def _choose_service_workers(args: argparse.Namespace, jobs: list[BatchJob], sheet: RunSheet,
                            compute: BatchCompute) -> int:
    if args.workers is not None:
        sheet.row("Workers", f"{args.workers} (--workers)")
        for warning in service_budget_warnings(jobs, args.workers, compute):
            sheet.warn("Workers", warning)
        return args.workers
    workers = service_worker_count(jobs, compute)
    if compute.shm_cap is not None and workers == compute.shm_cap < len(jobs):
        reason = "auto-selected to fit free shared memory"
    else:
        reason = "auto-selected from free memory and CPU cores"
    sheet.row("Workers", f"{workers}{RunSheet.SEP}{reason} (--workers N overrides)")
    return workers
