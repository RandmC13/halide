"""Run-sheet rows shared by the multi-frame commands (`batch`, `export`, `print`) — see
console.RunSheet for the block itself."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from halide.batch.orchestrator import BatchJob
from halide.cli._device_args import device_fallback_warning, device_row
from halide.cli.console import RunSheet
from halide.device import ComputeDevice


def frame_count(n: int) -> str:
    return f"{n} frame" if n == 1 else f"{n} frames"


def roll_row(sheet: RunSheet, input_dir: Path, n_frames: int, destination: str) -> None:
    # resolve(): `halide batch .` would otherwise name the roll "" (Path(".").name is empty).
    sheet.row("Roll", f"{input_dir.resolve().name}{RunSheet.SEP}{frame_count(n_frames)} → {destination}")


def compute_row(sheet: RunSheet, device: ComputeDevice) -> None:
    """The run sheet's Compute row (see cli/_device_args.py::device_row) plus, when set, why an
    `auto` choice fell back from GPU to CPU."""
    sheet.row("Compute", device_row(device))
    warning = device_fallback_warning(device)
    if warning:
        sheet.warn("Compute", warning)


def choose_workers(
    args: argparse.Namespace,
    jobs: list[BatchJob],
    sheet: RunSheet,
    *,
    default_count: Callable[[list[BatchJob]], int],
    budget_warning: Callable[[list[BatchJob], int], str | None],
) -> int:
    """An explicit --workers N wins (with a warning if it looks too many for free memory);
    otherwise the memory-aware default. Either way the choice goes on the sheet."""
    if args.workers is not None:
        sheet.row("Workers", f"{args.workers} (--workers)")
        warning = budget_warning(jobs, args.workers)
        if warning:
            sheet.warn("Workers", warning)
        return args.workers
    workers = default_count(jobs)
    sheet.row("Workers", f"{workers}{RunSheet.SEP}auto-selected from free memory and CPU cores "
                         "(--workers N overrides)")
    return workers
