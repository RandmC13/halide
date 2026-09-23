"""`halide contact` — a high-resolution contact sheet of an already-processed folder: halide's own
TIFF output (from invert/batch/print) or `halide export`'s PNG/JPEG files. Make one per set of
settings and compare them side by side."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from halide.batch.orchestrator import (
    TIFF_SUFFIXES,
    BatchJob,
    default_export_worker_count,
    export_memory_budget_warning,
    run_thumbnail_batch,
)
from halide.batch.progress import GridProgressRenderer
from halide.cli import console
from halide.cli._contact_sheet import add_contact_layout_arguments, write_contact_sheet
from halide.io.contact_sheet import check_sheet_path, is_contact_sheet

_SOURCE_SUFFIXES = TIFF_SUFFIXES + (".png", ".jpg", ".jpeg")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs", nargs="+",
        help="A folder of processed frames (TIFF output of invert/batch/print, or PNG/JPEG from "
        "export), or individual files, followed by the sheet to write",
    )
    add_contact_layout_arguments(parser)
    parser.add_argument("--workers", type=int, help="Number of parallel worker processes (default: auto-selected)")
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")


def _collect(inputs: list[str], sheet: Path) -> list[Path]:
    files: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            files.extend(sorted(f for f in item.iterdir() if f.is_file() and f.suffix.lower() in _SOURCE_SUFFIXES))
        elif item.exists():
            files.append(item)
        else:
            raise SystemExit(f"not found: {item}")
    # A sheet written into the folder it proofs — this one, or an earlier one — must not end up on
    # the next sheet of that folder as if it were a frame.
    return [
        f for f in files
        if f.resolve() != sheet.resolve() and not (f.suffix.lower() not in TIFF_SUFFIXES and is_contact_sheet(f))
    ]


def run(args: argparse.Namespace) -> int:
    if len(args.inputs) < 2:
        raise SystemExit("usage: halide contact <folder-or-files...> <sheet.jpg|.png>")
    sheet_path = Path(args.inputs[-1])
    try:
        check_sheet_path(sheet_path)
    except ValueError as exc:
        raise SystemExit(str(exc))
    files = _collect(args.inputs[:-1], sheet_path)
    if not files:
        print("No processed frames (TIFF/PNG/JPEG) found")
        return 1
    if sheet_path.exists() and not console.confirm_overwrite(sheet_path):
        return 1

    first = Path(args.inputs[0])
    default_title = first.name if first.is_dir() else first.parent.name or "contact sheet"
    tmp = Path(tempfile.mkdtemp(prefix="halide-contact-"))
    try:
        jobs = [
            BatchJob(input_path=f, output_path=None, thumbnail_path=tmp / f"{i:04d}.png")
            for i, f in enumerate(files)
        ]
        if args.workers is not None:
            workers = args.workers
            warning = export_memory_budget_warning(jobs, workers)
            if warning and not args.quiet:
                print(console.warning(warning))
        else:
            workers = default_export_worker_count(jobs)

        renderer = None if args.quiet else GridProgressRenderer(total=len(jobs), verb="proof")
        job_index = {job: i for i, job in enumerate(jobs)}
        if renderer:
            renderer.start()
        results = run_thumbnail_batch(
            jobs, thumbnail_long_edge=args.frame_width, max_workers=workers,
            on_start=(lambda job: renderer.mark_processing(job_index[job])) if renderer else None,
            on_result=(lambda r: renderer.report(job_index[r.job], r)) if renderer else None,
        )
        cancelled = len(results) < len(jobs)
        if renderer:
            if cancelled:
                done = {job_index[r.job] for r in results}
                renderer.cancel([i for i in range(len(jobs)) if i not in done])
            renderer.finish(cancelled=cancelled)
        if cancelled:
            return 130
        failures = [r for r in results if r.error]
        if renderer is None:
            for r in failures:
                print(console.error(f"{r.job.input_path.name}: {r.error}"))
        write_contact_sheet(args, jobs, results, sheet_path, default_title)
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
