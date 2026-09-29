"""Shared contact-sheet plumbing for `halide contact` and `halide batch --contact-sheet`: layout
flags, and turning a finished pool's thumbnails into the written sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.cli import console
from halide.io.contact_sheet_defaults import DEFAULT_COLUMNS, DEFAULT_FRAME_WIDTH
from halide.io.roll import Skipped

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.


def add_contact_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--columns", type=int, default=DEFAULT_COLUMNS, metavar="N",
        help=f"Frames per strip (default: {DEFAULT_COLUMNS})"
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        metavar="PX",
        help=f"Width of each frame on the sheet, in pixels (default: {DEFAULT_FRAME_WIDTH}; a "
        "six-across sheet is then about 6000 px wide)",
    )
    parser.add_argument("--title", help="Title printed at the top of the sheet (default: the folder's name)")


def _common_decision(records: list[dict | None]) -> str:
    """What the frames' recorded printing decisions have in common, for the sheet's header."""
    outputs = {r.get("output") for r in records if r}
    if len(outputs) != 1 or len(records) != sum(1 for r in records if r):
        return ""
    return {"print": "print", "flat": "flat (linear)"}.get(outputs.pop(), "")


def common_film_stock(records: list[dict | None]) -> str | None:
    """The film stock for the sheet's edge print: the one every frame's provenance names, if they
    agree (a folder of mixed rolls gets the generic edge print rather than a wrong stock)."""
    stocks = {r.get("film_stock") for r in records if r}
    return stocks.pop() if len(stocks) == 1 and None not in stocks and len(records) == sum(1 for r in records if r) else None


def write_contact_sheet(
    args: argparse.Namespace,
    jobs: list[BatchJob],
    results: list[BatchResult],
    sheet_path: str | Path,
    default_title: str,
    settings: str = "",
) -> None:
    """Assemble the sheet from every job's thumbnail (in job order — the order of the roll), with a
    'failed' tile for any frame that didn't make it, and write it."""
    from halide.io.contact_sheet import Tile, caption_from_provenance, load_thumbnail, render_sheet, write_sheet
    from halide.processing import _halide_version

    failed = {r.job for r in results if r.error}
    tiles, records = [], []
    for job in jobs:
        name = job.input_path.stem
        if job in failed or job.thumbnail_path is None or not Path(job.thumbnail_path).exists():
            tiles.append(Tile(name=name, image=None))
            continue
        image, record = load_thumbnail(job.thumbnail_path)
        records.append(record)
        tiles.append(Tile(name=name, image=image, caption=caption_from_provenance(record)))

    bits = [console.plural(len(jobs), "frame")]
    if settings:
        bits.append(settings)
    elif _common_decision(records):
        bits.append(_common_decision(records))
    bits.append(f"halide {_halide_version()}")
    sheet = render_sheet(
        tiles, title=args.title or default_title, subtitle="  ·  ".join(bits),
        frame_width=args.frame_width, columns=args.columns, film_stock=common_film_stock(records),
    )
    write_sheet(sheet_path, sheet)
    print(console.success(f"Contact sheet → {sheet_path} ({sheet.width}×{sheet.height} px)"))


def discover_processed_files(folder: Path, sheet_path: Path) -> tuple[list[Path], Skipped]:
    """Every already-processed frame in `folder` (halide's own TIFF output, or PNG/JPEG from
    `halide export`), excluding the sheet being written itself and any earlier contact sheet in
    that folder (the listing is halide.io.roll.list_scans; the second value is what it skipped) — shared by `halide contact`'s own directory case and `halide batch
    --skip-existing --contact-sheet`'s whole-roll rebuild."""
    from halide.io.roll import list_scans

    files, skipped = list_scans(folder, extra_suffixes=(".png", ".jpg", ".jpeg"))
    return [f for f in files if f.resolve() != sheet_path.resolve()], skipped


def write_sheet_from_folder(
    files: list[Path],
    sheet_path: str | Path,
    args: argparse.Namespace,
    default_title: str,
    *,
    quiet: bool = False,
) -> int:
    """Build a contact sheet by thumbnailing already-processed files fresh from disk — what
    `halide contact <processed_dir> <sheet>` does, and what `halide batch --skip-existing
    --contact-sheet` falls back to for the whole roll when some frames were never touched by this
    run's own worker pool (so never got a thumbnail from it). One shared implementation so the two
    callers can't drift. Returns 0 on success, 1 if any file failed to thumbnail, 130 if cancelled."""
    import shutil
    import tempfile

    from halide.batch.orchestrator import (
        BatchJob,
        default_export_worker_count,
        export_memory_budget_warning,
        run_thumbnail_batch,
    )
    from halide.batch.progress import cancel_notice, make_renderer

    if not files:
        print(f"No processed frames (TIFF/PNG/JPEG) found for {sheet_path}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="halide-contact-"))
    try:
        jobs = [
            BatchJob(input_path=f, output_path=None, thumbnail_path=tmp / f"{i:04d}.png")
            for i, f in enumerate(files)
        ]
        workers = getattr(args, "workers", None)
        if workers is not None:
            warning = export_memory_budget_warning(jobs, workers)
            if warning and not quiet:
                print(console.warning(warning))
        else:
            workers = default_export_worker_count(jobs)

        renderer = make_renderer(len(jobs), "proof", quiet=quiet)
        job_index = {job: i for i, job in enumerate(jobs)}
        if renderer:
            renderer.start()
        results = run_thumbnail_batch(
            jobs, thumbnail_long_edge=args.frame_width, max_workers=workers,
            on_start=(lambda job: renderer.mark_processing(job_index[job])) if renderer else None,
            on_result=(lambda r: renderer.report(job_index[r.job], r)) if renderer else None,
            on_cancel=cancel_notice(renderer),
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
        write_contact_sheet(args, jobs, results, sheet_path, default_title)
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
