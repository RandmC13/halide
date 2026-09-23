"""Shared contact-sheet plumbing for `halide contact` and `halide batch --contact-sheet`: layout
flags, and turning a finished pool's thumbnails into the written sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.cli import console
from halide.io.contact_sheet import (
    DEFAULT_COLUMNS,
    DEFAULT_FRAME_WIDTH,
    Tile,
    caption_from_provenance,
    load_thumbnail,
    render_sheet,
    write_sheet,
)
from halide.processing import _halide_version


def add_contact_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--columns", type=int, default=DEFAULT_COLUMNS, help=f"Frames per strip (default: {DEFAULT_COLUMNS})"
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Width of each frame on the sheet, in pixels (default: {DEFAULT_FRAME_WIDTH}; a "
        "six-across sheet is then about 6000 px wide)",
    )
    parser.add_argument("--title", help="Title printed at the top of the sheet (default: the input folder's name)")


def _common_decision(records: list[dict | None]) -> str:
    """What the frames' recorded printing decisions have in common, for the sheet's header."""
    outputs = {r.get("output") for r in records if r}
    if len(outputs) != 1 or len(records) != sum(1 for r in records if r):
        return ""
    return {"print": "print", "flat": "flat (linear)"}.get(outputs.pop(), "")


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

    bits = [f"{len(jobs)} frame(s)"]
    if settings:
        bits.append(settings)
    elif _common_decision(records):
        bits.append(_common_decision(records))
    bits.append(f"halide {_halide_version()}")
    sheet = render_sheet(
        tiles, title=args.title or default_title, subtitle="  ·  ".join(bits),
        frame_width=args.frame_width, columns=args.columns,
    )
    write_sheet(sheet_path, sheet)
    print(console.success(f"Contact sheet → {sheet_path} ({sheet.width}×{sheet.height} px)"))
