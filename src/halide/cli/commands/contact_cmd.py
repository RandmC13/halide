"""`halide contact` — a high-resolution contact sheet of an already-processed folder: halide's own
TIFF output (from invert/batch/print) or `halide export`'s PNG/JPEG files. Make one per set of
settings and compare them side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

from halide.cli import console
from halide.cli._contact_sheet import add_contact_layout_arguments, discover_processed_files, write_sheet_from_folder
from halide.cli._device_args import add_device_argument, device_row, requested_device_arg, resolve_device_arg
from halide.cli._help import add_workers_argument
from halide.cli._output_policy import (
    add_output_policy_arguments,
    is_interactive,
    policy_from_args,
    resolve_existing,
)
from halide.device import ComputeDevice
from halide.io.roll import Skipped

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs", nargs="+",
        help="A folder of developed frames (TIFFs from invert/batch/print, or PNG/JPEG from export), "
        "or individual files, followed by the contact sheet to write (.jpg or .png)",
    )
    add_contact_layout_arguments(parser)
    add_output_policy_arguments(parser)
    add_workers_argument(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")
    add_device_argument(parser)


def _collect(inputs: list[str], sheet: Path) -> tuple[list[Path], Skipped]:
    from halide.io.contact_sheet import is_contact_sheet
    from halide.io.roll import TIFF_SUFFIXES

    files: list[Path] = []
    skipped = Skipped()
    for item in map(Path, inputs):
        if item.is_dir():
            found, left_out = discover_processed_files(item, sheet)
            files.extend(found)
            skipped += left_out
        elif item.exists():
            files.append(item)
        else:
            raise SystemExit(f"not found: {item}")
    # A sheet written into the folder it proofs — this one, or an earlier one — must not end up on
    # the next sheet of that folder as if it were a frame. discover_processed_files() already
    # applies this to files found via a directory; this also covers individual file arguments.
    return [
        f for f in files
        if f.resolve() != sheet.resolve() and not (f.suffix.lower() not in TIFF_SUFFIXES and is_contact_sheet(f))
    ], skipped


def run(args: argparse.Namespace) -> int:
    from halide.io.contact_sheet import check_sheet_path

    if len(args.inputs) < 2:
        raise SystemExit("usage: halide contact <folder-or-files...> <sheet.jpg|.png>")
    sheet_path = Path(args.inputs[-1])
    try:
        check_sheet_path(sheet_path)
    except ValueError as exc:
        raise SystemExit(str(exc))
    files, skipped = _collect(args.inputs[:-1], sheet_path)
    if skipped and not args.quiet:
        print(console.dim(f"Skipped {skipped.describe()}"))
    if not files:
        print("No processed frames (TIFF/PNG/JPEG) found")
        return 1
    resolved = resolve_existing(
        [(sheet_path, sheet_path)], policy_from_args(args), interactive=is_interactive()
    )
    if not resolved:
        print(console.success(f"{sheet_path} already exists — skipped (--skip-existing)."))
        return 0

    # The thumbnails are made on the CPU (run_thumbnail_batch: the frames are already developed,
    # and what's left isn't worth uploading a frame for). So the device is resolved only when a GPU
    # was explicitly asked for (--device gpu or HALIDE_DEVICE=gpu), which still fails fast without
    # CuPy/a driver like every command (Ruling R7). `auto` is never resolved here: that would import
    # CuPy and make a CUDA context (~0.6 s, ~300 MB of GPU memory), and could warn "GPU not usable"
    # for a command that never uses one.
    note = ""
    if requested_device_arg(args) == "gpu":
        resolve_device_arg(args)
        note = f"{console.RunSheet.SEP}developed frames need no GPU"
    if not args.quiet:
        print(f"Compute: {device_row(ComputeDevice(kind='cpu'))}{note}")

    first = Path(args.inputs[0])
    default_title = first.name if first.is_dir() else first.parent.name or "contact sheet"
    return write_sheet_from_folder(files, sheet_path, args, default_title, quiet=args.quiet)
