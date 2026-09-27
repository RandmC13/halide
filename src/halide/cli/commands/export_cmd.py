"""`halide export` — convert a processed ACEScg TIFF (or a directory of them, e.g. the output of
`halide batch`) into delivery-ready sRGB PNG/JPEG."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from halide.batch.orchestrator import (
    TIFF_SUFFIXES,
    BatchJob,
    default_export_worker_count,
    export_memory_budget_warning,
    run_export_batch,
)
from halide.batch.progress import GridProgressRenderer
from halide.cli import console
from halide.cli._device_args import add_device_argument, device_fallback_warning, device_row, resolve_device_arg
from halide.cli._run_sheet import choose_workers, compute_row, roll_row

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.

_FORMATS = ("png", "jpg", "jpeg")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", help="Input ACEScg TIFF, or a directory of them (output of `halide invert`/`halide batch`)"
    )
    parser.add_argument(
        "output",
        help="Output image path (.png, .jpg, or .jpeg) for a single file, or an output directory "
        "when `input` is a directory",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality, 1-100 (default: 95; ignored for PNG)"
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=_FORMATS,
        help="Output format when `input` is a directory (default: png). Ignored for a single "
        "file, where the output path's own extension is used instead.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to append to output filenames when `input` is a directory (default: none)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel worker processes when `input` is a directory (default: "
        "auto-selected from available memory and CPU count, same as `halide batch`; pass this to "
        "override the auto-selected count)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the progress display when `input` is a directory"
    )
    add_device_argument(parser)


def _run_single(args: argparse.Namespace, input_path: Path, device) -> int:
    from halide import device as halide_device
    from halide.processing import export_delivery_image

    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    output_path = Path(args.output)
    if output_path.exists() and not console.confirm_overwrite(output_path):
        return 1

    fallback_warning = device_fallback_warning(device)
    if fallback_warning:
        print(console.warning(fallback_warning))

    start = time.monotonic()
    label = f"{console.VERB['export']} {input_path.name}... (exposing)"
    with console.themed_animation(
        console.ENLARGER_FRAMES, label, min_width=console.ENLARGER_MIN_SIZE[0],
        min_height=console.ENLARGER_MIN_SIZE[1], interval=0.45,
    ):
        warning = export_delivery_image(
            input_path, args.output, quality=args.quality,
            device=device, on_warning=lambda msg: print(console.warning(msg)),
        )
    halide_device.release_memory()
    if warning:
        print(console.warning(warning))

    elapsed = time.monotonic() - start
    size = output_path.stat().st_size
    print(
        console.success(
            f"{console.VERB_PAST['export']} → {output_path} "
            f"({console.human_time(elapsed)}, {console.human_bytes(size)}, {device_row(device)})"
        )
    )
    return 0


def _run_bulk(args: argparse.Namespace, input_dir: Path, device) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES)
    if not files:
        print(f"No TIFF files found in {input_dir}")
        return 1

    jobs = [
        BatchJob(input_path=f, output_path=output_dir / f"{f.stem}{args.suffix}.{args.format}")
        for f in files
    ]

    with console.RunSheet(quiet=args.quiet) as sheet:
        roll_row(sheet, input_dir, len(jobs), str(output_dir))
        compute_row(sheet, device)
        workers = choose_workers(
            args, jobs, sheet,
            default_count=lambda jobs: default_export_worker_count(jobs, device=device),
            budget_warning=export_memory_budget_warning,
        )

    renderer = None if args.quiet else GridProgressRenderer(total=len(jobs), verb="export")
    job_index = {job: i for i, job in enumerate(jobs)}

    def on_start(job):
        if renderer:
            renderer.mark_processing(job_index[job])

    def on_result(result):
        if renderer:
            renderer.report(job_index[result.job], result)

    if renderer:
        renderer.start()

    results = run_export_batch(
        jobs, quality=args.quality, max_workers=workers, on_result=on_result, on_start=on_start, device=device
    )

    cancelled = len(results) < len(jobs)
    if renderer:
        if cancelled:
            done_indices = {job_index[r.job] for r in results}
            not_started = [i for i in range(len(jobs)) if i not in done_indices]
            renderer.cancel(not_started)
        renderer.finish(cancelled=cancelled)

    for r in results:
        if r.warning:
            print(console.warning(f"{r.job.input_path.name}: {r.warning}"))

    failures = [r for r in results if r.error]
    if renderer is None:
        if cancelled:
            print(console.warning(f"Cancelled — {len(results)}/{len(jobs)} frames processed."))
        for r in failures:
            print(console.error(f"{r.job.input_path.name}: {r.error}"))

    if cancelled:
        return 130
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    # Resolved here regardless of which mode runs, so --device gpu still fails fast without
    # CuPy/a driver and every command's Compute row behaves the same way (Ruling R7). Both modes
    # export on it: the single file in this process, a directory in the worker pool.
    device = resolve_device_arg(args)
    input_path = Path(args.input)
    if input_path.is_dir():
        return _run_bulk(args, input_path, device)
    return _run_single(args, input_path, device)
