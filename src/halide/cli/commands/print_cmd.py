"""`halide print` — apply only the print stage (fitted exposure + paper grade, then the paper curve)
to a flat linear positive, or a directory of them.

The intended round trip: `halide invert --output flat` -> scene-level edits in darktable (crop,
spot removal, lens correction, denoise, global exposure — nothing that reshapes tone: no filmic/
sigmoid, curves, levels or local contrast) -> 32-bit float linear TIFF with its profile embedded ->
`halide print`. Setting the dynamic range is this command's job, not the editor's: a black point set
by hand before the curve would put the shadows on the wrong part of the paper.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from halide.batch.orchestrator import (
    TIFF_SUFFIXES,
    BatchJob,
    default_worker_count,
    memory_budget_warning,
    run_print_batch,
)
from halide.batch.progress import GridProgressRenderer
from halide.calibration.profile_store import load_tone_override, resolve_profile_path
from halide.cli import console
from halide.cli._run_sheet import choose_workers, roll_row
from halide.cli._calibration_args import add_tone_arguments, describe_resolved_tone, resolve_tone_params
from halide.core.types import ToneCurveParams
from halide.processing import PrintInputError, ScanColorError, print_scan


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="Flat linear positive TIFF (from `halide invert --output flat`, optionally edited "
        "elsewhere and re-exported as linear float TIFF with an embedded profile), or a directory "
        "of them",
    )
    parser.add_argument("output", help="Output TIFF path, or an output directory when `input` is a directory")
    add_tone_arguments(parser, allow_output_mode=False)
    parser.add_argument(
        "--profile",
        help="Use a saved profile's exposure/contrast override (from `halide calibrate`'s Fine-tune "
        "controls), if it has one. Its density calibration is ignored — a flat positive already has "
        "it applied.",
    )
    parser.add_argument("--suffix", default="", help="Suffix to append to output filenames when `input` is a directory")
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel worker processes when `input` is a directory (default: "
        "auto-selected from available memory and CPU count)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display when `input` is a directory")


def _resolve_tone(args: argparse.Namespace) -> ToneCurveParams:
    saved_tone = None
    if args.profile:
        try:
            saved_tone = load_tone_override(resolve_profile_path(args.profile))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    return resolve_tone_params(args, saved_tone=saved_tone)


def _run_single(args: argparse.Namespace, input_path: Path, tone_params: ToneCurveParams) -> int:
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    output_path = Path(args.output)
    if output_path.exists() and not console.confirm_overwrite(output_path):
        return 1

    start = time.monotonic()
    label = f"{console.VERB['print']} {input_path.name}..."
    try:
        with console.themed_animation(
            console.ENLARGER_FRAMES, label, min_width=console.ENLARGER_MIN_SIZE[0],
            min_height=console.ENLARGER_MIN_SIZE[1], interval=0.45,
        ):
            resolved, warning = print_scan(input_path, output_path, tone_params)
    except (ScanColorError, PrintInputError) as exc:
        raise SystemExit(str(exc))
    if warning:
        print(console.warning(warning))
    print(describe_resolved_tone(resolved))

    elapsed = time.monotonic() - start
    size = output_path.stat().st_size
    print(
        console.success(
            f"{console.VERB_PAST['print']} → {output_path} "
            f"({console.human_time(elapsed)}, {console.human_bytes(size)})"
        )
    )
    return 0


def _run_bulk(args: argparse.Namespace, input_dir: Path, tone_params: ToneCurveParams) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES)
    if not files:
        print(f"No TIFF files found in {input_dir}")
        return 1
    jobs = [BatchJob(input_path=f, output_path=output_dir / f"{f.stem}{args.suffix}.tif") for f in files]

    with console.RunSheet(quiet=args.quiet) as sheet:
        roll_row(sheet, input_dir, len(jobs), str(output_dir))
        workers = choose_workers(
            args, jobs, sheet, default_count=default_worker_count, budget_warning=memory_budget_warning
        )

    renderer = None if args.quiet else GridProgressRenderer(total=len(jobs), verb="print")
    job_index = {job: i for i, job in enumerate(jobs)}

    def on_start(job):
        if renderer:
            renderer.mark_processing(job_index[job])

    def on_result(result):
        if renderer:
            renderer.report(job_index[result.job], result)

    if renderer:
        renderer.start()

    results = run_print_batch(jobs, tone_params, max_workers=workers, on_result=on_result, on_start=on_start)

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
    tone_params = _resolve_tone(args)
    input_path = Path(args.input)
    if input_path.is_dir():
        return _run_bulk(args, input_path, tone_params)
    return _run_single(args, input_path, tone_params)
