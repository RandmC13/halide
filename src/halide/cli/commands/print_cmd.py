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
import contextlib
import time
from pathlib import Path

from halide.batch.orchestrator import (
    BatchJob,
    default_worker_count,
    memory_budget_warning,
    run_print_batch,
)
from halide.batch.progress import cancel_notice, make_renderer
from halide.calibration.profile_store import load_tone_override, resolve_profile_path
from halide.cli import console
from halide.cli._device_args import add_device_argument, device_fallback_warning, device_row, resolve_device_arg
from halide.cli._help import add_workers_argument
from halide.cli._output_policy import (
    add_output_policy_arguments,
    check_input_file,
    check_not_input,
    check_output_parent,
    is_interactive,
    policy_from_args,
    prepare_output_folder,
    resolve_bulk_jobs,
    resolve_existing,
)
from halide.io.roll import list_scans
from halide.cli._run_sheet import (
    choose_workers,
    compute_row,
    frame_count,
    print_frame_warnings,
    roll_row,
    skipped_row,
    start_compute,
)
from halide.cli._calibration_args import add_tone_arguments, describe_resolved_tone, resolve_tone_params
from halide.core.types import ToneCurveParams
from halide.device import ComputeDevice

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="Flat linear positive TIFF (from `halide invert --output flat`, optionally edited "
        "elsewhere and re-exported as linear float TIFF with an embedded profile), or a folder "
        "of them",
    )
    parser.add_argument("output", help="Output TIFF path, or an output folder when `input` is a folder")
    add_output_policy_arguments(parser)
    add_tone_arguments(parser, allow_output_mode=False)
    parser.add_argument(
        "--profile",
        help="Use a saved profile's exposure/contrast override (from `halide calibrate`'s Print "
        "controls), if it has one. Its density calibration is ignored: a flat positive already has "
        "it applied",
    )
    parser.add_argument("--suffix", default="", help="Text to add to each output file name when `input` is a folder")
    add_workers_argument(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")
    add_device_argument(parser)


def _resolve_tone(args: argparse.Namespace) -> ToneCurveParams:
    saved_tone = None
    if args.profile:
        try:
            saved_tone = load_tone_override(resolve_profile_path(args.profile))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    return resolve_tone_params(args, saved_tone=saved_tone)


def _run_single(args: argparse.Namespace, input_path: Path, tone_params: ToneCurveParams, device: ComputeDevice) -> int:
    from halide import device as halide_device
    from halide.processing import PrintInputError, ScanColorError, print_scan

    check_input_file(input_path)

    output_path = Path(args.output)
    check_output_parent(output_path)
    check_not_input([(input_path, output_path)], suggest_suffix=False)
    resolved = resolve_existing(
        [(input_path, output_path)], policy_from_args(args), interactive=is_interactive()
    )
    if not resolved:
        print(console.success(f"{output_path} already exists — skipped (--skip-existing)."))
        return 0

    fallback_warning = device_fallback_warning(device)
    if fallback_warning:
        print(console.warning(fallback_warning))

    start = time.monotonic()
    label = f"{console.VERB['print']} {input_path.name}..."
    try:
        with console.themed_animation(
            console.ENLARGER_FRAMES, label, min_width=console.ENLARGER_MIN_SIZE[0],
            min_height=console.ENLARGER_MIN_SIZE[1], interval=0.45,
        ):
            resolved, warning = print_scan(
                input_path, output_path, tone_params,
                device=device, on_warning=lambda msg: print(console.warning(msg)),
            )
    except (ScanColorError, PrintInputError) as exc:
        raise SystemExit(str(exc))
    halide_device.release_memory()
    if warning:
        print(console.warning(warning))
    print(describe_resolved_tone(resolved))

    elapsed = time.monotonic() - start
    size = output_path.stat().st_size
    print(
        console.success(
            f"{console.VERB_PAST['print']} → {output_path} "
            f"({console.human_time(elapsed)}, {console.human_bytes(size)}, {device_row(device)})"
        )
    )
    return 0


def _run_bulk(args: argparse.Namespace, input_dir: Path, tone_params: ToneCurveParams, device: ComputeDevice) -> int:
    output_dir = Path(args.output)
    prepare_output_folder(output_dir)

    files, left_out = list_scans(input_dir)
    if not files:
        print(f"No TIFF files found in {input_dir}")
        return 1
    jobs = [BatchJob(input_path=f, output_path=output_dir / f"{f.stem}{args.suffix}.tif") for f in files]

    jobs, skipped, _ = resolve_bulk_jobs(jobs, args, interactive=is_interactive())
    if not jobs:
        print(console.success(f"Nothing to do — every output in {output_dir} already exists (--skip-existing)."))
        return 0

    # The GPU service (if any) starts while the run sheet is open and stops once the pool is done.
    with contextlib.ExitStack() as stack:
        with console.RunSheet(quiet=args.quiet) as sheet:
            roll_row(sheet, input_dir, len(jobs), str(output_dir))
            skipped_row(sheet, left_out)
            if skipped:
                sheet.row("Skipping", f"{frame_count(skipped)} already developed")
            compute = start_compute(stack, sheet, jobs, device, "develop")
            compute_row(sheet, device, compute)
            workers = choose_workers(
                args, jobs, sheet,
                default_count=lambda jobs: default_worker_count(jobs, device=device),
                budget_warning=memory_budget_warning,
                device=device,
                compute=compute,
            )

        renderer = make_renderer(len(jobs), "print", quiet=args.quiet)
        job_index = {job: i for i, job in enumerate(jobs)}

        def on_start(job):
            if renderer:
                renderer.mark_processing(job_index[job])

        def on_result(result):
            if renderer:
                renderer.report(job_index[result.job], result)

        if renderer:
            renderer.start()

        results = run_print_batch(
            jobs, tone_params, max_workers=workers, on_result=on_result, on_start=on_start, device=device,
            on_cancel=cancel_notice(renderer),
            compute=compute,
        )

    cancelled = len(results) < len(jobs)
    if renderer:
        if cancelled:
            done_indices = {job_index[r.job] for r in results}
            not_started = [i for i in range(len(jobs)) if i not in done_indices]
            renderer.cancel(not_started)
        renderer.finish(cancelled=cancelled)

    print_frame_warnings(results)

    failures = [r for r in results if r.error]

    if cancelled:
        return 130
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    tone_params = _resolve_tone(args)
    input_path = Path(args.input)
    device = resolve_device_arg(args, isolated=input_path.is_dir())  # a bulk run's parent never computes
    if input_path.is_dir():
        return _run_bulk(args, input_path, tone_params, device)
    return _run_single(args, input_path, tone_params, device)
