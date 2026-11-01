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
    jobs_for_files,
    plan_compute,
    default_worker_count,
    memory_budget_warning,
    run_print_batch,
)
from halide.batch.progress import cancel_notice, make_renderer
from halide.calibration.profile_store import damaged_profile_message, load_tone_override, resolve_profile_path
from halide.cli import console
from halide.cli._device_args import add_device_argument, device_fallback_warning, device_row, resolve_device_arg
from halide.cli._naming import CLASH_FIX, add_name_argument, naming_for, stock_for_print_or_export
from halide.cli._help import add_workers_argument
from halide.cli._output_policy import (
    add_dry_run_argument,
    RESUME_NOTE,
    add_output_policy_arguments,
    check_output_folder,
    plan_bulk_jobs,
    refuse_dry_run_for_single_file,
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
from halide.cli._run_log import RunRecord, add_log_arguments, check_log_path
from halide.cli._run_sheet import (
    choose_workers,
    compute_row,
    dry_run_line,
    print_rename_list,
    frame_count,
    outputs_row,
    roll_row,
    cleanup_row,
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
    add_output_policy_arguments(
        parser, skip_help=f"Print only the frames whose output doesn't exist yet{RESUME_NOTE}"
    )
    add_dry_run_argument(parser)
    add_log_arguments(parser)
    add_tone_arguments(parser, allow_output_mode=False)
    parser.add_argument(
        "--profile",
        help="Use a saved profile's exposure/contrast override (from `halide calibrate`'s Print "
        "controls), if it has one. Its density calibration is ignored: a flat positive already has "
        "it applied",
    )
    add_name_argument(parser, "Text to add to each output file name when `input` is a folder")
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
        except ValueError as exc:
            raise SystemExit(damaged_profile_message(args.profile, exc)) from exc
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
        print(console.success(f"{output_path} already exists — skipped (--skip-existing)"))
        return 0

    fallback_warning = device_fallback_warning(device)
    if fallback_warning:
        print(console.warning(fallback_warning))

    start = time.monotonic()
    label = f"{console.VERB['print']} {input_path.name}..."
    try:
        with console.developing(label):
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
    dry = args.dry_run
    files, left_out = list_scans(input_dir)
    if not files:
        raise SystemExit(f"No TIFF files found in {input_dir} — is this the right folder?")
    name = naming_for(args, input_dir.resolve().name, stock_for_print_or_export(args, files))
    jobs = jobs_for_files(files, output_dir, name=name, extension=".tif")
    fix = CLASH_FIX if args.name else "rename one of them"
    check_log_path(args.log, [(j.input_path, j.output_path) for j in jobs])  # before any folder is made
    if dry:
        check_output_folder(output_dir)  # a dry run creates nothing
    else:
        prepare_output_folder(output_dir)

    all_jobs = jobs
    existing = 0
    if dry:  # never asks, never refuses for want of a terminal
        jobs, skipped, _, existing = plan_bulk_jobs(jobs, args, fix=fix)
    else:
        jobs, skipped, _ = resolve_bulk_jobs(jobs, args, interactive=is_interactive(), fix=fix)
    developing = {j.output_path for j in jobs}
    kept = [j.input_path.name for j in all_jobs if j.output_path not in developing]
    if not jobs:
        if dry:
            print(console.success(f"Dry run — nothing to do, every output in {output_dir} already exists (--skip-existing)"))
            return 0
        print(console.success(f"Nothing to do — every output in {output_dir} already exists (--skip-existing)"))
        return 0

    # The GPU service (if any) starts while the run sheet is open and stops once the pool is done.
    with contextlib.ExitStack() as stack:
        with console.RunSheet(quiet=args.quiet) as sheet:
            destination = f"{output_dir} (would be created)" if dry and not output_dir.exists() else str(output_dir)
            roll_row(sheet, input_dir, len(jobs), destination)
            skipped_row(sheet, left_out)
            cleanup_row(sheet, output_dir, dry)
            if skipped:
                sheet.row("Skipping", f"{frame_count(skipped)} already {console.VERB_PAST['print'].lower()}")
            if dry:
                outputs_row(sheet, policy_from_args(args), existing)
            # A dry run only plans: the GPU service is never started.
            compute = plan_compute(jobs, device, "develop") if dry else start_compute(stack, sheet, jobs, device, "develop")
            compute_row(sheet, device, compute, planned=dry)
            workers = choose_workers(
                args, jobs, sheet,
                default_count=lambda jobs: default_worker_count(jobs, device=device),
                budget_warning=memory_budget_warning,
                device=device,
                compute=compute,
            )

        if dry:
            print(dry_run_line("print", len(jobs), str(output_dir), kept=len(kept), existing=existing,
                               policy=policy_from_args(args)))
            if args.name:
                print_rename_list(jobs)
            return 0

        record = RunRecord(args, "print", output_dir)
        record.start(sheet, jobs, list(kept))
        try:
            renderer = make_renderer(len(jobs), "print", quiet=args.quiet)
            job_index = {job: i for i, job in enumerate(jobs)}

            def on_start(job):
                renderer.mark_processing(job_index[job])

            def on_result(result):
                record.add(result)
                renderer.report(job_index[result.job], result)

            renderer.start()
            results = run_print_batch(
                jobs, tone_params, max_workers=workers, on_result=on_result, on_start=on_start, device=device,
                on_cancel=cancel_notice(renderer), compute=compute,
            )

            cancelled = len(results) < len(jobs)
            if cancelled:
                done_indices = {job_index[r.job] for r in results}
                renderer.cancel([i for i in range(len(jobs)) if i not in done_indices])
            renderer.finish(cancelled=cancelled)
            record.finish(renderer)
        except BaseException as exc:
            record.abort(exc)
            raise

    failures = [r for r in results if r.error]

    if cancelled:
        return 130
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_dir():
        refuse_dry_run_for_single_file(args)  # before anything is resolved
    tone_params = _resolve_tone(args)
    device = resolve_device_arg(args, isolated=input_path.is_dir())  # a bulk run's parent never computes
    if input_path.is_dir():
        return _run_bulk(args, input_path, tone_params, device)
    return _run_single(args, input_path, tone_params, device)
