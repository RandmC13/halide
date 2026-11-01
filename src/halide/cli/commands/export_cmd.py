"""`halide export` — convert a processed ACEScg TIFF (or a directory of them, e.g. the output of
`halide batch`) into delivery-ready sRGB PNG/JPEG."""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

from halide.batch.orchestrator import (
    jobs_for_files,
    plan_compute,
    default_export_worker_count,
    export_memory_budget_warning,
    run_export_batch,
)
from halide.batch.progress import cancel_notice, make_renderer
from halide.cli import console
from halide.cli._device_args import add_device_argument, device_fallback_warning, device_row, resolve_device_arg
from halide.cli._naming import CLASH_FIX, add_name_argument, naming_for, stock_for_print_or_export
from halide.cli._help import add_workers_argument, jpeg_quality
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

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.

_FORMATS = ("png", "jpg", "jpeg")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", help="An ACEScg TIFF, or a folder of them (output of `halide invert`/`halide batch`)"
    )
    parser.add_argument(
        "output",
        help="Output image path (.png, .jpg, or .jpeg) for a single file, or an output folder "
        "when `input` is a folder",
    )
    add_output_policy_arguments(
        parser, skip_help=f"Export only the files whose output doesn't exist yet{RESUME_NOTE}"
    )
    add_dry_run_argument(parser)
    add_log_arguments(parser)
    parser.add_argument(
        "--quality", type=jpeg_quality, default=None, metavar="N",
        help="JPEG quality, 1-100 (default: 95); only for JPEG output"
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=_FORMATS,
        help="Output format when `input` is a folder (default: png); a single file follows its "
        "output name's extension instead",
    )
    add_name_argument(parser, "Text to add to each output file name when `input` is a folder (default: none)")
    add_workers_argument(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")
    add_device_argument(parser)


def _run_single(args: argparse.Namespace, input_path: Path, device) -> int:
    from halide import device as halide_device
    from halide.processing import ScanColorError, export_delivery_image

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
    label = f"{console.VERB['export']} {input_path.name}..."
    try:
        with console.developing(label):
            export_delivery_image(
                input_path, args.output, quality=args.quality,
                device=device, on_warning=lambda msg: print(console.warning(msg)),
            )
    except ScanColorError as exc:  # includes ScanInputError: not a TIFF, not RGB, ...
        raise SystemExit(str(exc))
    halide_device.release_memory()
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
    dry = args.dry_run
    files, left_out = list_scans(input_dir)
    if not files:
        raise SystemExit(f"No TIFF files found in {input_dir} — is this the right folder?")

    name = naming_for(args, input_dir.resolve().name, stock_for_print_or_export(args, files))
    jobs = jobs_for_files(files, output_dir, name=name, extension=f".{args.format}")
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
                sheet.row("Skipping", f"{frame_count(skipped)} already {console.VERB_PAST['export'].lower()}")
            if dry:
                outputs_row(sheet, policy_from_args(args), existing)
            # A dry run only plans: the GPU service is never started.
            compute = plan_compute(jobs, device, "export") if dry else start_compute(stack, sheet, jobs, device, "export")
            compute_row(sheet, device, compute, planned=dry)
            workers = choose_workers(
                args, jobs, sheet,
                default_count=lambda jobs: default_export_worker_count(jobs, device=device),
                budget_warning=export_memory_budget_warning,
                device=device,
                compute=compute,
            )

        if dry:
            print(dry_run_line("export", len(jobs), str(output_dir), kept=len(kept), existing=existing,
                               policy=policy_from_args(args)))
            if args.name:
                print_rename_list(jobs)
            return 0

        record = RunRecord(args, "export", output_dir)
        record.start(sheet, jobs, list(kept))
        try:
            renderer = make_renderer(len(jobs), "export", quiet=args.quiet)
            job_index = {job: i for i, job in enumerate(jobs)}

            def on_start(job):
                renderer.mark_processing(job_index[job])

            def on_result(result):
                record.add(result)
                renderer.report(job_index[result.job], result)

            renderer.start()
            results = run_export_batch(
                jobs, quality=args.quality, max_workers=workers, on_result=on_result, on_start=on_start,
                device=device, on_cancel=cancel_notice(renderer), compute=compute,
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
    # Resolved here regardless of which mode runs, so --device gpu still fails fast without
    # CuPy/a driver and every command's Compute row behaves the same way. Both modes
    # export on it: the single file in this process, a directory in the worker pool.
    input_path = Path(args.input)
    is_jpeg = (args.format if input_path.is_dir() else Path(args.output).suffix.lower().lstrip(".")) in ("jpg", "jpeg")
    if args.quality is not None and not is_jpeg:
        raise SystemExit("--quality only applies to JPEG output — drop it, or export to .jpg")
    if args.quality is None:
        args.quality = 95
    if not input_path.is_dir():
        refuse_dry_run_for_single_file(args)  # before the device is resolved
    device = resolve_device_arg(args, isolated=input_path.is_dir())  # a bulk run's parent never computes
    if input_path.is_dir():
        return _run_bulk(args, input_path, device)
    return _run_single(args, input_path, device)
