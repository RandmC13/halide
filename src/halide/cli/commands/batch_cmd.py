"""`halide batch` — process a directory of negative scans in parallel."""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

from halide.io.roll import list_scans
from halide.batch.orchestrator import BatchJob, default_worker_count, jobs_for_files, memory_budget_warning, run_batch
from halide.batch.progress import cancel_notice, make_renderer
from halide.cli import console
from halide.calibration.scan_consistency import assess_roll, most_common_settings, scan_gain
from halide.cli._device_args import add_device_argument, resolve_device_arg
from halide.cli._help import add_workers_argument
from halide.cli._output_policy import (
    add_output_policy_arguments,
    check_input_folder,
    is_interactive,
    policy_from_args,
    prepare_output_folder,
    resolve_bulk_jobs,
    resolve_existing,
)
from halide.cli._calibration_args import (
    add_calibration_arguments,
    manual_calibration_given,
    add_scan_arguments,
    add_stage_arguments,
    add_tone_arguments,
    maybe_save_profile,
    choose_calibration_source,
    resolve_density_profile,
    resolve_scan_reference,
    resolve_stage,
    resolve_tone_params,
)
from halide.cli._run_sheet import (
    choose_workers,
    compute_row,
    frame_count,
    print_frame_warnings,
    roll_row,
    skipped_row,
    start_compute,
)
from halide.cli._contact_sheet import add_contact_layout_arguments, write_contact_sheet
from halide.core.types import Stage

# The pipeline (numpy, Pillow, colour-science) is imported inside the functions that use it, so
# building the parser — `halide --help`, tab completion — doesn't load it.

_SEP = console.RunSheet.SEP


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="Folder of linear TIFF scans (one roll)")
    parser.add_argument(
        "output_dir", nargs="?",
        help="Folder to write the developed TIFFs into. Optional with --contact-sheet: leave it out to "
        "preview settings as a contact sheet without keeping any full-size TIFFs",
    )
    parser.add_argument(
        "--contact-sheet", metavar="SHEET",
        help="Also write a high-resolution contact sheet of the roll (.jpg or .png), each frame "
        "captioned with its print settings — for comparing settings. Without an output directory "
        "this is a preview: frames are developed exactly as a real run would, but only small "
        "thumbnails are kept (in a temporary folder, deleted afterwards)",
    )
    add_contact_layout_arguments(parser)
    parser.add_argument("--suffix", default="", help="Text to add to each output file name, before .tif")
    add_output_policy_arguments(parser, saves_profile=True)

    add_stage_arguments(parser)
    sources = add_calibration_arguments(parser)
    sources.add_argument(
        "--auto-density-roll",
        action="store_true",
        help="Estimate one shared density-balance profile from the whole roll, rather than "
        "per-frame (--auto-density) — generally more consistent across a roll, at the cost of "
        "not adapting to any single frame's content",
    )
    add_tone_arguments(parser)
    add_scan_arguments(parser)

    add_workers_argument(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")
    add_device_argument(parser)


def _match_scan_exposure(
    jobs, metadata, reference, reference_origin: str, unknown_reference_warning: str | None, sheet: console.RunSheet
):
    """Give every job the gain that puts it at the calibration's scan exposure (see
    calibration/scan_consistency.py). Returns (jobs, the reference used). `reference_origin` says
    where a known `reference` came from, for the run sheet; `unknown_reference_warning` is shown
    when there isn't one and the roll's most common setting has to stand in for it."""
    known = [settings for settings, _ in metadata.values() if settings is not None]
    if not known:
        raise SystemExit("--match-scan-exposure: none of these frames has camera exposure settings (EXIF) to match from")
    guessed = reference is None
    if guessed:
        reference = most_common_settings(known)
        reference_origin = "the roll's most common"
    matched, missing = [], []
    for job in jobs:
        settings, _ = metadata[str(job.input_path)]
        if settings is None:
            missing.append(job.input_path.name)
            matched.append(job)
        else:
            matched.append(replace(job, scan_gain=scan_gain(settings, reference)))
    adjusted = sum(1 for job in matched if abs(job.scan_gain - 1.0) > 1e-9)
    sheet.row(
        "Scan exposure",
        f"matched to {reference.describe()} ({reference_origin}){_SEP}"
        f"{adjusted}/{len(jobs)} frames adjusted",
    )
    if guessed and unknown_reference_warning:
        sheet.warn("Scan exposure", unknown_reference_warning)
    if missing:
        sheet.warn("Scan exposure", f"no camera exposure settings (EXIF) in {', '.join(missing)} — left unadjusted")
    return matched, reference


def _settings_summary(args: argparse.Namespace, stage: Stage, tone_params, scan_reference, matched: bool) -> str:
    """The settings a run used, for a contact sheet's header — so two sheets can be told apart."""
    if stage is Stage.INVERT_ONLY:
        source = "no density balance"
    elif args.profile:
        source = f"profile {Path(args.profile).stem}"
    elif args.auto_density_roll:
        source = "auto (roll)"
    elif args.auto_density:
        source = "auto (per frame)"
    else:
        source = "manual calibration"
    bits = [source]
    if tone_params.mode == "linear":
        bits.append("flat (linear)")
    else:
        bits.append("print")
        bits.append(f"grade {tone_params.contrast:.2f}" if tone_params.contrast is not None else "grade fitted")
        bits.append(f"exposure {tone_params.exposure:+.2f}" if tone_params.exposure is not None else "exposure fitted")
    if matched and scan_reference is not None:
        bits.append(f"scan exposure matched to {scan_reference.describe()}")
    return " · ".join(bits)


def _unknown_reference_warning(args: argparse.Namespace) -> str | None:
    """Why matching to the roll's most common scan exposure may leave an offset — or None for
    --auto-density-roll, whose calibration is measured from the roll itself, so it's exact."""
    if args.auto_density_roll:
        return None
    offset = "a constant colour offset remains across the roll"
    if args.profile:
        return (
            "the profile doesn't record the scan exposure it was calibrated at — if its calibration "
            f"frame was scanned differently, {offset} (pass --scan-reference FRAME, or save the "
            "profile again from that frame)"
        )
    return (
        "manual values don't record which scan they were measured on — if that frame was scanned "
        f"differently, {offset} (pass --scan-reference FRAME)"
    )


def _destination(args: argparse.Namespace) -> str:
    if args.output_dir is None:
        return f"contact sheet {args.contact_sheet} only (a preview — no full-size TIFFs kept)"
    destination = f"{args.output_dir}"
    return f"{destination} + contact sheet {args.contact_sheet}" if args.contact_sheet else destination


# R-050: the automatic tiers are approximate (CLAUDE.md, auto calibration's limits); say what's better.
AUTO_CALIBRATION_ADVICE = (
    "automatic estimate - for the most faithful colour, pick neutral points with `halide calibrate`"
)


def _calibration_text(args: argparse.Namespace, profile, saved_tone) -> str:
    if profile is None:
        return f"auto{_SEP}estimated separately for each frame"
    if args.profile:
        text = f"profile {args.profile}"
        if saved_tone is not None and (saved_tone.exposure is not None or saved_tone.contrast is not None):
            text += " (with its saved print settings)"
        return text
    (rm, _, bm), (rs, _, bs) = profile.white_balance, profile.density_scale
    return f"manual{_SEP}white balance R ×{rm:g} B ×{bm:g}, density scale R {rs:g} B {bs:g}"


def _output_text(stage: Stage, tone_params) -> str:
    if stage is Stage.DENSITY_ONLY:
        return "density balance only — not inverted (--density-only)"
    if tone_params.mode == "linear":
        return "flat positive for editing elsewhere (--output flat)"
    grade = f"{tone_params.contrast:.2f}" if tone_params.contrast is not None else "fitted per frame"
    exposure = f"{tone_params.exposure:+.2f}" if tone_params.exposure is not None else "fitted per frame"
    return f"print{_SEP}grade {grade}{_SEP}exposure {exposure}"


def run(args: argparse.Namespace) -> int:
    stage = resolve_stage(args)

    input_dir = Path(args.input_dir)
    check_input_folder(input_dir)
    if args.output_dir is None and not args.contact_sheet:
        raise SystemExit("give an output directory, --contact-sheet SHEET (a preview), or both")
    sheet_path = Path(args.contact_sheet) if args.contact_sheet else None
    if sheet_path is not None:
        from halide.io.contact_sheet import check_sheet_path

        try:
            check_sheet_path(sheet_path)
        except ValueError as exc:
            raise SystemExit(str(exc))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        prepare_output_folder(output_dir)
        files, left_out = list_scans(input_dir)
        jobs = jobs_for_files(files, output_dir, suffix=args.suffix)
    else:
        output_dir = None
        files, left_out = list_scans(input_dir)
        jobs = [BatchJob(input_path=f, output_path=None) for f in files]
    if not jobs:
        print(f"No TIFF files found in {input_dir}")
        return 1

    # A preview run's jobs (output_dir is None) keep nothing but a temp thumbnail, so there's
    # nothing there to protect; only real per-frame outputs and the contact sheet itself go
    # through the overwrite policy — combined into one decision, one prompt for the whole roll.
    if output_dir is not None:
        extra = [(sheet_path, sheet_path)] if sheet_path is not None else None
        jobs, skipped_frames, kept_outputs = resolve_bulk_jobs(
            jobs, args, interactive=is_interactive(), extra_pairs=extra
        )
        build_sheet = sheet_path is None or sheet_path in kept_outputs
    else:
        skipped_frames = 0
        # output_dir is None only reaches here with a contact sheet requested (checked above) — no
        # single "input" for a sheet, so only its own existence matters.
        build_sheet = bool(
            resolve_existing([(sheet_path, sheet_path)], policy_from_args(args), interactive=is_interactive())
        )

    if output_dir is not None and not jobs:
        if sheet_path is None or not build_sheet:
            print(console.success(f"Nothing to do — every output in {output_dir} already exists (--skip-existing)."))
            return 0
        # Nothing left to develop, but the sheet still needs building — from the whole,
        # already-complete folder, the same way `halide contact <output_dir> <sheet>` would.
        from halide.cli._contact_sheet import discover_processed_files, write_sheet_from_folder

        files, _ = discover_processed_files(output_dir, sheet_path)
        return write_sheet_from_folder(files, sheet_path, args, input_dir.name, quiet=args.quiet)
    if output_dir is None and not build_sheet:
        print(console.success(f"Nothing to do — {sheet_path} already exists (--skip-existing)."))
        return 0

    thumbnails = Path(tempfile.mkdtemp(prefix="halide-contact-")) if build_sheet else None
    try:
        if thumbnails is not None:
            jobs = [replace(job, thumbnail_path=thumbnails / f"{i:04d}.png") for i, job in enumerate(jobs)]
        return _run(
            args, stage, input_dir, jobs, output_dir=output_dir, skipped_frames=skipped_frames, build_sheet=build_sheet,
            left_out=left_out,
        )
    finally:
        if thumbnails is not None:
            shutil.rmtree(thumbnails, ignore_errors=True)


def _prepare(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob], sheet: console.RunSheet,
             stack: contextlib.ExitStack, skipped_frames: int = 0, left_out=None):
    """Everything decided before developing starts — scan checks, scan-exposure matching,
    calibration, print settings, how the GPU is used, worker count — each reported on the run sheet
    as it's decided. On a GPU the shared GPU service starts here and runs until `stack` closes.
    Returns (jobs, density_profile, tone_params, scan_reference, workers, device, compute)."""
    from halide.processing import ScanColorError, estimate_roll_density_profile, read_roll_scan_metadata

    roll_row(sheet, input_dir, len(jobs), _destination(args))
    skipped_row(sheet, left_out)
    if skipped_frames:
        sheet.row("Skipping", f"{frame_count(skipped_frames)} already developed")

    per_frame_auto = args.auto_density and not args.auto_density_roll
    matching = args.match_scan_exposure and not (stage is Stage.INVERT_ONLY or per_frame_auto)

    metadata = read_roll_scan_metadata([job.input_path for job in jobs])
    report = assess_roll({Path(path).name: meta for path, meta in metadata.items()})
    issues = report.summary_lines(exposure_corrected=matching)
    for line in issues:
        sheet.warn("Scans", line)
    if issues:
        sheet.note("Scans", f"details, and how to scan a roll consistently: halide check {input_dir}")
    elif report.exposure_groups or report.white_balance_groups:
        sheet.ok("Scans", "scanned consistently")

    scan_reference = None if args.auto_density_roll else resolve_scan_reference(args)
    if args.match_scan_exposure:
        if not matching:
            sheet.warn("Scan exposure", "--match-scan-exposure has no effect here: each frame is calibrated from itself")
        else:
            origin = (
                f"from {Path(args.scan_reference).name}" if args.scan_reference
                else "the profile's calibration scan"
            )
            jobs, scan_reference = _match_scan_exposure(
                jobs, metadata, scan_reference, origin, _unknown_reference_warning(args), sheet
            )

    if stage is Stage.INVERT_ONLY:
        density_profile, saved_tone = None, None  # unused by process_scan for this stage
        sheet.row("Calibration", "none — density balance skipped (--invert-only)")
    elif args.auto_density_roll:
        skipped: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with sheet.working("Calibration", f"estimating one profile from all {frame_count(len(jobs))}…"):
                try:
                    density_profile = estimate_roll_density_profile(
                        [job.input_path for job in jobs],
                        scan_gains={str(job.input_path): job.scan_gain for job in jobs},
                        on_skip=lambda path, exc: skipped.append(f"{path.name} ({exc})"),
                    )
                except ScanColorError as exc:
                    raise SystemExit(str(exc))
        saved_tone = None
        used = len(jobs) - len(skipped)
        sheet.row("Calibration", f"auto{_SEP}one profile for the whole roll, from {frame_count(used)}")
        for name in skipped:
            sheet.warn("Calibration", f"skipped unreadable {name}")
        for caught_warning in caught:
            sheet.warn("Calibration", str(caught_warning.message))
        sheet.note("Calibration", AUTO_CALIBRATION_ADVICE)
    else:
        density_profile, saved_tone = resolve_density_profile(args)  # profile may be None -> per-frame auto
        sheet.row("Calibration", _calibration_text(args, density_profile, saved_tone))
        if density_profile is None:
            sheet.note("Calibration", AUTO_CALIBRATION_ADVICE)

    saved_path = maybe_save_profile(args, density_profile, tone=saved_tone, scan=scan_reference, announce=False)
    if saved_path is not None:
        sheet.ok("Calibration", f"saved as {args.save_profile_as!r} — reuse with --profile {args.save_profile_as}")
    tone_params = resolve_tone_params(args, saved_tone=saved_tone)
    sheet.row("Output", _output_text(stage, tone_params))

    # Resolved after choose_calibration_source (called by _run, before this sheet opens) and
    # before the sheet closes — see CLAUDE.md's choose_calibration_source ordering note. The workers
    # develop on it (run_batch): on a GPU through the shared GPU service, or — if it can't be used —
    # with a CUDA context per worker, whose default count also fits the card's memory.
    device = resolve_device_arg(args, isolated=True)  # the parent never computes on the card (2.4-7)
    compute = start_compute(stack, sheet, jobs, device)
    compute_row(sheet, device, compute)

    workers = choose_workers(
        args, jobs, sheet,
        default_count=lambda jobs: default_worker_count(jobs, device=device),
        budget_warning=memory_budget_warning,
        device=device,
        compute=compute,
    )
    return jobs, density_profile, tone_params, scan_reference, workers, device, compute


def _run(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob], *,
         output_dir: Path | None = None, skipped_frames: int = 0, build_sheet: bool = True,
         left_out=None) -> int:
    if stage is not Stage.INVERT_ONLY:
        choose_calibration_source(args, "this roll")  # before the run sheet, and before anything reads args.profile
    # --auto-density-roll vs --profile/--auto-density is argparse's (exclusive group); only the
    # manual values need this hand check.
    if args.auto_density_roll and manual_calibration_given(args):
        raise SystemExit("manual overrides (--rm/--bm/--rs/--bs) can't be combined with --auto-density-roll")

    # The GPU service (if any) starts while the run sheet is open and stops once the pool is done.
    with contextlib.ExitStack() as stack:
        with console.RunSheet(quiet=args.quiet) as sheet:
            jobs, density_profile, tone_params, scan_reference, workers, device, compute = _prepare(
                args, stage, input_dir, jobs, sheet, stack, skipped_frames=skipped_frames, left_out=left_out
            )

        renderer = make_renderer(len(jobs), "invert", quiet=args.quiet)
        job_index = {job: i for i, job in enumerate(jobs)}

        def on_start(job):
            if renderer:
                renderer.mark_processing(job_index[job])

        def on_result(result):
            if renderer:
                renderer.report(job_index[result.job], result)

        if renderer:
            renderer.start()

        results = run_batch(
            jobs,
            stage,
            density_profile,
            tone_params,
            max_workers=workers,
            on_result=on_result,
            on_start=on_start,
            on_cancel=cancel_notice(renderer),
            thumbnail_long_edge=args.frame_width,
            device=device,
            compute=compute,
        )

    cancelled = len(results) < len(jobs)
    if renderer:
        if cancelled:
            done_indices = {job_index[r.job] for r in results}
            not_started = [i for i in range(len(jobs)) if i not in done_indices]
            renderer.cancel(not_started)
        renderer.finish(cancelled=cancelled)

    # Printed even with --quiet, like every run-sheet warning: e.g. a frame the GPU couldn't develop
    # and the CPU redid (same result, within the GPU tolerance, but the user should know).
    print_frame_warnings(results)

    failures = [r for r in results if r.error]

    if cancelled:
        return 130

    sheet_failed = False
    if args.contact_sheet and build_sheet:
        if skipped_frames and output_dir is not None:
            # Frames --skip-existing left alone never got a thumbnail from this run's own worker
            # pool, so the sheet has to cover the whole roll the way `halide contact <output_dir>
            # <sheet>` does: thumbnail every processed file on disk, not just this run's `jobs`.
            from halide.cli._contact_sheet import discover_processed_files, write_sheet_from_folder

            sheet_path = Path(args.contact_sheet)
            files, _ = discover_processed_files(output_dir, sheet_path)
            sheet_code = write_sheet_from_folder(files, sheet_path, args, input_dir.name, quiet=args.quiet)
            if sheet_code == 130:
                return 130
            sheet_failed = sheet_code != 0
        else:
            settings = _settings_summary(args, stage, tone_params, scan_reference, args.match_scan_exposure)
            write_contact_sheet(args, jobs, results, args.contact_sheet, input_dir.name, settings)
    return 1 if (failures or sheet_failed) else 0
