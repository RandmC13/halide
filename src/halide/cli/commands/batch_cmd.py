"""`halide batch` — process a directory of negative scans in parallel."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

from halide.batch.orchestrator import BatchJob, default_worker_count, discover_jobs, memory_budget_warning, run_batch
from halide.batch.progress import GridProgressRenderer
from halide.cli import console
from halide.calibration.scan_consistency import assess_roll, most_common_settings, scan_gain
from halide.cli._calibration_args import (
    add_calibration_arguments,
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
from halide.cli._run_sheet import choose_workers, frame_count, roll_row
from halide.cli._contact_sheet import add_contact_layout_arguments, write_contact_sheet
from halide.io.contact_sheet import check_sheet_path
from halide.processing import ScanColorError, Stage, estimate_roll_density_profile, read_roll_scan_metadata

_SEP = console.RunSheet.SEP


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="Directory of input linear TIFF scans")
    parser.add_argument(
        "output_dir", nargs="?",
        help="Directory to write output TIFFs into. Optional with --contact-sheet: leave it out to "
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
    parser.add_argument("--suffix", default="", help="Suffix to append to output filenames")

    add_stage_arguments(parser)
    add_calibration_arguments(parser)
    parser.add_argument(
        "--auto-density-roll",
        action="store_true",
        help="Estimate one shared density-balance profile from the whole roll, rather than "
        "per-frame (--auto-density) — generally more consistent across a roll, at the cost of "
        "not adapting to any single frame's content",
    )
    add_tone_arguments(parser)
    add_scan_arguments(parser)

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel worker processes (default: auto-selected from available memory "
        "and CPU count — full-resolution scans are memory-heavy enough that RAM, not CPU threads, "
        "is usually the real limit; pass this to override the auto-selected count)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")


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
    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")
    if args.output_dir is None and not args.contact_sheet:
        raise SystemExit("give an output directory, --contact-sheet SHEET (a preview), or both")
    if args.contact_sheet:
        try:
            check_sheet_path(args.contact_sheet)
        except ValueError as exc:
            raise SystemExit(str(exc))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs = discover_jobs(input_dir, output_dir, suffix=args.suffix)
    else:
        jobs = [BatchJob(input_path=job.input_path, output_path=None)
                for job in discover_jobs(input_dir, input_dir, suffix=args.suffix)]
    if not jobs:
        print(f"No TIFF files found in {input_dir}")
        return 1

    thumbnails = Path(tempfile.mkdtemp(prefix="halide-contact-")) if args.contact_sheet else None
    try:
        if thumbnails is not None:
            jobs = [replace(job, thumbnail_path=thumbnails / f"{i:04d}.png") for i, job in enumerate(jobs)]
        return _run(args, stage, input_dir, jobs)
    finally:
        if thumbnails is not None:
            shutil.rmtree(thumbnails, ignore_errors=True)


def _prepare(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob], sheet: console.RunSheet):
    """Everything decided before developing starts — scan checks, scan-exposure matching,
    calibration, print settings, worker count — each reported on the run sheet as it's decided.
    Returns (jobs, density_profile, tone_params, scan_reference, workers)."""
    roll_row(sheet, input_dir, len(jobs), _destination(args))

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
    else:
        sheet.note("Scans", "not checked — no camera EXIF or darktable history in these files")

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
    else:
        density_profile, saved_tone = resolve_density_profile(args)  # profile may be None -> per-frame auto
        sheet.row("Calibration", _calibration_text(args, density_profile, saved_tone))

    saved_path = maybe_save_profile(args, density_profile, tone=saved_tone, scan=scan_reference, announce=False)
    if saved_path is not None:
        sheet.ok("Calibration", f"saved as {args.save_profile_as!r} — reuse with --profile {args.save_profile_as}")
    tone_params = resolve_tone_params(args, saved_tone=saved_tone)
    sheet.row("Output", _output_text(stage, tone_params))

    workers = choose_workers(
        args, jobs, sheet, default_count=default_worker_count, budget_warning=memory_budget_warning
    )
    return jobs, density_profile, tone_params, scan_reference, workers


def _run(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob]) -> int:
    if stage is not Stage.INVERT_ONLY:
        choose_calibration_source(args, "this roll")  # before the run sheet, and before anything reads args.profile
    manual_given =args.rm is not None or args.bm is not None or args.rs != 1.0 or args.bs != 1.0
    other_source_given = bool(args.profile) or manual_given or args.auto_density
    if args.auto_density_roll and other_source_given:
        raise SystemExit(
            "--auto-density-roll cannot be combined with --profile/manual overrides/--auto-density"
        )

    with console.RunSheet(quiet=args.quiet) as sheet:
        jobs, density_profile, tone_params, scan_reference, workers = _prepare(args, stage, input_dir, jobs, sheet)

    renderer = None if args.quiet else GridProgressRenderer(total=len(jobs), verb="invert")
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
        thumbnail_long_edge=args.frame_width,
    )

    cancelled = len(results) < len(jobs)
    if renderer:
        if cancelled:
            done_indices = {job_index[r.job] for r in results}
            not_started = [i for i in range(len(jobs)) if i not in done_indices]
            renderer.cancel(not_started)
        renderer.finish(cancelled=cancelled)

    failures = [r for r in results if r.error]
    if renderer is None:
        if cancelled:
            print(console.warning(f"Cancelled — {len(results)}/{len(jobs)} frames processed."))
        for r in failures:
            print(console.error(f"{r.job.input_path.name}: {r.error}"))

    if cancelled:
        return 130
    if args.contact_sheet:
        settings = _settings_summary(args, stage, tone_params, scan_reference, args.match_scan_exposure)
        write_contact_sheet(args, jobs, results, args.contact_sheet, input_dir.name, settings)
    return 1 if failures else 0
