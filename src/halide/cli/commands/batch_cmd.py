"""`halide batch` — process a directory of negative scans in parallel."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from halide.batch.orchestrator import default_worker_count, discover_jobs, memory_budget_warning, run_batch
from halide.batch.progress import GridProgressRenderer
from halide.cli import console
from halide.calibration.scan_consistency import assess_roll, most_common_settings, scan_gain
from halide.cli._calibration_args import (
    add_calibration_arguments,
    add_scan_arguments,
    add_stage_arguments,
    add_tone_arguments,
    maybe_save_profile,
    resolve_density_profile,
    resolve_scan_reference,
    resolve_stage,
    resolve_tone_params,
)
from halide.processing import ScanColorError, Stage, estimate_roll_density_profile, read_roll_scan_metadata


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="Directory of input linear TIFF scans")
    parser.add_argument("output_dir", help="Directory to write output TIFFs into")
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


def _match_scan_exposure(jobs, metadata, reference, roll_estimate: bool):
    """Give every job the gain that puts it at the calibration's scan exposure (see
    calibration/scan_consistency.py). Returns (jobs, the reference used)."""
    known = [settings for settings, _ in metadata.values() if settings is not None]
    if not known:
        raise SystemExit("--match-scan-exposure: none of these frames has camera exposure settings (EXIF) to match from")
    if reference is None:
        reference = most_common_settings(known)
        if roll_estimate:
            print(f"Matching every frame to the roll's most common scan exposure ({reference.describe()})")
        else:
            print(console.warning(
                f"the profile doesn't record the scan exposure it was calibrated at, so frames are "
                f"matched to the roll's most common setting ({reference.describe()}) — if the "
                "calibration frame was scanned differently, a constant colour offset remains across "
                "the roll (pass --scan-reference FRAME, or `halide profile set-scan-reference`)"
            ))
    matched, missing = [], []
    for job in jobs:
        settings, _ = metadata[str(job.input_path)]
        if settings is None:
            missing.append(job.input_path.name)
            matched.append(job)
        else:
            matched.append(replace(job, scan_gain=scan_gain(settings, reference)))
    adjusted = sum(1 for job in matched if abs(job.scan_gain - 1.0) > 1e-9)
    print(f"Matching scan exposure to {reference.describe()}: {adjusted}/{len(jobs)} frame(s) adjusted")
    if missing:
        print(console.warning(f"no camera exposure settings (EXIF) in {', '.join(missing)} — left unadjusted"))
    return matched, reference


def run(args: argparse.Namespace) -> int:
    stage = resolve_stage(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs(input_dir, output_dir, suffix=args.suffix)
    if not jobs:
        print(f"No TIFF files found in {input_dir}")
        return 1

    manual_given = args.rm is not None or args.bm is not None or args.rs != 1.0 or args.bs != 1.0
    other_source_given = bool(args.profile) or manual_given or args.auto_density
    if args.auto_density_roll and other_source_given:
        raise SystemExit(
            "--auto-density-roll cannot be combined with --profile/manual overrides/--auto-density"
        )

    metadata = read_roll_scan_metadata([job.input_path for job in jobs])
    report = assess_roll({Path(path).name: meta for path, meta in metadata.items()})
    if report.has_issues and not args.quiet:
        for line in report.summary_lines():
            print(console.warning(line))
        print(f"  Run `halide check {input_dir}` for details and how to scan a roll consistently.")

    per_frame_auto = args.auto_density and not args.auto_density_roll
    scan_reference = None if args.auto_density_roll else resolve_scan_reference(args)
    if args.match_scan_exposure:
        if stage is Stage.INVERT_ONLY or per_frame_auto:
            print(console.warning("--match-scan-exposure has no effect here: each frame is calibrated from itself"))
        else:
            jobs, scan_reference = _match_scan_exposure(jobs, metadata, scan_reference, args.auto_density_roll)

    if stage is Stage.INVERT_ONLY:
        density_profile, saved_tone = None, None  # unused by process_scan for this stage
    elif args.auto_density_roll:
        print(f"Estimating a shared density-balance profile from {len(jobs)} frame(s)...")
        try:
            density_profile = estimate_roll_density_profile(
                [job.input_path for job in jobs], scan_gains={str(job.input_path): job.scan_gain for job in jobs}
            )
        except ScanColorError as exc:
            raise SystemExit(str(exc))
        saved_tone = None
    else:
        density_profile, saved_tone = resolve_density_profile(args)  # profile may be None -> per-frame auto

    maybe_save_profile(args, density_profile, tone=saved_tone, scan=scan_reference)
    tone_params = resolve_tone_params(args, saved_tone=saved_tone)

    if args.workers is not None:
        workers = args.workers
        warning = memory_budget_warning(jobs, workers)
        if warning and not args.quiet:
            print(console.warning(warning))
    else:
        workers = default_worker_count(jobs)
        if not args.quiet:
            print(f"Auto-selected {workers} worker process(es) based on available memory and CPU count "
                  "(pass --workers N to override)")

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
    return 1 if failures else 0
