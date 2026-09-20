"""`halide batch` — process a directory of negative scans in parallel."""

from __future__ import annotations

import argparse
from pathlib import Path

from halide.batch.orchestrator import default_worker_count, discover_jobs, memory_budget_warning, run_batch
from halide.batch.progress import GridProgressRenderer
from halide.cli import console
from halide.cli._calibration_args import (
    add_calibration_arguments,
    add_stage_arguments,
    add_tone_arguments,
    maybe_save_profile,
    resolve_density_profile,
    resolve_stage,
    resolve_tone_params,
)
from halide.processing import ScanColorError, Stage, estimate_roll_density_profile


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

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel worker processes (default: auto-selected from available memory "
        "and CPU count — full-resolution scans are memory-heavy enough that RAM, not CPU threads, "
        "is usually the real limit; pass this to override the auto-selected count)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress display")


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

    if stage is Stage.INVERT_ONLY:
        density_profile = None  # unused by process_scan for this stage; identity applies regardless
    elif args.auto_density_roll:
        print(f"Estimating a shared density-balance profile from {len(jobs)} frame(s)...")
        try:
            density_profile = estimate_roll_density_profile([job.input_path for job in jobs])
        except ScanColorError as exc:
            raise SystemExit(str(exc))
    else:
        density_profile = resolve_density_profile(args)  # may be None -> per-frame auto

    maybe_save_profile(args, density_profile)
    tone_params = resolve_tone_params(args)

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
