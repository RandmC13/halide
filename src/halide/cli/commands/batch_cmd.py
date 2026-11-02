"""`halide batch` — process a directory of negative scans in parallel."""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import shutil
import tempfile
import time
import warnings
from typing import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from halide import ignore_colour_matplotlib_warning
from halide.io.roll import find_rolls, list_scans
from halide.batch.orchestrator import (
    BatchJob,
    default_worker_count,
    jobs_for_files,
    memory_budget_warning,
    plan_compute,
    run_batch,
)
from halide.batch.progress import cancel_notice, make_renderer
from halide.cli import console
from halide.calibration.scan_consistency import assess_roll, most_common_settings, scan_gain
from halide.cli._device_args import add_device_argument, resolve_device_arg
from halide.cli._naming import CLASH_FIX, add_name_argument, naming_for
from halide.cli._help import add_workers_argument
from halide.cli._output_policy import (
    add_dry_run_argument,
    add_output_policy_arguments,
    check_input_folder,
    check_output_folder,
    OutputPolicy,
    plan_bulk_jobs,
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
    missing_profile_message,
    choose_calibration_source,
    resolve_density_profile,
    resolve_scan_reference,
    resolve_stage,
    check_no_ignored_flags,
    resolve_tone_params,
)
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
    parser.add_argument(
        "--into", metavar="PATH",
        help="Develop into PATH, resolved against each roll's scan folder (--into ../inversion puts "
        "Roll1/pre-processed's prints in Roll1/inversion). An alternative to the output folder",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Treat every folder of scans under the input folder as a roll, and develop each one in "
        "turn — with --into, or into a subfolder of the output folder named for the roll. Folders "
        "of halide's own output are skipped",
    )
    add_contact_layout_arguments(parser)
    add_name_argument(parser, "Text to add to each output file name, before .tif")
    add_output_policy_arguments(parser, saves_profile=True)
    add_dry_run_argument(parser)
    add_log_arguments(parser)

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


def _destination(args: argparse.Namespace, would_create: bool = False) -> str:
    if args.output_dir is None:
        return f"contact sheet {args.contact_sheet} only (a preview — no full-size TIFFs kept)"
    destination = f"{args.output_dir}"
    if would_create:
        destination += " (would be created)"
    return f"{destination} + contact sheet {args.contact_sheet}" if args.contact_sheet else destination


# The automatic tiers are approximate (see auto calibration's known limits); say what's better.
AUTO_CALIBRATION_ADVICE = (
    "automatic estimate - for the most faithful colour, pick neutral points with `halide open`"
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


def _no_calibration_text(input_dir: Path) -> str:
    """The Calibration row of a dry run given no calibration source: a real run would ask (in a
    terminal) or refuse; a dry run does neither, so it lists what the real prompt would offer."""
    from halide.calibration.profile_store import list_profiles

    readable = [(stem, profile) for stem, profile, problem in list_profiles() if problem is None]
    newest = sorted(readable, key=lambda item: item[1].created_at or "", reverse=True)[:3]
    text = "none given — a real run would ask you to choose"
    if newest:
        return f"{text} (saved: {', '.join(stem for stem, _ in newest)})"
    return f"{text}; no saved profiles yet (make one with: halide open {input_dir})"


def _check_profile_exists(name: str) -> None:
    from halide.calibration.profile_store import resolve_profile_path

    try:
        resolve_profile_path(name)
    except FileNotFoundError as exc:
        raise SystemExit(missing_profile_message(name, exc)) from exc


def _dry_save_profile_row(args: argparse.Namespace, sheet: console.RunSheet, profile, no_source: bool,
                          stage: Stage) -> None:
    """--save-profile-as in a dry run: nothing is saved, but the same refusal a real run would make
    (nothing single to save) still stops it, and a name already taken is warned about."""
    name = args.save_profile_as
    if not name:
        return
    if (not no_source or stage is Stage.INVERT_ONLY) and profile is None and not args.auto_density_roll:
        maybe_save_profile(args, None)  # raises the real run's explanation
    from halide.calibration.profile_store import find_profile, validate_profile_name

    try:
        validate_profile_name(name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    sheet.row("Calibration", f"would save as {name!r} — reuse with --profile {name}")
    if find_profile(name) is not None and policy_from_args(args) is not OutputPolicy.OVERWRITE:
        sheet.warn("Calibration", f"a profile named {name!r} already exists — a real run would ask (or --overwrite)")


def _output_text(stage: Stage, tone_params) -> str:
    if stage is Stage.DENSITY_ONLY:
        return "density balance only — not inverted (--density-only)"
    if tone_params.mode == "linear":
        return "flat positive for editing elsewhere (--output flat)"
    grade = f"{tone_params.contrast:.2f}" if tone_params.contrast is not None else "fitted per frame"
    exposure = f"{tone_params.exposure:+.2f} for the whole roll" if tone_params.exposure is not None else "fitted per frame"
    return f"print{_SEP}grade {grade}{_SEP}exposure {exposure}"


def _profile_stock(args: argparse.Namespace) -> str | None:
    """`{stock}` for --name: the --profile's film stock, when the template uses it. A profile that
    can't be loaded is reported where it is used, not here."""
    if not args.name or "{stock}" not in args.name or not args.profile:
        return None
    from halide.calibration.profile_store import load_profile, resolve_profile_path

    try:
        return load_profile(resolve_profile_path(args.profile)).film_stock
    except (OSError, ValueError):
        return None


@dataclass
class RollPlan:
    """One roll of a run: its scans, where it develops to, and what is left to do for it. `name` is
    set only with --recursive (a single-roll run names itself after its folder, as it always has)."""

    name: str | None
    input_dir: Path
    left_out: object
    output_dir: Path | None
    sheet_path: Path | None
    jobs: list[BatchJob]  # every frame of the roll
    todo: list[BatchJob] = field(default_factory=list)  # the frames left to develop
    skipped_frames: int = 0
    kept: list[str] = field(default_factory=list)
    build_sheet: bool = True
    existing: int = 0
    sheet_exists: bool = False

    @property
    def label(self) -> str:
        return self.name or self.input_dir.resolve().name


@dataclass
class RollOutcome:
    code: int
    developed: int = 0
    failed: int = 0


def _roll_destination(args: argparse.Namespace, folder: Path, name: str) -> Path | None:
    """Where a roll develops to: --into resolved against its scan folder, else the output folder
    (one subfolder per roll with --recursive), else nowhere (a contact-sheet preview)."""
    if args.into:
        into = Path(args.into)
        return into if into.is_absolute() else Path(os.path.normpath(folder / into))
    if args.output_dir is None:
        return None
    return Path(args.output_dir) / name if args.recursive else Path(args.output_dir)


def _check_recursive_save(args: argparse.Namespace) -> None:
    if args.recursive and args.save_profile_as and (args.auto_density or args.auto_density_roll):
        raise SystemExit("--save-profile-as with --recursive needs one calibration for every roll — each roll "
                         "gets its own estimate with --auto-density and --auto-density-roll; save one from a "
                         "single roll")


def _check_destination_flags(args: argparse.Namespace) -> None:
    if args.into and args.output_dir:
        raise SystemExit("--into and an output folder both say where to develop to — give one of them")
    if args.output_dir is None and not args.contact_sheet and not args.into:
        raise SystemExit("give an output directory, --into FOLDER, --contact-sheet SHEET (a preview), or both"
                         if args.recursive else
                         "give an output directory, --contact-sheet SHEET (a preview), or both")
    if args.output_dir is None and not args.into:
        if args.recursive:
            raise SystemExit("a contact-sheet preview of several rolls needs --into or an output folder to "
                             "put each sheet in")
        if args.name or args.suffix:
            flag = "--name" if args.name else "--suffix"
            raise SystemExit(f"{flag} names the developed frames, and a contact-sheet preview keeps none — "
                             f"give an output directory, or drop {flag}")
    if args.recursive and args.contact_sheet and Path(args.contact_sheet).name != args.contact_sheet:
        raise SystemExit(f"--contact-sheet {args.contact_sheet} — with --recursive give a file name only "
                         f"(it is written into each roll's output folder)")
    _check_recursive_save(args)


def _find_plans(args: argparse.Namespace, input_dir: Path) -> tuple[list[RollPlan], object]:
    """The rolls of this run, each with its jobs (outputs named, clashes refused). Returns (plans,
    what --recursive left out, or None). A destination is excluded from discovery by working out
    every candidate's destination first, so a new roll's `inversion/` is never taken for a roll."""
    if args.recursive:
        rolls, skipped = find_rolls(input_dir)
        destinations = {d.resolve() for r in rolls if (d := _roll_destination(args, r.folder, r.name))}
        if destinations:
            rolls, skipped = find_rolls(input_dir, exclude=lambda folder: folder.resolve() in destinations)
        if args.into and Path(args.into).is_absolute() and len(rolls) > 1:
            raise SystemExit(f"--into {args.into} is an absolute path, so all {len(rolls)} rolls would be "
                             f"developed into it — use a relative one such as ../inversion, which is "
                             f"resolved against each roll's scan folder")
        found = [(r.name, r.folder, r.files, r.left_out) for r in rolls]
    else:
        files, left_out = list_scans(input_dir, skip_positives=True)
        skipped = None
        found = [(None, input_dir, files, left_out)]
    stock = _profile_stock(args)
    plans = []
    for name, folder, files, left_out in found:
        label = name or input_dir.resolve().name
        output_dir = _roll_destination(args, folder, label)
        sheet_path = None
        if args.contact_sheet:
            sheet_path = output_dir / args.contact_sheet if args.recursive else Path(args.contact_sheet)
        if output_dir is not None:
            check_output_folder(output_dir)  # the folder itself is made when its roll starts
            jobs = jobs_for_files(files, output_dir, suffix=args.suffix,
                                  name=naming_for(args, label, stock) if args.name else None)
        else:
            jobs = [BatchJob(input_path=f, output_path=None) for f in files]
        plans.append(RollPlan(name, folder, left_out, output_dir, sheet_path, jobs))
    seen: dict[str, RollPlan] = {}
    for plan in plans:
        if plan.output_dir is None or not plan.jobs:
            continue
        first = seen.setdefault(str(plan.output_dir.resolve()).casefold(), plan)
        if first is not plan:
            raise SystemExit(f"{first.label} and {plan.label} would both be developed into {plan.output_dir} — "
                             f"give each roll its own output folder")
    return plans, skipped


def _decide_existing(args: argparse.Namespace, plans: list[RollPlan]) -> None:
    """Who develops what, after applying the overwrite policy to every roll's outputs at once — so
    a terminal is asked once, and a script refused once, however many rolls there are."""
    for plan in plans:
        plan.sheet_exists = bool(args.dry_run and plan.sheet_path is not None and plan.sheet_path.exists())
    if plans[0].output_dir is None:
        plan = plans[0]  # a contact-sheet preview: one roll, no full-size outputs to protect
        check_log_path(args.log, [(j.input_path, j.input_path) for j in plan.jobs])
        plan.todo = plan.jobs
        if args.dry_run:
            plan.build_sheet = not (policy_from_args(args) is OutputPolicy.SKIP_EXISTING and plan.sheet_path.exists())
        else:
            plan.build_sheet = bool(resolve_existing([(plan.sheet_path, plan.sheet_path)],
                                                     policy_from_args(args), interactive=is_interactive()))
        return
    extra = [(p.sheet_path, p.sheet_path) for p in plans if p.sheet_path is not None] or None
    fix = CLASH_FIX if args.name else "rename one of them"
    everything = [job for plan in plans for job in plan.jobs]
    if args.dry_run:  # never asks, never refuses for want of a terminal
        todo, _, kept_outputs, _ = plan_bulk_jobs(everything, args, extra_pairs=extra, fix=fix)
    else:
        todo, _, kept_outputs = resolve_bulk_jobs(everything, args, interactive=is_interactive(),
                                                  extra_pairs=extra, fix=fix)
    developing = {j.output_path for j in todo}
    for plan in plans:
        plan.todo = [j for j in plan.jobs if j.output_path in developing]
        plan.skipped_frames = len(plan.jobs) - len(plan.todo)
        plan.kept = [j.input_path.name for j in plan.jobs if j.output_path not in developing]
        plan.build_sheet = plan.sheet_path is None or plan.sheet_path in kept_outputs
        plan.existing = sum(1 for j in plan.jobs if j.output_path.exists()) if args.dry_run else 0


def _plan_args(args: argparse.Namespace, plan: RollPlan, first: bool) -> argparse.Namespace:
    """The arguments one roll runs with: where it develops to, and its own contact sheet. Only the
    first roll saves the profile — the calibration is the same for all of them."""
    if not (args.recursive or args.into):
        return args
    roll_args = copy.copy(args)
    roll_args.output_dir = plan.output_dir
    roll_args.contact_sheet = plan.sheet_path
    if not first:
        roll_args.save_profile_as = None
    return roll_args


def run(args: argparse.Namespace) -> int:
    check_no_ignored_flags(args)
    stage = resolve_stage(args)

    input_dir = Path(args.input_dir)
    check_input_folder(input_dir)
    _check_destination_flags(args)
    if args.contact_sheet:
        from halide.io.contact_sheet import check_sheet_path

        try:
            check_sheet_path(Path(args.contact_sheet))
        except ValueError as exc:
            raise SystemExit(str(exc))

    plans, rolls_skipped = _find_plans(args, input_dir)
    plans = [p for p in plans if p.jobs]
    if not plans:
        positives = rolls_skipped.halide_outputs if rolls_skipped else 0
        if positives or (not args.recursive and list_scans(input_dir, skip_positives=True)[1].positives):
            raise SystemExit(f"No scans to develop in {input_dir} — its TIFFs are halide positives. To re-print "
                             f"them use `halide print`; to develop again, point halide at the original scans")
        raise SystemExit(f"No TIFF files found in {input_dir} — is this the right folder? "
                         f"Subfolders are skipped (use --recursive)")
    _decide_existing(args, plans)

    def has_work(plan: RollPlan) -> bool:
        return bool(plan.todo) if plan.output_dir is not None else plan.build_sheet

    if not args.dry_run and any(has_work(p) for p in plans):
        if stage is not Stage.INVERT_ONLY:
            choose_calibration_source(args, "this roll" if len(plans) == 1 else "these rolls")  # before any sheet
        _check_recursive_save(args)  # the prompt may have chosen an automatic estimate
    # --auto-density-roll vs --profile/--auto-density is argparse's (exclusive group); only the
    # manual values need this hand check.
    if any(has_work(p) for p in plans) and args.auto_density_roll and manual_calibration_given(args):
        raise SystemExit("manual overrides (--rm/--bm/--rs/--bs) can't be combined with --auto-density-roll")

    if args.recursive:
        with console.RunSheet(quiet=args.quiet) as sheet:
            sheet.row("Rolls", f"{console.plural(len(plans), 'roll')}{_SEP}"
                               f"{frame_count(sum(len(p.jobs) for p in plans))} under {input_dir}")
            if rolls_skipped and (left := rolls_skipped.describe()):
                sheet.row("Skipped", left)

    started = time.monotonic()
    outcomes: list[RollOutcome] = []
    for i, plan in enumerate(plans):
        try:
            if plan.output_dir is not None and not args.dry_run:
                prepare_output_folder(plan.output_dir)
            outcome = _run_roll(_plan_args(args, plan, i == 0), stage, plan)
        except SystemExit as exc:
            if not args.recursive:
                raise
            print(console.error(f"{plan.label}: {exc.code if isinstance(exc.code, str) else 'stopped'}"))
            outcome = RollOutcome(1, failed=len(plan.todo))
        outcomes.append(outcome)
        if outcome.code == 130:
            break
    if not args.recursive:
        return outcomes[0].code
    return _overall_line(args, plans, outcomes, time.monotonic() - started)


def _overall_line(args: argparse.Namespace, plans: list[RollPlan], outcomes: list[RollOutcome],
                  seconds: float) -> int:
    """The line after the last roll of a --recursive run, and the exit code: 130 if cancelled, 1 if
    any roll or frame failed."""
    rolls = console.plural(len(plans), "roll")
    if any(o.code == 130 for o in outcomes):
        states = [f"{p.label} {'cancelled' if o.code == 130 else 'done'}" for p, o in zip(plans, outcomes)]
        states += [f"{p.label} not started" for p in plans[len(outcomes):]]
        print(console.cancelled(f"Cancelled — {', '.join(states)}"))
        return 130
    planned = sum(len(p.todo) for p in plans)
    if args.dry_run:
        print(console.success(f"Dry run — {rolls}{_SEP}{frame_count(planned)} would be developed"))
        return 0
    developed = sum(o.developed for o in outcomes)
    failed_rolls = [p.label for p, o in zip(plans, outcomes) if o.code != 0]
    if failed_rolls:
        print(f"{console.Style.YELLOW}{console.ICON_WARN}{console.Style.RESET} "
              f"{rolls}{_SEP}{developed} of {frame_count(planned)} developed "
              f"({planned - developed} failed: {', '.join(failed_rolls)}) in {console.human_time(seconds)}")
        return 1
    if not developed:
        print(console.success(f"{rolls}{_SEP}nothing new to develop"))
    else:
        print(console.success(f"{rolls}{_SEP}{frame_count(developed)} developed in {console.human_time(seconds)}"))
    return 0


def _run_roll(args: argparse.Namespace, stage: Stage, plan: RollPlan) -> RollOutcome:
    """One roll, from its early exits (nothing to do) to the end of its run."""
    input_dir, output_dir, sheet_path, jobs = plan.input_dir, plan.output_dir, plan.sheet_path, plan.todo
    title = plan.name or input_dir.name
    if args.dry_run and output_dir is None and not plan.build_sheet:
        print(console.success(
            f"Dry run — nothing to do, {sheet_path} already exists (--skip-existing)"
        ))
        return RollOutcome(0)
    if args.dry_run and not jobs:
        text = f"nothing to do, every output in {output_dir} already exists (--skip-existing)"
        if sheet_path is not None and plan.build_sheet:
            text += f"; the contact sheet {sheet_path} would be built from the folder"
        print(console.success(f"Dry run — {text}"))
        return RollOutcome(0)
    if output_dir is not None and not jobs:
        if sheet_path is None or not plan.build_sheet:
            print(console.success(f"Nothing to do — every output in {output_dir} already exists (--skip-existing)"))
            return RollOutcome(0)
        # Nothing left to develop, but the sheet still needs building — from the whole,
        # already-complete folder, the same way `halide contact <output_dir> <sheet>` would.
        from halide.cli._contact_sheet import discover_processed_files, write_sheet_from_folder

        files, _ = discover_processed_files(output_dir, sheet_path)
        return RollOutcome(write_sheet_from_folder(files, sheet_path, args, title, quiet=args.quiet))
    if output_dir is None and not plan.build_sheet:
        print(console.success(f"Nothing to do — {sheet_path} already exists (--skip-existing)"))
        return RollOutcome(0)

    thumbnails = Path(tempfile.mkdtemp(prefix="halide-contact-")) if plan.build_sheet and not args.dry_run else None
    try:
        if thumbnails is not None:
            jobs = [replace(job, thumbnail_path=thumbnails / f"{i:04d}.png") for i, job in enumerate(jobs)]
        return _run(
            args, stage, input_dir, jobs, roll_jobs=plan.jobs, output_dir=output_dir,
            skipped_frames=plan.skipped_frames, build_sheet=plan.build_sheet, left_out=plan.left_out,
            kept=plan.kept, existing=plan.existing, sheet_exists=plan.sheet_exists, roll_name=plan.name,
        )
    finally:
        if thumbnails is not None:
            shutil.rmtree(thumbnails, ignore_errors=True)


def _prepare(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob], sheet: console.RunSheet,
             stack: contextlib.ExitStack, skipped_frames: int = 0, left_out=None,
             roll_jobs: list[BatchJob] | None = None, existing: int = 0,
             sheet_exists: bool = False, roll_name: str | None = None):
    """Everything decided before developing starts — scan checks, scan-exposure matching,
    calibration, print settings, how the GPU is used, worker count — each reported on the run sheet
    as it's decided. On a GPU the shared GPU service starts here and runs until `stack` closes.
    `jobs` are the frames to develop; `roll_jobs` (default: the same) are every frame of the roll,
    which the roll-wide decisions are made from — they differ when resuming a roll.
    With --dry-run the same decisions are made from headers and small files only: nothing is
    created, saved, estimated or started, and what a real run would do reads as "would".
    Returns (jobs, density_profile, tone_params, scan_reference, workers, device, compute)."""
    from halide.processing import ScanColorError, estimate_roll_density_profile, read_roll_scan_metadata

    if roll_jobs is None:
        roll_jobs = jobs
    dry = args.dry_run
    would_create = dry and args.output_dir is not None and not Path(args.output_dir).exists()
    roll_row(sheet, input_dir, len(jobs), _destination(args, would_create), name=roll_name)
    skipped_row(sheet, left_out)
    cleanup_row(sheet, Path(args.output_dir) if args.output_dir else None, dry)
    if skipped_frames:
        sheet.row("Skipping", f"{frame_count(skipped_frames)} already developed")
    if dry:
        outputs_row(
            sheet, policy_from_args(args), existing, str(args.contact_sheet) if sheet_exists else None
        )

    per_frame_auto = args.auto_density and not args.auto_density_roll
    matching = args.match_scan_exposure and not (stage is Stage.INVERT_ONLY or per_frame_auto)

    metadata = read_roll_scan_metadata([job.input_path for job in roll_jobs])
    report = assess_roll({Path(path).name: meta for path, meta in metadata.items()})
    issues = report.summary_lines(exposure_corrected=matching)
    for line in issues:
        sheet.warn("Scans", line)
    if getattr(args, "exposure", None) is not None and report.exposure_inconsistent and not matching:
        sheet.warn("Scans", "one --exposure across scans digitized at different exposures prints the brighter "
                            "scans brighter: add --match-scan-exposure")
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
            roll_jobs, scan_reference = _match_scan_exposure(
                roll_jobs, metadata, scan_reference, origin, _unknown_reference_warning(args), sheet
            )
            gains = {job.input_path: job.scan_gain for job in roll_jobs}
            jobs = [replace(job, scan_gain=gains[job.input_path]) for job in jobs]

    no_source = dry and not (
        args.profile or manual_calibration_given(args) or args.auto_density or args.auto_density_roll
    )
    if stage is Stage.INVERT_ONLY:
        density_profile, saved_tone = None, None  # unused by process_scan for this stage
        sheet.row("Calibration", "none — density balance skipped (--invert-only)")
    elif no_source:
        density_profile, saved_tone = None, None
        sheet.row("Calibration", _no_calibration_text(input_dir))
    elif args.auto_density_roll and dry:
        density_profile, saved_tone = None, None
        sheet.row("Calibration", f"auto{_SEP}would estimate one profile for the whole roll from all "
                                 f"{frame_count(len(roll_jobs))}")
        sheet.note("Calibration", AUTO_CALIBRATION_ADVICE)
    elif args.auto_density_roll:
        skipped: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ignore_colour_matplotlib_warning()  # "always" would bring back colour's plotting notice
            with sheet.working("Calibration", f"estimating one profile from all {frame_count(len(roll_jobs))}…"):
                try:
                    roll_estimate = estimate_roll_density_profile(
                        [job.input_path for job in roll_jobs],
                        scan_gains={str(job.input_path): job.scan_gain for job in roll_jobs},
                        on_skip=lambda path, exc: skipped.append(f"{path.name} ({exc})"),
                    )
                except (ScanColorError, ValueError) as exc:  # ValueError: no frame could vote
                    raise SystemExit(str(exc))
        density_profile, saved_tone = roll_estimate.profile, None
        used = roll_estimate.used
        sheet.row("Calibration", f"auto{_SEP}one profile for the whole roll, from {frame_count(used)}")
        for name in skipped:
            sheet.warn("Calibration", f"skipped unreadable {name}")
        for caught_warning in caught:
            sheet.warn("Calibration", str(caught_warning.message))
        sheet.note("Calibration", AUTO_CALIBRATION_ADVICE)
    else:
        if dry and args.profile:
            _check_profile_exists(args.profile)  # the real error, without the real run's "did you mean?" prompt
        density_profile, saved_tone = resolve_density_profile(args)  # profile may be None -> per-frame auto
        sheet.row("Calibration", _calibration_text(args, density_profile, saved_tone))
        if density_profile is None:
            sheet.note("Calibration", AUTO_CALIBRATION_ADVICE)

    if dry:
        _dry_save_profile_row(args, sheet, density_profile, no_source, stage)
        saved_path = None
    else:
        saved_path = maybe_save_profile(args, density_profile, tone=saved_tone, scan=scan_reference, announce=False)
    if saved_path is not None:
        sheet.ok("Calibration", f"saved as {args.save_profile_as!r} — reuse with --profile {args.save_profile_as}")
    tone_params = resolve_tone_params(args, saved_tone=saved_tone)
    sheet.row("Output", _output_text(stage, tone_params))

    # Resolved after choose_calibration_source (called by _run, before this sheet opens) and
    # before the sheet closes (the calibration prompt must come first). The workers
    # develop on it (run_batch): on a GPU through the shared GPU service, or — if it can't be used —
    # with a CUDA context per worker, whose default count also fits the card's memory.
    device = resolve_device_arg(args, isolated=True)  # the parent never computes on the card
    # A dry run only plans: the GPU service is never started.
    compute = plan_compute(jobs, device, "develop") if dry else start_compute(stack, sheet, jobs, device)
    compute_row(sheet, device, compute, planned=dry)

    workers = choose_workers(
        args, jobs, sheet,
        default_count=lambda jobs: default_worker_count(jobs, device=device),
        budget_warning=memory_budget_warning,
        device=device,
        compute=compute,
    )
    return jobs, density_profile, tone_params, scan_reference, workers, device, compute


def _run(args: argparse.Namespace, stage: Stage, input_dir: Path, jobs: list[BatchJob], *,
         roll_jobs: list[BatchJob] | None = None, output_dir: Path | None = None, skipped_frames: int = 0,
         build_sheet: bool = True,
         left_out=None, kept: Sequence[str] = (), existing: int = 0,
         sheet_exists: bool = False, roll_name: str | None = None) -> RollOutcome:
    # The GPU service (if any) starts while the run sheet is open and stops once the pool is done.
    with contextlib.ExitStack() as stack:
        with console.RunSheet(quiet=args.quiet) as sheet:
            jobs, density_profile, tone_params, scan_reference, workers, device, compute = _prepare(
                args, stage, input_dir, jobs, sheet, stack, skipped_frames=skipped_frames, left_out=left_out,
                roll_jobs=roll_jobs, existing=existing, sheet_exists=sheet_exists, roll_name=roll_name,
            )

        if args.dry_run:
            print(dry_run_line(
                "batch", len(jobs), str(output_dir), kept=len(kept), existing=existing,
                policy=policy_from_args(args),
                sheet=str(args.contact_sheet) if sheet_exists else None,
                preview_sheet=str(args.contact_sheet) if output_dir is None else None,
            ))
            if args.name:
                print_rename_list(jobs)
            return RollOutcome(0)

        record = RunRecord(args, "develop", output_dir)
        record.start(sheet, jobs, list(kept))
        try:
            renderer = make_renderer(len(jobs), "invert", quiet=args.quiet, names=[j.input_path.name for j in jobs])
            job_index = {job: i for i, job in enumerate(jobs)}

            def on_start(job):
                renderer.mark_processing(job_index[job])

            def on_result(result):
                record.add(result)
                renderer.report(job_index[result.job], result)

            renderer.start()
            results = run_batch(
                jobs, stage, density_profile, tone_params, max_workers=workers, on_result=on_result,
                on_start=on_start, on_cancel=cancel_notice(renderer), thumbnail_long_edge=args.frame_width,
                device=device, compute=compute,
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
    outcome = RollOutcome(0, developed=len(results) - len(failures), failed=len(failures))

    if cancelled:
        return replace(outcome, code=130)

    sheet_failed = False
    if args.contact_sheet and build_sheet:
        if skipped_frames and output_dir is not None:
            # Frames --skip-existing left alone never got a thumbnail from this run's own worker
            # pool, so the sheet has to cover the whole roll the way `halide contact <output_dir>
            # <sheet>` does: thumbnail every processed file on disk, not just this run's `jobs`.
            from halide.cli._contact_sheet import discover_processed_files, write_sheet_from_folder

            sheet_path = Path(args.contact_sheet)
            files, _ = discover_processed_files(output_dir, sheet_path)
            sheet_code = write_sheet_from_folder(files, sheet_path, args, roll_name or input_dir.name, quiet=args.quiet)
            if sheet_code == 130:
                return replace(outcome, code=130)
            sheet_failed = sheet_code != 0
        else:
            settings = _settings_summary(args, stage, tone_params, scan_reference, args.match_scan_exposure)
            write_contact_sheet(args, jobs, results, args.contact_sheet, roll_name or input_dir.name, settings)
    return replace(outcome, code=1 if (failures or sheet_failed) else 0)
