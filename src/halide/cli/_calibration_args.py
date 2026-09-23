"""Shared CLI argument definitions + resolution for pipeline stage and calibration source
selection, used by both the `invert` and `batch` subcommands so they don't duplicate the same
flags/logic."""

from __future__ import annotations

import argparse
import difflib
import sys

from halide.calibration.profile_store import (
    list_profiles,
    load_profile,
    load_tone_override,
    resolve_profile_path,
    save_named_profile,
)
from halide.cli import console
from halide.core.types import DensityProfile, ToneCurveParams
from halide.processing import Stage


def _suggest_profile_name(name: str) -> str | None:
    names = [n for n, _ in list_profiles()]
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    return matches[0] if matches else None


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--invert-only", action="store_true", help="Skip density balance (invert + tone-render only)"
    )
    stage_group.add_argument(
        "--density-only", action="store_true", help="Skip inversion and tone-render (density balance only)"
    )


def add_calibration_arguments(
    parser: argparse.ArgumentParser, *, allow_auto: bool = True, allow_pick: bool = False
) -> None:
    parser.add_argument(
        "--profile", help="A saved calibration profile — either a file path or a saved profile's name"
    )
    parser.add_argument("--rm", type=float, help="Red channel white-balance multiplier (manual calibration)")
    parser.add_argument("--bm", type=float, help="Blue channel white-balance multiplier (manual calibration)")
    parser.add_argument(
        "--rs", type=float, default=1.0, help="Red channel density-balance scale (manual calibration, default: 1.0)"
    )
    parser.add_argument(
        "--bs", type=float, default=1.0, help="Blue channel density-balance scale (manual calibration, default: 1.0)"
    )
    if allow_auto:
        parser.add_argument(
            "--auto-density",
            action="store_true",
            help="Automatically estimate density balance from the image itself (approximate — "
            "prefer a saved --profile from a real calibration when you have one)",
        )
    if allow_pick:
        parser.add_argument(
            "--pick",
            action="store_true",
            help="Interactively pick shadow/highlight neutral points in a small GUI window, then "
            "use that calibration for this run only (combine with --save-profile-as to also keep "
            "it for later)",
        )
    parser.add_argument(
        "--save-profile-as",
        metavar="NAME",
        help="Save the calibration profile used for this run (however it was obtained — manual, "
        "loaded, automatic, or picked) under NAME for reuse via --profile NAME next time",
    )


def add_tone_arguments(parser: argparse.ArgumentParser, *, allow_output_mode: bool = True) -> None:
    """`allow_output_mode=False` is for `halide print`, whose whole job is the print output — it
    only takes the exposure/contrast flags."""
    if allow_output_mode:
        parser.add_argument(
            "--output",
            dest="output_mode",
            choices=("print", "flat"),
            default=None,
            help="print (default): a finished print — exposure and paper grade fitted to this "
            "negative, then the paper curve. flat: white balance + density balance + invert + one "
            "global exposure scale only, a minimal-bias linear positive for editing elsewhere "
            "(e.g. darktable, then `halide print`)",
        )
        parser.add_argument(
            "--linear-output", action="store_true", help="Same as --output flat (kept for compatibility)"
        )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Print exposure (where the negative sits on the paper curve, in density units). "
        "Default: fitted per image so the negative's highlights land on the paper's highlight "
        "point. Pass a value to pin it, e.g. to match a look across a whole roll.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="Paper grade, 0-1 (1.0 = the untouched reference paper; lower = softer). Default: "
        "fitted per image so the negative's density range fills the paper's range (capped at 1.0), "
        "or a value saved into the resolved --profile's calibration if it has one.",
    )


def resolve_stage(args: argparse.Namespace) -> Stage:
    if args.invert_only:
        return Stage.INVERT_ONLY
    if args.density_only:
        return Stage.DENSITY_ONLY
    return Stage.FULL


def resolve_tone_params(args: argparse.Namespace, saved_tone: ToneCurveParams | None = None) -> ToneCurveParams:
    """Precedence for exposure/contrast: an explicit CLI flag wins, then a tone override saved
    into the resolved calibration profile (see calibration/profile_store.py's `tone` sidecar,
    written by the GUI's Fine-tune controls), then None = fitted per image (see
    core.tone_render.fit_print). The flat/linear output is CLI-flag-only — never inherited from
    `saved_tone`, deliberately (silently changing output format felt like the wrong kind of thing
    for a saved profile to do by default)."""
    exposure = args.exposure if args.exposure is not None else (saved_tone.exposure if saved_tone else None)
    contrast = args.contrast if args.contrast is not None else (saved_tone.contrast if saved_tone else None)
    output_mode = getattr(args, "output_mode", None)
    linear_output = getattr(args, "linear_output", False)
    if linear_output and output_mode == "print":
        raise SystemExit("--linear-output conflicts with --output print")
    flat = linear_output or output_mode == "flat"
    return ToneCurveParams(mode="linear" if flat else "paper", exposure=exposure, contrast=contrast)


def describe_resolved_tone(resolved) -> str:
    """One line naming the printing decision actually made, so it's visible and reproducible
    (pass the same numbers back as --exposure/--contrast to pin them)."""
    if resolved.mode == "linear":
        return f"flat output: exposure scale ×{resolved.linear_scale:.4g}"
    return f"print: grade (--contrast) {resolved.contrast:.3f}, exposure (--exposure) {resolved.exposure:+.3f}"


def resolve_density_profile(
    args: argparse.Namespace,
) -> tuple[DensityProfile | None, ToneCurveParams | None]:
    """Returns (profile, saved_tone) - profile=None means "compute automatically per-frame" (only
    call this when the pipeline stage actually needs a density profile at all, i.e. not
    Stage.INVERT_ONLY); saved_tone is an optional exposure/contrast override that came bundled with
    the resolved profile (from a saved profile's "tone" sidecar, or from the GUI's Fine-tune
    controls during --pick), to be passed into resolve_tone_params."""
    manual_given = args.rm is not None or args.bm is not None or args.rs != 1.0 or args.bs != 1.0
    auto_given = getattr(args, "auto_density", False)
    pick_given = getattr(args, "pick", False)

    if sum([bool(args.profile), manual_given, auto_given, pick_given]) > 1:
        raise SystemExit(
            "--profile, manual overrides (--rm/--bm/--rs/--bs), --auto-density, and --pick are "
            "mutually exclusive"
        )

    if args.profile:
        try:
            path = resolve_profile_path(args.profile)
            return load_profile(path), load_tone_override(path)
        except FileNotFoundError as exc:
            suggestion = _suggest_profile_name(args.profile)
            if suggestion and console.confirm(f"No profile named '{args.profile}' — did you mean '{suggestion}'?"):
                path = resolve_profile_path(suggestion)
                return load_profile(path), load_tone_override(path)
            hint = f" — did you mean '{suggestion}'?" if suggestion else ""
            raise SystemExit(f"{exc}{hint}") from exc
    if manual_given:
        return (
            DensityProfile(
                white_balance=(args.rm if args.rm is not None else 1.0, 1.0, args.bm if args.bm is not None else 1.0),
                density_scale=(args.rs, 1.0, args.bs),
                source="manual",
            ),
            None,
        )
    if auto_given:
        return None, None
    if pick_given:
        # Deferred import: the rest of the CLI must not require Qt/a display (see
        # cli/commands/calibrate_cmd.py's own deferred import for the same reason). --pick is
        # currently invert-only (add_calibration_arguments(allow_pick=...) gates this), so
        # args.input is always present whenever pick_given is True.
        from halide.gui.quick_pick import run_quick_pick

        result = run_quick_pick(args.input)
        if result is None:
            raise SystemExit("no calibration picked — closed without using a calibration")
        return result

    if sys.stdin.isatty():
        options = []
        if hasattr(args, "pick"):
            options.append(("pick", "Pick shadow/highlight points interactively now"))
        options.append(("auto", "Use --auto-density (quick automatic estimate)"))
        options.append(("cancel", "Cancel"))
        choice = console.menu("No calibration source given for this image. What would you like to do?", options)
        if choice == "pick":
            from halide.gui.quick_pick import run_quick_pick

            result = run_quick_pick(args.input)
            if result is None:
                raise SystemExit("no calibration picked — closed without using a calibration")
            return result
        if choice == "auto":
            return None, None

    pick_hint = " --pick to choose interactively," if hasattr(args, "pick") else ""
    raise SystemExit(
        "density balance requires a calibration source: --profile <name-or-path> (see `halide "
        "calibrate --save-profile-as NAME` to create one), manual overrides "
        f"(--rm/--bm/--rs/--bs), --auto-density for a quick approximate guess,{pick_hint} or "
        "--invert-only to skip density balance entirely"
    )


def maybe_save_profile(
    args: argparse.Namespace, profile: DensityProfile | None, tone: ToneCurveParams | None = None
) -> None:
    """Save the resolved profile under --save-profile-as, if given. `profile=None` means no
    single profile was computed here (per-frame --auto-density produces a different profile per
    image; --invert-only skips density balance entirely) — that is an error if the user asked to
    save one. `tone` (e.g. from a --pick session's Fine-tune controls) is saved alongside it."""
    save_as = getattr(args, "save_profile_as", None)
    if not save_as:
        return
    if profile is None:
        raise SystemExit(
            "--save-profile-as needs a single resolved profile to save, but none was computed here "
            "(per-frame --auto-density produces a different profile per image, and --invert-only "
            "skips density balance entirely — use --auto-density-roll in `batch` for a single "
            "shared automatic profile, or a manual/--profile source instead)"
        )
    path = save_named_profile(profile, save_as, tone=tone)
    print(console.success(f"Saved calibration profile as {save_as!r} ({path})"))
