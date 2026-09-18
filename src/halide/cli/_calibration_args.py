"""Shared CLI argument definitions + resolution for pipeline stage and calibration source
selection, used by both the `invert` and `batch` subcommands so they don't duplicate the same
flags/logic."""

from __future__ import annotations

import argparse

from halide.calibration.profile_store import load_profile, resolve_profile_path, save_named_profile
from halide.core.types import DensityProfile, ToneCurveParams
from halide.processing import Stage


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--invert-only", action="store_true", help="Skip density balance (invert + tone-render only)"
    )
    stage_group.add_argument(
        "--density-only", action="store_true", help="Skip inversion and tone-render (density balance only)"
    )


def add_calibration_arguments(parser: argparse.ArgumentParser, *, allow_auto: bool = True) -> None:
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
    parser.add_argument(
        "--save-profile-as",
        metavar="NAME",
        help="Save the calibration profile used for this run (however it was obtained — manual, "
        "loaded, or automatic) under NAME for reuse via --profile NAME next time",
    )


def add_tone_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--linear-output", action="store_true", help="Skip the tone-render curve; write unbounded linear values"
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Tone-render curve exposure offset. Default: auto-computed per image from its own "
        "shadow statistics (recommended — a single fixed value doesn't correctly position every "
        "scan's density range). Pass an explicit value to pin it, e.g. to match a look across a "
        "whole roll.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.5,
        help="Tone-render curve contrast, 0-1 (default: 0.5). 1.0 is the untouched reference paper "
        "curve (very high contrast — can amplify small calibration residuals into visible color "
        "casts); lower is the digital equivalent of a softer paper grade.",
    )


def resolve_stage(args: argparse.Namespace) -> Stage:
    if args.invert_only:
        return Stage.INVERT_ONLY
    if args.density_only:
        return Stage.DENSITY_ONLY
    return Stage.FULL


def resolve_tone_params(args: argparse.Namespace) -> ToneCurveParams:
    return ToneCurveParams(
        mode="linear" if args.linear_output else "paper", exposure=args.exposure, contrast=args.contrast
    )


def resolve_density_profile(args: argparse.Namespace) -> DensityProfile | None:
    """Returns None to mean "compute automatically per-frame" — only call this when the pipeline
    stage actually needs a density profile at all (i.e. not Stage.INVERT_ONLY)."""
    manual_given = args.rm is not None or args.bm is not None or args.rs != 1.0 or args.bs != 1.0
    auto_given = getattr(args, "auto_density", False)

    if sum([bool(args.profile), manual_given, auto_given]) > 1:
        raise SystemExit(
            "--profile, manual overrides (--rm/--bm/--rs/--bs), and --auto-density are mutually exclusive"
        )

    if args.profile:
        return load_profile(resolve_profile_path(args.profile))
    if manual_given:
        return DensityProfile(
            white_balance=(args.rm if args.rm is not None else 1.0, 1.0, args.bm if args.bm is not None else 1.0),
            density_scale=(args.rs, 1.0, args.bs),
            source="manual",
        )
    if auto_given:
        return None
    raise SystemExit(
        "density balance requires a calibration source: --profile <path>, manual overrides "
        "(--rm/--bm/--rs/--bs), --auto-density, or --invert-only to skip density balance entirely"
    )


def maybe_save_profile(args: argparse.Namespace, profile: DensityProfile | None) -> None:
    """Save the resolved profile under --save-profile-as, if given. `profile=None` means no
    single profile was computed here (per-frame --auto-density produces a different profile per
    image; --invert-only skips density balance entirely) — that is an error if the user asked to
    save one."""
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
    path = save_named_profile(profile, save_as)
    print(f"Saved calibration profile as {save_as!r} ({path})")
