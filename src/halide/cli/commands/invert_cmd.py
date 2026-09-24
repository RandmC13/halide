"""`halide invert` — process a single negative scan into a positive image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from halide.cli import console
from halide.cli._calibration_args import (
    add_calibration_arguments,
    add_scan_arguments,
    add_stage_arguments,
    add_tone_arguments,
    describe_resolved_tone,
    maybe_save_profile,
    choose_calibration_source,
    resolve_density_profile,
    resolve_scan_reference,
    resolve_stage,
    resolve_tone_params,
)
from halide.calibration.scan_consistency import scan_gain as compute_scan_gain
from halide.io.scan_metadata import read_scan_metadata
from halide.processing import ScanColorError, Stage, process_scan


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input linear TIFF scan of a negative")
    parser.add_argument("output", help="Output TIFF path")
    add_stage_arguments(parser)
    add_calibration_arguments(parser, allow_pick=True)
    add_tone_arguments(parser)
    add_scan_arguments(parser)


def _resolve_scan_gain(args: argparse.Namespace, calibrated_here: bool) -> tuple[float, object]:
    """(gain for this frame, scan settings to record if the calibration is saved). A calibration
    solved from this very frame (--pick, --auto-density) is at this frame's scan exposure by
    definition, so there's nothing to match."""
    frame_settings, _ = read_scan_metadata(args.input)
    if calibrated_here:
        if args.match_scan_exposure:
            print(console.warning("--match-scan-exposure has no effect here: the calibration comes from this frame itself"))
        return 1.0, frame_settings

    reference = resolve_scan_reference(args)
    if not args.match_scan_exposure:
        if reference is not None and frame_settings is not None and frame_settings != reference:
            print(console.warning(
                f"this frame was digitized at {frame_settings.describe()}, the profile at "
                f"{reference.describe()} — its colours will be shifted, visibly so for a stop or more (pass --match-scan-exposure "
                "to correct, see `halide check`)"
            ))
        return 1.0, reference
    if reference is None:
        raise SystemExit(
            "--match-scan-exposure needs the camera exposure the profile was calibrated at, but it "
            "isn't recorded in this profile — pass --scan-reference FRAME (the frame you calibrated "
            "on), or save the profile again from that frame so it records it"
        )
    if frame_settings is None:
        raise SystemExit(f"--match-scan-exposure: {args.input} has no camera exposure settings (EXIF) to match from")
    gain = compute_scan_gain(frame_settings, reference)
    print(f"Matching scan exposure: {frame_settings.describe()} → {reference.describe()} (×{gain:.3g})")
    return gain, reference


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    stage = resolve_stage(args)
    if stage is Stage.INVERT_ONLY:
        density_profile, saved_tone = None, None
    else:
        choose_calibration_source(args, "this image")  # sets args.profile etc. before anything reads them
        density_profile, saved_tone = resolve_density_profile(args)
    manual_given = args.rm is not None or args.bm is not None or args.rs != 1.0 or args.bs != 1.0
    calibrated_here = stage is not Stage.INVERT_ONLY and not args.profile and not manual_given
    gain, scan_reference = _resolve_scan_gain(args, calibrated_here)
    maybe_save_profile(args, density_profile, tone=saved_tone, scan=scan_reference)
    tone_params = resolve_tone_params(args, saved_tone=saved_tone)

    output_path = Path(args.output)
    if output_path.exists() and not console.confirm_overwrite(output_path):
        return 1

    start = time.monotonic()
    label = f"{console.VERB['invert']} {input_path.name}..."
    try:
        with console.themed_animation(
            console.TANK_FRAMES,
            label,
            min_width=console.TANK_MIN_SIZE[0],
            min_height=console.TANK_MIN_SIZE[1],
            interval=0.5,
        ):
            resolved = process_scan(args.input, args.output, stage, density_profile, tone_params, scan_gain=gain)
    except ScanColorError as exc:
        raise SystemExit(str(exc))
    if resolved is not None:
        print(describe_resolved_tone(resolved))

    elapsed = time.monotonic() - start
    size = output_path.stat().st_size
    print(
        console.success(
            f"{console.VERB_PAST['invert']} → {output_path} "
            f"({console.human_time(elapsed)}, {console.human_bytes(size)})"
        )
    )
    return 0
