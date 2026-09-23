"""`halide invert` — process a single negative scan into a positive image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from halide.cli import console
from halide.cli._calibration_args import (
    add_calibration_arguments,
    add_stage_arguments,
    add_tone_arguments,
    describe_resolved_tone,
    maybe_save_profile,
    resolve_density_profile,
    resolve_stage,
    resolve_tone_params,
)
from halide.processing import ScanColorError, Stage, process_scan


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input linear TIFF scan of a negative")
    parser.add_argument("output", help="Output TIFF path")
    add_stage_arguments(parser)
    add_calibration_arguments(parser, allow_pick=True)
    add_tone_arguments(parser)


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    stage = resolve_stage(args)
    if stage is Stage.INVERT_ONLY:
        density_profile, saved_tone = None, None
    else:
        density_profile, saved_tone = resolve_density_profile(args)
    maybe_save_profile(args, density_profile, tone=saved_tone)
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
            resolved = process_scan(args.input, args.output, stage, density_profile, tone_params)
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
