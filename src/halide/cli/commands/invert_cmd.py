"""`halide invert` — process a single negative scan into a positive image."""

from __future__ import annotations

import argparse

from halide.cli._calibration_args import (
    add_calibration_arguments,
    add_stage_arguments,
    add_tone_arguments,
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
    stage = resolve_stage(args)
    density_profile = None if stage is Stage.INVERT_ONLY else resolve_density_profile(args)
    maybe_save_profile(args, density_profile)
    tone_params = resolve_tone_params(args)

    try:
        process_scan(args.input, args.output, stage, density_profile, tone_params)
    except ScanColorError as exc:
        raise SystemExit(str(exc))

    return 0
