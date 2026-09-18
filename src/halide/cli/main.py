from __future__ import annotations

import argparse
import sys

from halide.cli.commands import batch_cmd, calibrate_cmd, export_cmd, invert_cmd, profile_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halide", description="Invert scanned color negative film into a positive image."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    invert_parser = subparsers.add_parser("invert", help="Invert a single negative scan")
    invert_cmd.add_arguments(invert_parser)

    batch_parser = subparsers.add_parser("batch", help="Invert a directory of negative scans in parallel")
    batch_cmd.add_arguments(batch_parser)

    export_parser = subparsers.add_parser(
        "export", help="Convert a processed ACEScg TIFF into a delivery-ready sRGB PNG/JPEG"
    )
    export_cmd.add_arguments(export_parser)

    profile_parser = subparsers.add_parser("profile", help="Manage saved calibration profiles")
    profile_cmd.add_arguments(profile_parser)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Launch the anchor-frame calibration picker GUI"
    )
    calibrate_cmd.add_arguments(calibrate_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "invert":
        return invert_cmd.run(args)
    if args.command == "batch":
        return batch_cmd.run(args)
    if args.command == "export":
        return export_cmd.run(args)
    if args.command == "profile":
        return profile_cmd.run(args)
    if args.command == "calibrate":
        return calibrate_cmd.run(args)

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
