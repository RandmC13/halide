from __future__ import annotations

import argparse
import sys

from halide.cli import console
from halide.cli.commands import batch_cmd, calibrate_cmd, export_cmd, invert_cmd, print_cmd, profile_cmd

_SUBCOMMANDS = ("invert", "batch", "print", "export", "profile", "calibrate")


def _is_top_level_help(argv: list[str]) -> bool:
    """True if `-h`/`--help` would be handled by the top-level parser rather than a subcommand's
    own subparser — i.e. it appears before any subcommand token. Used to decide whether to print
    `console.help_banner()` (see its own docstring for why this isn't done via `formatter_class`
    instead)."""
    for token in argv:
        if token in _SUBCOMMANDS:
            return False
        if token in ("-h", "--help"):
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halide", description="Develop scanned color negative film into a positive image."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full Python tracebacks for unexpected errors instead of a short message "
        "(must come before the subcommand, e.g. `halide --debug invert ...`; HALIDE_DEBUG=1 works "
        "anywhere)",
    )
    # Not required: a bare `halide` shows a welcome screen (see main()) rather than argparse's
    # plain "the following arguments are required: command" error.
    subparsers = parser.add_subparsers(dest="command", required=False)

    invert_parser = subparsers.add_parser("invert", help="Invert a single negative scan")
    invert_cmd.add_arguments(invert_parser)

    batch_parser = subparsers.add_parser("batch", help="Invert a directory of negative scans in parallel")
    batch_cmd.add_arguments(batch_parser)

    print_parser = subparsers.add_parser(
        "print",
        help="Print a flat positive (from `invert --output flat`, optionally edited in darktable) "
        "onto the paper curve",
    )
    print_cmd.add_arguments(print_parser)

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
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if _is_top_level_help(raw_args):
        print(console.help_banner())

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print(console.welcome_screen())
        return 0

    if args.command == "invert":
        return invert_cmd.run(args)
    if args.command == "batch":
        return batch_cmd.run(args)
    if args.command == "print":
        return print_cmd.run(args)
    if args.command == "export":
        return export_cmd.run(args)
    if args.command == "profile":
        return profile_cmd.run(args)
    if args.command == "calibrate":
        return calibrate_cmd.run(args)

    parser.error(f"unknown command {args.command!r}")
    return 2


def run_cli(argv: list[str] | None = None) -> int:
    """The installed `halide` console script's actual entry point (see pyproject.toml) — wraps
    main() with clean error/Ctrl+C presentation via halide.cli.console.run_guarded. main() itself
    stays exception-raising/untouched so the integration tests, which call it directly, keep
    exercising its real SystemExit contract."""
    from halide.cli.console import run_guarded

    return run_guarded(main, argv)


if __name__ == "__main__":
    sys.exit(run_cli())
