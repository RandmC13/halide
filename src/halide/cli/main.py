from __future__ import annotations

import argparse
import sys

from halide.cli import console
from halide.cli._help import DEBUG_HELP, HELP_FORMATTER, examples, wrapped
from halide.cli.commands import (
    batch_cmd,
    calibrate_cmd,
    check_cmd,
    contact_cmd,
    export_cmd,
    gpu_cmd,
    invert_cmd,
    print_cmd,
    profile_cmd,
)

_SUBCOMMANDS = (
    "invert", "batch", "print", "export", "contact", "check", "profile", "calibrate", "gpu",
)


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


def _version() -> str:
    """The installed version, or the source tree's own when halide isn't installed as a package."""
    from importlib import metadata

    try:
        return metadata.version("halide")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halide", description="Develop scanned color negative film into a positive image.",
        formatter_class=HELP_FORMATTER,
        epilog=examples(
            "halide calibrate roll16/",
            "halide batch roll16/ roll16-developed/ --profile \"Portra 400\"",
            "halide invert negative.tif positive.tif --pick",
        ),
    )
    parser.add_argument("--version", action="version", version=f"halide {_version()}")
    parser.add_argument("--debug", action="store_true", help=DEBUG_HELP)
    # Not required: a bare `halide` shows a welcome screen (see main()) rather than argparse's
    # plain "the following arguments are required: command" error.
    subparsers = parser.add_subparsers(dest="command", required=False)

    def add(name: str, summary: str, module, *sample_lines: str):
        sub = subparsers.add_parser(
            name, help=summary, description=wrapped(summary + "."),
            formatter_class=HELP_FORMATTER, epilog=examples(*sample_lines),
        )
        # After the subcommand too. SUPPRESS: only set when given, so it can't overwrite the
        # top-level flag with False.
        sub.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=DEBUG_HELP)
        module.add_arguments(sub)

    add("invert", "Invert a single negative scan", invert_cmd,
        "halide invert negative.tif positive.tif --pick",
        'halide invert negative.tif positive.tif --profile "Portra 400"',
        "halide invert negative.tif flat.tif --auto-density --output flat")
    add("batch", "Invert a folder of negative scans in parallel", batch_cmd,
        'halide batch roll16/ roll16-developed/ --profile "Portra 400"',
        "halide batch roll16/ --contact-sheet preview.jpg",
        "halide batch roll16/ roll16-developed/ --skip-existing")
    add("print", "Print a flat positive (from `invert --output flat`, optionally edited in "
        "darktable) onto the paper curve", print_cmd,
        "halide print flat_edited.tif print.tif",
        "halide print flat-edited/ prints/ --contrast 0.8")
    add("export", "Convert developed ACEScg TIFFs into delivery-ready sRGB PNG/JPEG", export_cmd,
        "halide export positive.tif positive.png",
        "halide export roll16-developed/ roll16-jpeg/ --format jpg --quality 92")
    add("contact", "Make a high-resolution contact sheet of developed frames (TIFF, or "
        "PNG/JPEG from export)", contact_cmd,
        "halide contact roll16-developed/ roll16-sheet.jpg",
        "halide contact roll16-jpeg/ sheet.png --columns 4 --title \"Roll 16\"")
    add("check", "Check that a roll's scans were made consistently (camera settings, white "
        "balance, edits)", check_cmd,
        "halide check roll16/")

    profile_parser = subparsers.add_parser(
        "profile", help="Manage saved calibration profiles",
        description=wrapped("Manage saved calibration profiles: list, show, rename, delete, or edit the "
        "details recorded with one."),
        formatter_class=HELP_FORMATTER,
    )
    profile_parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=DEBUG_HELP)
    profile_cmd.add_arguments(profile_parser)

    add("calibrate", "Open the picker: click neutral points on a roll and save a calibration "
        "profile", calibrate_cmd,
        "halide calibrate roll16/",
        'halide calibrate --profile "Portra 400"')
    add("gpu", "Find an NVIDIA GPU and report or add optional GPU support", gpu_cmd,
        "halide gpu",
        "halide gpu --install")

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
    if args.command == "contact":
        return contact_cmd.run(args)
    if args.command == "check":
        return check_cmd.run(args)
    if args.command == "profile":
        return profile_cmd.run(args)
    if args.command == "calibrate":
        return calibrate_cmd.run(args)
    if args.command == "gpu":
        return gpu_cmd.run(args)

    parser.error(f"unknown command {args.command!r}")
    return 2


def run_cli(argv: list[str] | None = None) -> int:
    """The installed `halide` console script's actual entry point (see pyproject.toml) — wraps
    main() with clean error/Ctrl+C presentation via halide.cli.console.run_guarded. main() itself
    stays exception-raising/untouched so the integration tests, which call it directly, keep
    exercising its real SystemExit contract.

    Afterwards (so the note never lands inside a command's own output) it keeps tab completion for
    the shell it was started from (zsh, bash or fish) installed and current — see
    halide.cli.completion for why there's no `completion` command."""
    from halide.cli.completion import maybe_install_completion
    from halide.cli.console import run_guarded

    # A logged run (`halide batch ... > log.txt 2>&1`) must keep its order: stdout is block-buffered
    # when it isn't a terminal, so the run sheet, progress lines and warnings would otherwise arrive
    # out of step with stderr's messages.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)

    code = run_guarded(main, argv)
    maybe_install_completion(build_parser())
    return code


if __name__ == "__main__":
    sys.exit(run_cli())
