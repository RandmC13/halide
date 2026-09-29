"""Shared help text and argparse `type=` checks for the CLI: the examples epilog every subcommand
ends with, the one `--workers` definition, and the range checks on print and calibration numbers
(a paper grade of -1 or a red multiplier of 0 used to be accepted and 'developed' happily).

Deliberately free of numpy and the pipeline: building the parser must stay cheap."""

from __future__ import annotations

import argparse
import math
import textwrap

# Kept out of the top-level parser's own kwargs so every subcommand (and `profile`'s own
# subcommands) gets the same wrapped-paragraph-preserving formatter.
HELP_FORMATTER = argparse.RawDescriptionHelpFormatter

MAX_CONTRAST = 2.0

DEBUG_HELP = "Show the full Python traceback for an unexpected error (HALIDE_DEBUG=1 does the same)"


def wrapped(text: str) -> str:
    """A description wrapped to the terminal-safe width, since the raw formatter doesn't wrap."""
    return textwrap.fill(text, width=78)


def examples(*lines: str) -> str:
    """The "examples:" epilog: a few real command lines, printed as written (the formatter keeps
    line breaks), so keep each under ~80 columns."""
    return "examples:\n" + "\n".join(f"  {line}" for line in lines)


def _finite(value: str, what: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"{what} must be a finite number, not {value}")
    return number


def finite_float(value: str) -> float:
    """--exposure: any real number (it's a shift in density units, so zero and negatives are fine)."""
    return _finite(value, "the exposure")


def positive_float(value: str) -> float:
    """--rm/--bm/--rs/--bs: a multiplier or power, so it has to be above zero."""
    number = _finite(value, "the value")
    if number <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} isn't usable: this has to be above zero (1.0 leaves the channel as it is)"
        )
    return number


def contrast_grade(value: str) -> float:
    """--contrast (paper grade): above 0, up to MAX_CONTRAST. 1.0 is the untouched reference paper."""
    number = _finite(value, "the grade")
    if not 0 < number <= MAX_CONTRAST:
        raise argparse.ArgumentTypeError(
            f"a paper grade of {value} isn't usable: pick above 0 and at most {MAX_CONTRAST:g} "
            "(1.0 is the reference paper, lower is softer)"
        )
    return number


def worker_count(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} isn't a number of workers (use a whole number, 1 or more)") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value} workers isn't possible: use 1 or more")
    return number


WORKERS_HELP = (
    "Number of frames to process at once (default: chosen from free memory and CPU cores; "
    "memory, not cores, is usually the limit for full-size scans)"
)


def add_workers_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=worker_count, metavar="N", help=WORKERS_HELP)
