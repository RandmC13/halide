"""`halide calibrate` — launch the anchor-frame calibration picker GUI."""

from __future__ import annotations

import argparse
import os
import sys

from halide.cli import console


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="TIFF negative to auto-load on startup (optional)",
    )


def run(args: argparse.Namespace) -> int:
    # A missing display isn't a catchable Python exception here — Qt's xcb platform plugin hits a
    # native assertion and aborts the whole process outright (SIGABRT) rather than raising, so this
    # needs to be ruled out *before* ever calling into the GUI toolkit, not caught afterward (verified
    # directly: constructing a headless QApplication with no DISPLAY/WAYLAND_DISPLAY set aborts the
    # process, doesn't raise). This only covers the common "forgot X11 forwarding over SSH" case on
    # Linux/X11 specifically; an invalid-but-set DISPLAY, or other exotic display-server failures,
    # can still hit the same native abort.
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SystemExit(
            "couldn't launch the calibration picker: no display available (DISPLAY/WAYLAND_DISPLAY "
            "isn't set) — this needs a real display; if you're on a remote/SSH session, try a local "
            "one or forward X11"
        )

    from halide.gui.app import (
        main as run_gui,
    )  # deferred: don't require Qt/a display for the rest of the CLI

    print(console.framed([f"{console.Style.BOLD}halide{console.Style.RESET} · calibration picker"]))
    try:
        run_gui(initial_calibrate_path=args.path)
    except Exception as exc:  # noqa: BLE001 -- a GUI-toolkit failure, not a domain error
        raise SystemExit(
            f"couldn't launch the calibration picker window ({exc})"
        ) from exc
    return 0
