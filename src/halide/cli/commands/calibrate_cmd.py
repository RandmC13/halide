"""`halide calibrate` — launch the anchor-frame calibration picker GUI."""

from __future__ import annotations

import argparse
import os
import sys

from halide.cli import console


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path", nargs="?", default=None, help="TIFF negative to auto-load on startup (optional)"
    )


def run(args: argparse.Namespace) -> int:
    # A missing display isn't a catchable Python exception here — dearpygui/GLFW hits a native
    # assertion and aborts the whole process outright rather than raising, so this needs to be
    # ruled out *before* ever calling into the GUI toolkit, not caught afterward. This only covers
    # the common "forgot X11 forwarding over SSH" case on Linux/X11 specifically; an invalid-but-set
    # DISPLAY, or other exotic display-server failures, can still hit the same native abort.
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "couldn't launch the calibration picker: no display available (DISPLAY/WAYLAND_DISPLAY "
            "isn't set) — this needs a real display; if you're on a remote/SSH session, try a local "
            "one or forward X11"
        )

    from halide.gui.app import main as run_gui  # deferred: don't require dearpygui/a display for the rest of the CLI

    print(f"{console.rule(10)}\n{console.Style.BOLD}halide{console.Style.RESET} · calibration picker\n")
    try:
        run_gui(initial_calibrate_path=args.path)
    except Exception as exc:  # noqa: BLE001 -- a GUI-toolkit failure, not a domain error
        raise SystemExit(f"couldn't launch the calibration picker window ({exc})") from exc
    return 0
