"""`halide open` — open the halide window: load a roll, pick neutral points on any frames or use a saved
profile, and save profiles."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from halide.cli import console
from halide.cli._device_args import add_device_argument, device_fallback_warning, device_row, resolve_device_arg


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs",
        nargs="*",
        help="A roll folder, or one or more TIFF scans, to open in the window (optional)",
    )
    parser.add_argument(
        "--profile",
        help="Reopen a saved profile (its name or a path) with its neutral points, extra information and roll folder",
    )
    add_device_argument(parser)


def run(args: argparse.Namespace) -> int:
    if args.profile and args.inputs:
        raise SystemExit(
            "--profile reopens a saved profile together with its own roll, so a roll or scan "
            "can't be given as well — drop one of them"
        )
    if args.profile:  # fail on an unknown name now, not after the window has opened
        from halide.calibration.profile_store import resolve_profile_path
        from halide.cli._calibration_args import missing_profile_message

        try:
            resolve_profile_path(args.profile)
        except FileNotFoundError as exc:
            raise SystemExit(missing_profile_message(args.profile, exc)) from exc

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
            "couldn't open the halide window: no display available (DISPLAY/WAYLAND_DISPLAY "
            "isn't set) — this needs a real display; if you're on a remote/SSH session, try a local "
            "one or forward X11"
        )

    # Resolved before the window opens, so --device gpu fails fast without CuPy/a driver. The
    # picker's contact-sheet window develops every frame on it (batch's own workers); the picker's
    # own previews and picking stay on the CPU.
    device = resolve_device_arg(args)

    try:
        from halide.gui.app import main as run_gui  # deferred: don't require Qt/a display for the rest of the CLI
    except ImportError as exc:
        raise SystemExit(f"couldn't open the halide window ({exc})") from exc

    print(console.framed([f"{console.Style.BOLD}halide{console.Style.RESET} · window"]))
    print(f"Compute: {device_row(device)}")
    fallback_warning = device_fallback_warning(device)
    if fallback_warning:
        print(console.warning(fallback_warning))
    # Any error from here on is the window's own and keeps its real message (--debug shows the traceback).
    saved = run_gui(inputs=args.inputs, profile=args.profile, device=device)
    if saved is not None:
        print(next_step(*saved))
    return 0


def next_step(name: str, roll: Path | None) -> str:
    """What to run after saving profile `name` in the window, on the roll it was made from (or a
    placeholder when the points were picked on loose scans)."""
    source = shlex.quote(str(roll)) if roll is not None else "ROLL_DIR"
    out = shlex.quote(f"{roll}-developed") if roll is not None else "OUT_DIR"
    return "\n".join([
        console.success(f"Saved profile '{name}'"),
        f"next: halide batch {source} {out} --profile {shlex.quote(name)}",
    ])
