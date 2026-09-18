"""`halide calibrate` — launch the anchor-frame calibration picker GUI."""

from __future__ import annotations

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path", nargs="?", default=None, help="TIFF negative to auto-load on startup (optional)"
    )


def run(args: argparse.Namespace) -> int:
    from halide.gui.app import main as run_gui  # deferred: don't require dearpygui/a display for the rest of the CLI

    run_gui(initial_calibrate_path=args.path)
    return 0
