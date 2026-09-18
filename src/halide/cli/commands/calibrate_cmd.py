"""`halide calibrate` — launch the anchor-frame calibration picker GUI."""

from __future__ import annotations

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass  # the GUI itself prompts for a TIFF path to load


def run(args: argparse.Namespace) -> int:
    from halide.gui.app import main as run_gui  # deferred: don't require dearpygui/a display for the rest of the CLI

    run_gui()
    return 0
