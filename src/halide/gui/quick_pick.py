"""Standalone, blocking calibration picker for `halide invert --pick`: a one-shot alternative to
the persistent `halide calibrate` app, for a user who just wants to manually pick shadow/highlight
points for a single image without building a reusable named profile.

Reuses `main_window.MainWindow` as-is (magnifier, markers, live preview, the auto-detection overlay/
comparison, Fine-tune) via its `is_pick_session=True` mode - none of that picking machinery is
specific to the profile-saving workflow, all of it is directly useful for picking quickly and
accurately. The only real difference is the entry point itself: a blocking function that returns a
value, rather than the persistent tabbed app's own event loop (see `gui/app.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui import theme
from halide.gui.main_window import MainWindow


def run_quick_pick(path: str) -> tuple[DensityProfile, ToneCurveParams | None] | None:
    """Open a standalone picker window pre-loaded with `path`, block until the user either clicks
    "Develop" (returns the picked (DensityProfile, tone_override) - tone_override is None unless
    the Print drawer pinned exposure/grade) or closes the window without doing so (returns None).

    Runs a local QEventLoop rather than a full QApplication.exec() so this function can return a
    value once the user's done, rather than exiting the process - new territory for this codebase's
    GUI code (every other entry point runs the full event loop and exits the process).
    """
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)

    window = MainWindow(show_load_controls=False, is_pick_session=True)
    window.setWindowTitle("halide · quick calibrate")
    window.load_files([Path(path)])

    result_holder: list[tuple[DensityProfile, ToneCurveParams | None] | None] = [None]
    loop = QEventLoop()

    def on_completed(value: object) -> None:
        result_holder[0] = value
        loop.quit()

    window.pickCompleted.connect(on_completed)
    window.show()
    loop.exec()

    return result_holder[0]
