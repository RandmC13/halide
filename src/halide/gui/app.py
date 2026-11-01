"""Entry point for the halide GUI: the calibration picker, one small purpose-built window — not a
full image editor, deliberately (crop/dust/creative editing stays in darktable/GIMP/whatever you
already use)."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from halide.device import ComputeDevice
from halide.gui import theme
from halide.gui.main_window import MainWindow


def quiet_qt_noise() -> None:
    """Silence one Qt warning that halide can do nothing about: on some Wayland compositors Qt's
    text-input support prints "Trying to disable … but 0x0 is focused" whenever focus moves between
    surfaces, a bookkeeping message with no effect on the window. QT_LOGGING_RULES still wins, so it
    can be turned back on for debugging."""
    from PySide6.QtCore import QLoggingCategory

    QLoggingCategory.setFilterRules("qt.qpa.wayland.textinput.warning=false")


def main(inputs: list[str] | None = None, profile: str | None = None, device: ComputeDevice | None = None) -> None:
    """`inputs`: a roll folder, or one or more scans. `profile`: a saved profile (path or name) to
    reopen - its points and extra information, and its roll if the folder is still there. `device`:
    the resolved --device, which the contact sheet window develops its frames on (None = CPU)."""
    quiet_qt_noise()
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)

    window = MainWindow(show_load_controls=True, device=device)
    window.setWindowTitle("halide · calibration picker")
    window.show()
    if profile:
        from halide.calibration.profile_store import resolve_profile_path

        window.open_profile(resolve_profile_path(profile))
    elif inputs:
        paths = [Path(p) for p in inputs]
        if len(paths) == 1 and paths[0].is_dir():
            window.load_roll(paths[0])
        else:
            window.load_files(paths)

    app.exec()


if __name__ == "__main__":
    main(sys.argv[1:])
