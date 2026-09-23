"""Entry point for the halide GUI: the anchor-frame calibration picker, one small purpose-built
window — not a full image editor, deliberately (see the project's build plan for why: crop/dust/
creative editing stays in darktable/GIMP/whatever you already use)."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from halide.gui import theme
from halide.gui.main_window import MainWindow


def main(initial_calibrate_path: str | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)

    window = MainWindow(show_load_controls=True)
    window.setWindowTitle("halide · calibration picker")
    if initial_calibrate_path:
        window.load_image(initial_calibrate_path)
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
