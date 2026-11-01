"""The picker's top bar and the film-carrier surround of the image."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from halide.gui import theme
from halide.gui.main_window import ImageView, MainWindow
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _plain(label):
    return re.sub(r"<[^>]+>", "", label.text())


def test_top_bar_counter_chip_and_contact_sheet_button(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(3):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    window = MainWindow()
    window.show()
    assert window.frame_counter.text() == ""
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    assert _plain(window.frame_counter) == "FRAME 01 / 03"
    QTest.keyClick(window, Qt.Key.Key_Right)
    _pump(app, lambda: _frame_ready(window, 1))
    assert _plain(window.frame_counter) == "FRAME 02 / 03"
    assert not window.proof_button.isEnabled()
    before = window.profile_chip.text()
    window._on_saved("X", "msg")
    assert window.profile_chip.text() != before and "X" in _plain(window.profile_chip)
    window._closing = True
    window.close()


def test_contact_sheet_button_hidden_in_a_pick_session(app):
    window = MainWindow(is_pick_session=True)
    assert window.proof_button.isHidden()
    window._closing = True
    window.close()


def test_image_view_paints_rebate_around_the_image(app):
    view = ImageView()
    view.resize(400, 300)
    pixmap = QPixmap(40, 30)
    pixmap.fill(QColor("#336699"))
    view.set_image(pixmap)
    image = view.grab().toImage()
    assert image.pixelColor(3, 3) == QColor(theme.FILM_REBATE)
    assert image.pixelColor(200, 150) == QColor("#336699")
