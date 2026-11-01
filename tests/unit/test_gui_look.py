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
from halide.gui.image_view import ImageView
from halide.gui.main_window import MainWindow
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
    assert window.frame_counter.text() == "" and window.windowTitle() == "halide"
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    assert _plain(window.frame_counter) == "FRAME 01 / 03" and window.windowTitle() == "halide · roll"
    QTest.keyClick(window, Qt.Key.Key_Right)
    _pump(app, lambda: _frame_ready(window, 1))
    assert _plain(window.frame_counter) == "FRAME 02 / 03"
    assert not window.proof_button.isEnabled()
    before = window.profile_chip.text()
    assert window.last_saved is None
    window._on_saved("X", "msg")
    assert window.profile_chip.text() != before and "X" in _plain(window.profile_chip)
    assert window.last_saved == ("X", roll)  # what `halide open` names in its next step
    window._closing = True
    window.close()


def test_contact_sheet_button_hidden_in_a_pick_session(app):
    window = MainWindow(is_pick_session=True)
    window.show()  # a window never shown makes Qt crash as the test process exits
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


def test_band_labels_are_the_density_at_each_band_centre():
    from halide.gui.step_wedge import band_labels

    labels = band_labels((0.71, 1.50), 11)
    assert len(labels) == 11 and labels[0] == "0.75" and labels[-1] == "1.46"


def _roll(tmp_path, name, count=2):
    roll = tmp_path / name
    roll.mkdir()
    for i in range(count):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    return roll


def test_a_new_roll_forgets_the_saved_profiles_name_in_the_top_bar(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(MainWindow, "_confirm_new_calibration", lambda self: True)
    window = MainWindow()
    window.load_roll(_roll(tmp_path, "first"))
    window.show()
    _pump(app, lambda: _frame_ready(window, 0))
    window._on_saved("Portra 400", "saved")
    assert window.save_stamp.isVisible()
    window.load_roll(_roll(tmp_path, "second"))
    _pump(app, lambda: _frame_ready(window, 0))
    assert not window.save_stamp.isVisible() and window.profile_chip.isVisible()
    assert _plain(window.profile_chip) == "New calibration" and window._profile_name == ""
    window.close()


def test_long_roll_and_profile_names_are_cut_so_the_top_bar_buttons_fit(tmp_path, monkeypatch, app):
    from PySide6.QtCore import QSize

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = MainWindow()
    window.setFixedSize(QSize(900, 700))
    long_name = "2026-10-holiday-in-the-mountains-portra-400-pushed-one-stop-roll-16"
    window.load_roll(_roll(tmp_path, long_name))
    window.show()
    _pump(app, lambda: _frame_ready(window, 0))
    window._on_saved("Kodak Portra 400 at box speed, home C-41 kit, camera scan", "saved")
    app.processEvents()
    assert window.roll_label.toolTip() == long_name.upper() and "…" in window.roll_label.text()
    from PySide6.QtWidgets import QPushButton

    top = [b for b in window.findChildren(QPushButton) if b.text() in ("Contact sheet", "Profiles", "Load roll…")]
    assert len(top) == 3
    for button in top:  # whole, and inside the window
        assert button.mapTo(window, button.rect().bottomRight()).x() <= window.width()
        assert button.width() >= button.sizeHint().width()
    # ... and nothing on the left runs into them.
    assert window.save_stamp.geometry().right() < window.proof_button.geometry().left()
    assert window.roll_label.width() >= window.roll_label.sizeHint().width()
    window.close()


def test_drawer_headers_carry_their_title_and_a_disabled_one_is_not_a_hand(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = MainWindow()
    window.load_roll(_roll(tmp_path, "roll"))
    window.show()
    _pump(app, lambda: _frame_ready(window, 0))
    assert [d.header.text() for d in window._accordion.drawers] == ["Extra information", "Print", "Details"]
    assert not window.print_drawer.isEnabled()  # the negative view: no print to adjust
    assert window.print_drawer.header.cursor().shape() == Qt.CursorShape.ArrowCursor
    window.positive_button.click()
    assert window.print_drawer.header.cursor().shape() == Qt.CursorShape.PointingHandCursor
    window.close()


def test_the_carrier_uses_the_contact_sheets_arrow(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    window = MainWindow()
    window.load_roll(_roll(tmp_path, "roll"))
    window.show()
    _pump(app, lambda: _frame_ready(window, 0))
    assert window.image_view._edge_print == "1  →1A"
    window.close()
