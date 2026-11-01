"""Picker interactions driven by real mouse and key events on Qt's offscreen platform: clicking the
image, selecting and removing points, the point rows, the Negative/Positive switch, the drawers, the
Clear buttons and when Save and Build contact sheet are enabled. A small synthetic roll is loaded by
the real loaders; events go to the actual widgets through QTest."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from halide.gui import main_window as mw
from halide.gui.main_window import MainWindow
from halide.gui.point_list import _Row
from halide.gui.sampling import display_to_full_res_coords, snap_to_representative_pixel
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame

LEFT = Qt.MouseButton.LeftButton
SHADOW = (8, 3)  # displayed-image pixel in the top (shadow) half
HIGHLIGHT = (8, 12)  # ... in the bottom (highlight) half


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(MainWindow, "_confirm_close", lambda self, count: True)
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(2):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    win = MainWindow()
    win.load_roll(roll)
    win.show()
    _pump(app, lambda: _frame_ready(win, 0))
    yield win
    win.close()


def _click_image(window, xy):
    """Left-click the displayed-image pixel `xy` on the image view."""
    view = window.image_view
    rect, scale = view._image_rect(), view._scale()
    pos = QPoint(round(rect.x() + (xy[0] + 0.5) * scale), round(rect.y() + (xy[1] + 0.5) * scale))
    QTest.mouseClick(view, LEFT, pos=pos)


def _click_marker(window, index):
    """Click where point `index` is drawn: snapping can move a point away from where it was clicked."""
    point = window.session.points[index]
    _click_image(window, (point.x // window.stride, point.y // window.stride))


def _click_row(window, index):
    QTest.mouseClick(window.point_list._rows[index], LEFT)


def test_clicking_the_image_adds_a_point_where_the_click_landed(window):
    _click_image(window, SHADOW)
    assert len(window.session.points) == 1
    point = window.session.points[0]
    h, w = window.full_image.shape[:2]
    x, y = display_to_full_res_coords(*SHADOW, window.stride, w, h)
    snapped_x, snapped_y, _ = snap_to_representative_pixel(window.full_image, x, y)
    assert (point.x, point.y) == (snapped_x, snapped_y)  # exactly where the picker snaps that click


def test_clicking_a_point_again_selects_it_and_delete_backspace_escape_work(window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    assert len(window.session.points) == 2 and window.session.selected == 1
    _click_marker(window, 0)  # on an existing point: select, don't add
    assert len(window.session.points) == 2 and window.session.selected == 0
    QTest.keyClick(window, Qt.Key.Key_Delete)
    assert len(window.session.points) == 1
    _click_marker(window, 0)
    assert window.session.selected == 0
    QTest.keyClick(window, Qt.Key.Key_Backspace)
    assert window.session.points == []
    _click_image(window, SHADOW)
    assert window.session.selected == 0  # a new point starts selected
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.session.selected is None and len(window.session.points) == 1


def test_clicking_a_row_on_the_current_frame_selects_it_and_flashes_its_marker(window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    assert window.session.selected == 1
    assert isinstance(window.point_list._rows[0], _Row)
    _click_row(window, 0)
    assert window.session.selected == 0
    assert window.image_view._flash == 1


def test_positive_and_negative_buttons_swap_caption_note_and_print_drawer(window):
    negative = (window.caption.text(), window.source_note.text())
    assert not window.print_drawer.isEnabled()
    QTest.mouseClick(window.positive_button, LEFT)
    assert window.caption.text() != negative[0] and window.source_note.text() != negative[1]
    assert window.print_drawer.isEnabled()
    QTest.mouseClick(window.negative_button, LEFT)
    assert (window.caption.text(), window.source_note.text()) == negative
    assert not window.print_drawer.isEnabled()


def test_opening_a_drawer_closes_the_other_and_details_takes_the_spare_height(window):
    extra, print_, details = window._accordion.drawers
    panel = window._panel
    list_stretch = lambda: panel.stretch(panel.indexOf(window.point_list))  # noqa: E731
    assert list_stretch() == 1

    QTest.mouseClick(extra.header, LEFT)
    assert extra.header.isChecked()
    QTest.mouseClick(details.header, LEFT)
    assert details.header.isChecked() and not extra.header.isChecked()
    assert list_stretch() == 0

    QTest.mouseClick(details.header, LEFT)
    assert not any(d.header.isChecked() for d in (extra, print_, details))
    assert list_stretch() == 1


@pytest.mark.parametrize(("button", "left"), [("clear_frame_button", 1), ("clear_all_button", 0)])
def test_clear_buttons_remove_points_only_when_confirmed(window, monkeypatch, app, button, left):
    # Two points on the second frame, then back to the first for one more: Clear frame must take
    # only the first frame's point, Clear all every point.
    QTest.keyClick(window, Qt.Key.Key_Right)
    _pump(app, lambda: _frame_ready(window, 1))
    _click_image(window, HIGHLIGHT)
    QTest.keyClick(window, Qt.Key.Key_Left)
    _pump(app, lambda: _frame_ready(window, 0))
    _click_image(window, SHADOW)
    monkeypatch.setattr(MainWindow, "_confirm", lambda self, question: False)
    QTest.mouseClick(getattr(window, button), LEFT)
    assert len(window.session.points) == 2
    monkeypatch.setattr(MainWindow, "_confirm", lambda self, question: True)
    QTest.mouseClick(getattr(window, button), LEFT)
    assert len(window.session.points) == left
    assert all(p.frame == window.session.frames[1].path for p in window.session.points)


def test_save_and_contact_sheet_need_two_separated_points(window):
    assert not window.primary_button.isEnabled() and not window.proof_button.isEnabled()
    _click_image(window, SHADOW)
    assert not window.primary_button.isEnabled() and not window.proof_button.isEnabled()
    _click_image(window, HIGHLIGHT)
    assert window.primary_button.isEnabled() and window.proof_button.isEnabled()


def test_drawer_summaries_follow_the_film_stock_and_the_print_sliders(window):
    extra, print_, _details = window._accordion.drawers
    QTest.mouseClick(window.positive_button, LEFT)  # the Print drawer only works on the positive
    assert extra.header.summary == "film stock, process, scanner" and print_.header.summary == "fitted per frame"
    window.details_form.fields["film_stock"].setText("Portra 400")
    assert extra.header.summary == "Portra 400"
    window.print_controls.exposure.setValue(window.print_controls.exposure.value() + 10)
    assert print_.header.summary == "pinned for every frame"
    window.print_controls.reset_button.click()
    assert print_.header.summary == "fitted per frame"


def _show_with_points(window, app, size):
    window.setFixedSize(*size)
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    app.processEvents()


def test_point_list_uses_the_spare_height_at_1080p(window, app):
    from halide.gui.point_list import ROW_HEIGHT

    _show_with_points(window, app, (1056, 756))
    assert window.point_list.viewport().height() >= 8 * ROW_HEIGHT


def test_nothing_in_the_panel_is_clipped_at_the_smallest_window_with_print_open(window, app):
    from PySide6.QtWidgets import QWidget

    _show_with_points(window, app, (900, 700))
    QTest.mouseClick(window.positive_button, LEFT)
    QTest.mouseClick(window.print_drawer.header, LEFT)
    window.reliability_note.setText("Fit reliable over D 0.9-1.3 only - add a point near the ends")
    window.reliability_note.setVisible(True)
    app.processEvents()
    panel = window._panel.parentWidget()
    inside = panel.rect()
    for child in panel.findChildren(QWidget):
        if child.isVisibleTo(panel):
            rect = child.rect().translated(child.mapTo(panel, QPoint(0, 0)))
            assert inside.contains(rect), (type(child).__name__, rect, inside)


def test_show_likely_neutrals_sits_by_the_switch_and_marks_the_image(window, app):
    import numpy as np

    assert not window.details_drawer.isAncestorOf(window.auto_overlay)
    mask = np.zeros(window.display.shape[:2], dtype=bool)
    mask[:4, :4] = True
    window._auto_mask = mask  # the synthetic frame has no candidates of its own
    before = window.image_view._pixmap.toImage()
    QTest.mouseClick(window.auto_overlay, LEFT)
    assert window.auto_overlay.isChecked()
    assert window.image_view._pixmap.toImage() != before
