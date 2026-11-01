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
    assert print_.header.summary == "previewing on this frame"
    window.print_controls.pin_button.click()
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


# --- Print sliders: a preview on this frame, or pinned for the whole roll ----------------------------


def _positive_with_a_fit(window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    QTest.mouseClick(window.positive_button, LEFT)


def _shown(window):
    return window.image_view._pixmap.toImage()


def test_moving_a_print_slider_previews_this_frame_only(window, monkeypatch):
    _positive_with_a_fit(window)
    stale = []
    monkeypatch.setattr(window, "_proof_is_stale", lambda: stale.append(1))
    before, thumb_before = _shown(window), window._thumbnail(window.session.frames[1]).copy()
    controls = window.print_controls
    controls.exposure.setValue(controls.exposure.value() + 40)
    assert _shown(window) != before
    assert window.session.tone_override is None and window.session.sidecars()["tone"] is None
    assert (window._thumbnail(window.session.frames[1]) == thumb_before).all()  # the other frame
    assert controls.status.text() == "Previewing on this frame" and stale == []


def test_use_for_the_whole_roll_pins_and_saves_the_print(window, monkeypatch):
    _positive_with_a_fit(window)
    stale = []
    monkeypatch.setattr(window, "_proof_is_stale", lambda: stale.append(1))
    controls = window.print_controls
    controls.exposure.setValue(controls.exposure.value() + 40)
    controls.pin_button.click()
    pinned = window.session.tone_override
    assert pinned is not None and pinned.exposure == pytest.approx(controls.exposure.value() / 100)
    assert window.session.sidecars()["tone"] == pinned and stale == [1]
    assert controls.status.text() == "Pinned for every frame (saved with the profile)"
    assert not controls.pin_button.isEnabled()  # nothing new to pin


def test_changing_frame_drops_a_preview_but_keeps_a_pin(window, app):
    _positive_with_a_fit(window)
    controls = window.print_controls
    controls.exposure.setValue(controls.exposure.value() + 40)
    controls.pin_button.click()
    pinned = window.session.tone_override
    controls.grade.setValue(controls.grade.value() - 20)  # a preview on top of the pin
    assert window._preview_tone is not None
    QTest.keyClick(window, Qt.Key.Key_Right)
    _pump(app, lambda: _frame_ready(window, 1))
    assert window._preview_tone is None and window.session.tone_override == pinned
    assert controls.grade.value() == round(pinned.contrast * 100)  # the sliders show the pin again
    controls.reset_button.click()
    assert window.session.tone_override is None and controls.status.text() == "Fitted per frame"


# --- undo, and dropping a roll on the window ----------------------------------------------------------


def _undo(window):
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)


@pytest.fixture
def active(window):
    """Window shortcuts (Ctrl+Z, Delete) fire only in the active window, and a window closed by an
    earlier test can still hold that on the offscreen platform."""
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)
    return window


def test_undo_takes_back_an_added_point_once(active, window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    _undo(window)
    assert len(window.session.points) == 1 and window.status_label.full_text() == "Took back 1 point"
    _undo(window)  # one level only
    assert len(window.session.points) == 1


def test_undo_restores_a_removed_point_and_its_selection(active, window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)
    kept = list(window.session.points)
    _click_marker(window, 1)
    QTest.keyClick(window, Qt.Key.Key_Delete)
    assert len(window.session.points) == 1
    _undo(window)
    assert window.session.points == kept and window.session.selected == 1
    assert window.status_label.full_text() == "Restored 1 point"


@pytest.mark.parametrize("button", ["clear_frame_button", "clear_all_button"])
def test_undo_restores_cleared_points(active, window, monkeypatch, button):
    monkeypatch.setattr(window, "_confirm", lambda question: True)
    for xy in (SHADOW, HIGHLIGHT, (4, 10)):
        _click_image(window, xy)
    kept = list(window.session.points)
    getattr(window, button).click()
    assert window.session.points == []
    _undo(window)
    assert window.session.points == kept and window.status_label.full_text() == "Restored 3 points"


def test_undo_with_nothing_to_undo_does_nothing(active, window):
    before = window.status_label.full_text()
    _undo(window)
    assert window.session.points == [] and window.status_label.full_text() == before


def _drop(window, paths):
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, data, LEFT, Qt.KeyboardModifier.NoModifier)
    window.dropEvent(event)


def test_dropping_a_folder_loads_it_as_a_roll_after_asking(window, tmp_path, monkeypatch, app):
    asked = []
    monkeypatch.setattr(window, "_confirm_new_calibration", lambda: asked.append(1) or True)
    other = tmp_path / "Roll17"
    other.mkdir()
    for i in range(3):
        _write_frame(other / f"f_{i}.tiff", seed=i)
    _drop(window, [other])
    assert asked == [1] and window.session.roll_folder.name == "Roll17" and len(window.session.frames) == 3
    _pump(app, lambda: _frame_ready(window, 0))


def test_dropping_tiffs_loads_them_as_scans(window, tmp_path, monkeypatch, app):
    monkeypatch.setattr(window, "_confirm_new_calibration", lambda: True)
    loose = [_write_frame(tmp_path / f"loose_{i}.tif", seed=i) or tmp_path / f"loose_{i}.tif" for i in range(2)]
    _drop(window, loose)
    assert [f.path.name for f in window.session.frames] == ["loose_0.tif", "loose_1.tif"]
    _pump(app, lambda: _frame_ready(window, 0))


def test_dropping_anything_else_says_what_to_drop(window, tmp_path):
    jpeg = tmp_path / "print.jpg"
    jpeg.write_bytes(b"not a scan")
    frames = list(window.session.frames)
    _drop(window, [jpeg])
    assert window.session.frames == frames
    assert window.status_label.full_text() == "Drop a folder of scans, or TIFF scans"
