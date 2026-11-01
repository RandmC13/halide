"""The picker's animations: under HALIDE_MOTION=0 (the suite's default) each lands in its end state
at once."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPixmap

from halide.gui import motion
from tests.unit.test_gui_interaction import HIGHLIGHT, SHADOW, _click_image, app, window  # noqa: F401


def test_duration_is_zero_with_motion_off_and_itself_otherwise(monkeypatch):
    assert motion.duration(250) == 0
    monkeypatch.delenv("HALIDE_MOTION")
    assert motion.duration(250) == 250


def test_animate_with_motion_off_jumps_to_the_end(app):
    from PySide6.QtCore import QObject

    seen, done = [], []
    assert motion.animate(QObject(), 0, 1, 200, seen.append, lambda: done.append(True)) is None
    assert seen == [1] and done == [True]


def test_strip_frames_are_fully_in(window):
    strip = window.filmstrip.strip
    assert strip._appear and all(a == 1.0 for a in strip._appear)


def test_view_switch_highlight_sits_under_the_checked_button(window):
    switch = window.view_switch
    assert switch.highlight_rect() == switch.negative_button.geometry().toRectF()
    window.positive_button.click()
    assert switch.highlight_rect() == switch.positive_button.geometry().toRectF()
    assert window.image_view._blend == 1.0 and window.image_view._old_pixmap is None


def test_new_point_lands_fully(window):
    _click_image(window, SHADOW)
    assert window.image_view._probe == 1.0 and window.image_view._probe_number is None
    assert window.wedge._drop == 0.0


def test_save_stamp_shows_after_a_save_and_hides_when_points_change(window):
    _click_image(window, SHADOW)
    window._on_saved("Portra 400", "saved")
    assert window.save_stamp.isVisible() and "PORTRA 400" in window.save_stamp.text()
    _click_image(window, HIGHLIGHT)
    assert not window.save_stamp.isVisible()
