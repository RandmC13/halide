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
    motion.animate(QObject(), "x", 0, 1, 200, seen.append, lambda: done.append(True))
    assert seen == [1] and done == [True]


def _settle(app, until, timeout=5.0):
    import time

    from PySide6.QtCore import QCoreApplication, QEvent

    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # let finished animations delete themselves
        time.sleep(0.01)
    assert until()


def test_a_new_image_after_a_finished_fade_does_not_touch_the_deleted_animation(app, monkeypatch):
    # With motion on, a finished cross-fade deletes itself; the next picture (the next frame after a
    # Negative/Positive switch) once stopped that dead animation and wedged the picker.
    from halide.gui.image_view import ImageView

    monkeypatch.delenv("HALIDE_MOTION")
    view = ImageView()
    view.resize(200, 150)
    view.set_image(QPixmap(40, 30))
    view.set_image(QPixmap(40, 30), fade=True)
    _settle(app, lambda: view._blend == 1.0 and view._old_pixmap is None)
    view.set_image(QPixmap(40, 30))  # raised RuntimeError: Internal C++ object already deleted
    view.set_image(QPixmap(40, 30), fade=True)
    _settle(app, lambda: view._blend == 1.0)


def test_a_second_animation_with_the_same_name_replaces_the_first(app, monkeypatch):
    from PySide6.QtCore import QObject, QVariantAnimation

    monkeypatch.delenv("HALIDE_MOTION")
    owner, first, second = QObject(), [], []
    motion.animate(owner, "x", 0, 1, 1000, first.append)
    motion.animate(owner, "x", 0, 1, 50, second.append)
    running = [a for a in owner.findChildren(QVariantAnimation) if a.state() == QVariantAnimation.State.Running]
    assert len(running) == 1
    _settle(app, lambda: second and second[-1] == 1.0)
    assert not first or first[-1] < 1.0  # the first was stopped, never finished


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
    assert not window.profile_chip.isVisible()  # the stamp stands where the profile chip was
    _click_image(window, HIGHLIGHT)
    assert not window.save_stamp.isVisible() and window.profile_chip.isVisible()
