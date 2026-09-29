"""The picker window's own behaviour, on Qt's offscreen platform (no display needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QRect
from PySide6.QtWidgets import QApplication, QDialog

from halide.calibration import anchors
from halide.calibration.anchors import NeutralPoint
from halide.gui import main_window as mw
from halide.gui.filmstrip import _Strip
from halide.gui.main_window import ElidedLabel, MainWindow, window_size_for
from halide.io.scan_metadata import ScanSettings

SCAN = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
SCALE = (1.12, 1.0, 0.78)
WB = (0.73, 1.0, 1.05)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _point(frame, green_density, deviation=(0.0, 0.0, 0.0)):
    d = np.array([np.log10(WB[c]) + green_density / SCALE[c] for c in range(3)]) + np.asarray(deviation)
    return NeutralPoint(frame=frame, x=1, y=1, rgb=tuple(10.0**-d), scan=SCAN)


# --- window size (D-3) ------------------------------------------------------------------------


def test_window_size_on_1920x1080_is_1056x756():
    size = window_size_for(QRect(0, 0, 1920, 1080))
    assert (size.width(), size.height()) == (1056, 756)


def test_window_size_on_2560x1440_is_capped_at_1400x900():
    size = window_size_for(QRect(0, 0, 2560, 1440))
    assert (size.width(), size.height()) == (1400, 900)


def test_window_size_on_a_small_laptop_is_the_minimum():
    size = window_size_for(QRect(0, 0, 1366, 768))
    assert (size.width(), size.height()) == (900, 700)


# --- contact sheet window tracking (F11) ------------------------------------------------------


class _FakeProof(QDialog):
    """Stands in for ProofWindow (which would start real render workers)."""

    built: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(args[0])
        self.setAttribute(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WidgetAttribute.WA_DeleteOnClose)
        self.stale = False
        _FakeProof.built.append(self)

    def mark_stale(self, reproof):
        self.stale = True


def _window_with_points(monkeypatch, tmp_path, app):
    monkeypatch.setattr(mw, "ProofWindow", _FakeProof)
    _FakeProof.built = []
    window = MainWindow()
    paths = [tmp_path / n for n in ("a.tif", "b.tif", "c.tif")]
    window.session.set_roll(paths, [SCAN] * 3, tmp_path)
    for path, density in zip(paths, (0.4, 0.9, 1.5)):
        window.session.add_point(_point(path, density))
    window._refresh_points()
    return window


def test_rebuilt_contact_sheet_is_still_tracked_after_the_old_one_is_deleted(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    window._on_proof()
    window._on_proof()
    window._on_proof()
    first, second, third = _FakeProof.built
    # the old windows' deferred `destroyed` arrives only now, after the rebuilds
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert window._proof is third
    window.session.add_point(_point(tmp_path / "a.tif", 1.1, deviation=(0.05, 0.0, 0.0)))
    window._points_changed()
    assert third.stale and not second.stale
    window._proof.close()


# --- status line ------------------------------------------------------------------------------


def test_status_line_elides_and_keeps_the_full_text_as_its_tooltip(app):
    label = ElidedLabel()
    label.resize(120, 20)
    message = "Error: something went wrong reading /very/long/path/to/some/scan.tif at byte 12345"
    label.set_full_text(message)
    assert label.text().endswith("…") and len(label.text()) < len(message)
    assert label.toolTip() == message and label.full_text() == message


# --- filmstrip / step wedge -------------------------------------------------------------------


def test_a_failed_frame_reports_its_reason_for_the_tooltip(app):
    strip = _Strip()
    strip.set_frames([1, 2])
    strip.set_pixmap(1, None, failed=True, reason="not a linear TIFF")
    assert strip.failure_at(strip.frame_rect(1).center()) == "not a linear TIFF"
    assert strip.failure_at(strip.frame_rect(0).center()) is None
    strip.mark_failed(0, "unreadable")
    assert strip.failure_at(strip.frame_rect(0).center()) == "unreadable"


def test_measuring_note_clears_when_no_roll_is_loaded(monkeypatch, tmp_path, app):
    window = MainWindow()
    window._refresh_points()  # no roll (Continue without)
    assert window.wedge._measuring is False
    window.session.set_roll([tmp_path / "a.tif"], [SCAN], tmp_path)
    window._refresh_points()
    assert window.wedge._measuring is True


def test_window_repaints_itself_when_reactivated(monkeypatch, app):
    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "_repaint_all", lambda: calls.append(1))
    monkeypatch.setattr(window, "isActiveWindow", lambda: True)
    window.changeEvent(QEvent(QEvent.Type.ActivationChange))
    assert calls == [1]


# --- reliability wording (R16) ----------------------------------------------------------------


def test_reliability_says_not_reliable_anywhere_instead_of_a_one_point_range(tmp_path, monkeypatch):
    monkeypatch.setattr(anchors, "ASSUMED_PICK_ERROR", 0.06)  # a fit so loose that no density is within CC 5
    points = [_point(tmp_path / f"{i}.tif", d, deviation=dev) for i, (d, dev) in
              enumerate([(1.50, (0.0, 0.0, 0.0)), (1.56, (0.0, 0.0, 0.0)), (1.62, (0.0, 0.0, 0.0))])]
    reliability = anchors.fit_reliability(points, SCAN, (0.2, 2.2))
    assert reliability is not None and reliability.is_limited
    assert not reliability.reliable_anywhere
    note = reliability.warning()
    assert "not reliable anywhere" in note and "D 1." not in note
