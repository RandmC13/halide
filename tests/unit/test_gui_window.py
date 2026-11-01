"""The picker window's own behaviour, on Qt's offscreen platform (no display needed)."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QRect, QThread, Signal
from PySide6.QtWidgets import QApplication, QDialog, QLabel

from halide.calibration import anchors
from halide.calibration.anchors import NeutralPoint
from halide.gui import main_window as mw
from halide.gui.filmstrip import _Strip
from halide.gui.main_window import MainWindow, window_size_for
from halide.gui.widgets import ElidedLabel
from halide.io.scan_metadata import ScanSettings
from tests.unit._gui_helpers import ParkedLoader, record_started_threads

SCAN = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
SCALE = (1.12, 1.0, 0.78)
WB = (0.73, 1.0, 1.05)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _closing_never_blocks_on_the_question(monkeypatch):
    """Tests that close a window holding unsaved points shouldn't wait on the modal question."""
    monkeypatch.setattr(MainWindow, "_confirm_close", lambda self, count: True)


def _point(frame, green_density, deviation=(0.0, 0.0, 0.0)):
    d = np.array([np.log10(WB[c]) + green_density / SCALE[c] for c in range(3)]) + np.asarray(deviation)
    return NeutralPoint(frame=frame, x=1, y=1, rgb=tuple(10.0**-d), scan=SCAN)


# --- window size (the user's 1920x1080 size is pinned) --------------------------------------------


def test_window_size_on_1920x1080_is_1056x756():
    size = window_size_for(QRect(0, 0, 1920, 1080))
    assert (size.width(), size.height()) == (1056, 756)


def test_window_size_on_2560x1440_is_capped_at_1400x900():
    size = window_size_for(QRect(0, 0, 2560, 1440))
    assert (size.width(), size.height()) == (1400, 900)


def test_window_size_on_a_small_laptop_is_the_minimum():
    size = window_size_for(QRect(0, 0, 1366, 768))
    assert (size.width(), size.height()) == (900, 700)


# --- contact sheet window tracking ------------------------------------------------------------


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


def test_a_span_under_a_tenth_names_the_spot_not_a_rounded_range():
    text = anchors.Reliability(worst_cc_at_ends=12.0, reliable_range=(1.03, 1.11)).warning()
    assert text.startswith("Fit reliable only near D 1.07 - add a point")
    wide = anchors.Reliability(worst_cc_at_ends=12.0, reliable_range=(0.93, 1.31)).warning()
    assert wide.startswith("Fit reliable over D 0.9-1.3 only")


def test_reliability_says_not_reliable_anywhere_instead_of_a_one_point_range(tmp_path, monkeypatch):
    monkeypatch.setattr(anchors, "ASSUMED_PICK_ERROR", 0.06)  # a fit so loose that no density is within CC 5
    points = [_point(tmp_path / f"{i}.tif", d, deviation=dev) for i, (d, dev) in
              enumerate([(1.50, (0.0, 0.0, 0.0)), (1.56, (0.0, 0.0, 0.0)), (1.62, (0.0, 0.0, 0.0))])]
    reliability = anchors.fit_reliability(points, SCAN, (0.2, 2.2))
    assert reliability is not None and reliability.is_limited
    assert not reliability.reliable_anywhere
    note = reliability.warning()
    assert "not reliable anywhere" in note and "D 1." not in note


# --- point-list names (the 1080p picker must read as before) --------------------------------------


def _list_names(app, tmp_path, stem):
    window = MainWindow()
    window.setFixedSize(1056, 756)  # the 1920x1080 picker
    path = tmp_path / f"{stem}.tif"
    window.session.set_roll([path], [SCAN], tmp_path)
    for density in (1.0, 1.1):
        window.session.add_point(_point(path, density))
    window._refresh_points()
    window.show()
    for _ in range(4):
        app.processEvents()
    from halide.gui.point_list import _ElidedNameLabel

    labels = window.point_list.findChildren(_ElidedNameLabel)
    return window, labels


def test_a_normal_frame_name_is_not_elided_at_the_1080p_panel_width(app, tmp_path):
    window, labels = _list_names(app, tmp_path, "IMG_0138")
    assert labels and all(label.text() == label._full == "IMG_0138" for label in labels)
    window.close()


def test_a_row_shows_the_frame_number_then_the_stem(app, tmp_path):
    window, _labels = _list_names(app, tmp_path, "IMG_0138")
    texts = [label.text() for label in window.point_list._rows[0].findChildren(QLabel)]
    assert "1" in texts and any(t.startswith("IMG_") for t in texts) and not any("#" in t for t in texts)
    window.close()


def test_a_very_long_frame_name_is_elided_with_the_full_path_as_tooltip(app, tmp_path):
    window, labels = _list_names(app, tmp_path, "IMG_0138_scanned_at_the_lab_on_a_tuesday_afternoon")
    assert labels and all(label.text().endswith("…") and label.text().startswith("IMG_") for label in labels)
    assert all("tuesday_afternoon" in label.toolTip() for label in labels)
    window.close()


# --- late signals, one load in flight, closing mid-rebuild ---------------------


def _bare_roll(window, tmp_path, count=3):
    paths = [tmp_path / f"f{i}.tif" for i in range(count)]
    window.session.set_roll(paths, [SCAN] * count, tmp_path)
    window.filmstrip.strip.set_frames([f.number for f in window.session.frames])
    return paths


def test_late_signals_from_a_replaced_roll_are_dropped(tmp_path, app):
    window = MainWindow()
    (path, *_) = _bare_roll(window, tmp_path)
    window._generation = 5  # the roll that started these loads has since been replaced
    preview = np.full((8, 8, 3), 0.1, dtype=np.float32)

    window._on_preview_ready(4, 0, preview, None, None)
    assert window.session.frames[0].preview is None  # not stored: it belongs to the old roll

    window._current = 0
    window._on_frame_loaded(4, path, np.ones((8, 8, 3), dtype=np.float32), None)
    assert window.full_image is None and window.display is None

    window._on_preview_ready(5, 0, preview, None, None)  # the current generation still lands
    assert window.session.frames[0].preview is preview
    window.close()


def test_only_one_frame_load_is_in_flight_and_the_latest_frame_wins(monkeypatch, tmp_path, app):
    import threading

    ParkedLoader.reset()
    started = record_started_threads(monkeypatch)
    monkeypatch.setattr(mw, "FrameLoader", ParkedLoader)
    window = MainWindow()
    a, b, c = _bare_roll(window, tmp_path)
    window._select_frame(0)
    window._load_current_frame()  # asking again while frame 0 loads starts nothing
    window._select_frame(1)  # scrubbing along the strip: neither of these starts a load ...
    window._select_frame(2)
    assert ParkedLoader.started_for == [a]
    ParkedLoader.release.set()  # ... until the one in flight ends: then only where the user landed
    deadline = time.monotonic() + 30
    while (ParkedLoader.started_for != [a, c] or window._threads) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert ParkedLoader.started_for == [a, c]  # frame b was never loaded
    window.close()
    assert started and all(not thread.isRunning() for thread in started)


def test_closing_the_picker_mid_frame_load_waits_for_the_loader(monkeypatch, tmp_path, app):
    """A window destroyed while a FrameLoader still runs aborts Qt. close() must not
    return until every thread it started has stopped, including one that is mid-load."""
    ParkedLoader.reset()
    started = record_started_threads(monkeypatch)
    monkeypatch.setattr(mw, "FrameLoader", ParkedLoader)
    window = MainWindow()
    _bare_roll(window, tmp_path)
    window._select_frame(0)
    assert len(started) == 1 and started[0].isRunning()  # mid-load, and it will not finish by itself
    window.close()
    assert all(not thread.isRunning() for thread in started)


class _ParkedRenderer(QThread):
    """Stands in for ProofRenderer.run: a rebuild that stays busy until the test lets it finish
    (deliberately ignoring the interruption request, so it is still running when closeEvent looks)."""

    release = None

    def run(self):
        _ParkedRenderer.release.wait(30)


def _real_proof_windows(monkeypatch, tmp_path, app):
    import threading

    from halide.gui import proof_window

    _ParkedRenderer.release = threading.Event()
    monkeypatch.setattr(proof_window.ProofRenderer, "run", _ParkedRenderer.run)
    window = MainWindow()  # the real ProofWindow, unlike _window_with_points
    paths = _bare_roll(window, tmp_path)
    for path, density in zip(paths, (0.4, 0.9, 1.5)):
        window.session.add_point(_point(path, density))
    window._refresh_points()
    return window, proof_window


def test_closing_a_proof_window_mid_render_hands_its_renderer_to_the_stopping_set(monkeypatch, tmp_path, app):
    window, proof_window = _real_proof_windows(monkeypatch, tmp_path, app)
    window._on_proof()
    first = window._proof
    renderer = first._renderer
    deadline = time.monotonic() + 10
    while not renderer.isRunning() and time.monotonic() < deadline:
        time.sleep(0.01)
    late = []
    monkeypatch.setattr(proof_window.ProofWindow, "_on_frame_done", lambda self, *a: late.append(a))
    first.close()  # must return at once even though the renderer is still busy
    assert renderer.isRunning() and renderer in proof_window._STOPPING
    assert renderer.isInterruptionRequested()
    renderer.frameDone.emit(0, None, None, "late")  # disconnected: nothing reaches the closed window
    assert late == []
    _ParkedRenderer.release.set()
    proof_window.wait_for_stopping_renderers()
    assert not renderer.isRunning() and not proof_window._STOPPING
    window.close()


def test_closing_the_picker_mid_rebuild_waits_for_every_renderer(monkeypatch, tmp_path, app):
    window, proof_window = _real_proof_windows(monkeypatch, tmp_path, app)
    window._on_proof()
    first = window._proof._renderer
    window._on_proof()  # a rebuild: the first sheet is closed and stops in the background
    second = window._proof._renderer
    assert first in proof_window._STOPPING and first is not second
    threading_release = _ParkedRenderer.release
    import threading

    threading.Timer(0.3, threading_release.set).start()  # the workers wind down while close waits
    window.close()  # closeEvent: the current sheet closes, then every stopping renderer is waited for
    assert not first.isRunning() and not second.isRunning()
    assert not proof_window._STOPPING


# --- one red widget per window ------------------------------------------------------------------


def test_only_the_one_primary_button_is_red(app):
    from PySide6.QtWidgets import QPushButton

    from halide.gui import theme

    # In the stylesheet, RED colours nothing but the primary button's states.
    blocks = [b for b in theme.STYLESHEET.split("}") if theme.RED in b or theme.RED_HOVER in b or theme.RED_ACTIVE in b]
    assert blocks and all('QPushButton[role="primary"]' in b for b in blocks)
    # And each window has exactly one such button.
    for window in (MainWindow(), MainWindow(is_pick_session=True)):
        primaries = [b for b in window.findChildren(QPushButton) if b.property("role") == "primary"]
        assert primaries == [window.primary_button]
        window.close()


# --- closing and reopening never silently throws points away -------------------------------------


def _close_asks(window, monkeypatch, answer):
    asked = []
    monkeypatch.setattr(window, "_confirm_close", lambda n: asked.append(n) or answer)
    return asked


def test_closing_with_unsaved_points_and_keep_picking_leaves_the_window_open(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    stopped = []
    monkeypatch.setattr(window, "_stop_loaders", lambda **kw: stopped.append(kw))
    asked = _close_asks(window, monkeypatch, False)
    window.show()
    assert not window.close()
    assert asked == [3] and window.isVisible() and not stopped


def test_closing_with_unsaved_points_and_close_closes(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    asked = _close_asks(window, monkeypatch, True)
    window.show()
    assert window.close()
    assert asked == [3] and not window.isVisible()


def test_closing_without_points_or_after_a_save_asks_nothing(monkeypatch, tmp_path, app):
    empty = MainWindow()
    asked = _close_asks(empty, monkeypatch, False)
    assert empty.close() and not asked
    window = _window_with_points(monkeypatch, tmp_path, app)
    window._on_saved("x", "saved")
    asked = _close_asks(window, monkeypatch, False)
    assert window.close() and not asked


def test_one_point_added_after_a_save_asks_again(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    window._on_saved("x", "saved")
    window.session.add_point(_point(window.session.frames[0].path, 2.0))
    asked = _close_asks(window, monkeypatch, True)
    window.close()
    assert asked == [4]


def _saved_profile(window, tmp_path):
    from halide.calibration.profile_store import save_named_profile

    s = window.session
    return save_named_profile(s.profile_to_save(), "p", profiles_dir=tmp_path / "profiles", **s.sidecars())


def test_reopening_a_profile_unchanged_then_closing_asks_nothing(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    path = _saved_profile(window, tmp_path)
    monkeypatch.setattr(window, "_confirm_new_calibration", lambda: True)
    (tmp_path / "a.tif").touch()  # the recorded roll folder must hold scans, or nothing is reopened
    monkeypatch.setattr(window, "load_roll", lambda *a, **k: None)  # (they aren't real ones)
    window.open_profile(path)
    assert window.session.points
    asked = _close_asks(window, monkeypatch, False)
    assert window.close() and not asked


def test_opening_an_unreadable_profile_keeps_the_points_and_asks_nothing(monkeypatch, tmp_path, app):
    window = _window_with_points(monkeypatch, tmp_path, app)
    before = list(window.session.points)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    monkeypatch.setattr(window, "_confirm_new_calibration", lambda: pytest.fail("asked"))
    window.open_profile(bad)
    assert window.session.points == before
    assert "Couldn't open bad.json" in window.status_label.toolTip()


def test_grade_slider_never_reaches_zero(app):
    from halide.core.types import ToneCurveParams
    from halide.gui.drawers import PrintControls

    controls = PrintControls()
    assert controls.grade.minimum() == 1
    emitted = []
    controls.changed.connect(lambda tone, _flat: emitted.append(tone))
    controls.set_override(ToneCurveParams(mode="paper", exposure=0.0, contrast=0.0))  # an old saved zero
    controls.exposure.setValue(10)
    assert emitted[-1].contrast == pytest.approx(0.01)


def test_coverage_dots_stack_close_points_and_only_points_to_act_on_get_numbers():
    from halide.gui import theme
    from halide.gui.step_wedge import WedgeTick, dot_stacks, labelled

    xs = [100.0, 101.0, 102.5, 200.0]
    stacks = dot_stacks(xs)
    assert [s[1] for s in stacks[:3]] == [0, 1, 2] and stacks[0][2] == 3  # one pile of three
    assert stacks[3] == (200.0, 0, 1)
    calm = [WedgeTick(n + 1, 1.0, theme.AGREE_CALM) for n in range(30)]
    many = [float(10 * n) for n in range(30)]
    assert labelled(calm, many) == []  # thirty agreeing points: no numbers at all
    ticks = list(calm)
    ticks[3] = WedgeTick(4, 1.0, theme.AGREE_CALM, selected=True)
    ticks[10] = WedgeTick(11, 1.0, theme.AGREE_RED)
    ticks[11] = WedgeTick(12, 1.0, theme.AGREE_AMBER)  # 10 px from point 11: would overprint it
    assert sorted(labelled(ticks, many)) == [3, 10]
