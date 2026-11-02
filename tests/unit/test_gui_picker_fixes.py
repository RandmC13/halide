"""Picker behaviours on Qt's offscreen platform: which mouse button does what, the status line,
stale per-frame state, retrying a failed load, keyboard stepping, scrolling the point list."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QSlider

from halide.calibration.anchors import NeutralPoint
from halide.gui import main_window as mw
from halide.gui.filmstrip import _Strip
from halide.gui.image_view import ImageView
from halide.gui.main_window import MainWindow
from halide.gui.point_list import PointList, _Row
from halide.gui.roll import PointView
from halide.io.scan_metadata import ScanSettings
from tests.unit._gui_helpers import ParkedLoader

SCAN = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
BUTTONS = [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(MainWindow, "_confirm_close", lambda self, count: True)
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)  # never wait on a modal box
    ParkedLoader.reset()
    monkeypatch.setattr(mw, "FrameLoader", ParkedLoader)  # a frame load that never finishes by itself
    yield
    ParkedLoader.release.set()  # so a failing test can't leave a thread running


def _roll(window, tmp_path, count=3):
    paths = [tmp_path / f"f{i}.tif" for i in range(count)]
    window.session.set_roll(paths, [SCAN] * count, tmp_path)
    window.filmstrip.strip.set_frames([f.number for f in window.session.frames])
    return paths


def _view(index):
    point = NeutralPoint(frame=__import__("pathlib").Path(f"/x/f{index}.tif"), x=1, y=1, rgb=(0.5, 0.5, 0.5), scan=SCAN)
    return PointView(index=index, point=point, frame_number=index + 1, green_density=1.0, agreement=None)


# --- only the left button acts ----------------------------------------------------------------


@pytest.mark.parametrize("button", BUTTONS)
def test_only_the_left_button_adds_a_point(app, button):
    view = ImageView()
    view.resize(400, 300)
    view.set_image(__import__("PySide6.QtGui", fromlist=["QPixmap"]).QPixmap(40, 30))
    clicks = []
    view.clicked.connect(lambda x, y: clicks.append((x, y)))
    QTest.mouseClick(view, button, pos=QPoint(200, 150))
    assert clicks == []
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    assert len(clicks) == 1


@pytest.mark.parametrize("button", BUTTONS)
def test_only_the_left_button_selects_a_row(app, button):
    row = _Row(_view(0), selected=False)
    seen = []
    row.clicked.connect(seen.append)
    QTest.mouseClick(row, button)
    assert seen == []
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert seen == [0]


@pytest.mark.parametrize("button", BUTTONS)
def test_only_the_left_button_picks_a_frame(app, button):
    strip = _Strip()
    strip.set_frames([1, 2])
    seen = []
    strip.frameClicked.connect(seen.append)
    QTest.mouseClick(strip, button, pos=strip.frame_rect(1).center())
    assert seen == []
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=strip.frame_rect(1).center())
    assert seen == [1]


# --- status line ------------------------------------------------------------------------------


def test_loading_does_not_overwrite_a_more_useful_status(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._sticky("Reopened 'p': 2 points")
    window._select_frame(0)
    assert window.status_label.full_text() == "Reopened 'p': 2 points"
    preview = np.full((8, 8, 3), 0.1, dtype=np.float32)
    for i in range(3):
        window._on_preview_ready(window._generation, i, preview, None, None)
    assert window.status_label.full_text() == "Reopened 'p': 2 points"  # not "Roll loaded"
    window.close()


def test_a_frame_the_person_chooses_ends_a_sticky_status(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._sticky("Reopened 'p': 2 points")
    window._select_frame(0)
    window._go_to_frame(1)  # a filmstrip click or an arrow key
    assert window.status_label.full_text() == "Loading f1.tif…"
    window.close()


def test_an_error_is_replaced_once_the_next_frame_loads(tmp_path, app):
    window = MainWindow()
    paths = _roll(window, tmp_path)
    window._select_frame(0)
    window._on_frame_loaded(window._generation, paths[0], None, "disk on fire")
    window._frame_load_finished()
    window._frame_loader = None
    window._go_to_frame(1)
    assert window.status_label.full_text() == "Loading f1.tif…"
    window.close()


def test_loading_still_reports_itself_on_an_idle_status(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(0)
    assert window.status_label.full_text() == "Loading f0.tif…"
    window._select_frame(1)
    assert window.status_label.full_text() == "Loading f1.tif…"
    window.close()


def test_reopened_status_counts_points_with_plural(tmp_path, app, monkeypatch):
    from halide.calibration.profile_store import save_named_profile

    window = MainWindow()
    paths = _roll(window, tmp_path)
    for d in (0.4, 0.9):
        window.session.add_point(NeutralPoint(frame=paths[0], x=1, y=1, rgb=(10.0**-d,) * 3, scan=SCAN))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    saved = save_named_profile(window.session.profile_to_save(), "p", overwrite=True, **window.session.sidecars())
    window.close()
    reopened = MainWindow()
    (tmp_path / "f0.tif").touch()
    monkeypatch.setattr(reopened, "_start_rolls", lambda *a, **k: None)  # the folder holds no real scans
    reopened.open_profile(saved)
    assert reopened.status_label.full_text().startswith("Reopened 'p': 2 points")
    reopened.close()


# --- per-frame state --------------------------------------------------------------------------


def test_pending_flash_dies_with_the_frame_change_and_the_new_roll(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(0)
    window._pending_flash = 3
    window._select_frame(1)
    assert window._pending_flash is None
    window._pending_flash = 3
    window._stop_loaders = lambda wait=False: None
    window._start_roll(list(p.path for p in window.session.frames), tmp_path)
    assert window._pending_flash is None
    window.close()


def test_row_click_on_another_frame_flashes_there(tmp_path, app):
    window = MainWindow()
    paths = _roll(window, tmp_path)
    window._select_frame(0)
    window.session.add_point(NeutralPoint(frame=paths[2], x=1, y=1, rgb=(0.5,) * 3, scan=SCAN))
    window._on_row_clicked(0)
    assert window._current == 2 and window._pending_flash == 1
    window.close()


def test_changing_frame_clears_the_previous_frames_numbers(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(0)
    window._display_estimate = object()
    window._auto_mask = np.zeros((2, 2), dtype=bool)
    window._stretch = (0.2, 0.7)
    window._select_frame(1)
    assert window._display_estimate is None and window._auto_mask is None and window._stretch == (0.0, 1.0)
    window.close()


def test_a_failed_load_sets_frame_error_and_keeps_the_numbers_empty(tmp_path, app):
    window = MainWindow()
    paths = _roll(window, tmp_path)
    window._select_frame(0)
    window._on_frame_loaded(window._generation, paths[0], None, "disk on fire")
    assert window.session.frames[0].error == "disk on fire"
    assert window._display_estimate is None and window._auto_mask is None
    window.close()


def test_clicking_a_frame_whose_load_failed_retries_it(tmp_path, app):
    window = MainWindow()
    paths = _roll(window, tmp_path)
    window._select_frame(0)
    window._on_frame_loaded(window._generation, paths[0], None, "disk on fire")
    window._frame_load_finished()  # the loader thread has ended
    ParkedLoader.started_for.clear()
    window._frame_loader = None
    window._select_frame(0)  # a click on the frame already shown
    assert ParkedLoader.started_for == [paths[0]]
    assert window.session.frames[0].error is None
    window.close()


# --- other windows and profiles ---------------------------------------------------------------


def test_loading_another_roll_closes_the_contact_sheet(monkeypatch, tmp_path, app):
    class Fake(QDialog):
        def __init__(self, *a, **k):
            super().__init__(a[0])

        def mark_stale(self, _):
            pass

    monkeypatch.setattr(mw, "ProofWindow", Fake)
    window = MainWindow()
    paths = _roll(window, tmp_path)
    window.session.add_point(NeutralPoint(frame=paths[0], x=1, y=1, rgb=(0.2,) * 3, scan=SCAN))
    window.session.add_point(NeutralPoint(frame=paths[0], x=2, y=1, rgb=(0.05,) * 3, scan=SCAN))
    window._on_proof()
    proof = window._proof
    assert proof is not None and proof.isVisible()
    window._start_roll(paths, tmp_path)
    assert window._proof is None and not proof.isVisible()
    window.close()


def test_opening_a_profile_whose_roll_folder_has_no_scans_asks_and_keeps_the_points(tmp_path, app, monkeypatch):
    from halide.calibration.profile_store import save_named_profile

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    window = MainWindow()
    paths = _roll(window, tmp_path)
    for d in (0.4, 0.9):
        window.session.add_point(NeutralPoint(frame=paths[0], x=1, y=1, rgb=(10.0**-d,) * 3, scan=SCAN))
    saved = save_named_profile(window.session.profile_to_save(), "p", overwrite=True, **window.session.sidecars())
    window.close()
    reopened = MainWindow()  # tmp_path holds no .tif files, so the recorded roll folder is empty
    asked = []
    monkeypatch.setattr(MainWindow, "_roll_not_found", lambda self, name, folder: asked.append(folder))
    reopened.open_profile(saved)
    assert asked == [tmp_path] and len(reopened.session.points) == 2  # they still count in the fit
    reopened.close()


# --- keyboard ---------------------------------------------------------------------------------


def test_left_and_right_step_through_the_frames(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(0)
    QTest.keyClick(window, Qt.Key.Key_Right)
    assert window._current == 1
    QTest.keyClick(window, Qt.Key.Key_Right)
    QTest.keyClick(window, Qt.Key.Key_Right)  # the last frame stays put
    assert window._current == 2
    QTest.keyClick(window, Qt.Key.Key_Left)
    assert window._current == 1
    window.close()


def test_arrows_still_step_frames_right_after_clicking_the_filmstrip(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window.show()
    strip = window.filmstrip.strip
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=strip.frame_rect(0).center())
    QTest.keyClick(window.filmstrip, Qt.Key.Key_Right)  # a real click leaves the strip focused
    assert window._current == 1
    window.close()


def test_a_focused_slider_keeps_its_arrow_keys(tmp_path, app):
    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(1)
    window.positive_button.setChecked(True)
    window.show()
    slider = window.print_controls.findChildren(QSlider)[0]
    slider.setFocus()
    before = slider.value()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    assert slider.value() != before and window._current == 1
    window.close()


# --- point list scrolls to a new point --------------------------------------------------------


def test_a_new_point_past_the_visible_rows_is_scrolled_into_view(app):
    window = QApplication.instance() and PointList()
    window.resize(300, 150)
    window.show()
    window.set_rows([_view(i) for i in range(9)], 8)
    for _ in range(5):
        app.processEvents()
    assert window.verticalScrollBar().value() > 0
    window.close()


# --- the roll's estimate ----------------------------------------------------------------------


def test_the_positive_follows_the_rolls_estimate_as_previews_arrive(tmp_path, app):
    from halide.core.types import DensityProfile

    window = MainWindow()
    _roll(window, tmp_path)
    window._select_frame(0)
    window.positive_button.setChecked(True)
    preview = np.full((8, 8, 3), 0.1, dtype=np.float32)
    first = DensityProfile(white_balance=(0.5, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0), source="auto")
    window._on_preview_ready(window._generation, 0, preview, first, None)
    window._follow_roll_estimate()  # what the throttle timer runs
    assert window._positive_profile == window.session.roll_estimate(tmp_path)
    assert window.source_note.text().endswith(" · from 1 frame")
    for i in (1, 2):
        estimate = DensityProfile(white_balance=(0.7 + i, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0), source="auto")
        window._on_preview_ready(window._generation, i, preview, estimate, None)
    # The last preview re-renders at once, with every frame's vote.
    assert window.session.roll_votes(tmp_path) == 3
    assert window._positive_profile.white_balance[0] == pytest.approx(1.7)
    assert window.source_note.text().endswith(" · from 3 frames")
    window.close()
