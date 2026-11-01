"""Edit points on a profile whose points came from several rolls: every roll opens side by side in the
filmstrip, a moved roll is asked about once, and nothing about the rolls is stored in the profile."""

import json
import re

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from halide.calibration import anchors
from halide.calibration.anchors import NeutralPoint
from halide.calibration.profile_store import save_named_profile
from halide.core.types import DensityProfile
from halide.gui import filmstrip
from halide.gui import main_window as mw
from halide.gui.main_window import MainWindow
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _roll(tmp_path, name, count):
    roll = tmp_path / name
    roll.mkdir()
    for i in range(count):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    return roll


@pytest.fixture
def two_rolls(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    one, two = _roll(tmp_path, "Roll16", 3), _roll(tmp_path, "Roll17", 2)
    picks = [one / "frame_01.tiff", one / "frame_02.tiff", two / "frame_00.tiff"]
    points = [NeutralPoint(frame=f, x=1, y=1, rgb=(0.2, 0.2, 0.2), scan=None) for f in picks]
    profile = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))
    path = save_named_profile(profile, "both", anchors=[anchors.point_to_dict(p) for p in points])
    return tmp_path, one, two, path


def _open(app, path):
    window = MainWindow()
    window.show()
    window.activateWindow()
    window.open_profile(path)
    _pump(app, lambda: _frame_ready(window, 0))
    return window


def _close(window):
    window._confirm_close = lambda count: True
    window.close()


def _counter(window):
    return re.sub(r"<[^>]+>", "", window.frame_counter.text())


def test_both_rolls_open_in_order_with_their_own_numbers_and_markers(app, two_rolls):
    _, one, two, path = two_rolls
    window = _open(app, path)
    session, strip = window.session, window.filmstrip.strip
    assert session.rolls == [one, two] and session.roll_folder is None
    assert [f.number for f in session.frames] == [1, 2, 3, 1, 2]
    assert strip._counts == [0, 1, 1, 1, 0]
    assert not window.proof_button.isEnabled() and "one roll" in window.proof_button.toolTip()
    assert window.roll_label.text() == "2 ROLLS" and window.roll_label.toolTip() == "ROLL16\nROLL17"
    assert window.windowTitle() == "halide · 2 rolls"
    _close(window)


def test_the_counter_names_the_roll_and_left_right_cross_into_the_next(app, two_rolls):
    window = _open(app, two_rolls[3])
    assert _counter(window) == "FRAME 01 / 03 · ROLL 1 OF 2"
    for _ in range(2):
        QTest.keyClick(window, Qt.Key.Key_Right)
    assert _counter(window) == "FRAME 03 / 03 · ROLL 1 OF 2"
    QTest.keyClick(window, Qt.Key.Key_Right)
    _pump(app, lambda: _frame_ready(window, 3))
    assert _counter(window) == "FRAME 01 / 02 · ROLL 2 OF 2"
    _close(window)


def test_the_second_roll_sits_one_roll_gap_further_right(app, two_rolls):
    window = _open(app, two_rolls[3])
    strip = window.filmstrip.strip
    first_of_second = strip.frame_rect(3).x()
    strip.set_frames([1, 2, 3, 1, 2])
    assert first_of_second - strip.frame_rect(3).x() == filmstrip.ROLL_GAP
    assert strip.frame_rect(2).x() == filmstrip.GAP + 2 * (filmstrip.FRAME_WIDTH + filmstrip.GAP)
    _close(window)


def test_a_moved_roll_is_asked_about_once_and_the_other_still_loads(app, two_rolls, monkeypatch):
    tmp_path, one, two, path = two_rolls
    one.rename(tmp_path / "Roll16-moved")
    asked = []
    monkeypatch.setattr(MainWindow, "_roll_not_found", lambda self, name, folder: asked.append(folder))
    window = _open(app, path)
    assert asked == [one]
    assert window.session.rolls == [two] and window.session.roll_folder == two
    assert len(window.session.points) == 3  # the moved roll's points still count
    _close(window)


def test_saving_from_a_multi_roll_session_records_no_roll(app, two_rolls):
    tmp_path, *_, path = two_rolls
    window = _open(app, path)
    session = window.session
    assert "roll" not in session.sidecars()
    profile = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))
    saved = save_named_profile(profile, "again", **session.sidecars())
    data = json.loads(saved.read_text())
    assert "roll" not in data and len(data["anchors"]) == 3
    _close(window)


def test_reopened_stays_on_the_status_line_once_the_rolls_have_loaded(app, two_rolls):
    window = _open(app, two_rolls[3])
    assert window.status_label.full_text().startswith("Reopened 'both': 3 points")
    _close(window)


def test_find_roll_naming_a_roll_already_open_loads_it_once(app, two_rolls, monkeypatch):
    tmp_path, one, two, path = two_rolls
    one.rename(tmp_path / "Roll16-moved")
    monkeypatch.setattr(MainWindow, "_roll_not_found", lambda self, name, folder: two)  # the wrong folder
    window = _open(app, path)
    assert window.session.rolls == [two] and len(window.session.frames) == 2
    _close(window)
