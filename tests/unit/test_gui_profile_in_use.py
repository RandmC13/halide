"""A saved profile used on another roll: Use for this roll keeps the open roll, the profile's points
stay in the fit (listed "on <roll>"), new points refine it, and Save updates it or saves a new one."""

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from halide.calibration.profile_store import load_anchors, save_named_profile
from halide.gui import main_window as mw
from halide.gui.main_window import MainWindow
from halide.gui.point_list import _Row
from halide.gui.save_dialog import SaveProfileDialog
from tests.unit.test_gui_interaction import HIGHLIGHT, SHADOW, _click_image
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _roll(tmp_path, name, count=2):
    roll = tmp_path / name
    roll.mkdir()
    for i in range(count):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i + (10 if name != "Roll16" else 0))
    return roll


def _window(app, roll):
    window = MainWindow()
    window.show()
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    return window


def _close(window):
    window._confirm_close = lambda count: True
    window.close()


@pytest.fixture
def setup(tmp_path, monkeypatch, app):
    """A profile 'Portra 400' made from two points on Roll16, and a window with Roll17 open."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    first = _window(app, _roll(tmp_path, "Roll16"))
    _click_image(first, SHADOW)
    _click_image(first, HIGHLIGHT)
    session = first.session
    sidecars = session.sidecars()
    sidecars["roll"] = None
    profile_path = save_named_profile(session.profile_to_save(), "Portra 400", **sidecars)
    old_points = list(session.points)
    _close(first)
    window = _window(app, _roll(tmp_path, "Roll17"))
    yield window, profile_path, old_points
    _close(window)


def _rows(window):
    return window.point_list.findChildren(_Row)


def test_use_for_this_roll_keeps_the_roll_and_shows_the_profile_as_a_card(setup):
    window, path, old_points = setup
    frames = [f.path for f in window.session.frames]
    window.use_profile(path)
    assert [f.path for f in window.session.frames] == frames and window.session.roll_folder.name == "Roll17"
    assert window.session.points == old_points and window.session.profile() is not None  # the positive's fit
    assert _rows(window) == []  # the card stands in for the rows
    texts = [label.text() for label in window.point_list.findChildren(mw.QLabel)]
    assert "Portra 400" in texts and "2 points on Roll16" in texts and "Click points on this roll to refine it" in texts
    assert "Using" in window.profile_chip.text() and "Portra 400" in window.profile_chip.text()
    window.positive_button.setChecked(True)
    assert window.source_note.text() == "Profile Portra 400 · 2 points"


def test_new_points_are_listed_after_the_profiles_own_marked_on_their_roll(setup):
    window, path, _ = setup
    window.use_profile(path)
    _click_image(window, SHADOW)
    rows = _rows(window)
    assert len(rows) == 3
    first_row = [label.text() for label in rows[0].findChildren(mw.QLabel)]
    assert "on Roll16" in first_row
    assert "+ 1 new point" in window.profile_chip.text()


def test_a_deleted_frame_still_says_missing(setup):
    window, path, old_points = setup
    old_points[0].frame.unlink()
    window.use_profile(path)
    _click_image(window, SHADOW)
    names = [[label.text() for label in row.findChildren(mw.QLabel)] for row in _rows(window)]
    assert sum("frame_00 (missing)" in texts for texts in names) == 2  # both of the profile's points were on it
    assert not any("on Roll16" in texts for texts in names)


def test_update_adds_the_new_points_to_the_profile(setup, monkeypatch):
    window, path, old_points = setup
    window.use_profile(path)
    _click_image(window, SHADOW)
    new = window.session.new_points()

    def update(dialog):
        assert dialog.update_button.text() == "Update 'Portra 400'"
        dialog.update_button.click()

    monkeypatch.setattr(SaveProfileDialog, "exec", update)
    window.primary_button.click()
    saved, _roll = load_anchors(path)
    assert len(saved) == len(old_points) + len(new) == 3
    assert window.session.new_points() == []  # they're the profile's now


def test_save_as_new_leaves_the_old_profile_byte_identical(setup, monkeypatch):
    window, path, _ = setup
    before = path.read_bytes()
    window.use_profile(path)
    _click_image(window, SHADOW)

    def save_as_new(dialog):
        dialog._name_input.setText("Portra 400 · Roll17")
        dialog._save_button.click()

    monkeypatch.setattr(SaveProfileDialog, "exec", save_as_new)
    window.primary_button.click()
    assert path.read_bytes() == before
    assert len(load_anchors(path.parent / "Portra 400 · Roll17.json")[0]) == 3


def test_clear_all_keeps_the_profiles_points(setup, monkeypatch):
    window, path, old_points = setup
    window.use_profile(path)
    _click_image(window, SHADOW)
    asked = []
    monkeypatch.setattr(window, "_confirm", lambda question: asked.append(question) or True)
    window.clear_all_button.click()
    assert asked == ["Remove the 1 new point? The profile's own points stay."]
    assert window.session.points == old_points
