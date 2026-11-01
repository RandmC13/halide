"""The right panel's three states and the develop side's fields (no run is started)."""

import shlex
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from halide.calibration.profile_store import save_named_profile
from halide.gui import main_window as mw
from halide.gui.develop_panel import DevelopPanel
from halide.gui.main_window import MainWindow
from tests.unit.test_gui_interaction import HIGHLIGHT, SHADOW, _click_image
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _roll(tmp_path, name="Roll16", count=2):
    roll = tmp_path / name
    roll.mkdir()
    for i in range(count):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    return roll


@pytest.fixture
def window(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    win = MainWindow()
    win._confirm_close = lambda count: True
    win._confirm_new_calibration = lambda: True
    win.show()
    win.load_roll(_roll(tmp_path))
    _pump(app, lambda: _frame_ready(win, 0))
    yield win
    win.close()


def _fit(window):
    _click_image(window, SHADOW)
    _click_image(window, HIGHLIGHT)


def _develop_side(window):
    _fit(window)
    window.primary_button.click()
    return window.develop_panel


def _command(panel):
    return shlex.split(panel.command.toPlainText())


# --- the three states ---------------------------------------------------------------------------


def test_the_panel_turns_over_and_back(window):
    _fit(window)
    assert window._stack.currentIndex() == 0
    window.primary_button.click()
    assert window._stack.currentIndex() == 1
    window.develop_panel.back_button.click()
    assert window._stack.currentIndex() == 0
    window._turn_to(2)
    assert window._stack.currentIndex() == 2
    window.summary_back.click()
    assert window._stack.currentIndex() == 0


def test_the_turn_ends_with_the_page_in_place_even_with_motion_on(window, monkeypatch, app):
    monkeypatch.delenv("HALIDE_MOTION")
    _fit(window)
    window.primary_button.click()
    _pump(app, lambda: window.develop_panel.pos().x() == 0, timeout=3)
    assert window.develop_panel.geometry() == window._stack.rect()


def test_clicking_a_point_on_the_develop_side_goes_back_to_the_points(window):
    _develop_side(window)
    _click_image(window, (4, 6))
    assert window._stack.currentIndex() == 0


def test_loading_a_roll_returns_to_the_points(window, tmp_path):
    _develop_side(window)
    window.load_roll(_roll(tmp_path, "Roll17"))
    assert window._stack.currentIndex() == 0


# --- the bottom buttons -------------------------------------------------------------------------


def test_save_and_develop_follow_the_fit(window):
    assert not window.save_button.isEnabled() and not window.primary_button.isEnabled()
    _click_image(window, SHADOW)
    assert not window.save_button.isEnabled() and not window.primary_button.isEnabled()
    _click_image(window, HIGHLIGHT)
    assert window.save_button.isEnabled() and window.primary_button.isEnabled()
    assert window.primary_button.text() == "Develop roll…" and window.save_button.text() == "Save profile"


def test_a_profile_in_use_enables_both_buttons(window, tmp_path):
    _fit(window)
    path = save_named_profile(window.session.profile_to_save(), "Portra", **window.session.sidecars())
    window.session.reset()
    window._refresh_points()
    assert not window.save_button.isEnabled()
    window.use_profile(path)
    assert window.save_button.isEnabled() and window.primary_button.isEnabled()


def test_several_rolls_or_loose_scans_cannot_be_developed(window, tmp_path, app):
    one, two = _roll(tmp_path, "A"), _roll(tmp_path, "B")
    window._start_rolls([(one, sorted(one.iterdir())), (two, sorted(two.iterdir()))])
    _pump(app, lambda: _frame_ready(window, 0))
    _fit(window)
    assert window.save_button.isEnabled() and not window.primary_button.isEnabled()
    assert "one roll folder" in window.primary_button.toolTip()
    window.load_files(sorted(one.iterdir()))
    _pump(app, lambda: _frame_ready(window, 0))
    _fit(window)
    assert not window.primary_button.isEnabled() and "one roll folder" in window.primary_button.toolTip()


def test_the_pick_session_keeps_its_single_develop_button(tmp_path, app):
    window = MainWindow(is_pick_session=True)
    assert window.primary_button.text() == "Develop"
    assert not hasattr(window, "save_button") and not hasattr(window, "develop_panel")
    assert window._stack.count() == 1
    window.close()


# --- the develop side ---------------------------------------------------------------------------


def test_defaults(window):
    panel = _develop_side(window)
    roll = window.session.roll_folder
    assert Path(panel.into_input.text()) == roll / "developed"
    assert (panel.output.index(), panel.names.text(), panel.existing.index(), panel.sharing.index()) == (0, "{stem}", 0, 0)
    assert not panel.match_exposure.isVisibleTo(panel) and not panel.sharing_note.isVisibleTo(panel)
    s = panel.settings()
    assert s.output == "print" and s.existing == "ask" and s.sharing == "none" and not s.match_scan_exposure
    assert _command(panel)[:3] == ["halide", "batch", str(roll)]


def test_each_field_changes_the_command(window):
    panel = _develop_side(window)
    panel.choice.set_index(1)  # use once
    base = _command(panel)
    assert "--output" not in base and "--name" not in base and "--overwrite" not in base
    panel.output.set_index(1)
    assert _command(panel)[-2:] == ["--output", "flat"]
    panel.output.set_index(0)
    panel.names.setText("{roll}_{n:02}")
    assert "--name" in _command(panel) and "{roll}_{n:02}" in _command(panel)
    panel.names.setText("")
    assert "--name" not in _command(panel)  # a blank template is the default
    panel.existing.set_index(1)
    assert "--overwrite" in _command(panel)
    panel.existing.set_index(2)
    assert "--skip-existing" in _command(panel) and "--overwrite" not in _command(panel)
    panel.into_input.setText("/elsewhere/prints")
    assert _command(panel)[2:4] == [str(window.session.roll_folder), "/elsewhere/prints"]
    assert "/elsewhere/prints/halide-calibration.json" in _command(panel)


def test_into_is_remembered_while_the_roll_is_the_same(window):
    panel = _develop_side(window)
    panel.into_input.setText("/scratch/out")
    panel.back_button.click()
    window.primary_button.click()
    assert panel.into_input.text() == "/scratch/out"


def test_sharing_shows_its_note(window):
    panel = _develop_side(window)
    panel.sharing.set_index(1)
    assert panel.sharing_note.isVisibleTo(panel)
    assert panel.sharing_note.full_text() == f"sRGB, full size, in {panel.into() / 'sharing'}"
    assert panel.settings().sharing == "jpg"
    panel.sharing.set_index(2)
    assert panel.settings().sharing == "png"


def test_the_example_line_and_a_refused_template(window):
    panel = _develop_side(window)
    panel.choice.set_index(1)
    panel.names.setText("{roll}_{n:02}")
    assert panel.example.full_text() == "frame_00.tiff → Roll16_01.tiff"
    panel.names.setText("{bogus}")
    assert panel.example.full_text() == ""
    assert panel.error.full_text() and not panel.develop_button.isEnabled()
    assert panel.develop_button.toolTip() == panel.error.full_text()
    panel.names.setText("{stem}")
    assert panel.develop_button.isEnabled() and not panel.error.isVisibleTo(panel)


def test_the_button_counts_the_frames(window):
    panel = _develop_side(window)
    panel.choice.set_index(1)
    assert panel.develop_button.text() == "Develop 2 frames"


def test_scan_exposure_row_only_with_differing_exposures(window):
    panel = _develop_side(window)
    assert not panel.match_exposure.isVisibleTo(panel)
    panel.back_button.click()
    window._exposure_differs = True
    window.primary_button.click()
    assert panel.match_exposure.isVisibleTo(panel) and panel.exposure_label.isVisibleTo(panel)
    panel.choice.set_index(1)
    panel.match_exposure.setChecked(True)
    assert "--match-scan-exposure" in _command(panel)


# --- the calibration choice ---------------------------------------------------------------------


def test_unsaved_points_need_a_choice_before_developing(window):
    panel = _develop_side(window)
    assert panel.choice_value() is None
    assert not panel.develop_button.isEnabled()
    assert "Choose" in panel.develop_button.toolTip()
    panel.choice.set_index(1)
    assert panel.choice_value() == "once" and panel.develop_button.isEnabled()
    assert not panel.name_input.isVisibleTo(panel)
    assert _command(panel)[-1] == str(panel.into() / "halide-calibration.json")


def test_save_as_a_profile_asks_for_a_valid_new_name(window, tmp_path):
    panel = _develop_side(window)
    panel.choice.set_index(0)
    assert panel.choice_value() == "save" and panel.name_input.isVisibleTo(panel)
    assert not panel.develop_button.isEnabled()  # no name yet
    panel.name_input.setText(".hidden")
    assert panel.name_error.isVisibleTo(panel) and not panel.develop_button.isEnabled()
    panel.name_input.setText("Portra 400")
    assert not panel.name_error.isVisibleTo(panel) and panel.develop_button.isEnabled()
    assert _command(panel)[-1] == str(tmp_path / "config" / "halide" / "profiles" / "Portra 400.json")
    save_named_profile(window.session.profile_to_save(), "Portra 400")
    panel.name_input.setText("portra 400")
    assert "already exists" in panel.name_error.full_text() and not panel.develop_button.isEnabled()


def test_the_name_is_prefilled_from_the_last_save(window):
    _fit(window)
    window._profile_name = "Ektar"
    window._saved_points = []  # changed since, so it isn't "saved" any more
    window.primary_button.click()
    assert window.develop_panel.name_input.text() == "Ektar"


def test_a_saved_profile_in_use_needs_no_choice(window):
    _fit(window)
    path = save_named_profile(window.session.profile_to_save(), "Portra", **window.session.sidecars())
    window.session.reset()
    window.use_profile(path)
    window.primary_button.click()
    panel = window.develop_panel
    assert panel.choice_value() == "profile" and panel.develop_button.isEnabled()
    assert "Portra" in panel.profile_label.text() and "saved" in panel.profile_label.text()
    assert not panel.choice.isVisibleTo(panel)
    assert _command(panel)[-1] == str(path)


def test_new_points_on_a_profile_offer_add_or_this_roll_only(window):
    _fit(window)
    path = save_named_profile(window.session.profile_to_save(), "Portra", **window.session.sidecars())
    window._saved_points = list(window.session.points)  # saved, so using the profile asks nothing
    window.use_profile(path)
    _click_image(window, (4, 6))
    window.primary_button.click()
    panel = window.develop_panel
    assert panel.choice.buttons[0].text() == "Add this point to 'Portra'"
    assert panel.choice.buttons[1].text() == "Use it for this roll only"
    assert panel.choice_value() is None and not panel.develop_button.isEnabled()
    panel.choice.set_index(0)
    assert panel.choice_value() == "add" and _command(panel)[-1] == str(path)
    panel.choice.set_index(1)
    assert panel.choice_value() == "roll" and _command(panel)[-1] == str(panel.into() / "halide-calibration.json")


def test_pressing_develop_emits_the_settings_and_the_choice(window):
    panel = _develop_side(window)
    panel.choice.set_index(1)
    got = []
    panel.developClicked.connect(lambda s, c: got.append((s, c)))
    panel.develop_button.click()
    assert got and got[0][1] == "once" and got[0][0] == panel.settings()


def test_a_new_calibration_asks_for_the_choice_again(window):
    panel = _develop_side(window)
    panel.choice.set_index(1)
    panel.back_button.click()
    window.primary_button.click()
    assert panel.choice_value() == "once"  # the same calibration: the choice stands
    panel.back_button.click()
    _click_image(window, (4, 6))
    window.primary_button.click()
    assert panel.choice_value() is None


def test_standalone_panel_is_inert_until_prepared(app):
    DevelopPanel()  # builds without a roll
