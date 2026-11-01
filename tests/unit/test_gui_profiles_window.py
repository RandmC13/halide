"""The Profiles window: the list (newest first, with when each was made and its points), what a
damaged profile allows, rename inline and delete."""

import json
from datetime import date

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from halide.calibration.profile_store import default_profiles_dir, save_named_profile
from halide.core.types import DensityProfile
from halide.gui.profiles_window import ProfilesWindow, made_text

PROFILE = DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78), film_stock="Kodak Portra 400")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def anchor(frame):
        return {"frame": f"/rolls/Roll16/{frame}.tif", "x": 1, "y": 1, "rgb": [0.1, 0.1, 0.1], "scan": None}

    made = {"Gold 200": "2026-09-02T10:00:00+00:00", "Portra 400": "2026-10-18T10:00:00+00:00", "Ektar": None}
    for name, created in made.items():
        anchors = [anchor("a"), anchor("a"), anchor("b")] if name == "Portra 400" else None
        path = save_named_profile(PROFILE, name, anchors=anchors)
        data = json.loads(path.read_text())
        data["created_at"] = created
        path.write_text(json.dumps(data))
    (default_profiles_dir() / "Broken.json").write_text("{ not json")
    return default_profiles_dir()


def _column(window, column):
    return [window.table.item(row, column).text() for row in range(window.table.rowCount())]


def test_the_list_is_newest_first_with_its_date_and_points(profiles, app):
    window = ProfilesWindow()
    assert _column(window, 0)[:2] == ["Portra 400", "Gold 200"]
    portra = _column(window, 0).index("Portra 400")
    assert window.table.item(portra, 3).text() == "3 points on 2 frames"
    assert window.table.item(portra, 2).text() == made_text("2026-10-18T10:00:00+00:00")
    assert window.table.item(_column(window, 0).index("Gold 200"), 3).text() == "—"  # made without the window
    assert window.count_label.text() == "4 saved · newest first"
    window.close()


def test_made_reads_today_for_a_profile_saved_today():
    today = date(2026, 11, 1)
    assert made_text("2026-11-01T12:00:00", today=today) == "today"
    assert made_text("2026-10-18T12:00:00", today=today) == "2026-10-18"
    assert made_text(None) == ""


def test_a_damaged_profile_says_why_and_can_only_be_deleted(profiles, app):
    window = ProfilesWindow()
    window.set_roll_open(True)
    window.table.selectRow(_column(window, 0).index("Broken"))
    assert window.table.item(window.table.currentRow(), 1).text().startswith("damaged — ")
    assert window.delete_button.isEnabled()
    assert not any(b.isEnabled() for b in (window.rename_button, window.edit_button, window.use_button))
    window.close()


def test_use_for_this_roll_needs_an_open_roll(profiles, app):
    window = ProfilesWindow()
    window.table.selectRow(0)
    assert not window.use_button.isEnabled()
    window.set_roll_open(True)
    assert window.use_button.isEnabled()
    used = []
    window.useRequested.connect(used.append)
    window.use_button.click()
    assert used == [profiles / "Portra 400.json"]
    window.close()


def test_rename_inline_validates_as_you_type(profiles, app):
    window = ProfilesWindow()
    window.table.selectRow(0)
    window.rename_button.click()
    assert window._rename_bar.isVisibleTo(window) and not window._bar.isVisibleTo(window)
    window.name_input.setText("gold 200")  # another profile's name, by case
    assert not window.rename_confirm.isEnabled() and "already exists" in window.name_error.text()
    window.name_input.setText("Portra 400 · home")
    assert window.rename_confirm.isEnabled()
    window.rename_confirm.click()
    assert (profiles / "Portra 400 · home.json").exists() and not (profiles / "Portra 400.json").exists()
    assert window.selected_name() == "Portra 400 · home" and window._bar.isVisibleTo(window)
    window.close()


def test_delete_asks_then_removes_the_file(profiles, app, monkeypatch):
    window = ProfilesWindow()
    window.table.selectRow(_column(window, 0).index("Ektar"))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    window.delete_button.click()
    assert (profiles / "Ektar.json").exists()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.delete_button.click()
    assert not (profiles / "Ektar.json").exists() and "Ektar" not in _column(window, 0)
    window.close()


def test_the_profiles_button_opens_the_window_and_edit_points_reopens(profiles, app, tmp_path, monkeypatch):
    from halide.gui.main_window import MainWindow

    window = MainWindow()
    opened = []
    monkeypatch.setattr(window, "open_profile", opened.append)
    profiles_window = window.show_profiles()
    profiles_window.edit_button.click()  # the newest is selected
    assert opened == [profiles / "Portra 400.json"]
    assert not profiles_window.use_button.isEnabled()  # no roll open
    window.close()
    assert not profiles_window.isVisible()


def test_a_calibration_file_from_elsewhere_changes_nothing_in_the_profiles_folder_until_saved(profiles, app, tmp_path, monkeypatch):
    from halide.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_warn_missing_frames", lambda self: None)  # its frames aren't on disk
    monkeypatch.setattr(MainWindow, "_confirm_close", lambda self, count: True)
    monkeypatch.setattr(MainWindow, "_roll_not_found", lambda self, name, folder: None)  # its roll isn't on disk

    elsewhere = tmp_path / "Roll16-developed" / "halide-calibration.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text((profiles / "Portra 400.json").read_text())
    before = sorted(p.name for p in profiles.iterdir())
    window = MainWindow()
    window.open_profile(elsewhere)
    assert "Calibration file" in window.profile_chip.text() and "halide-calibration.json" in window.profile_chip.toolTip()
    assert window._profile_name == ""  # Save asks for a name rather than offering "halide-calibration"
    assert sorted(p.name for p in profiles.iterdir()) == before
    window._closing = True
    window.close()
