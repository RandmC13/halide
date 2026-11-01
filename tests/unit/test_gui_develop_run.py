"""Developing a roll from the window: the real `halide batch` (and `halide export`) as a child, its
progress followed, the summary after. The prints must be the terminal's, pixel for pixel."""

import json
import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from halide.calibration.profile_store import default_profiles_dir
from halide.cli.main import main
from halide.gui import main_window as mw
from halide.io.tiff import read_tiff, read_tiff_description
from tests.unit.test_gui_develop_panel import _develop_side, _roll
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump

_ONCE, _SAVE = 1, 0  # the calibration choice's buttons for unsaved points


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)
    win = mw.MainWindow()
    win._confirm_close = lambda count: True
    win._confirm_new_calibration = lambda: True
    win.show()
    win.load_roll(_roll(tmp_path))
    _pump(app, lambda: _frame_ready(win, 0))
    yield win
    win._runner.cancel()
    win._runner.wait()
    win.close()


def _develop(window, app, choice=_ONCE, **setup):
    panel = _develop_side(window)
    panel.choice.set_index(choice)
    for key, index in setup.items():
        getattr(panel, key).set_index(index)
    panel.develop_button.click()
    _pump(app, lambda: not window._runner.is_running() and window._stack.currentIndex() == 2, timeout=120)
    return panel


def _provenance(path):
    data = json.loads(read_tiff_description(path))
    return {k: v for k, v in data.items() if "time" not in k and "created" not in k and "path" not in k}


def test_use_once_develops_exactly_what_batch_does(window, app, tmp_path):
    panel = _develop(window, app)
    into = window._dev["settings"].into
    calibration = into / "halide-calibration.json"
    assert calibration.exists()
    assert not default_profiles_dir().exists() or not list(default_profiles_dir().iterdir())
    assert (into / "halide-log.txt").exists()

    terminal = tmp_path / "terminal"
    assert main(["batch", str(window.session.roll_folder), str(terminal), "--profile", str(calibration),
                 "--workers", "1"]) == 0
    prints = sorted(p.name for p in into.glob("*.tiff"))
    assert prints == sorted(p.name for p in terminal.glob("*.tiff")) and len(prints) == 2
    for name in prints:
        assert np.array_equal(read_tiff(into / name).image, read_tiff(terminal / name).image)
        assert _provenance(into / name) == _provenance(terminal / name)

    summary = window.summary_panel
    assert summary.end_line.text().startswith("Developed 2 frames")
    labels = [summary._rows.itemAtPosition(r, 0).widget().text() for r in range(summary._rows.rowCount())
              if summary._rows.itemAtPosition(r, 0)]
    assert "Prints" in labels and "Log" in labels
    assert all(window.filmstrip.strip.print_phase(i) == "printed" for i in range(2))
    assert not panel.cancel_button.isVisible() and window.load_button.isEnabled()


def test_save_as_a_profile_saves_before_developing(window, app):
    panel = _develop_side(window)
    panel.choice.set_index(_SAVE)
    panel.name_input.setText("Portra 400")
    panel.develop_button.click()
    saved = default_profiles_dir() / "Portra 400.json"
    assert saved.exists()  # before the child has even started
    assert window._dev["settings"].profile == saved
    _pump(app, lambda: not window._runner.is_running() and window._stack.currentIndex() == 2, timeout=120)
    assert not (window._dev["settings"].into / "halide-calibration.json").exists()


@pytest.mark.parametrize("answer", ["replace", "skip", None])
def test_prints_already_there_are_asked_about_once(window, app, answer, monkeypatch):
    _develop(window, app)
    into = window._dev["settings"].into
    first = {p: p.stat().st_mtime_ns for p in into.glob("*.tiff")}
    asked = []
    monkeypatch.setattr(window, "_ask_existing", lambda n, total, where: asked.append((n, total)) or answer)
    window._turn_to(1)
    window.develop_panel.develop_button.click()
    assert asked == [(2, 2)]
    if answer is None:
        assert not window._runner.is_running() and window._stack.currentIndex() == 1
        return
    if answer == "skip":  # every print is there: nothing runs, and the window says so
        assert not window._runner.is_running() and "Nothing to do" in window.notice_bar.activity.text()
        return
    assert window._dev["settings"].existing == answer
    _pump(app, lambda: not window._runner.is_running() and window._stack.currentIndex() == 2, timeout=120)
    changed = {p for p, t in first.items() if p.stat().st_mtime_ns != t}
    assert changed == set(first)


_SLOW = r'''
import signal, sys, time
stop = []
signal.signal(signal.SIGINT, lambda *_: stop.append(1))
print("start\t0\tframe_00.tiff", flush=True)
print("done\t0\tok\tgrade 0.9", flush=True)
print("start\t1\tframe_01.tiff", flush=True)
for _ in range(400):
    if stop:
        break
    time.sleep(0.05)
print("done\t1\tok\tgrade 0.9", flush=True)
print("summary\tPrints\tgrade 0.9", flush=True)
print("end\t%d\t2\t%d" % (2 - len(stop), len(stop)), flush=True)
sys.exit(130 if stop else 0)
'''


@pytest.fixture
def slow(tmp_path, monkeypatch):
    script = tmp_path / "slow.py"
    script.write_text(_SLOW)
    monkeypatch.setattr(mw, "batch_command", lambda settings: [sys.executable, str(script)])


def test_cancel_ends_as_the_cli_does_and_carry_on_skips_existing(window, app, slow):
    panel = _develop_side(window)
    panel.choice.set_index(_ONCE)
    panel.develop_button.click()
    _pump(app, lambda: window._dev["done"] == 1)
    assert panel.develop_button.text() == "Developing… 1 of 2" and panel.cancel_button.isVisible()
    assert not window.load_button.isEnabled() and not panel.output.isEnabled()
    assert window.filmstrip.strip.print_phase(1) in ("expose", "tray")
    panel.cancel_button.click()
    _pump(app, lambda: not window._runner.is_running() and window._stack.currentIndex() == 2)
    assert window.summary_panel.end_line.text() == "Cancelled — 1 of 2 frames developed"
    assert window.summary_panel.carry_on_button.isVisible()
    window.summary_panel.carry_on_button.click()
    assert window._dev["settings"].existing == "skip" and window._runner.is_running()


def test_closing_during_a_run_asks_and_stops_the_child(window, app, slow, monkeypatch):
    panel = _develop_side(window)
    panel.choice.set_index(_ONCE)
    panel.develop_button.click()
    _pump(app, lambda: window._dev["done"] == 1)
    monkeypatch.setattr(window, "_confirm_stop_developing", lambda: False)
    window.close()
    assert window.isVisible() and window._runner.is_running()
    monkeypatch.setattr(window, "_confirm_stop_developing", lambda: True)
    window.close()
    assert not window._runner.is_running() and window._dev["cancelled"]


def test_a_child_that_fails_to_start_shows_an_error_and_stays(window, app, tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "batch_command", lambda s: [sys.executable, "-c", "import sys; sys.exit('halide: bad flag')"])
    panel = _develop_side(window)
    panel.choice.set_index(_ONCE)
    panel.develop_button.click()
    _pump(app, lambda: not window._runner.is_running())
    assert window._stack.currentIndex() == 1 and panel.develop_button.isEnabled()
    assert "bad flag" in window.notice_bar.notices()["error"].title


@pytest.mark.parametrize("fmt,index", [("jpg", 1), ("png", 2)])
def test_sharing_copies(window, app, fmt, index):
    _develop(window, app, sharing=index)
    sharing = window._dev["settings"].sharing_dir
    assert sorted(p.suffix for p in sharing.iterdir()) == [f".{fmt}"] * 2
    assert window._dev["sharing"].startswith("2 files, ")


def test_the_summary_opens_the_folder_and_saves_more_for_sharing(window, app, monkeypatch):
    _develop(window, app)
    opened = []
    monkeypatch.setattr(mw.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    window.summary_panel.open_button.click()
    assert opened == [str(window._dev["settings"].into)]
    monkeypatch.setattr(mw.QMessageBox, "exec", lambda box: box.buttons()[1].click())  # PNG
    window.summary_panel.share_button.click()
    _pump(app, lambda: not window._runner.is_running() and window._dev["phase"] == "ended", timeout=120)
    assert len(list(window._dev["settings"].sharing_dir.glob("*.png"))) == 2


def test_a_developed_rolls_calibration_file_can_be_kept_as_a_profile(window, app, monkeypatch):
    _develop(window, app)
    calibration = window._dev["settings"].into / "halide-calibration.json"
    other = mw.MainWindow()
    other._confirm_close = lambda count: True
    other.open_profile(calibration)
    _pump(app, lambda: other.session.frames and other._frame_loader is None)
    assert "Profile file" in other.profile_chip.text() and "halide-calibration.json" in other.profile_chip.toolTip()

    def save(dialog):
        dialog._name_input.setText("Portra 400")
        dialog._on_save_clicked()

    monkeypatch.setattr(mw.SaveProfileDialog, "exec", save)
    other.save_button.click()
    kept = default_profiles_dir() / "Portra 400.json"
    assert json.loads(kept.read_text())["anchors"] == json.loads(calibration.read_text())["anchors"]
    other.close()


def test_the_summarys_contact_sheet_is_saved_beside_the_prints_and_shown(window, app, monkeypatch):
    _develop(window, app)
    opened = []
    monkeypatch.setattr(mw.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    window.summary_panel.contact_button.click()
    _pump(app, lambda: opened or "error" in window.notice_bar.notices(), timeout=120)
    into = window._dev["settings"].into
    assert opened == [str(into / "developed-contact-sheet.jpg")] and (into / "developed-contact-sheet.jpg").exists()


def test_nothing_replaces_the_roll_while_it_develops(window, app, slow):
    panel = _develop_side(window)
    panel.choice.set_index(_ONCE)
    panel.develop_button.click()
    _pump(app, lambda: window._dev["done"] == 1)
    frames = list(window.session.frames)
    window.load_roll(window.session.roll_folder)
    window.load_files([frames[0].path])
    assert window.session.frames == frames and window._stack.currentIndex() == 1
    assert "Developing" in window.notice_bar.activity.text()


def test_declining_either_question_on_close_keeps_the_run_going(window, app, slow, monkeypatch):
    panel = _develop_side(window)
    panel.choice.set_index(_ONCE)
    panel.develop_button.click()
    _pump(app, lambda: window._dev["done"] == 1)
    monkeypatch.setattr(window, "_confirm_stop_developing", lambda: True)
    monkeypatch.setattr(window, "_confirm_close", lambda count: False)  # keep the unsaved points
    window.close()
    assert window.isVisible() and window._runner.is_running()
