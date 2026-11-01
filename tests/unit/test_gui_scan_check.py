"""The window runs the roll's scan check as it loads: frames scanned differently carry a ⚠ on the
strip with the reason as its tooltip, and the status line says so once."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from halide.gui import main_window as mw
from halide.gui.main_window import MainWindow
from halide.io.scan_metadata import ScanSettings
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame

S30 = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
S60 = ScanSettings(exposure_time=1 / 60, f_number=8.0, iso=100.0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _open(tmp_path, monkeypatch, app, exposures):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(len(exposures)):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    # The EXIF itself is read and tested in io/scan_metadata; here only what the window does with it.
    monkeypatch.setattr(mw, "read_roll_scan_metadata", lambda paths: {str(p): (s, None) for p, s in zip(paths, exposures)})
    window = MainWindow()
    window.show()
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    return window


def test_frames_scanned_at_another_exposure_are_flagged_on_the_strip(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, [S30, S60, S30, S30])
    strip = window.filmstrip.strip
    assert list(strip._flags) == [1]
    below = strip.frame_rect(1).bottomLeft()
    below.setY(below.y() + 8)  # the lower rebate, where the ⚠ sits
    assert strip.tooltip_at(below) == f"Scanned differently: digitized at {S60.describe()}, most of the roll at {S30.describe()}"
    assert strip.tooltip_at(strip.frame_rect(0).center()) is None
    notice = window.notice_bar.notices()["scan"]
    assert notice.chip == "⚠ 1 scanned differently"
    assert notice.title == "1 frame was scanned differently from most of the roll"
    assert notice.rows == [("FRAME 02", f"digitized at {S60.describe()}, most of the roll at {S30.describe()}", 1)]
    window._closing = True
    window.close()


def test_a_consistent_roll_has_no_flags(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, [S30, S30])
    assert window.filmstrip.strip._flags == {}
    assert "scan" not in window.notice_bar.notices()
    window._closing = True
    window.close()


def test_flags_show_for_a_roll_opened_by_a_relative_path(tmp_path, monkeypatch, app):
    # `halide open Roll16` passes a relative path; the session holds absolute ones.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(3):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    exposures = [S30, S60, S30]
    monkeypatch.setattr(mw, "read_roll_scan_metadata", lambda paths: {str(p): (s, None) for p, s in zip(paths, exposures)})
    window = MainWindow()
    window.show()
    window.load_roll(Path("roll"))
    _pump(app, lambda: _frame_ready(window, 0))
    assert list(window.filmstrip.strip._flags) == [1]
    window._closing = True
    window.close()
