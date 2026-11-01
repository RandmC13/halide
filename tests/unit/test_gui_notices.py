"""The line under the picture: chips for what's true of the roll, a list each opens over the picture,
activity text that fades, and errors that wait to be read."""

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from halide.gui import notices
from halide.gui.notices import Notice, NoticeBar


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _bar():
    host = QWidget()
    host.resize(700, 500)
    bar = NoticeBar()
    bar.set_host(host)
    return host, bar


def test_a_chip_opens_its_list_and_a_row_goes_to_its_frame(app):
    host, bar = _bar()
    bar.set_notice("scan", Notice(chip="⚠ 2 scanned differently", title="2 frames were scanned differently",
                                  rows=[("FRAME 03", "digitized at 1/60 s", 2), ("FRAME 09", "tone modules", 8)]))
    went = []
    bar.frameRequested.connect(went.append)
    chip = bar._chips["scan"]
    chip.click()
    assert bar.list_open() == "scan" and bar._panel.isVisibleTo(host)
    texts = [label.text() for label in bar._panel.findChildren(QLabel)]
    assert "FRAME 09" in texts and any(t.startswith("tone") for t in texts)
    rows = [b for b in bar._panel.findChildren(QPushButton) if b.toolTip() == "Go to this frame"]
    rows[1].click()
    assert went == [8] and bar.list_open() is None  # going there closes the list
    chip.click()  # opens again after going to a frame
    assert bar.list_open() == "scan" and bar._panel.isVisibleTo(host)
    assert "FRAME 09" in [label.text() for label in bar._panel.findChildren(QLabel)]
    chip.click()  # and a second click closes it
    assert bar.list_open() is None and not bar._panel.isVisibleTo(host)


def test_a_long_list_shows_eight_rows_then_how_many_more(app):
    host, bar = _bar()
    bar.set_notice("scan", Notice(chip="⚠ 17", title="17", rows=[(f"FRAME {i}", "why", i) for i in range(17)]))
    bar.toggle("scan")
    rows = [b for b in bar._panel.findChildren(QPushButton) if b.toolTip() == "Go to this frame"]
    assert len(rows) == 8
    assert any(label.text().startswith("…and 9 more") for label in bar._panel.findChildren(QLabel))


def test_activity_fades_but_a_loading_line_lasts(app, monkeypatch):
    host, bar = _bar()
    bar.show_activity("Restored 3 points")
    assert bar.activity.full_text() == "Restored 3 points" and bar._fade.isActive()
    bar._fade.timeout.emit()
    assert bar.activity.full_text() == ""
    bar.show_activity("Loading IMG_0152.tif…", lasting=True)
    assert not bar._fade.isActive()


def test_an_error_stays_until_it_has_been_read(app):
    host, bar = _bar()
    bar.set_notice("error", Notice(chip="✕ Couldn't open x.json", title="Couldn't open x.json: not JSON", error=True))
    bar.show_activity("Loading…")  # other messages don't push it out
    assert "error" in bar.notices()
    bar.toggle("error")
    bar.close_list()
    assert "error" not in bar.notices()


def test_skipped_files_and_the_scan_check_are_chips_in_the_window(tmp_path, monkeypatch, app):
    from halide.gui.main_window import MainWindow
    from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(2):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    (roll / "notes.txt").write_text("not a scan")
    window = MainWindow()
    window.show()
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    skipped = window.notice_bar.notices()["skipped"]
    assert skipped.chip == "1 skipped" and "not being scans" in skipped.title
    assert window.view_mark.text() == "NEGATIVE" and "real whites look DARK here" in window.caption.text()
    window.positive_button.setChecked(True)
    assert window.view_mark.text() == "POSITIVE"
    window._closing = True
    window.close()
