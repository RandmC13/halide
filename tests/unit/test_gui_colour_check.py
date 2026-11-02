"""Once a roll's previews are in, the window names frames whose colour runs away from the rest of
the roll: a quiet chip whose list goes to each frame, and the direction word on the strip."""

import pytest
from PySide6.QtWidgets import QApplication

from halide.core.types import DensityProfile
from halide.gui import main_window as mw
from halide.gui import theme
from halide.gui.main_window import MainWindow
from halide.gui.notices import Notice, NoticeBar, _chip_style
from halide.io.scan_metadata import ScanSettings
from tests.unit.test_gui_picker_smoke import _frame_ready, _pump, _write_frame

S30 = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
NEUTRAL = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0), source="auto")


def _short_of_blue(cc):
    return DensityProfile(white_balance=(1.0, 1.0, 10 ** (-cc / 100)), density_scale=(1.0, 1.0, 1.0), source="auto")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _open(tmp_path, monkeypatch, app, count):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(count):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)
    monkeypatch.setattr(mw, "read_roll_scan_metadata", lambda paths: {str(p): (S30, None) for p in paths})
    window = MainWindow()
    window.show()
    window.load_roll(roll)
    _pump(app, lambda: _frame_ready(window, 0))
    return window


def _close(window):
    window._closing = True
    window.close()


def _vote(window, estimates):
    for frame, estimate in zip(window.session.frames, estimates):
        frame.estimate = estimate
    window._show_colour_check()


def test_no_chip_while_the_roll_loads(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, 6)
    window.notice_bar.set_notice("colour", Notice(chip="x", title="x"))
    window.load_roll(tmp_path / "roll")  # a roll starting over clears it until its previews are in
    assert "colour" not in window.notice_bar.notices()
    _close(window)


def test_a_frame_that_runs_warm_is_named_with_its_pack(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, 7)
    _vote(window, [NEUTRAL] * 6 + [_short_of_blue(20)])
    notice = window.notice_bar.notices()["colour"]
    assert notice.chip == "1 runs warm"
    assert notice.title == "1 frame runs warmer than the rest of the roll"
    assert notice.rows == [("FRAME 07", "runs warm · CC 20Y", 6)]
    assert notice.quiet
    assert notice.footer.startswith("They'll print that way, as the film recorded")
    assert window.filmstrip.strip._marks == {6: "warm"}
    _close(window)


def test_frames_that_differ_both_ways(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, 8)
    _vote(window, [NEUTRAL] * 6 + [_short_of_blue(15), _short_of_blue(-30)])
    notice = window.notice_bar.notices()["colour"]
    assert notice.chip == "2 differ in colour"
    assert notice.title == "2 frames differ in colour from the rest of the roll"
    assert [row[2] for row in notice.rows] == [7, 6]  # largest first
    assert window.filmstrip.strip._marks == {6: "warm", 7: "cool"}
    _close(window)


def test_nothing_is_named_on_a_short_roll(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, 5)
    _vote(window, [NEUTRAL] * 4 + [_short_of_blue(40)])
    assert "colour" not in window.notice_bar.notices()
    assert window.filmstrip.strip._marks == {}
    _close(window)


def test_a_row_goes_to_its_frame(tmp_path, monkeypatch, app):
    window = _open(tmp_path, monkeypatch, app, 7)
    _vote(window, [NEUTRAL] * 6 + [_short_of_blue(20)])
    seen = []
    window.notice_bar.frameRequested.connect(seen.append)
    window.notice_bar._on_frame(6)
    assert seen == [6]
    _close(window)


def test_a_quiet_chip_is_in_the_plain_text_colour(app):
    assert f"color: {theme.TEXT};" in _chip_style(Notice(chip="x", title="x", quiet=True), False)
    assert f"color: {theme.TEXT_WARNING};" in _chip_style(Notice(chip="x", title="x"), False)


def test_the_strip_paints_the_word_under_the_marked_frame_only(app):
    from halide.gui.filmstrip import REBATE, _Strip

    strip = _Strip()
    strip.set_frames([1, 2])
    blank = strip.grab().toImage()
    strip.set_marks({1: "magenta"})
    marked = strip.grab().toImage()

    def changed(index):
        rect = strip.frame_rect(index)
        y0 = rect.bottom() + 1
        return sum(
            blank.pixelColor(x, y) != marked.pixelColor(x, y)
            for x in range(rect.x() + 16, rect.right() - 16)
            for y in range(y0, y0 + REBATE)
        )

    assert changed(1) > 0
    assert changed(0) == 0
