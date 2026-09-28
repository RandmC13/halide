"""Unit tests for halide.batch.progress's contact-sheet layout: how frames split into strips, which
layout gets picked for a given terminal size, and that the redrawn frame never outgrows the
terminal (the in-place redraw moves the cursor up by the frame's own line count, which silently
corrupts the display if that frame is taller than the screen). Rendering is checked on ANSI-stripped
text so the assertions are about visible layout, not color codes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from halide.batch import progress
from halide.batch.orchestrator import BatchJob, BatchResult
from halide.batch.progress import GridProgressRenderer
from halide.cli.console import SPROCKET

_ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(progress.time, "sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _colour(monkeypatch):
    # These tests read the coloured layout (pending frames as blocks); the no-colour look has its own.
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)


def _visible_lines(renderer: GridProgressRenderer) -> list[str]:
    return [_ANSI.sub("", line) for line in renderer._frame().splitlines()]


def _content_rows(lines: list[str]) -> list[str]:
    return [line for line in lines if line.strip().startswith("│")]


def _ok(i: int) -> BatchResult:
    return BatchResult(job=BatchJob(Path(f"f{i}.tif"), Path(f"o{i}.tif")), error=None)


def _complete(renderer: GridProgressRenderer, indices) -> None:
    for i in indices:
        renderer.mark_processing(i)
        renderer.report(i, _ok(i))


def test_frames_split_into_strips_of_six_with_short_leftover_strip():
    r = GridProgressRenderer(total=14, terminal_size=(120, 60))
    rows = _content_rows(_visible_lines(r))
    assert [row.count("███") for row in rows] == [6, 6, 2]


def test_narrow_terminal_shortens_strips():
    r = GridProgressRenderer(total=7, terminal_size=(30, 60))
    rows = _content_rows(_visible_lines(r))
    assert [row.count("███") for row in rows] == [3, 3, 1]


def test_sprocket_rows_match_adjacent_strip_width():
    r = GridProgressRenderer(total=8, terminal_size=(120, 60))
    lines = _visible_lines(r)
    for i, line in enumerate(lines):
        if line.strip().startswith("│"):
            assert len(lines[i - 1]) == len(line)
            assert len(lines[i + 1]) == len(line)
            assert lines[i - 1].strip().startswith(SPROCKET)
            assert lines[i + 1].strip().startswith(SPROCKET)


def test_full_layout_separates_strips_with_a_blank_line_when_it_fits():
    r = GridProgressRenderer(total=12, terminal_size=(80, 40))
    lines = _visible_lines(r)
    first_row = next(i for i, line in enumerate(lines) if line.strip().startswith("│"))
    # frames / sprocket / blank / sprocket / frames
    assert lines[first_row + 1].strip().startswith(SPROCKET)
    assert lines[first_row + 2].strip() == ""
    assert lines[first_row + 3].strip().startswith(SPROCKET)
    assert lines[first_row + 4].strip().startswith("│")


def test_compact_layout_shares_sprocket_rows_when_full_layout_is_too_tall():
    # 36 frames = 6 strips: the full layout needs 25 lines, more than a 24-row terminal can hold.
    r = GridProgressRenderer(total=36, terminal_size=(80, 24))
    lines = _visible_lines(r)
    rows = [i for i, line in enumerate(lines) if line.strip().startswith("│")]
    assert len(rows) == 6
    assert all(b - a == 2 for a, b in zip(rows, rows[1:]))  # exactly one shared sprocket row between


@pytest.mark.parametrize("total", [1, 5, 14, 36, 100, 400])
@pytest.mark.parametrize("size", [(80, 24), (80, 12), (40, 30), (200, 60)])
def test_frame_never_exceeds_terminal_height_and_never_changes_height(total, size, capsys):
    r = GridProgressRenderer(total=total, terminal_size=size)
    heights = {len(r._frame().splitlines())}
    r.start()
    _complete(r, range(total))
    heights.add(len(r._frame().splitlines()))
    assert len(heights) == 1
    # One line for the "Developing N frame(s)..." header above the redrawn frame, one for the cursor.
    assert heights.pop() <= size[1] - 2


def test_overflow_scrolls_finished_strips_off_the_top_but_keeps_counting_them(capsys):
    r = GridProgressRenderer(total=100, terminal_size=(80, 24))
    r.start()
    _complete(r, range(12))
    lines = _visible_lines(r)
    assert any("12 frames above" in line for line in lines)
    assert any("Developed 12/100" in line for line in lines)
    rows = _content_rows(lines)
    # The first visible strip is the next one to develop — nothing done in it yet.
    assert "███" in rows[0]


def test_overflow_shows_frames_below_before_scrolling_starts(capsys):
    r = GridProgressRenderer(total=100, terminal_size=(80, 24))
    lines = _visible_lines(r)
    assert not any("above" in line for line in lines)
    hidden_below = 100 - sum(row.count("███") for row in _content_rows(lines))
    assert any(f"{hidden_below} frames below" in line for line in lines)


def test_cancelled_frames_draw_hollow(capsys):
    r = GridProgressRenderer(total=3, terminal_size=(80, 40))
    r.start()
    _complete(r, [0])
    r.cancel([1, 2])
    rows = _content_rows(_visible_lines(r))
    assert rows[0].count("░░░") == 2


@pytest.mark.parametrize("columns", [20, 30, 40, 80])
def test_no_line_wider_than_terminal(columns, capsys, monkeypatch):
    # A line that wraps takes two physical rows while the redraw's cursor-up only counts one, which
    # leaves stale rows piling up above the sheet — the status line (with elapsed/remaining) is the
    # one that can actually get long.
    monkeypatch.setattr(progress.time, "monotonic", iter(range(0, 10_000, 7)).__next__)
    r = GridProgressRenderer(total=100, terminal_size=(columns, 40))
    r.start()
    _complete(r, range(40))
    r.report(40, BatchResult(job=BatchJob(Path("x.tif"), Path("y.tif")), error="boom"))
    assert all(len(line) < columns for line in _visible_lines(r))


def test_a_note_is_part_of_the_frame_and_never_wraps(capsys):
    """The "finishing the frames in progress" notice after Ctrl-C goes on the sheet itself, so the
    in-place redraw's line count stays right; trimmed like the status line, never wrapped."""
    r = GridProgressRenderer(total=3, terminal_size=(30, 40))
    r.start()
    before = len(_visible_lines(r))
    r.note("Finishing the frames in progress — Ctrl-C again to stop now")
    lines = _visible_lines(r)
    assert len(lines) == before + 1
    assert lines[-1].strip().startswith("Finishing the frames")
    assert all(len(line) < 30 for line in lines)
    assert r._last_frame_lines == len(lines)


class _Tty:
    def __init__(self, tty):
        self.tty, self.text = tty, ""

    def isatty(self):
        return self.tty

    def write(self, text):
        self.text += text

    def flush(self):
        pass


@pytest.mark.parametrize("tty", [True, False])
def test_cancel_notice_shows_only_in_a_terminal(monkeypatch, tty):
    out = _Tty(tty)
    monkeypatch.setattr(progress.sys, "stdout", out)
    progress.cancel_notice(None)("Finishing the frames in progress — Ctrl-C again to stop now")  # --quiet
    assert ("Finishing the frames in progress — Ctrl-C again to stop now" in out.text) is tty
    r = GridProgressRenderer(total=2, terminal_size=(80, 40))
    progress.cancel_notice(r)("note")
    assert (r._note == "note") is tty


def _fail(i: int) -> BatchResult:
    return BatchResult(job=BatchJob(Path(f"f{i}.tif"), Path(f"o{i}.tif")), error="boom")


def _finish_text(renderer, capsys, **kwargs) -> str:
    capsys.readouterr()
    renderer.finish(**kwargs)
    return _ANSI.sub("", capsys.readouterr().out).strip()


@pytest.mark.parametrize(
    "verb, ok_line, failed_line",
    [
        ("batch", "Developed 3 frames in", "Developed 2 of 3 frames (1 failed) in"),
        ("invert", "Developed 3 frames in", "Developed 2 of 3 frames (1 failed) in"),
        ("export", "Exported 3 files in", "Exported 2 of 3 files (1 failed) in"),
        ("print", "Printed 3 frames in", "Printed 2 of 3 frames (1 failed) in"),
        ("proof", "Contact sheet complete", "Proofed 2 of 3 frames (1 failed) in"),
    ],
)
def test_finish_message_per_command_and_with_failures(verb, ok_line, failed_line, capsys):
    for make in (
        lambda: GridProgressRenderer(total=3, verb=verb, terminal_size=(80, 40)),
        lambda: progress.PlainProgressRenderer(3, verb),
        lambda: progress.QuietProgressRenderer(3, verb),
    ):
        r = make()
        r.start()
        for i in range(3):
            r.report(i, _ok(i))
        assert ok_line in _finish_text(r, capsys)

        r = make()
        r.start()
        r.report(0, _ok(0))
        r.report(1, _ok(1))
        r.report(2, _fail(2))
        text = _finish_text(r, capsys)
        assert failed_line in text  # the success count excludes the failure (4-2)
        assert "Contact sheet complete" not in text


def test_the_status_line_counts_only_frames_that_developed():
    r = GridProgressRenderer(total=3, terminal_size=(80, 40))
    r.start()
    r.report(0, _ok(0))
    r.report(1, _fail(1))
    assert "Developed 1/3 — 1 failed" in "\n".join(_visible_lines(r))


def test_cancelled_says_how_many_developed_and_that_nothing_is_half_written(capsys):
    r = progress.PlainProgressRenderer(6, "batch")
    r.start()
    r.report(0, _ok(0))
    r.report(1, _ok(1))
    text = _finish_text(r, capsys, cancelled=True)
    assert text == "✗ Cancelled - 2 of 6 frames developed; nothing half-written"


class _Screen:
    """A tiny terminal: text, \\r, \\n, cursor-up and clear-to-end-of-line — enough to replay what
    the renderer really writes and look at what would be on screen."""

    def __init__(self):
        self.rows, self.row, self.col = [""], 0, 0

    def feed(self, text):
        for m in re.finditer(r"\033\[([0-9;]*)([A-Za-z])|(.)", text, re.S):
            code, final, ch = m.groups()
            if final == "A":
                self.row = max(0, self.row - int(code or 1))
            elif final == "K":
                self.rows[self.row] = self.rows[self.row][: self.col]
            elif final:
                pass  # colour
            elif ch == "\n":
                self.row += 1
                self.col = 0
                while len(self.rows) <= self.row:
                    self.rows.append("")
            elif ch == "\r":
                self.col = 0
            else:
                line = self.rows[self.row].ljust(self.col)
                self.rows[self.row] = line[: self.col] + ch + line[self.col + 1 :]
                self.col += 1

    def text(self):
        return "\n".join(self.rows).rstrip("\n")


def test_the_cancel_note_is_gone_from_the_final_screen_and_the_separator_stays(capsys):
    r = GridProgressRenderer(total=3, terminal_size=(80, 40))
    r.start()
    r.mark_processing(0)
    r.note("Finishing the frames in progress — Ctrl-C again to stop now")
    r.cancel([0, 1, 2])
    r.finish(cancelled=True)
    screen = _Screen()
    screen.feed(capsys.readouterr().out)
    text = screen.text()
    assert "Finishing the frames" not in text
    lines = text.split("\n")
    end = next(i for i, line in enumerate(lines) if "Cancelled" in line)
    assert lines[end - 1] == "" and lines[end - 2].strip().startswith("[ ")  # status, blank, then the line


def test_no_colour_grid_has_no_escape_colours_and_states_differ_by_shape(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    r = GridProgressRenderer(total=3, terminal_size=(80, 40))
    r.start()
    r.mark_processing(1)
    r.report(0, _ok(0))
    frame = r._frame()
    assert not re.search(r"\033\[[0-9;]*m", frame)  # only cursor/line control may remain
    row = _content_rows([_ANSI.sub("", line) for line in frame.splitlines()])[0]
    assert row.count("███") == 1 and "▒▒▒" in row and "···" in row


def test_plain_renderer_writes_one_line_per_frame_and_no_cursor_movement(capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR")  # stdout under capture isn't a terminal: no colour
    r = progress.PlainProgressRenderer(2, "batch")
    r.start()
    r.mark_processing(0)
    ok = BatchResult(job=BatchJob(Path("IMG_0138.tif"), Path("o.tif")), error=None, detail="grade 0.88 exp +0.39",
                     seconds=4.1)
    r.report(0, ok)
    r.report(1, _fail(1))
    r.finish()
    out = capsys.readouterr().out
    assert "✓ IMG_0138.tif  grade 0.88 exp +0.39  4.1s\n" in out
    assert "✗ f1.tif: boom\n" in out
    assert "Developed 1 of 2 frames (1 failed) in" in out
    assert "\033" not in out
    assert out.count("f1.tif") == 1  # the failure isn't repeated at the end


def test_make_renderer_picks_by_terminal(monkeypatch):
    monkeypatch.setattr(progress.sys, "stdout", _Tty(True))
    monkeypatch.setenv("TERM", "xterm")
    assert type(progress.make_renderer(2, "batch")) is GridProgressRenderer
    monkeypatch.setenv("TERM", "dumb")
    assert type(progress.make_renderer(2, "batch")) is progress.PlainProgressRenderer
    monkeypatch.setattr(progress.sys, "stdout", _Tty(False))
    monkeypatch.setenv("TERM", "xterm")
    assert type(progress.make_renderer(2, "batch")) is progress.PlainProgressRenderer
    assert type(progress.make_renderer(2, "batch", quiet=True)) is progress.QuietProgressRenderer


def test_quiet_renderer_prints_only_the_end_line_and_failures(capsys):
    r = progress.QuietProgressRenderer(2, "batch")
    r.start()
    r.report(0, _ok(0))
    r.report(1, _fail(1))
    assert capsys.readouterr().out == ""
    r.finish()
    out = _ANSI.sub("", capsys.readouterr().out).splitlines()
    assert out[0].startswith("Developed 1 of 2 frames (1 failed) in")
    assert out[1] == "  ✗ f1.tif: boom"
