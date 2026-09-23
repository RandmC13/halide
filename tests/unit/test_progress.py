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
