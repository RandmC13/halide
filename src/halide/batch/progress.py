"""Terminal rendering for batch progress. This module only draws — it knows nothing about the
pipeline math or multiprocessing in halide.batch.orchestrator, and is driven purely by
BatchResult callbacks so it can be swapped out (or skipped, e.g. --quiet) without touching
processing logic at all. Color/style/wording primitives come from halide.cli.console, the shared
place that policy lives, so this stays purely "how to draw a grid," not a second copy of it.
"""

from __future__ import annotations

import shutil
import sys
import time

from halide.batch.orchestrator import BatchResult
from halide.cli.console import SPROCKET, VERB, VERB_PAST, Style, human_time

_BLOCK = "██"
_HOLLOW = "░░"

# Color states read as an actual developing process, brightest to darkest: pending frames look like
# undeveloped film (a light, uniform orange base), an active frame pulses between two orange shades
# so it visibly reads as "alive," and a finished frame darkens down to a near-black grey — a real
# color brightness fade, not a glyph-density trick. (A glyph-density fade — e.g. cycling ░▒▓ — was
# tried for the export animation's paper and turned out to read backwards on a dark terminal, sparse
# fill looking darker than dense fill; that trap doesn't apply here since this is real ANSI color
# brightness, which behaves the way you'd expect.)
_PENDING_COLOR = "\033[38;5;223m"  # light/pale orange — undeveloped film base
_PROCESSING_COLORS = ("\033[38;5;208m", "\033[38;5;214m")  # two shades to pulse between
_FINISHING_COLOR = "\033[38;5;130m"  # medium dark orange — mid-fade
_DONE_COLOR = "\033[38;5;236m"  # near-black grey — fully developed
_STATE_COLOR = {
    "pending": _PENDING_COLOR,
    "finishing": _FINISHING_COLOR,
    "done": _DONE_COLOR,
    "error": Style.RED,
    "cancelled": Style.DIM,
}

_TERMINAL_STATES = ("done", "error", "cancelled")


class GridProgressRenderer:
    """A single row of frames — a window onto a longer strip of film — framed top and bottom by a
    dense row of sprocket-hole ticks, redrawn in place as jobs start and complete. Real film frames
    are in sequence, so the window shows jobs in true left-to-right order (no shuffling): jobs
    `[window_start : window_start + window_width]`. Once every cell currently in the window is
    terminal (done/error/cancelled) and more jobs remain, the window advances to the next slice,
    via a short left-to-right reveal rather than an instant cut.

    `_states` still tracks every job by its real index regardless of what's currently visible, so
    `finish()`'s totals/elapsed/ETA are always accurate — windowing only changes what's drawn, not
    what's tracked.

    Sprocket rows are sized from the row content's own *visible* width (computed analytically from
    the window size and the fixed per-cell layout, not from `len()` of a string that may contain
    invisible ANSI color codes) — this is what keeps them pixel-aligned with the content row at any
    window size, unlike an earlier version that sized them from a separate `columns` count.

    Both the per-cell finishing fade and the window-to-window transition are synchronous (small
    `time.sleep()` calls on whatever thread is calling `report()`/`mark_processing()` — there's no
    background thread here, deliberately, matching this renderer's existing callback-driven design),
    so they add a small fixed delay — roughly ~100ms per completed job, ~300-400ms per window
    transition — to when the caller (halide.batch.orchestrator's job-dispatch loop) can submit the
    next job. For this tool's real workload (several seconds per full-resolution frame), that's
    proportionally negligible; it would matter for a batch of very fast/tiny jobs, which isn't this
    tool's use case.

    The window transition reveals the new window cell-by-cell (a wipe) rather than crossfading the
    row character-by-character: each cell's rendered text already contains ANSI color codes, and a
    literal character-level slide (like halide.cli.console.slide_strings, used safely elsewhere for
    plain uncolored text) would risk slicing through an escape sequence mid-code, corrupting the
    terminal's color state. A whole-cell reveal sidesteps that entirely and arguably reads better
    anyway — a real film frame doesn't partially exist mid-transition.

    Each redraw rebuilds the whole frame as one string first and derives how far to move the cursor
    up for the *next* redraw from that string's own line count, rather than maintaining a separate
    expected-line-count calculation that has to be kept in exact sync by hand — that drift was a
    real bug in an earlier version of this renderer.
    """

    _CELL_VISIBLE_WIDTH = 5  # " XX │" per cell, visible width only (excludes ANSI color codes)

    def __init__(self, total: int, verb: str = "invert"):
        self.total = total
        self.verb = verb
        self.completed = 0
        self.failures: list[tuple[str, str]] = []  # (filename, error message)
        self._states = ["pending"] * total
        self.window_start = 0
        self.window_width = self._compute_window_width(total)
        self._start_time: float | None = None
        self._last_frame_lines = 0
        self._tick = 0

    @classmethod
    def _compute_window_width(cls, remaining: int) -> int:
        # A window wide enough to feel like a strip, narrow enough to read at a glance, but never
        # wider than what the terminal can actually show pixel-aligned, and never wider than the
        # number of jobs actually left to display.
        columns, _ = shutil.get_terminal_size(fallback=(80, 24))
        cells_that_fit = max(1, (columns - 8) // cls._CELL_VISIBLE_WIDTH)
        return max(1, min(7, cells_that_fit, max(remaining, 1)))

    def _window_jobs(self) -> range:
        w = min(self.window_width, self.total - self.window_start)
        return range(self.window_start, self.window_start + w)

    def _cell_glyph(self, job_index: int) -> str:
        state = self._states[job_index]
        if state == "processing":
            return f"{_PROCESSING_COLORS[self._tick % 2]}{_BLOCK}{Style.RESET}"
        glyph = _HOLLOW if state == "cancelled" else _BLOCK
        return f"{_STATE_COLOR[state]}{glyph}{Style.RESET}"

    def _window_cells(self) -> list[str]:
        return [self._cell_glyph(j) for j in self._window_jobs()]

    @staticmethod
    def _content_row(cells: list[str]) -> str:
        return "│" + "│".join(f" {c} " for c in cells) + "│"

    def _sprocket_row(self, visible_width: int) -> str:
        pattern = (SPROCKET + " ") * (visible_width // 2 + 1)
        return f"{Style.DIM}{pattern[:visible_width]}{Style.RESET}"

    def _status_text(self) -> str:
        verb_past = VERB_PAST.get(self.verb, "Processed")
        text = f"{verb_past} {self.completed}/{self.total}"
        if self.failures:
            text += f" — {len(self.failures)} failed"
        if self.completed >= 2 and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = self.total - self.completed
            if rate > 0 and remaining > 0:
                text += f" · {human_time(elapsed)} elapsed · ~{human_time(remaining / rate)} remaining"
            else:
                text += f" · {human_time(elapsed)} elapsed"
        return text

    def _frame_from_cells(self, cells: list[str]) -> str:
        content = self._content_row(cells)
        visible_width = 1 + len(cells) * self._CELL_VISIBLE_WIDTH
        sprocket = self._sprocket_row(visible_width)
        lines = [f"    {sprocket}", f"    {content}", f"    {sprocket}", "", f"    [ {self._status_text()} ]"]
        return "\n".join(lines) + "\n"

    def _draw(self, frame: str) -> None:
        sys.stdout.write(Style.cursor_up(self._last_frame_lines))
        for line in frame.splitlines():
            sys.stdout.write(f"{Style.CLEAR_LINE}{line}\n")
        self._last_frame_lines = frame.count("\n")
        sys.stdout.flush()

    def start(self) -> None:
        self._start_time = time.monotonic()
        verb = VERB.get(self.verb, "Processing")
        print(f"{Style.BOLD}{verb} {self.total} frame(s)...{Style.RESET}")
        self._redraw()  # draw the initial all-pending window immediately, not on the first report()

    def _redraw(self) -> None:
        self._tick += 1
        self._draw(self._frame_from_cells(self._window_cells()))

    def mark_processing(self, index: int) -> None:
        self._states[index] = "processing"
        self._redraw()

    def _slide_to_new_window(self, old_cells: list[str], new_cells: list[str]) -> None:
        if len(old_cells) != len(new_cells):
            self._redraw()  # window width changed between windows (e.g. a resize) — just cut
            return
        for revealed in range(1, len(new_cells) + 1):
            self._draw(self._frame_from_cells(new_cells[:revealed] + old_cells[revealed:]))
            time.sleep(0.06)

    def report(self, index: int, result: BatchResult) -> None:
        if result.error:
            self._states[index] = "error"
            self.failures.append((result.job.input_path.name, result.error))
        else:
            self._states[index] = "finishing"
            self._redraw()
            time.sleep(0.08)
            self._states[index] = "done"
        self.completed += 1
        self._redraw()

        job_indices = list(self._window_jobs())
        if job_indices and all(self._states[j] in _TERMINAL_STATES for j in job_indices):
            next_start = job_indices[-1] + 1
            if next_start < self.total:
                old_cells = self._window_cells()
                self.window_start = next_start
                self.window_width = self._compute_window_width(self.total - next_start)
                new_cells = self._window_cells()
                self._slide_to_new_window(old_cells, new_cells)

    def cancel(self, indices: list[int]) -> None:
        """Mark jobs that never got a result (not started, or in flight when Ctrl+C landed) as
        cancelled and redraw one final time — called instead of report() for those indices. Indices
        outside the currently-visible window are also marked (harmless — they just aren't drawn),
        so callers don't need to know about windowing at all."""
        for i in indices:
            self._states[i] = "cancelled"
        self._redraw()

    def finish(self, cancelled: bool = False) -> None:
        verb_past = VERB_PAST.get(self.verb, "Processed").lower()
        elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        if cancelled:
            not_started = self.total - self.completed
            print(
                f"\n{Style.YELLOW}{Style.BOLD}Cancelled{Style.RESET} — {self.completed}/{self.total} "
                f"frames {verb_past}, {not_started} not started, {len(self.failures)} failed."
            )
        elif self.failures:
            print(f"\n{Style.YELLOW}{Style.BOLD}Completed with {len(self.failures)} failure(s){Style.RESET} "
                  f"— {self.completed}/{self.total} frames {verb_past} in {human_time(elapsed)}:")
            for name, error in self.failures:
                print(f"  {Style.YELLOW}✗{Style.RESET} {name}: {error}")
        else:
            print(
                f"\n{Style.GREEN}{Style.BOLD}Contact sheet complete{Style.RESET} — "
                f"{self.total}/{self.total} frames {verb_past} in {human_time(elapsed)}."
            )
