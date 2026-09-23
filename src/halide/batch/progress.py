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

_FRAME = "███"  # three cells wide by one tall ≈ a 3:2 35mm frame at a terminal's ~1:2 glyph aspect
_HOLLOW = "░░░"

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
    """The whole batch drawn at once as a contact sheet: the roll cut into strips of (up to) six
    frames, each strip edged above and below by a row of sprocket holes, redrawn in place as jobs
    start and complete. Frames stay in true job order, left-to-right then top-to-bottom, exactly as a
    proofed roll reads. The last strip is only as long as the frames left over, like a real roll's
    short end strip.

    Nothing moves between strips — an earlier version showed one strip-sized window at a time and
    wiped across to the next slice when it finished, which read as busy rather than as film. Here the
    only motion is per-frame: an active frame pulses, a finished one briefly fades before settling
    to near-black. The sheet itself develops in place.

    The layout is chosen once, at construction, from the terminal size, and then held fixed so every
    redraw has the same height (the in-place redraw only clears the lines it writes, so a frame that
    shrank would leave stale lines behind, and one taller than the screen corrupts the cursor-up
    redraw entirely). In order of preference:
      - "full": each strip gets its own top and bottom sprocket rows, with a blank gap between
        strips — the actual contact-sheet look.
      - "compact": adjacent strips share one sprocket row and there are no gaps.
      - "scroll": compact, but only as many strips as fit are shown, starting from the first strip
        with anything still undeveloped — so finished strips drop off the top as work proceeds —
        with a dim count of the frames hidden above and below.
    Strips shorten below six frames only when the terminal is too narrow for six.

    `_states` tracks every job by its real index regardless of what's visible, so the status line
    and `finish()` totals are always accurate.

    Sprocket rows are sized from the strip's *visible* width (computed from the per-cell layout, not
    from `len()` of a string containing invisible ANSI color codes), which keeps them aligned with
    the frame row. Each redraw builds the whole frame as one string and derives the next cursor-up
    distance from that string's own line count — hand-maintained line counting drifted in an
    earlier version.

    The per-frame finishing fade is a small synchronous `time.sleep()` (~80ms per completed job) on
    whatever thread calls `report()` — no background thread, deliberately — which is negligible
    against several seconds per full-resolution frame.
    """

    _CELL_VISIBLE_WIDTH = 6  # " ███ │" per cell, visible width only (excludes ANSI color codes)
    _MAX_STRIP_LENGTH = 6  # real 35mm negatives are cut and sleeved in strips of six
    _INDENT = "    "

    def __init__(self, total: int, verb: str = "invert", terminal_size: tuple[int, int] | None = None):
        self.total = total
        self.verb = verb
        self.completed = 0
        self.failures: list[tuple[str, str]] = []  # (filename, error message)
        self._states = ["pending"] * total
        self._start_time: float | None = None
        self._last_frame_lines = 0
        self._tick = 0

        columns, rows = terminal_size or shutil.get_terminal_size(fallback=(80, 24))
        self._columns = columns
        self.strip_length = max(1, min(self._MAX_STRIP_LENGTH, (columns - 8) // self._CELL_VISIBLE_WIDTH))
        self._strips = [
            range(start, min(start + self.strip_length, total)) for start in range(0, total, self.strip_length)
        ] or [range(0)]
        self.layout, self.visible_strips = self._choose_layout(len(self._strips), max_lines=rows - 2)

    @staticmethod
    def _choose_layout(n_strips: int, max_lines: int) -> tuple[str, int]:
        # Line counts include the blank line + status line under the sheet. `max_lines` already
        # leaves room for the header line above the sheet and the cursor's own line below it.
        if 4 * n_strips + 1 <= max_lines:
            return "full", n_strips
        if 2 * n_strips + 3 <= max_lines:
            return "compact", n_strips
        return "scroll", max(1, min(n_strips, (max_lines - 5) // 2))

    def _first_visible_strip(self) -> int:
        if self.layout != "scroll":
            return 0
        first_undeveloped = next(
            (
                i
                for i, strip in enumerate(self._strips)
                if any(self._states[j] not in _TERMINAL_STATES for j in strip)
            ),
            len(self._strips),
        )
        return max(0, min(first_undeveloped, len(self._strips) - self.visible_strips))

    def _cell_glyph(self, job_index: int) -> str:
        state = self._states[job_index]
        if state == "processing":
            return f"{_PROCESSING_COLORS[self._tick % 2]}{_FRAME}{Style.RESET}"
        glyph = _HOLLOW if state == "cancelled" else _FRAME
        return f"{_STATE_COLOR[state]}{glyph}{Style.RESET}"

    def _content_row(self, strip: range) -> str:
        return "│" + "│".join(f" {self._cell_glyph(j)} " for j in strip) + "│"

    def _visible_width(self, strip: range) -> int:
        return 1 + len(strip) * self._CELL_VISIBLE_WIDTH

    @staticmethod
    def _sprocket_row(visible_width: int) -> str:
        pattern = (SPROCKET + " ") * (visible_width // 2 + 1)
        return f"{Style.DIM}{pattern[:visible_width]}{Style.RESET}"

    def _sheet_lines(self) -> list[str]:
        first = self._first_visible_strip()
        strips = self._strips[first : first + self.visible_strips]
        lines: list[str] = []
        if self.layout == "full":
            for i, strip in enumerate(strips):
                sprocket = self._sprocket_row(self._visible_width(strip))
                lines += ([""] if i else []) + [sprocket, self._content_row(strip), sprocket]
            return lines

        # Compact/scroll: one sprocket row between neighbours, as wide as the wider of the two (only
        # ever the short final strip differs).
        widths = [self._visible_width(strip) for strip in strips]
        lines.append(self._sprocket_row(widths[0]))
        for i, strip in enumerate(strips):
            lines.append(self._content_row(strip))
            below = max(widths[i], widths[i + 1]) if i + 1 < len(strips) else widths[i]
            lines.append(self._sprocket_row(below))
        if self.layout == "scroll":
            above = strips[0].start if strips else 0
            below = self.total - (strips[-1].stop if strips else 0)
            lines.insert(0, self._hidden_note(above, "above"))
            lines.append(self._hidden_note(below, "below"))
        return lines

    def _hidden_note(self, count: int, where: str) -> str:
        if not count:
            return ""
        text = f"{SPROCKET} {count} frames {where}"
        if len(self._INDENT) + len(text) >= self._columns:
            text = f"{SPROCKET} {count} {where}"
        return f"{Style.DIM}{text}{Style.RESET}"

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

    def _fit_status(self, text: str) -> str:
        # A wrapped line takes two terminal rows but counts as one for the next redraw's cursor-up,
        # which strands stale rows above the sheet — so the status line (the only one whose length
        # isn't bounded by the layout) is trimmed to fit rather than allowed to wrap.
        room = self._columns - 1 - len(self._INDENT) - len("[  ]")
        return text if len(text) <= room else text[: max(0, room - 1)] + "…"

    def _frame(self) -> str:
        lines = [f"{self._INDENT}{line}" if line else "" for line in self._sheet_lines()]
        lines += ["", f"{self._INDENT}[ {self._fit_status(self._status_text())} ]"]
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
        self._redraw()  # draw the initial all-pending sheet immediately, not on the first report()

    def _redraw(self) -> None:
        self._tick += 1
        self._draw(self._frame())

    def mark_processing(self, index: int) -> None:
        self._states[index] = "processing"
        self._redraw()

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


    def cancel(self, indices: list[int]) -> None:
        """Mark jobs that never got a result (not started, or in flight when Ctrl+C landed) as
        cancelled and redraw one final time — called instead of report() for those indices."""
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
