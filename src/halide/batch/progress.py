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
from halide.cli import console
from halide.cli.console import (
    ICON_FAIL,
    ICON_OK,
    SPROCKET,
    VERB,
    VERB_PAST,
    Style,
    dim,
    human_time,
    interactive_output,
    plural,
    use_color,
)

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
    "error": "RED",
    "cancelled": "DIM",
}

# Without colour (NO_COLOR, TERM=dumb) the states have to tell apart by shape instead: the sheet
# would otherwise be one uniform block.
_PLAIN_GLYPH = {
    "pending": "···",
    "processing": "▒▒▒",
    "finishing": "▓▓▓",
    "done": _FRAME,
    "error": "✗✗✗",
    "cancelled": _HOLLOW,
}


def _state_color(state: str) -> str:
    """The colour code for a frame state — "" when colour is off (see console.use_color())."""
    if not use_color():
        return ""
    code = _STATE_COLOR[state]
    return getattr(Style, code) if code in ("RED", "DIM") else code


_TERMINAL_STATES = ("done", "error", "cancelled")


class _Renderer:
    """What every progress display shares: the tallies and the end line. `completed` counts frames
    that came back, failed ones included; `succeeded` is only those that developed."""

    def __init__(self, total: int, verb: str = "invert"):
        self.total = total
        self.verb = verb
        self.completed = 0
        self.failures: list[tuple[str, str]] = []  # (filename, error message)
        self._start_time: float | None = None
        self._note: str | None = None  # one line under the status line (note())

    @property
    def succeeded(self) -> int:
        return self.completed - len(self.failures)

    def _record_failure(self, result: BatchResult) -> None:
        self.failures.append((result.job.input_path.name, result.error))

    def _announce(self) -> None:
        print(f"{Style.BOLD}{VERB.get(self.verb, 'Processing')} {plural(self.total, 'frame')}{Style.RESET}", flush=True)

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._announce()

    def mark_processing(self, index: int) -> None:
        pass

    def note(self, message: str) -> None:
        print(dim(message), flush=True)

    def cancel(self, indices: list[int]) -> None:
        pass

    def _finish_lines(self, cancelled: bool, noun: str | None, list_failures: bool = True) -> None:
        elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        if cancelled:
            print(console.cancelled(console.cancelled_message(self.verb, self.succeeded, self.total, noun)))
        else:
            message = console.finish_message(self.verb, self.succeeded, self.total, len(self.failures), elapsed, noun)
            if self.failures:
                print(f"{Style.YELLOW}{Style.BOLD}{message}{Style.RESET}")
            else:
                print(f"{Style.GREEN}{Style.BOLD}{ICON_OK} {message}{Style.RESET}")
        if list_failures:
            for name, err in self.failures:
                print(f"  {Style.YELLOW}{ICON_FAIL}{Style.RESET} {name}: {err}")
        sys.stdout.flush()

    def finish(self, cancelled: bool = False, noun: str | None = None) -> None:
        self._finish_lines(cancelled, noun)


class GridProgressRenderer(_Renderer):
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
        super().__init__(total, verb)
        self._states = ["pending"] * total
        self._last_frame_lines = 0
        self._tick = 0
        self._note: str | None = None  # one line under the status line (note())

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
        if not use_color():
            return _PLAIN_GLYPH[state]
        if state == "processing":
            return f"{_PROCESSING_COLORS[self._tick % 2]}{_FRAME}{Style.RESET}"
        glyph = _HOLLOW if state == "cancelled" else _FRAME
        return f"{_state_color(state)}{glyph}{Style.RESET}"

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
        text = f"{verb_past} {self.succeeded}/{self.total}"
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
        if self._note:
            room = self._columns - 1 - len(self._INDENT)  # trimmed, never wrapped: see _fit_status
            note = self._note if len(self._note) <= room else self._note[: max(0, room - 1)] + "…"
            lines.append(f"{self._INDENT}{dim(note)}")
        return "\n".join(lines) + "\n"

    def _draw(self, frame: str) -> None:
        sys.stdout.write(Style.cursor_up(self._last_frame_lines))
        for line in frame.splitlines():
            sys.stdout.write(f"{Style.CLEAR_LINE}{line}\n")
        self._last_frame_lines = frame.count("\n")
        sys.stdout.flush()

    def start(self) -> None:
        super().start()
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
            self._record_failure(result)
        else:
            self._states[index] = "finishing"
            self._redraw()
            time.sleep(0.08)
            self._states[index] = "done"
        self.completed += 1
        self._redraw()

    def note(self, message: str) -> None:
        """Show `message` on its own line under the status line from now on — part of the frame, so
        the in-place redraws keep counting their lines right (a plain print() in between would
        strand a stale row)."""
        self._note = message
        self._redraw()

    def cancel(self, indices: list[int]) -> None:
        """Mark jobs that never got a result (not started, or in flight when Ctrl+C landed) as
        cancelled and redraw one final time — called instead of report() for those indices. The
        "finishing the frames in progress" note is over by now, so the last frame drops it."""
        for i in indices:
            self._states[i] = "cancelled"
        self._note = None
        self._redraw()

    def finish(self, cancelled: bool = False, noun: str | None = None) -> None:
        print()
        self._finish_lines(cancelled, noun)


class PlainProgressRenderer(_Renderer):
    """The same run for a log: no cursor movement, no colour, one line per finished frame —

        ✓ IMG_0138.tif  grade 0.88 exp +0.39  4.1s
        ✗ IMG_0139.tif: the scan is gamma-encoded ...

    then the end line. Used whenever stdout isn't a terminal that can redraw (a pipe, a file,
    TERM=dumb). Lines are flushed as they are written, so a tail -f follows the run."""

    def report(self, index: int, result: BatchResult) -> None:  # noqa: ARG002 -- same interface as the grid
        self.completed += 1
        name = result.job.input_path.name
        if result.error:
            self._record_failure(result)
            print(f"{ICON_FAIL} {name}: {result.error}", flush=True)
            return
        parts = [f"{ICON_OK} {name}"]
        if result.detail:
            parts.append(result.detail)
        if result.seconds is not None:
            parts.append(human_time(result.seconds))
        print("  ".join(parts), flush=True)

    def finish(self, cancelled: bool = False, noun: str | None = None) -> None:
        self._finish_lines(cancelled, noun, list_failures=False)


class QuietProgressRenderer(_Renderer):
    """--quiet: no progress at all, only the end line (and the failures, which are errors)."""

    def _announce(self) -> None:
        pass

    def report(self, index: int, result: BatchResult) -> None:  # noqa: ARG002
        self.completed += 1
        if result.error:
            self._record_failure(result)

    def finish(self, cancelled: bool = False, noun: str | None = None) -> None:
        self._finish_lines(cancelled, noun)


def make_renderer(total: int, verb: str, quiet: bool = False) -> _Renderer:
    """The progress display for a run: the contact-sheet grid in a terminal, plain lines when
    stdout is a file or pipe (or TERM=dumb), only the end line under --quiet. Every kind has the
    same interface, so a command never asks which one it got."""
    if quiet:
        return QuietProgressRenderer(total, verb)
    if interactive_output():
        return GridProgressRenderer(total, verb)
    return PlainProgressRenderer(total, verb)


def cancel_notice(renderer: GridProgressRenderer | None):
    """The `on_cancel` callback for the batch runners (orchestrator._run_pool): shows why a
    cancelled batch is still running — the frames in progress finishing — in a terminal only, on
    the progress sheet if there is one, else (--quiet) as one plain line."""
    def show(message: str) -> None:
        if not sys.stdout.isatty():
            return
        if renderer is not None:
            renderer.note(message)
        else:
            print(dim(message), flush=True)

    return show
