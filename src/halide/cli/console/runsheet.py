"""RunSheet: the aligned settings block that batch/export/print show before developing."""

from __future__ import annotations

import contextlib
import shutil
import textwrap
from halide.cli.console.style import ICON_OK, ICON_WARN, Style, framed, full_width_rule, warning
from halide.cli.console.animation import MINI_SPINNERS, animation


class RunSheet:
    """The settings/checks a multi-frame run prints before its progress display, as one aligned
    block framed by sprocket rules — like the job ticket that travels with a roll through a lab —
    instead of a run of unrelated sentences:

        ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫
          Roll           Roll16 · 37 frames → out/
          Scans          ⚠ exported with 14 different raw white balances (R 5%, B 7% apart)
                           — shifts colour frame to frame; re-export with one fixed white balance
                         details: halide check Roll16
          Workers        8 · auto-selected from free memory and CPU cores
        ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫

    Use it as a context manager: rows are held until it closes (even on an error, so the error
    prints below a finished sheet) and then framed by the house rule-sizing convention (see
    framed()) — so an interactive prompt while the run is being set up comes before the sheet, not
    in the middle of it. A slow step shows a spinner row via `working()`, which prints what's held
    so far first, so the spinner appears in place. Consecutive rows with the same label stack
    under one label. Values wrap with a hanging indent, so a long warning never runs back under
    the labels. With `quiet`, nothing is printed except warnings, as plain `warning()` lines."""

    # Joins the parts of one row's value. The space before the dot is non-breaking (textwrap only
    # breaks on ASCII whitespace), so a wrapped row never starts with a stray "·".
    SEP = " · "

    LABEL_WIDTH = 15  # fits the longest label in use ("Scan exposure") plus a gap
    _INDENT = "  "
    _MAX_TEXT_WIDTH = 100  # long prose past this is hard to read, whatever the terminal width

    def __init__(self, *, quiet: bool = False):
        self.quiet = quiet
        self._held: list[str] = []
        self._streaming = False  # True once the top rule is out and rows print as they come
        self._last_label: str | None = None

    def _lines(self, label: str, text: str, icon: str = "", icon_style: str = "", text_style: str = "") -> list[str]:
        width = min(shutil.get_terminal_size((80, 24)).columns, self._MAX_TEXT_WIDTH)
        prefix_width = len(self._INDENT) + self.LABEL_WIDTH + (2 if icon else 0)
        chunks = textwrap.wrap(text, max(20, width - 1 - prefix_width), break_on_hyphens=False) or [""]
        shown_label = "" if label == self._last_label else label
        lines = []
        for i, chunk in enumerate(chunks):
            head = f"{Style.DIM}{shown_label:<{self.LABEL_WIDTH}}{Style.RESET}" if i == 0 else " " * self.LABEL_WIDTH
            mark = (f"{icon_style}{icon}{Style.RESET} " if i == 0 else "  ") if icon else ""
            body = f"{text_style}{chunk}{Style.RESET}" if text_style else chunk
            lines.append(f"{self._INDENT}{head}{mark}{body}")
        return lines

    def _add(self, label: str, text: str, **style: str) -> None:
        lines = self._lines(label, text, **style)
        if self._streaming:
            print("\n".join(lines))
        else:
            self._held.extend(lines)
        self._last_label = label

    def _start_streaming(self) -> None:
        if not self._streaming:
            print("\n".join([full_width_rule(), *self._held]))
            self._held = []
            self._streaming = True

    def row(self, label: str, text: str) -> None:
        if not self.quiet:
            self._add(label, text)

    def note(self, label: str, text: str) -> None:
        """A dim, secondary row — a pointer to more detail, not a setting in its own right."""
        if not self.quiet:
            self._add(label, text, text_style=Style.DIM)

    def ok(self, label: str, text: str) -> None:
        if not self.quiet:
            self._add(label, text, icon=ICON_OK, icon_style=Style.GREEN)

    def warn(self, label: str, text: str) -> None:
        if self.quiet:
            print(warning(text))
        else:
            self._add(label, text, icon=ICON_WARN, icon_style=Style.YELLOW)

    def working(self, label: str, text: str):
        """Context manager: a spinner row while a slow step runs, cleared when it ends — follow it
        with the step's real result row."""
        if self.quiet:
            return contextlib.nullcontext()
        self._start_streaming()
        glyphs = MINI_SPINNERS["aperture_iris"]
        # One line only: animation redraws in place by line count.
        frames = [self._lines(label, text, icon=g, icon_style=Style.CYAN)[0] for g in glyphs]
        return animation(frames, interval=0.12)

    def close(self) -> None:
        if self._streaming:
            print(full_width_rule())
        elif self._held:
            print(framed(self._held))
        self._held, self._streaming, self._last_label = [], False, None

    def __enter__(self) -> "RunSheet":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
