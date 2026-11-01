"""The in-place spinner, and the single developing frame the one-file commands show."""

from __future__ import annotations

import sys
import threading
from typing import Sequence
from halide.cli.console.style import PULSE_COLORS, Style, interactive_output, use_color


IRIS = ("◐", "◓", "◑", "◒")  # a lens iris opening and closing


class animation:
    """Context manager: cycles a list of pre-rendered one-line frames on a background thread,
    redrawing in place until the block exits, then clears what was drawn. Prints the label once
    (no animation) when stdout isn't a real terminal, so piped output stays plain text."""

    def __init__(self, frames: Sequence[str], label: str = "", interval: float = 0.12):
        self.frames = list(frames)
        self.label = label
        self.interval = interval
        self._active = interactive_output()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_lines = 0

    def _draw(self, frame: str) -> None:
        text = f"{frame} {self.label}" if self.label else frame
        lines = text.splitlines()
        sys.stdout.write(Style.cursor_up(self._last_lines))
        for line in lines:
            sys.stdout.write(f"{Style.CLEAR_LINE}{line}\n")
        self._last_lines = len(lines)
        sys.stdout.flush()

    def _clear_drawn(self) -> None:
        if self._last_lines:
            sys.stdout.write(Style.cursor_up(self._last_lines))
            for _ in range(self._last_lines):
                sys.stdout.write(f"{Style.CLEAR_LINE}\n")
            sys.stdout.write(Style.cursor_up(self._last_lines))
            sys.stdout.flush()
        self._last_lines = 0

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            self._draw(self.frames[i % len(self.frames)])
            i += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "animation":
        if self._active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        elif self.label:
            print(self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            self._clear_drawn()


def developing(label: str) -> animation:
    """One frame of the batch contact sheet, pulsing in its colours beside `label`, for the
    single-file commands. Without colour a pulse is invisible, so the iris turns instead."""
    if not use_color():
        return animation(IRIS, label=label, interval=0.12)
    frames = [f"│ {colour}███{Style.RESET} │" for colour in PULSE_COLORS]
    return animation(frames, label=label, interval=0.5)
