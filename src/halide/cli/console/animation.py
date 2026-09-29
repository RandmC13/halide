"""Spinners and the themed darkroom animations."""

from __future__ import annotations

import random
import shutil
import signal
import sys
import threading
from typing import Sequence
from halide.cli.console.style import Style, interactive_output


# A small arsenal of single-line, photography-themed spinner glyph cycles — used as the fallback
# when a terminal is too small for a full multi-line themed animation (see animation/
# themed_animation below), and available for any future spot that just needs a bit of life without
# a whole diorama. Picking one at random rather than always using the same glyph gives the CLI some
# run-to-run visual variety instead of leaning on one motif everywhere.
MINI_SPINNERS: dict[str, tuple[str, ...]] = {
    "aperture_iris": ("◐", "◓", "◑", "◒"),  # a lens iris opening/closing
    "shutter_blades": ("▖", "▘", "▝", "▗"),  # shutter blades snapping around a frame
    "reel_spin": ("◴", "◷", "◶", "◵"),  # a spinning film reel/take-up spool
    "light_meter": ("◢", "◣", "◤", "◥"),  # a swinging light-meter needle
    "developer_ripple": ("·", "∘", "○", "∘"),  # a ripple in developer liquid during agitation
}


def random_mini_spinner_frames() -> list[str]:
    glyphs = random.choice(list(MINI_SPINNERS.values()))
    return [f"{Style.CYAN}{g}{Style.RESET}" for g in glyphs]


def slide_strings(a: str, b: str, steps: int = 6) -> list[str]:
    """A sequence of `steps` equal-length strings sliding from `a` to `b` by literal substring
    slicing of `a + b`: frame 0 is exactly `a`, the last frame is exactly `b`, and the frames in
    between crossfade `a`'s tail into `b`'s head character-by-character — a real slide, not a fake
    instant cut. `a` and `b` must be the same length. Shared by the batch grid's window-to-window
    slide and the enlarger animation's paper swap, rather than reimplementing the same trick twice.
    """
    if len(a) != len(b):
        raise ValueError(f"slide_strings requires equal-length strings, got {len(a)} and {len(b)}")
    width = len(a)
    steps = max(2, steps)
    combined = a + b
    offsets = [round(i * width / (steps - 1)) for i in range(steps)]
    return [combined[o : o + width] for o in offsets]


def _terminal_fits(min_width: int, min_height: int) -> bool:
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns >= min_width and size.lines >= min_height


class animation:
    """Context manager: cycles a list of pre-rendered frame strings (single-line, like a plain
    spinner glyph, or multi-line ASCII art) on a background thread, redrawing in place until the
    block exits, then clears exactly what was drawn. Each redraw measures the frame it just wrote
    (line count) fresh rather than assuming a fixed height, so mixed-height frame sets are handled
    correctly and a redraw can never drift out of sync with reality — the same lesson the batch
    grid's own redraw fix applies (see halide.batch.progress).

    Prints the label once (no animation) when stdout isn't a real terminal, so redirected/piped
    output stays clean plain text instead of filling with control codes.

    Pass `min_size=(columns, lines)` to make this a *themed* animation that only plays `frames` when
    the terminal is at least that big — otherwise (or if the terminal is resized smaller than that
    mid-run, detected via SIGWINCH where available) it falls back to `fallback_frames`, or a random
    themed mini-spinner (see MINI_SPINNERS) if none is given. Plain single-line spinners (no
    `min_size`) skip all of this — there's nothing to fall back from.
    """

    def __init__(
        self,
        frames: Sequence[str],
        label: str = "",
        interval: float = 0.12,
        *,
        min_size: tuple[int, int] | None = None,
        fallback_frames: Sequence[str] | None = None,
    ):
        self.label = label
        self.interval = interval
        self._min_size = min_size
        self._fallback_frames = list(fallback_frames) if fallback_frames else None
        self._active = interactive_output()
        self._degraded = False
        if self._min_size is not None and not _terminal_fits(*self._min_size):
            self.frames = self._fallback_frames or random_mini_spinner_frames()
            self._degraded = True
        else:
            self.frames = list(frames)
        self._stop = threading.Event()
        self._resized = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_lines = 0
        self._prev_handler = None

    def _on_resize(self, signum, frame) -> None:  # noqa: ARG002 -- required signal handler signature
        self._resized.set()

    def _degrade(self) -> None:
        self._clear_drawn()
        self.frames = self._fallback_frames or random_mini_spinner_frames()
        self._degraded = True

    def _draw(self, frame: str) -> None:
        text = f"{frame}\n{self.label}" if "\n" in frame else f"{frame} {self.label}"
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
            if self._resized.is_set():
                self._resized.clear()
                if not self._degraded and self._min_size is not None and not _terminal_fits(*self._min_size):
                    self._degrade()
                    i = 0
            frame = self.frames[i % len(self.frames)]
            self._draw(frame)
            i += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "animation":
        if self._min_size is not None and hasattr(signal, "SIGWINCH"):
            self._prev_handler = signal.signal(signal.SIGWINCH, self._on_resize)
        if self._active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        elif self.label:
            print(self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._prev_handler is not None:
            signal.signal(signal.SIGWINCH, self._prev_handler)
        if self._active:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=0.5)
            self._clear_drawn()


class spinner(animation):
    """Backward-compatible single-line spinner (the aperture-iris glyph cycle) — a thin wrapper
    around `animation`. New code wanting a themed multi-line animation should use
    `themed_animation` directly instead."""

    def __init__(self, label: str):
        glyphs = MINI_SPINNERS["aperture_iris"]
        frames = [f"{Style.CYAN}{g}{Style.RESET}" for g in glyphs]
        super().__init__(frames, label=label, interval=0.12)


def themed_animation(
    frames: Sequence[str], label: str, *, min_width: int, min_height: int, interval: float = 0.5
) -> animation:
    """A multi-line themed animation that gracefully degrades to a mini-spinner when the terminal
    is smaller than `min_width`x`min_height`, either from the start or if resized smaller mid-run
    (see `animation`'s SIGWINCH handling)."""
    return animation(frames, label=label, interval=interval, min_size=(min_width, min_height))


# `invert` — developing-tank agitation: the single most recognizable manual action in analog film
# processing — flip the loaded tank upside down, hold briefly, flip it back, repeat, so developer
# reaches the film evenly. The tank body is drawn with box-drawing characters that are themselves
# vertically symmetric (┌─────┐ over │▓▓▓▓▓│ reads the same whichever way up), so the only thing
# that actually needs to move between frames is the lid — on top when upright, on the bottom when
# inverted — which is both the simplest possible way to draw this and an accurate one, not a
# simplification: the lid is genuinely the one part of a real tank whose position changes.
def tank_frames() -> list[str]:
    """The tank animation's frames, built when asked so they follow use_color()."""
    lid_up = f"{Style.DIM} ▄▄▄▄▄ {Style.RESET}"
    lid_down = f"{Style.DIM} ▀▀▀▀▀ {Style.RESET}"
    top = "┌─────┐"
    fill = f"│{Style.ORANGE}▓▓▓▓▓{Style.RESET}│"
    bottom = "└─────┘"
    bench = f"{Style.DIM}{'▔' * 9}{Style.RESET}"
    return [
        "\n".join([lid_up, top, fill, fill, bottom, bench]),
        "\n".join([top, fill, fill, bottom, lid_down, bench]),
    ]


TANK_MIN_SIZE = (16, 8)  # (columns, lines) — see themed_animation's fallback behavior below this


# `export` — an enlarger projecting a negative's image onto photographic paper, the real darkroom
# analog of what this command does (turn the processed working file into a viewable/deliverable
# image) — which is also why round 1 already named this command's verb "Printing," not "exporting."
# Uses plain ASCII `\ / X` for the light cone, not Unicode diagonal box-drawing characters — those
# were tried first and reported as visibly misaligned in a real terminal (many monospace fonts
# don't render them at true single-character width); plain ASCII is guaranteed single-width
# everywhere. Only the paper's fill content changes across frames, in two stages that loop as a
# whole rather than one shade cycling forever: it darkens as if developing (▓→▒→░, bright blank
# paper to a dark image — see the note below on why that's the *opposite* glyph order you'd
# naively expect), then — once fully dark — slides out for a fresh (bright) sheet via slide_strings
# (the same slide technique the batch grid's window transition uses) before the cycle repeats, so
# the loop reads as expose → develop → reload paper → expose the next one, not an oscillating color.
_ENLARGER_LAMP = ["   ▄▄▄▄▄", "   █ ● █", "   ▀▀▀▀▀"]
_ENLARGER_CONE = ["    \\ /", "     X", "    / \\"]
_ENLARGER_TRAY_TOP = " ┌───────┐"
_ENLARGER_TRAY_BOTTOM = " └───────┘"


def _enlarger_frame(paper: str) -> str:
    tray_mid = f" │ {paper} │"
    return "\n".join([*_ENLARGER_LAMP, *_ENLARGER_CONE, _ENLARGER_TRAY_TOP, tray_mid, _ENLARGER_TRAY_BOTTOM])


# Counter-intuitive but confirmed live: on a typical dark-background terminal, "░" (sparse fill)
# reads as visually DARKER than "▓" (dense fill) — a sparse character shows mostly background
# (dark) with a few light dots, while a dense one shows mostly foreground (light). So blank paper
# (should look bright) is "▓▓▓▓▓" and a developed image (should look dark) is "░░░░░", not the
# other way around — the glyph names' apparent light/dark implication is backwards for this purpose.
_ENLARGER_DEVELOP_STAGES = ["▓▓▓▓▓", "▒▒▒▒▒", "░░░░░"]
_ENLARGER_SWAP_STAGES = slide_strings("░░░░░", "▓▓▓▓▓", steps=6)[1:]  # drop the ░░░░░ duplicate

ENLARGER_FRAMES = [_enlarger_frame(paper) for paper in _ENLARGER_DEVELOP_STAGES + _ENLARGER_SWAP_STAGES]
ENLARGER_MIN_SIZE = (24, 12)  # (columns, lines) — see themed_animation's fallback behavior above
