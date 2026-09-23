"""Shared CLI presentation layer: color/style constants, TTY-aware interactive helpers (spinner /
confirm / menu), formatting helpers, and the top-level exception boundary used by the installed
`halide` console script.

Deliberately separate from halide.batch.progress (which owns the batch-grid rendering itself) so
every command shares one consistent look and feel without either module pulling in the other's
concerns. This is the single place CLI-facing wording/color/interactivity policy lives, the same
role halide.processing plays for the actual processing pipeline (see its own module docstring).

The visual identity here is deliberately themed around the darkroom process this tool automates
(developing/printing a negative) rather than being generic tool chrome — see VERB/VERB_PAST and
SPROCKET below — but semantic clarity always wins over theme: success/failure stay ✓/✗, colors stay
in the same semantic slots (green=success, yellow=warning, red=error) everywhere they're used.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence


class Style:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    @staticmethod
    def cursor_up(n: int) -> str:
        return f"\033[{n}A" if n > 0 else ""


ICON_OK = "✓"  # ✓
ICON_FAIL = "✗"  # ✗
ICON_WARN = "⚠"  # ⚠

# A literal sprocket-hole tick, used as a lightweight section divider / filmstrip border — the one
# piece of pure decoration in this module, used sparingly (see rule()).
SPROCKET = "▫"  # ▫

# Real darkroom terms mapped onto what each command actually does, so headers/spinners/finish
# banners read consistently instead of each command inventing its own wording. `export` used to
# be "Printing" too, back when no separate print step existed; now that `halide print` is the real
# printing step (paper exposure + grade + paper curve), `export` is just a format conversion and
# says so, so the two can't be confused.
VERB = {"invert": "Developing", "export": "Exporting", "batch": "Developing", "print": "Printing", "proof": "Proofing"}
VERB_PAST = {"invert": "Developed", "export": "Exported", "batch": "Developed", "print": "Printed", "proof": "Proofed"}

# manual/auto/anchor/colorchecker — see core.types.CalibrationSource.
SOURCE_COLOR = {
    "manual": Style.DIM,
    "auto": Style.YELLOW,
    "anchor": Style.CYAN,
    "colorchecker": Style.GREEN,
}

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


def success(message: str) -> str:
    return f"{Style.GREEN}{ICON_OK}{Style.RESET} {message}"


def error(message: str) -> str:
    return f"{Style.RED}{ICON_FAIL} Error:{Style.RESET} {message}"


def warning(message: str) -> str:
    return f"{Style.YELLOW}{ICON_WARN} Warning:{Style.RESET} {message}"


def dim(message: str) -> str:
    return f"{Style.DIM}{message}{Style.RESET}"


def rule(columns: int = 20) -> str:
    """A dim row of sprocket-hole ticks, used as a section divider that doubles as a light
    filmstrip motif — e.g. framing `profile list` or a batch run's header/footer.

    Sizing convention (the user's, keep to it): framing one or two lines of output, the rule is as
    wide as the text so it neatly wraps it; framing longer output, it spans the whole terminal line
    — a fixed-width rule over long output looks like it stops abruptly. Use framed() /
    rule_fitting() / full_width_rule() rather than guessing a column count."""
    # Strip the trailing space *inside* the style codes: stripping the whole string never removed
    # it (the RESET code came after it), so every rule was one character wider than _rule_width().
    ticks = ((SPROCKET + " ") * max(1, columns)).rstrip()
    return f"{Style.DIM}{ticks}{Style.RESET}"


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def visible_width(text: str) -> int:
    return len(_ANSI_ESCAPE.sub("", text))


def rule_fitting(width: int) -> str:
    """A rule as wide as `width` visible characters (one over for an even width, since the ticks
    alternate with spaces) — for framing short output so the ticks neatly wrap the text."""
    return rule(max(1, (width + 2) // 2))


def full_width_rule() -> str:
    """A rule spanning one whole terminal line — for framing long output, where a short rule would
    stop abruptly partway across."""
    return rule(max(1, (shutil.get_terminal_size((80, 24)).columns + 1) // 2))


def framed(lines: list[str]) -> str:
    """Sprocket-rule framing, following the house convention: output of one or two lines gets rules
    sized to the text itself; anything longer gets rules spanning the whole terminal line."""
    top = rule_fitting(max(map(visible_width, lines), default=1)) if len(lines) <= 2 else full_width_rule()
    return "\n".join([top, *lines, top])


def _rule_width(columns: int = 20) -> int:
    """The visible width of `rule(columns)` — matches its construction exactly (columns pairs of
    "▫ ", minus the one trailing space `.rstrip()` removes), so anything centered against this can
    never drift out of sync with `rule()` itself the way hand-guessed padding already has once."""
    return columns * 2 - 1


def _banner(tagline: str) -> str:
    return "\n".join([rule(), f"{Style.BOLD}{tagline}{Style.RESET}", rule()])


def welcome_screen() -> str:
    """Shown when `halide` is run with no subcommand — replaces argparse's bare "the following
    arguments are required: command" error with something that actually orients a non-programmer
    photographer, instead of just failing."""
    return "\n".join(
        [
            _banner("h a l i d e".center(_rule_width())),
            "develop scanned color negative film".center(_rule_width()),  # not bold, outside _banner
            rule(),
            "",
            "  New here? Start with:",
            "    halide calibrate --save-profile-as NAME   solve a calibration once",
            "    halide invert negative.tif positive.tif   develop a single scan",
            "    halide check  in_dir/                     check a roll was scanned consistently",
            "    halide batch  in_dir/ out_dir/            develop a whole roll",
            "    halide batch  in_dir/ --contact-sheet s.jpg  preview settings as a contact sheet",
            "",
            "  Editing yourself? Develop flat, edit in darktable, then print:",
            "    halide invert negative.tif flat.tif --output flat",
            "    halide print  flat_edited.tif print.tif",
            "",
            "  Run `halide --help` for the full command list.",
        ]
    )


def help_banner() -> str:
    """A small themed banner printed before `halide --help`'s own (argparse-generated) output —
    deliberately printed directly rather than via a custom HelpFormatter: argparse's own
    add_subparsers() internally re-invokes the parser's formatter_class just to compute each
    subcommand's prog-prefix string (e.g. "halide invert"), completely unrelated to rendering
    user-facing help text — a HelpFormatter.format_help() override meant for the latter ends up
    hijacking the former too, corrupting every subcommand's usage line with this banner's own text.
    Printing it as a plain string before argparse ever runs sidesteps that internal reuse entirely."""
    plain_tagline = "halide · develop scanned color negative film"
    # The default rule (20 columns -> 39 visible chars) is narrower than this tagline (44 visible
    # chars) — centering against it would be a no-op (str.center() can't shrink below the string's
    # own length), so this banner uses a wider rule specifically sized to leave room either side.
    columns = 24
    width = _rule_width(columns)
    tagline = f"{Style.BOLD}halide{Style.RESET} · develop scanned color negative film"
    centered = tagline.center(width + len(Style.BOLD) + len(Style.RESET))
    assert len(plain_tagline) < width, "help_banner()'s rule must stay wider than its tagline"
    return f"{rule(columns)}\n{centered}\n{rule(columns)}\n"


def human_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


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
        self._active = sys.stdout.isatty()
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
        else:
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
_TANK_LID_UP = f"{Style.DIM} ▄▄▄▄▄ {Style.RESET}"
_TANK_LID_DOWN = f"{Style.DIM} ▀▀▀▀▀ {Style.RESET}"
_TANK_TOP = "┌─────┐"
_TANK_FILL = f"│{Style.ORANGE}▓▓▓▓▓{Style.RESET}│"
_TANK_BOTTOM = "└─────┘"
_TANK_BENCH = f"{Style.DIM}{'▔' * 9}{Style.RESET}"

TANK_FRAMES = [
    "\n".join([_TANK_LID_UP, _TANK_TOP, _TANK_FILL, _TANK_FILL, _TANK_BOTTOM, _TANK_BENCH]),
    "\n".join([_TANK_TOP, _TANK_FILL, _TANK_FILL, _TANK_BOTTOM, _TANK_LID_DOWN, _TANK_BENCH]),
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


def confirm(prompt: str, default: bool = False) -> bool:
    """TTY-aware yes/no prompt. Returns `default` immediately, with nothing printed, when stdin
    isn't a real terminal — a script/CI run must never block waiting on input that can't come."""
    if not sys.stdin.isatty():
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        reply = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not reply:
        return default
    return reply in ("y", "yes")


def confirm_overwrite(path: str | Path) -> bool:
    """True = proceed with the write. Non-interactively, always proceeds (today's silent-overwrite
    behavior, matching `batch`) rather than blocking a script on input it'll never get."""
    if not sys.stdin.isatty():
        return True
    return confirm(f"{path} already exists — overwrite?")


def menu(prompt: str, options: Sequence[tuple[str, str]]) -> str | None:
    """TTY-aware numbered menu. `options` is a list of (key, label) pairs; returns the chosen key,
    or None if not interactive, or the user gave no valid choice (blank input, EOF, out of range)."""
    if not sys.stdin.isatty():
        return None
    print(prompt)
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}. {label}")
    try:
        reply = input("> ").strip()
    except EOFError:
        return None
    if not reply.isdigit():
        return None
    index = int(reply) - 1
    if 0 <= index < len(options):
        return options[index][0]
    return None


def run_guarded(main_func: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    """The top-level exception boundary for the installed console script (and `python -m
    halide.cli.main`) — NOT used by main() itself. The integration test suite calls main() directly
    and relies on it raising SystemExit for domain errors rather than swallowing it into a return
    code (see tests/integration/test_*_cli.py's pytest.raises(SystemExit, ...) usage); only this
    wrapper, which nothing tests against directly, is meant to be user-facing.

    A simple `"--debug" in argv` scan (rather than parsing argv through argparse a second time)
    decides whether an unexpected error re-raises with its full traceback — deliberately independent
    of exactly where `--debug` appears, and of whether argparse itself would even accept it there.
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    debug = "--debug" in raw_args or os.environ.get("HALIDE_DEBUG") == "1"
    try:
        return main_func(argv)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print(warning("Cancelled."), file=sys.stderr)
        return 130
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            print(error(code), file=sys.stderr)
            return 1
        return code if isinstance(code, int) else 0
    except Exception as exc:  # noqa: BLE001 -- last-resort presentation layer, see docstring above
        if debug:
            raise
        print(error(f"halide hit an unexpected error: {exc}"), file=sys.stderr)
        print(dim("(re-run with --debug, or set HALIDE_DEBUG=1, for the full traceback)"), file=sys.stderr)
        return 1
