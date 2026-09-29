"""Colours, symbols, rules and the plain-language wording helpers every CLI output shares."""

from __future__ import annotations

import os
import re
import shutil
import sys


def use_color() -> bool:
    """Whether output may carry colour codes. FORCE_COLOR (set, not "0") wins; otherwise no colour
    when NO_COLOR is set (any value, https://no-color.org), when TERM=dumb, or when stdout isn't a
    terminal — so a piped or logged run is plain text."""
    force = os.environ.get("FORCE_COLOR")
    if force and force != "0":
        return True
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def interactive_output() -> bool:
    """True when stdout is a terminal that can redraw in place (not piped, not TERM=dumb)."""
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


_COLOR_CODES = {
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "ORANGE": "\033[38;5;208m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
}


class _StyleMeta(type):
    """Colour and weight codes are looked up when used, not when this module loads, so they follow
    use_color() at that moment (NO_COLOR, a pipe). Cursor movement and line clearing are not colour
    and stay real: a terminal with NO_COLOR still redraws in place."""

    def __getattr__(cls, name: str) -> str:
        code = _COLOR_CODES.get(name)
        if code is None:
            raise AttributeError(name)
        return code if use_color() else ""


class Style(metaclass=_StyleMeta):
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
_SOURCE_COLOR_NAME = {"manual": "DIM", "auto": "YELLOW", "anchor": "CYAN", "colorchecker": "GREEN"}


def source_color(source: str) -> str:
    return getattr(Style, _SOURCE_COLOR_NAME.get(source, "DIM"))


def plural(n: int, noun: str) -> str:
    """"1 frame", "2 frames" — one place for every count with a noun."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


# What each command works on, for its end line ("Developed 36 of 37 frames", "Exported 12 files").
NOUN = {"invert": "frame", "batch": "frame", "export": "file", "print": "frame", "proof": "frame"}


def finish_message(verb: str, ok: int, total: int, failed: int, elapsed: float, noun: str | None = None) -> str:
    """The one end line every multi-frame command prints: "Developed 37 frames in 42s", or with
    failures "Developed 36 of 37 frames (1 failed) in 42s". `ok` is the count that really finished
    without an error. A contact sheet says "Contact sheet complete" only when every frame made it."""
    noun = noun or NOUN.get(verb, "frame")
    past = VERB_PAST.get(verb, "Processed")
    took = f"in {human_time(elapsed)}"
    if failed:
        return f"{past} {ok} of {plural(total, noun)} ({failed} failed) {took}"
    if verb == "proof":
        return f"Contact sheet complete — {plural(total, noun)} {took}"
    return f"{past} {plural(total, noun)} {took}"


def cancelled_message(verb: str, ok: int, total: int, noun: str | None = None) -> str:
    """The one Ctrl-C message: everything finished is on disk whole, nothing else was started."""
    noun = noun or NOUN.get(verb, "frame")
    past = VERB_PAST.get(verb, "Processed").lower()
    return f"Cancelled - {ok} of {plural(total, noun)} {past}; nothing half-written"


def success(message: str) -> str:
    return f"{Style.GREEN}{ICON_OK}{Style.RESET} {message}"


def error(message: str) -> str:
    return f"{Style.RED}{ICON_FAIL} Error:{Style.RESET} {message}"


def warning(message: str) -> str:
    return f"{Style.YELLOW}{ICON_WARN} Warning:{Style.RESET} {message}"


def cancelled(message: str = "Cancelled - nothing half-written") -> str:
    """The Ctrl-C line, worded the same everywhere (see cancelled_message)."""
    return f"{Style.RED}{ICON_FAIL}{Style.RESET} {message}"


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
