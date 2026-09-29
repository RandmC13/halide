"""Interactive prompts: confirm, one-line input and numbered menus (all safe without a terminal)."""

from __future__ import annotations

import sys
from typing import Sequence


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


def prompt_line(prompt: str) -> str | None:
    """TTY-aware single-line free-text prompt. Returns the typed line (possibly empty, meaning
    "leave as-is" to a caller offering that convention), or None if stdin isn't a real terminal
    or input was cut off (EOF/Ctrl-D) — distinct from "" so a non-interactive caller can tell
    "nothing typed" apart from "not interactive at all"."""
    if not sys.stdin.isatty():
        return None
    try:
        return input(prompt)
    except EOFError:
        return None


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
