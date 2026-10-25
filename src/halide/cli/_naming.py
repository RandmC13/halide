"""Output naming templates (`--name "{roll}_{n:02}"`): fields, rendering, file-safe values, clashes.
Stdlib only — the parser imports this module."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("stem", "roll", "n", "stock")
HELP = "fields: {stem} (the scan's name), {roll}, {n} (frame number; {n:02} pads to 2 digits), {stock}"
_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_PAD = re.compile(r"0[1-9]")


class TemplateError(ValueError):
    pass


def safe_field(value: str) -> str:
    """A field value made safe inside a file name on Linux, macOS and Windows."""
    return _UNSAFE.sub("-", value)


@dataclass(frozen=True)
class Template:
    text: str
    parts: tuple  # (literal, field or None, spec)

    def render(self, *, stem: str, roll: str, n: int, stock: str | None) -> str:
        values = {"stem": stem, "roll": roll, "stock": stock or ""}
        out = []
        for literal, field, spec in self.parts:
            out.append(literal)
            if field == "n":
                out.append(format(n, spec[1:] + "d" if spec else "d"))
            elif field is not None:
                out.append(safe_field(values[field]))
        name = "".join(out).strip()
        if not name:
            raise TemplateError(f"--name {self.text!r} renders an empty name — {HELP}")
        return name


def parse_template(text: str) -> Template:
    def bad(why: str) -> TemplateError:
        return TemplateError(f"--name {text!r}: {why} — {HELP}")

    if "/" in text or "\\" in text:
        raise bad("a name can't contain a folder separator")
    try:
        pieces = list(string.Formatter().parse(text))
    except ValueError as exc:
        raise bad(str(exc)) from exc
    parts = []
    for literal, field, spec, conversion in pieces:
        if field is None:
            parts.append((literal, None, ""))
            continue
        if field not in FIELDS or conversion:
            raise bad(f"unknown field {{{field}}}")
        if spec and not (field == "n" and _PAD.fullmatch(spec)):
            raise bad(f"{{{field}:{spec}}} isn't supported (only {{n:02}}-style zero padding)")
        parts.append((literal, field, ":" + spec if spec else ""))
    if not any(field for _, field, _ in parts) and not text:
        raise bad("it's empty")
    template = Template(text, tuple(parts))
    template.render(stem="x", roll="x", n=1, stock="x")  # an always-empty template fails here
    return template


def find_clashes(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path, Path]]:
    """Outputs two inputs would share, compared case-insensitively (macOS's default disk is)."""
    seen: dict[str, Path] = {}
    clashes = []
    for src, out in pairs:
        key = str(out).casefold()
        if key in seen:
            clashes.append((seen[key], src, out))
        else:
            seen[key] = src
    return clashes
