"""Output naming templates (`--name "{roll}_{n:02}"`): fields, rendering, file-safe values.
Stdlib only — the parser imports this module."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

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
        if name.startswith("."):
            # folder runs skip dotfiles, so a later print, export or contact sheet would miss it
            raise TemplateError(f"--name {self.text!r} gives {name!r}, a hidden file — start it with a field or a letter")
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
    template.render(stem="x", roll="x", n=1, stock=None)  # profiles without a film stock render it empty
    return template

