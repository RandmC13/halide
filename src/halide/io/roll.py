"""The one place a folder of scans is listed: batch, print, export, check, contact and the picker all
call `list_scans`, so they agree on which files are frames and in what order (F09).

Stdlib only (no Pillow until a picture file needs telling apart from a contact sheet): imported on
the CLI's start-up path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TIFF_SUFFIXES = (".tif", ".tiff")
RAW_SUFFIXES = (".nef", ".cr2", ".cr3", ".arw", ".raf", ".dng", ".orf", ".rw2")
_PICTURE_SUFFIXES = (".jpg", ".jpeg", ".png")
_MACOS_JUNK = ("._", ".ds_store")


@dataclass
class Skipped:
    """What a folder listing left out, by reason — one run-sheet row, never one line per file."""

    hidden: int = 0  # names starting with "." (macOS AppleDouble `._*`, `.DS_Store`, halide's partial files)
    macos: int = 0  # of `hidden`, the ones that are macOS's own
    contact_sheets: int = 0
    other: dict[str, int] = field(default_factory=dict)  # not a TIFF, by what it looks like
    folders: int = 0

    def __bool__(self) -> bool:
        return bool(self.hidden or self.contact_sheets or self.other or self.folders)

    def __iadd__(self, more: "Skipped") -> "Skipped":
        self.hidden += more.hidden
        self.macos += more.macos
        self.contact_sheets += more.contact_sheets
        self.folders += more.folders
        for label, n in more.other.items():
            self.other[label] = self.other.get(label, 0) + n
        return self

    def describe(self) -> str:
        """e.g. "2 hidden macOS files, 1 JPEG, 1 folder"."""
        parts = []
        if self.hidden:
            kind = "hidden macOS files" if self.macos == self.hidden else "hidden files"
            parts.append(f"{self.hidden} {kind}" if self.hidden > 1 else f"1 {kind[:-1]}")
        if self.contact_sheets:
            parts.append(f"{self.contact_sheets} contact sheet" + ("s" if self.contact_sheets > 1 else ""))
        for label, n in sorted(self.other.items()):
            parts.append(f"{n} {label if n == 1 else label + 's'}")
        if self.folders:
            parts.append(f"{self.folders} folder" + ("s" if self.folders > 1 else ""))
        return ", ".join(parts)


def _kind(suffix: str) -> str:
    if suffix in (".jpg", ".jpeg"):
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix in RAW_SUFFIXES:
        return "raw file"
    return "non-TIFF file"


def list_scans(
    folder: Path | str, *, extra_suffixes: tuple[str, ...] = ()
) -> tuple[list[Path], Skipped]:
    """The scans in `folder` (`.tif`/`.tiff`, any case), sorted by name, and what was skipped and why.

    Skipped: names starting with "." (which covers macOS's `._*` copies and halide's own
    partial-write files), halide's contact sheets, other files, and subfolders. `extra_suffixes`
    (e.g. `.png`, `.jpg`) admits more file types as frames — `halide contact` proofs exported
    pictures too — while halide's own contact sheets stay excluded."""
    folder = Path(folder)
    accepted = TIFF_SUFFIXES + tuple(extra_suffixes)
    scans: list[Path] = []
    skipped = Skipped()
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        name = entry.name
        if name.startswith("."):
            skipped.hidden += 1
            if name.lower().startswith(_MACOS_JUNK):
                skipped.macos += 1
            continue
        if entry.is_dir():
            skipped.folders += 1
            continue
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in _PICTURE_SUFFIXES and _is_contact_sheet(entry):
            skipped.contact_sheets += 1
        elif suffix in accepted:
            scans.append(entry)
        else:
            label = _kind(suffix)
            skipped.other[label] = skipped.other.get(label, 0) + 1
    return scans, skipped


def _is_contact_sheet(path: Path) -> bool:
    from halide.io.contact_sheet import is_contact_sheet

    return is_contact_sheet(path)
