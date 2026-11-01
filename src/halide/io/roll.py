"""The one place a folder of scans is listed: batch, print, export, check, contact and the picker all
call `list_scans`, so they agree on which files are frames and in what order.

Stdlib only (no Pillow until a picture file needs telling apart from a contact sheet): imported on
the CLI's start-up path."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

TIFF_SUFFIXES = (".tif", ".tiff")
RAW_SUFFIXES = (".nef", ".cr2", ".cr3", ".arw", ".raf", ".dng", ".orf", ".rw2")
_PICTURE_SUFFIXES = (".jpg", ".jpeg", ".png")
_MACOS_JUNK = ("._", ".ds_store")
# halide's own run log (batch/runlog.py), written into output folders; never a frame.
RUN_LOG_NAME = "halide-log.txt"

_DIGITS = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """Sort key that compares the numbers inside a name as numbers, so `frame_2` comes before
    `frame_10` — the frame order `{n}`, contact sheets and logs all follow. Zero-padded camera names
    (IMG_0138) sort exactly as plain text did. The plain name breaks ties ("f01" vs "f1")."""
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in _DIGITS.split(name) if p)
    return parts, name


@dataclass
class Skipped:
    """What a folder listing left out, by reason — one run-sheet row, never one line per file."""

    hidden: int = 0  # names starting with "." (macOS AppleDouble `._*`, `.DS_Store`, halide's partial files)
    macos: int = 0  # of `hidden`, the ones that are macOS's own
    contact_sheets: int = 0
    other: dict[str, int] = field(default_factory=dict)  # not a TIFF, by what it looks like
    folders: int = 0
    positives: int = 0  # halide's own output, when the listing was asked to leave it out

    def __bool__(self) -> bool:
        return bool(self.hidden or self.contact_sheets or self.other or self.folders or self.positives)

    def __iadd__(self, more: "Skipped") -> "Skipped":
        self.hidden += more.hidden
        self.macos += more.macos
        self.contact_sheets += more.contact_sheets
        self.folders += more.folders
        self.positives += more.positives
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
        if self.positives:
            parts.append(f"{self.positives} halide positive" + ("s" if self.positives > 1 else ""))
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
    folder: Path | str, *, extra_suffixes: tuple[str, ...] = (), skip_positives: bool = False
) -> tuple[list[Path], Skipped]:
    """The scans in `folder` (`.tif`/`.tiff`, any case), sorted in natural order, and what was skipped and why.

    Skipped: names starting with "." (which covers macOS's `._*` copies and halide's own
    partial-write files), halide's contact sheets, other files, and subfolders. `extra_suffixes`
    (e.g. `.png`, `.jpg`) admits more file types as frames — `halide contact` proofs exported
    pictures too — while halide's own contact sheets stay excluded. `skip_positives` also leaves out
    TIFFs halide itself wrote (a developing run must not re-develop its own output); a header that
    can't be read doesn't count as one, so the frame fails later with its real error."""
    folder = Path(folder)
    accepted = TIFF_SUFFIXES + tuple(extra_suffixes)
    scans: list[Path] = []
    skipped = Skipped()
    for entry in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
        name = entry.name
        if name.startswith("."):
            skipped.hidden += 1
            if name.lower().startswith(_MACOS_JUNK):
                skipped.macos += 1
            continue
        if name == RUN_LOG_NAME:
            continue  # halide's own run log: never a frame, and not worth a "Skipped" mention
        if entry.is_dir():
            skipped.folders += 1
            continue
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in _PICTURE_SUFFIXES and _is_contact_sheet(entry):
            skipped.contact_sheets += 1
        elif suffix in accepted:
            if skip_positives and suffix in TIFF_SUFFIXES and _is_halide_positive(entry):
                skipped.positives += 1
            else:
                scans.append(entry)
        else:
            label = _kind(suffix)
            skipped.other[label] = skipped.other.get(label, 0) + 1
    return scans, skipped


def _is_contact_sheet(path: Path) -> bool:
    from halide.io.contact_sheet import is_contact_sheet

    return is_contact_sheet(path)


def _is_halide_positive(path: Path) -> bool:
    from halide.io.tiff import read_tiff_description
    from halide.processing import read_provenance

    try:
        return read_provenance(read_tiff_description(path)) is not None
    except Exception:  # an unreadable header is a scan that will fail with its own error later
        return False


@dataclass(frozen=True)
class Roll:
    """One folder of scans under a folder of rolls. `left_out` is what listing it skipped."""

    name: str
    folder: Path
    files: list[Path]
    left_out: Skipped


@dataclass
class RollsSkipped:
    """Folders under the input that are not rolls — one run-sheet row, never one line per folder."""

    halide_outputs: int = 0  # holding only halide's own positives
    destinations: int = 0  # where this run is about to write

    def describe(self) -> str:
        """e.g. "2 folders of halide output, 1 output folder of this run"."""
        parts = []
        if self.halide_outputs:
            parts.append(f"{self.halide_outputs} folder{'s' if self.halide_outputs > 1 else ''} of halide output")
        if self.destinations:
            parts.append(f"{self.destinations} output folder{'s' if self.destinations > 1 else ''} of this run")
        return ", ".join(parts)


def find_rolls(
    top: Path | str, exclude: Callable[[Path], bool] = lambda p: False
) -> tuple[list[Roll], RollsSkipped]:
    """Every folder under `top` (itself included) that holds scans is a roll, in natural order.

    Hidden and symlinked folders are not entered (no loops). A folder left with nothing but halide
    positives is an earlier run's output, and a folder `exclude` names is this run's own destination:
    neither is a roll, so a second run never develops the first one's results. A roll is named for
    the folder directly under `top` that holds it; two scan folders under one such folder get their
    path below `top` joined with " - " instead, so names stay unique."""
    top = Path(top)
    skipped = RollsSkipped()
    found: list[tuple[tuple[str, ...], Path, list[Path], Skipped]] = []
    for here, dirnames, _ in os.walk(top, followlinks=False):
        dirnames[:] = sorted(
            (d for d in dirnames if not d.startswith(".") and not os.path.islink(os.path.join(here, d))),
            key=natural_key,
        )
        folder = Path(here)
        files, left_out = list_scans(folder, skip_positives=True)
        if not files:
            if left_out.positives:
                skipped.halide_outputs += 1
        elif exclude(folder):
            skipped.destinations += 1
        else:
            found.append((folder.relative_to(top).parts, folder, files, left_out))
    heads = [parts[0] if parts else top.resolve().name for parts, *_ in found]
    rolls = []
    for (parts, folder, files, left_out), head in zip(found, heads):
        name = head if heads.count(head) == 1 else " - ".join(parts) or head
        rolls.append(Roll(name, folder, files, left_out))
    return rolls, skipped
