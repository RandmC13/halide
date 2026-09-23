"""`halide check` — report whether a roll's scans were made consistently enough for one calibration
to be exactly valid across it (see calibration/scan_consistency.py), from file headers only."""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

from halide.batch.orchestrator import TIFF_SUFFIXES
from halide.calibration.scan_consistency import SCANNING_GUIDANCE, assess_roll
from halide.cli import console
from halide.processing import read_roll_scan_metadata


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="A directory of scans (one roll), or individual scan TIFFs")


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            files.extend(sorted(f for f in item.iterdir() if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES))
        elif item.exists():
            files.append(item)
        else:
            raise SystemExit(f"not found: {item}")
    return files


def _names(names: list[str], limit: int = 6) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", … (+{len(names) - limit} more)" if len(names) > limit else "")


def run(args: argparse.Namespace) -> int:
    files = _collect(args.inputs)
    if not files:
        print("No TIFF files found")
        return 1
    report = assess_roll({f.name: meta for f, meta in zip(files, read_roll_scan_metadata(files).values())})

    print(console.full_width_rule())
    print(f"Checked {len(files)} scan(s)")

    print(f"\n{console.Style.BOLD}Camera exposure when digitizing{console.Style.RESET}")
    for setting, names in sorted(report.exposure_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {setting:>18s}  {len(names):3d} frame(s)  {_names(names)}")
    if report.missing_exif:
        print(f"  {'(no EXIF)':>18s}  {len(report.missing_exif):3d} frame(s)  {_names(report.missing_exif)}")
    if report.exposure_inconsistent:
        print(console.warning(
            f"{report.exposure_spread_stops:.1f} stops between the brightest and darkest scan — "
            "correctable with --match-scan-exposure on invert/batch"
        ))
    elif report.exposure_groups:
        print(console.success("consistent"))

    print(f"\n{console.Style.BOLD}Raw white balance (darktable exports){console.Style.RESET}")
    if not report.white_balance_groups:
        print("  (no darktable history found — can't check)")
    for wb, names in sorted(report.white_balance_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  R {wb[0]:.3f} G {wb[1]:.3f} B {wb[2]:.3f}  {len(names):3d} frame(s)  {_names(names)}")
    if report.white_balance_inconsistent:
        print(console.warning(
            "white balance differs between frames — not correctable after export (it's applied in "
            "the camera's own colour space); re-export with one fixed white balance for the roll "
            "(\"as shot\" is only fixed if the camera was set to a fixed white balance when scanning)"
        ))
    elif report.white_balance_groups:
        print(console.success("consistent"))

    print(f"\n{console.Style.BOLD}Tone/colour processing in the raw converter{console.Style.RESET}")
    if report.tonal_modules:
        for name, modules in sorted(report.tonal_modules.items()):
            print(console.warning(f"{name}: {', '.join(modules)} active — this export isn't linear; disable and re-export"))
    elif any(d is not None for d in report.darktable.values()):
        print(console.success("none active"))
    else:
        print("  (no darktable history found — can't check)")

    width = min(shutil.get_terminal_size((100, 24)).columns, 100)
    print(f"\n{textwrap.fill(SCANNING_GUIDANCE, width=width, initial_indent='  ', subsequent_indent='  ')}")
    print(console.full_width_rule())
    return 1 if report.has_issues else 0
