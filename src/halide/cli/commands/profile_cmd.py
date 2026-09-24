"""`halide profile` — list/show/rename/delete saved calibration profiles."""

from __future__ import annotations

import argparse
import sys

from halide.calibration.profile_store import (
    EDITABLE_FIELDS,
    default_profiles_dir,
    delete_profile,
    list_profiles,
    load_profile,
    load_scan_reference,
    rename_profile,
    resolve_profile_path,
    update_profile,
)
from halide.cli import console

# (field attr name, CLI flag dest, human label) — drives both the `edit` subparser's flags and
# the interactive prompt loop, so the two stay in sync automatically.
_EDIT_FIELD_LABELS = {
    "film_stock": "Film stock",
    "process": "Process",
    "scanner": "Scanner",
    "notes": "Notes",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="profile_command", required=True)

    subparsers.add_parser("list", help="List saved calibration profiles")

    show_parser = subparsers.add_parser("show", help="Show one saved profile's details")
    show_parser.add_argument("name", help="Profile name or file path")

    rename_parser = subparsers.add_parser("rename", help="Rename a saved profile")
    rename_parser.add_argument("old_name")
    rename_parser.add_argument("new_name")

    delete_parser = subparsers.add_parser("delete", help="Delete a saved profile")
    delete_parser.add_argument("name")

    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit a saved profile's film stock, process, scanner, or notes",
        description="Edit a saved profile's metadata fields (film stock, process, scanner, "
        "notes) — the calibration data itself (white balance / density scale) isn't editable "
        "here, only what you've recorded about it. With no flags and a real terminal, prompts "
        "for each field interactively; otherwise pass one or more flags to set fields directly.",
    )
    edit_parser.add_argument("name")
    edit_parser.add_argument("--film-stock", dest="film_stock", help="Set the film stock (pass '' to clear)")
    edit_parser.add_argument("--process", help="Set the process (pass '' to clear)")
    edit_parser.add_argument("--scanner", help="Set the scanner (pass '' to clear)")
    edit_parser.add_argument("--notes", help="Set the notes (pass '' to clear)")


def _run_list(args: argparse.Namespace) -> int:
    profiles = list_profiles()
    if not profiles:
        print(f"No saved profiles in {default_profiles_dir()}")
        return 0

    lines = [f"Saved profiles in {default_profiles_dir()}:"]
    for name, profile in profiles:
        detail_bits = [b for b in (profile.film_stock, profile.process, profile.scanner) if b]
        detail = f" ({', '.join(detail_bits)})" if detail_bits else ""
        source_color = console.SOURCE_COLOR.get(profile.source, console.Style.DIM)
        source = f"{source_color}{profile.source}{console.Style.RESET}"
        lines.append(f"  {console.Style.BOLD}{name}{console.Style.RESET}{detail} — source: {source}, "
                     f"created: {profile.created_at or 'unknown'}")
    print(console.framed(lines))
    return 0


def _run_show(args: argparse.Namespace) -> int:
    try:
        path = resolve_profile_path(args.name)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    profile = load_profile(path)
    print(f"Path:           {path}")
    print(f"Name:           {profile.name}")
    print(f"White balance:  {profile.white_balance}")
    print(f"Density scale:  {profile.density_scale}")
    print(f"Film stock:     {profile.film_stock or '(not set)'}")
    print(f"Process:        {profile.process or '(not set)'}")
    print(f"Scanner:        {profile.scanner or '(not set)'}")
    print(f"Source:         {profile.source}")
    print(f"Created:        {profile.created_at or 'unknown'}")
    print(f"Notes:          {profile.notes or '(not set)'}")
    scan = load_scan_reference(path)
    print(f"Scanned at:     {scan.describe() if scan else '(not recorded — pass --scan-reference FRAME with --match-scan-exposure)'}")
    return 0


def _run_rename(args: argparse.Namespace) -> int:
    try:
        new_path = rename_profile(args.old_name, args.new_name)
    except (FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc))
    print(console.success(f"Renamed {args.old_name!r} to {args.new_name!r} ({new_path})"))
    return 0


def _run_delete(args: argparse.Namespace) -> int:
    try:
        delete_profile(args.name)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(console.success(f"Deleted profile {args.name!r}"))
    return 0


def _interactive_edit_fields(profile) -> dict[str, str | None]:
    print(
        f"Editing profile {profile.name!r} — press Enter to leave a field as-is, or type a "
        "single '-' to clear it."
    )
    fields: dict[str, str | None] = {}
    for attr, label in _EDIT_FIELD_LABELS.items():
        current = getattr(profile, attr)
        typed = console.prompt_line(f"{label} [{current or '(not set)'}]: ")
        if not typed:
            continue
        fields[attr] = None if typed == "-" else typed
    return fields


def _run_edit(args: argparse.Namespace) -> int:
    directory = default_profiles_dir()
    path = directory / f"{args.name}.json"
    if not path.exists():
        raise SystemExit(f"no saved profile named {args.name!r} in {directory}")
    profile = load_profile(path)

    flag_fields = {field: getattr(args, field) for field in EDITABLE_FIELDS if getattr(args, field) is not None}
    if flag_fields:
        fields: dict[str, str | None] = {k: (v or None) for k, v in flag_fields.items()}
    elif sys.stdin.isatty():
        fields = _interactive_edit_fields(profile)
        if not fields:
            print(console.dim("No changes made."))
            return 0
    else:
        raise SystemExit(
            "halide profile edit needs either field flags (--film-stock/--process/--scanner/"
            "--notes) or an interactive terminal to prompt for changes"
        )

    update_profile(args.name, **fields)
    print(console.success(f"Updated profile {args.name!r} ({path})"))
    return 0


def run(args: argparse.Namespace) -> int:
    handlers = {
        "list": _run_list,
        "show": _run_show,
        "rename": _run_rename,
        "delete": _run_delete,
        "edit": _run_edit,
    }
    return handlers[args.profile_command](args)
