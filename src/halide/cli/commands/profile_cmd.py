"""`halide profile` — list/show/rename/delete saved calibration profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from halide.calibration.profile_store import (
    EDITABLE_FIELDS,
    ProfileNameError,
    damaged_profile_message,
    default_profiles_dir,
    delete_profile,
    list_profiles,
    load_anchors,
    load_profile,
    load_scan_reference,
    rename_profile,
    resolve_profile_path,
    suggest_profile_name,
    update_profile,
    validate_profile_name,
)
from halide.cli import console
from halide.cli._output_policy import is_interactive

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
    delete_parser.add_argument(
        "--yes", action="store_true", help="Delete without asking for confirmation (needed outside a terminal)"
    )

    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit a saved profile's film stock, process, scanner, or notes",
        description="Edit a saved profile's metadata fields (film stock, process, scanner, "
        "notes) — the calibration data itself (white balance / density scale) isn't editable "
        "here, only what you've recorded about it. With no flags and a real terminal, prompts "
        "for each field interactively; otherwise pass one or more flags to set fields directly.",
    )
    edit_parser.add_argument("name")
    edit_parser.add_argument("--film-stock", dest="film_stock", help="Set the film stock (pass - to clear)")
    edit_parser.add_argument("--process", help="Set the process (pass - to clear)")
    edit_parser.add_argument("--scanner", help="Set the scanner (pass - to clear)")
    edit_parser.add_argument("--notes", help="Set the notes (pass - to clear)")


def _run_list(args: argparse.Namespace) -> int:
    profiles = list_profiles()
    if not profiles:
        print(f"No saved profiles in {default_profiles_dir()}")
        return 0

    lines = [f"Saved profiles in {default_profiles_dir()}:"]
    for name, profile, problem in profiles:
        if problem is not None:
            lines.append(f"  {console.dim(f'{name}   (unreadable: {problem})')}")
            continue
        detail_bits = [b for b in (profile.film_stock, profile.process, profile.scanner) if b]
        detail = f" ({', '.join(detail_bits)})" if detail_bits else ""
        source_color = console.source_color(profile.source)
        source = f"{source_color}{profile.source}{console.Style.RESET}"
        lines.append(f"  {console.Style.BOLD}{name}{console.Style.RESET}{detail} — source: {source}, "
                     f"created: {profile.created_at or 'unknown'}")
    print(console.framed(lines))
    return 0


def _run_show(args: argparse.Namespace) -> int:
    try:
        path = resolve_profile_path(args.name)
    except FileNotFoundError as exc:
        suggestion = suggest_profile_name(args.name)
        hint = f" — did you mean '{suggestion}'?" if suggestion else ""
        raise SystemExit(f"{exc}{hint}")
    try:
        profile = load_profile(path)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(damaged_profile_message(path.stem, exc))
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
    print(f"Digitized at:   {scan.describe() if scan else '(not recorded — pass --scan-reference FRAME with --match-scan-exposure)'}")
    _show_roll(path)
    return 0


def _show_roll(path: Path) -> None:
    """Where the picker's points came from (profiles saved by `halide calibrate`), flagging a roll
    or frames that have been moved or deleted since. Older profiles recorded paths relative to the
    folder the picker was started in; those are read against the current folder, as the picker does."""
    records, roll = load_anchors(path)
    warn = f"                {console.Style.YELLOW}{console.ICON_WARN} {{}}{console.Style.RESET}"
    if roll:
        roll_path = Path(roll)
        if roll_path.is_absolute() or roll_path.is_dir():
            roll_path = Path(os.path.abspath(roll_path))
            print(f"Roll:           {roll_path}")
            if not roll_path.is_dir():
                print(warn.format("not found - moved or deleted?"))
        else:  # an older profile's relative path; which folder it was relative to isn't recorded
            print(f"Roll:           {roll}")
            print(warn.format("recorded relative to the folder halide calibrate ran in, and not in this one"))
            print("                  reopen it in halide calibrate --profile and use Find roll… to record its full path")
    if records:
        frames = {Path(r["frame"]) for r in records}
        missing = sum(not f.is_file() for f in frames)
        note = f" ({missing} missing)" if missing else ""
        print(f"Points:         {len(records)} on {console.plural(len(frames), 'frame')}{note}")


def _reject_path_like(value: str, command: str) -> None:
    """`profile rename`/`delete`/`edit` take a bare saved-profile name, never a file path - unlike
    `profile show` and `--profile`, which accept both (2.5-12). Caught here with a message that
    says so, rather than a confusing "no saved profile named '/long/path/to/it.json'" from the
    name-only lookup these three subcommands use."""
    if "/" in value or "\\" in value or value.lower().endswith(".json"):
        raise SystemExit(f"`profile {command}` takes a profile name - file paths work with `profile show` and `--profile`")


def _run_rename(args: argparse.Namespace) -> int:
    _reject_path_like(args.old_name, "rename")
    _reject_path_like(args.new_name, "rename")
    try:
        new_path = rename_profile(args.old_name, args.new_name)
    except FileNotFoundError as exc:
        suggestion = suggest_profile_name(args.old_name)
        hint = f" — did you mean '{suggestion}'?" if suggestion else ""
        raise SystemExit(f"{exc}{hint}")
    except (FileExistsError, ProfileNameError) as exc:
        raise SystemExit(str(exc))
    print(console.success(f"Renamed {args.old_name!r} to {args.new_name!r} ({new_path})"))
    return 0


def _delete_confirmation_detail(path: Path) -> str:
    """Points count and calibration date for the delete confirmation prompt - falls back to a
    plain notice for a damaged profile rather than blocking deletion of a file that can't even be
    read (arguably the most common reason to want to delete one)."""
    try:
        profile = load_profile(path)
    except (ValueError, json.JSONDecodeError, OSError):
        return "damaged - can't read its details"
    records, _ = load_anchors(path)
    bits = []
    if records:
        frames = {r["frame"] for r in records}
        bits.append(f"{console.plural(len(records), 'point')} on {console.plural(len(frames), 'frame')}")
    bits.append(f"created {profile.created_at or 'unknown date'}")
    return ", ".join(bits)


def _run_delete(args: argparse.Namespace) -> int:
    _reject_path_like(args.name, "delete")
    try:
        name = validate_profile_name(args.name)
    except ProfileNameError as exc:
        raise SystemExit(str(exc))
    directory = default_profiles_dir()
    path = directory / f"{name}.json"
    if not path.exists():
        suggestion = suggest_profile_name(name)
        hint = f" — did you mean '{suggestion}'?" if suggestion else ""
        raise SystemExit(f"no saved profile named {name!r} in {directory}{hint}")

    if not args.yes:
        if is_interactive():
            question = f"Delete profile '{name}' ({_delete_confirmation_detail(path)})? This can't be undone"
            if not console.confirm(question):
                raise SystemExit("Nothing deleted")
        else:
            raise SystemExit(f"refusing to delete {name!r} without confirmation - pass --yes (no terminal to ask in)")

    delete_profile(name)
    print(console.success(f"Deleted profile {name!r}"))
    return 0


def _interactive_edit_fields(profile) -> dict[str, str | None]:
    print(
        f"Editing profile {profile.name!r} — press Enter to leave a field as-is, or type a "
        "single '-' to clear it"
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
    _reject_path_like(args.name, "edit")
    directory = default_profiles_dir()
    path = directory / f"{args.name}.json"
    if not path.exists():
        suggestion = suggest_profile_name(args.name)
        hint = f" — did you mean '{suggestion}'?" if suggestion else ""
        raise SystemExit(f"no saved profile named {args.name!r} in {directory}{hint}")
    try:
        profile = load_profile(path)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(damaged_profile_message(args.name, exc))

    flag_fields = {field: getattr(args, field) for field in EDITABLE_FIELDS if getattr(args, field) is not None}
    if flag_fields:
        # One convention to clear a field: `-`, as in the prompt (the flag also takes '', undocumented).
        fields: dict[str, str | None] = {k: (None if v in ("", "-") else v) for k, v in flag_fields.items()}
    elif sys.stdin.isatty():
        fields = _interactive_edit_fields(profile)
        if not fields:
            print(console.dim("No changes made"))
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
