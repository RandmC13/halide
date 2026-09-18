"""`halide profile` — list/show/rename/delete saved calibration profiles."""

from __future__ import annotations

import argparse

from halide.calibration.profile_store import (
    default_profiles_dir,
    delete_profile,
    list_profiles,
    load_profile,
    rename_profile,
    resolve_profile_path,
)


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


def _run_list(args: argparse.Namespace) -> int:
    profiles = list_profiles()
    if not profiles:
        print(f"No saved profiles in {default_profiles_dir()}")
        return 0

    print(f"Saved profiles in {default_profiles_dir()}:")
    for name, profile in profiles:
        detail_bits = [b for b in (profile.film_stock, profile.process, profile.scanner) if b]
        detail = f" ({', '.join(detail_bits)})" if detail_bits else ""
        print(f"  {name}{detail} — source: {profile.source}, created: {profile.created_at or 'unknown'}")
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
    return 0


def _run_rename(args: argparse.Namespace) -> int:
    try:
        new_path = rename_profile(args.old_name, args.new_name)
    except (FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc))
    print(f"Renamed {args.old_name!r} to {args.new_name!r} ({new_path})")
    return 0


def _run_delete(args: argparse.Namespace) -> int:
    try:
        delete_profile(args.name)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(f"Deleted profile {args.name!r}")
    return 0


def run(args: argparse.Namespace) -> int:
    handlers = {
        "list": _run_list,
        "show": _run_show,
        "rename": _run_rename,
        "delete": _run_delete,
    }
    return handlers[args.profile_command](args)
