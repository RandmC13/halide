"""Persist a DensityProfile as JSON so a film-stock/process/scanner calibration can be reused
across a whole roll (and across sessions) instead of re-solved per image — this is the mechanism
behind "calibrate once per stock, not per image."

Profiles can be referenced either by a full file path or by a bare name, resolved against a
default directory (`~/.config/halide/profiles/` on Linux, `~/Library/Application Support/halide/
profiles/` on macOS, or `$XDG_CONFIG_HOME/halide/profiles/` if set).
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.atomic import atomic_output
from halide.io.scan_metadata import ScanSettings


class ProfileError(ValueError):
    """Base for a profile-name/identity problem that should reach the user as a plain message,
    never a raw traceback or an OS error string."""


class ProfileNameError(ProfileError):
    """A profile name that can't be used at all - empty, contains a path separator, starts with
    '.'/'-', or is too long. Raised before the name ever reaches a Path (2.2-1: an unvalidated
    name let `../evil` delete or overwrite a file outside the profiles directory)."""


class ProfileExistsError(ProfileError):
    """A save would silently replace an existing profile of the same name (compared
    case-insensitively - see find_profile) and `overwrite` wasn't passed."""

    def __init__(self, name: str, existing_path: Path) -> None:
        super().__init__(f"a profile named {name!r} already exists ({existing_path})")
        self.name = name
        self.existing_path = existing_path


_MAX_NAME_LENGTH = 64


def validate_profile_name(name: str) -> str:
    """Reject a profile name that could escape the profiles directory or hit an OS limit, before
    it's ever turned into a Path. Returns the name with surrounding whitespace stripped."""
    stripped = name.strip()
    if not stripped:
        raise ProfileNameError("Profile names can't be empty")
    if "/" in stripped or "\\" in stripped or "\0" in stripped:
        raise ProfileNameError(
            "Profile names can't contain / or \\ - they name a file in halide's profiles folder"
        )
    if stripped.startswith(".") or stripped.startswith("-"):
        raise ProfileNameError("Profile names can't start with '.' or '-'.")
    if len(stripped) > _MAX_NAME_LENGTH:
        raise ProfileNameError(f"Profile names can be at most {_MAX_NAME_LENGTH} characters")
    return stripped


def default_profiles_dir() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home)
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "halide" / "profiles"
    else:
        base = Path.home() / ".config"
    return base / "halide" / "profiles"


def find_profile(name: str, profiles_dir: Path | None = None) -> Path | None:
    """The saved profile file for `name`: an exact match first, then a case-insensitive
    (`casefold`) match - so 'Portra' and 'portra' resolve to the same file, matching what actually
    happens on macOS's default case-insensitive-but-preserving filesystem (2.2-12). None if no
    file matches either way."""
    directory = profiles_dir or default_profiles_dir()
    exact = directory / f"{name}.json"
    if exact.exists():
        return exact
    if not directory.exists():
        return None
    target = name.casefold()
    for path in directory.glob("*.json"):
        if path.stem.casefold() == target:
            return path
    return None


def save_profile(
    profile: DensityProfile,
    path: str | Path,
    *,
    tone: ToneCurveParams | None = None,
    scan: ScanSettings | None = None,
    anchors: list[dict] | None = None,
    roll: str | None = None,
) -> None:
    """`tone`, if given, is a GUI-picked exposure/contrast override (the calibration picker's Print
    drawer, gui/drawers.py) saved as an extra "tone" sidecar key alongside the profile's own fields -
    deliberately not merged into DensityProfile itself (see cli/_calibration_args.py/core/types.py: the
    density profile stays a pure calibration concept, tone stays a separate, optional per-run
    rendering choice that a profile can merely *suggest* a default for). Linear-output mode is
    deliberately never included here - see load_tone_override's docstring.

    `scan`, if given, is the digitizing camera exposure the calibration frame was scanned at (a
    "scan" sidecar, same reasoning as "tone"): a profile is only exactly valid at the scan exposure
    it was solved at, and --match-scan-exposure needs this reference to correct other frames to it
    (see calibration/scan_consistency.py).

    `anchors`/`roll`, if given, record how the calibration picker built this profile: every neutral
    point (frame, position, raw RGB, that frame's scan exposure - see
    calibration/anchors.py::point_to_dict) and the roll folder, so reopening the profile in the
    picker restores them. Sidecars like the others; nothing else reads them."""
    data = asdict(profile)
    if tone is not None:
        data["tone"] = {"exposure": tone.exposure, "contrast": tone.contrast}
    if scan is not None:
        data["scan"] = asdict(scan)
    if anchors is not None:
        data["anchors"] = anchors
    if roll is not None:
        data["roll"] = roll
    with atomic_output(Path(path)) as tmp:
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def load_profile(path: str | Path) -> DensityProfile:
    path = Path(path)
    data = json.loads(path.read_text())
    try:
        return DensityProfile(
            white_balance=tuple(data["white_balance"]),
            density_scale=tuple(data["density_scale"]),
            name=data.get("name"),
            film_stock=data.get("film_stock"),
            process=data.get("process"),
            scanner=data.get("scanner"),
            created_at=data.get("created_at"),
            source=data.get("source", "manual"),
            notes=data.get("notes"),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required field {exc}") from exc


def resolve_profile_path(name_or_path: str | Path, profiles_dir: Path | None = None) -> Path:
    """Accept either an existing file path or a bare profile name, looked up in the profiles
    directory as "<name>.json"."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    directory = profiles_dir or default_profiles_dir()
    named = directory / (candidate.name if candidate.suffix == ".json" else f"{candidate.name}.json")
    if named.exists():
        return named

    raise FileNotFoundError(
        f"no such profile file {str(name_or_path)!r}, and no saved profile named "
        f"{candidate.stem!r} found in {directory}"
    )


def save_named_profile(
    profile: DensityProfile,
    name: str,
    profiles_dir: Path | None = None,
    tone: ToneCurveParams | None = None,
    scan: ScanSettings | None = None,
    anchors: list[dict] | None = None,
    roll: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Save `profile` under `name` in the profiles directory, stamping its name and creation time.
    `tone`/`scan`/`anchors`/`roll` are optional sidecars to save alongside it - see save_profile.

    Raises ProfileNameError for an unusable name (validate_profile_name), and ProfileExistsError
    if a profile of that name already exists (compared case-insensitively, via find_profile) and
    `overwrite` isn't True (2.2-2/2.2-12/2.6-5: a re-run with the same --save-profile-as name, or a
    name differing only by case, used to overwrite a real calibration with no warning). Overwriting
    an existing name-differing-only-by-case match reuses that file's own on-disk name/casing rather
    than creating a second file."""
    name = validate_profile_name(name)
    directory = profiles_dir or default_profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    existing = find_profile(name, directory)
    if existing is not None and not overwrite:
        raise ProfileExistsError(name, existing)
    path = existing if existing is not None else directory / f"{name}.json"
    stamped = replace(profile, name=name, created_at=datetime.now(timezone.utc).isoformat())
    save_profile(stamped, path, tone=tone, scan=scan, anchors=anchors, roll=roll)
    return path


def load_tone_override(path: str | Path) -> ToneCurveParams | None:
    """Read back an optional saved exposure/contrast override (see save_profile's `tone` param).
    Returns None for any profile file without one - including every profile saved before this
    existed, which is the point: fully backward compatible, no migration needed. Deliberately never
    restores linear-output mode - that stays a CLI-flag/preview-only choice, not something a saved
    profile silently changes about a future run's output format."""
    data = json.loads(Path(path).read_text())
    tone_data = data.get("tone")
    if tone_data is None:
        return None
    return ToneCurveParams(mode="paper", exposure=tone_data["exposure"], contrast=tone_data["contrast"])


def load_scan_reference(path: str | Path) -> ScanSettings | None:
    """The scan exposure a profile was calibrated at, if recorded (profiles saved before this
    existed have none — pass --scan-reference FRAME for those)."""
    scan = json.loads(Path(path).read_text()).get("scan")
    if not scan:
        return None
    return ScanSettings(exposure_time=scan["exposure_time"], f_number=scan["f_number"], iso=scan["iso"])


def load_anchors(path: str | Path) -> tuple[list[dict], str | None]:
    """The calibration picker's record of how a profile was built (see save_profile's `anchors`/
    `roll`): ([], None) for any profile without one - hand-solved, CLI-saved, or older profiles."""
    data = json.loads(Path(path).read_text())
    return list(data.get("anchors") or []), data.get("roll")


def list_profiles(profiles_dir: Path | None = None) -> list[tuple[str, DensityProfile | None, str | None]]:
    """(filename stem, profile, problem) triples for every *.json file in the profiles directory,
    sorted by name. A file that can't be read - permission denied, corrupt/truncated JSON, or
    missing a required field - is never skipped silently and never crashes the whole listing (2.2-4:
    the previous version's own docstring promised this but only caught JSON errors, not OSError):
    it's listed with `profile=None` and a plain-English `problem` string, for the caller to show."""
    directory = profiles_dir or default_profiles_dir()
    if not directory.exists():
        return []
    results: list[tuple[str, DensityProfile | None, str | None]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            results.append((path.stem, load_profile(path), None))
        except (ValueError, json.JSONDecodeError) as exc:
            results.append((path.stem, None, str(exc)))
        except OSError as exc:
            results.append((path.stem, None, exc.strerror or str(exc)))
    return results


def suggest_profile_name(name: str, profiles_dir: Path | None = None) -> str | None:
    """A "did you mean" suggestion for a mistyped profile name - the closest saved, readable
    profile name by string similarity, or None if none is close enough. Shared by every profile
    subcommand (show/rename/edit/delete) and --profile's own not-found error (2.2-7)."""
    names = [stem for stem, profile, problem in list_profiles(profiles_dir) if problem is None]
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    return matches[0] if matches else None


def damaged_profile_message(name: str, exc: Exception) -> str:
    """A plain message for a saved profile file that exists but can't be parsed/read (corrupt or
    hand-edited JSON, a missing required field) - so a raw JSONDecodeError/ValueError never reaches
    the user directly (2.2-5)."""
    return (
        f"profile {name!r} is damaged and can't be read ({exc}) - re-save it from "
        f"`halide calibrate --profile {name}`, or delete it with `halide profile delete {name}`"
    )


def rename_profile(old_name: str, new_name: str, profiles_dir: Path | None = None) -> Path:
    old_name = validate_profile_name(old_name)
    new_name = validate_profile_name(new_name)
    directory = profiles_dir or default_profiles_dir()
    old_path = directory / f"{old_name}.json"
    if not old_path.exists():
        raise FileNotFoundError(f"no saved profile named {old_name!r} in {directory}")
    new_path = directory / f"{new_name}.json"

    # Case-insensitive, like save_named_profile (2.2-12): "a" -> "B" is still a collision if "b"
    # is already a saved profile - but not when the only match find_profile finds is old_path
    # itself, which is exactly what a case-only rename ("portra400" -> "Portra400") looks like to
    # a case-insensitive lookup.
    existing = find_profile(new_name, directory)
    if existing is not None and existing.resolve() != old_path.resolve():
        raise FileExistsError(f"a profile named {new_name!r} already exists in {directory}")

    # Rewrites the raw JSON rather than round-tripping through DensityProfile, so the optional
    # "tone"/"scan" sidecars survive a rename (round-tripping used to silently drop "tone").
    data = json.loads(old_path.read_text())
    data["name"] = new_name
    content = json.dumps(data, indent=2) + "\n"

    if old_path.name == new_path.name:
        # Renaming onto the exact same name (not even a case change) - just restamp the "name"
        # field in place. This has to be handled separately: falling through to the general branch
        # below would atomically overwrite old_path with the new content and then immediately
        # unlink it (old_path IS new_path here), deleting the profile outright.
        with atomic_output(old_path) as tmp:
            tmp.write_text(content)
        return old_path

    if old_path.name.casefold() == new_path.name.casefold():
        # A case-only change (e.g. "portra400" -> "Portra400"): asking the filesystem to rename a
        # path onto one that differs only in case can be treated as "the same file" and silently
        # no-op the case change, on a case-insensitive-but-preserving filesystem (macOS/Windows) -
        # not reproducible on this dev sandbox's Linux filesystem, but real on the user's. Moving
        # the old file to a name the filesystem can't confuse with either case first, then writing
        # the new path fresh, means no single rename call ever has to disambiguate "same file, new
        # case" - and the original survives under its backup name if the write fails partway.
        backup = directory / f".{old_name}.halide-case-rename-{os.getpid()}.json"
        old_path.replace(backup)
        try:
            with atomic_output(new_path) as tmp:
                tmp.write_text(content)
        except BaseException:
            backup.replace(old_path)  # put it back under its original name/case
            raise
        backup.unlink(missing_ok=True)
        return new_path

    with atomic_output(new_path) as tmp:
        tmp.write_text(content)
    old_path.unlink()
    return new_path


# Metadata fields `halide profile edit` (and update_profile below) may change. Deliberately
# excludes the solved calibration data (white_balance/density_scale) and identity fields
# (name/created_at/source) — those are either produced by a calibration method, not typed by
# hand, or already have their own dedicated operation (rename_profile).
EDITABLE_FIELDS = ("film_stock", "process", "scanner", "notes")


def update_profile(name: str, profiles_dir: Path | None = None, **fields: str | None) -> Path:
    """Update one or more editable metadata fields on a saved profile in place. Leaves the
    calibration data, name, source, and created_at untouched. Raises ValueError for any field
    not in EDITABLE_FIELDS, and FileNotFoundError if `name` isn't a saved profile."""
    name = validate_profile_name(name)
    unknown = set(fields) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"not an editable profile field: {', '.join(sorted(unknown))}")
    directory = profiles_dir or default_profiles_dir()
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved profile named {name!r} in {directory}")
    # Edits the raw JSON rather than round-tripping through DensityProfile (as rename_profile does),
    # so optional sidecars ("tone", "scan", ...) survive an edit - round-tripping dropped them.
    data = json.loads(path.read_text())
    data.update(fields)
    with atomic_output(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2) + "\n")
    return path


def delete_profile(name: str, profiles_dir: Path | None = None) -> None:
    name = validate_profile_name(name)
    directory = profiles_dir or default_profiles_dir()
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved profile named {name!r} in {directory}")
    path.unlink()
