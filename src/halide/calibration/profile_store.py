"""Persist a DensityProfile as JSON so a film-stock/process/scanner calibration can be reused
across a whole roll (and across sessions) instead of re-solved per image — this is the mechanism
behind "calibrate once per stock, not per image."

Profiles can be referenced either by a full file path or by a bare name, resolved against a
default directory (`~/.config/halide/profiles/`, or `$XDG_CONFIG_HOME/halide/profiles/` if set).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.scan_metadata import ScanSettings


def default_profiles_dir() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "halide" / "profiles"


def save_profile(
    profile: DensityProfile,
    path: str | Path,
    *,
    tone: ToneCurveParams | None = None,
    scan: ScanSettings | None = None,
    anchors: list[dict] | None = None,
    roll: str | None = None,
) -> None:
    """`tone`, if given, is a GUI-picked exposure/contrast override (see gui/preview_popup.py's
    Fine-tune controls) saved as an extra "tone" sidecar key alongside the profile's own fields -
    deliberately not merged into DensityProfile itself (see calibration_args.py/core/types.py: the
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
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


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
) -> Path:
    """Save `profile` under `name` in the profiles directory, stamping its name and creation time.
    `tone`/`scan`/`anchors`/`roll` are optional sidecars to save alongside it - see save_profile."""
    directory = profiles_dir or default_profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamped = replace(profile, name=name, created_at=datetime.now(timezone.utc).isoformat())
    path = directory / f"{name}.json"
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


def list_profiles(profiles_dir: Path | None = None) -> list[tuple[str, DensityProfile]]:
    """(filename stem, profile) pairs for every valid saved profile, sorted by name. A single
    unreadable file is skipped with a warning rather than making every other profile inaccessible."""
    directory = profiles_dir or default_profiles_dir()
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            results.append((path.stem, load_profile(path)))
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping unreadable profile {path}: {exc}")
    return results


def rename_profile(old_name: str, new_name: str, profiles_dir: Path | None = None) -> Path:
    directory = profiles_dir or default_profiles_dir()
    old_path = directory / f"{old_name}.json"
    if not old_path.exists():
        raise FileNotFoundError(f"no saved profile named {old_name!r} in {directory}")
    new_path = directory / f"{new_name}.json"
    if new_path.exists():
        raise FileExistsError(f"a profile named {new_name!r} already exists in {directory}")

    # Rewrites the raw JSON rather than round-tripping through DensityProfile, so the optional
    # "tone"/"scan" sidecars survive a rename (round-tripping used to silently drop "tone").
    data = json.loads(old_path.read_text())
    data["name"] = new_name
    new_path.write_text(json.dumps(data, indent=2) + "\n")
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
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def delete_profile(name: str, profiles_dir: Path | None = None) -> None:
    directory = profiles_dir or default_profiles_dir()
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved profile named {name!r} in {directory}")
    path.unlink()
