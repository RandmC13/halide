import os
from dataclasses import replace

import pytest

from halide.calibration.profile_store import (
    ProfileExistsError,
    ProfileNameError,
    damaged_profile_message,
    delete_profile,
    find_profile,
    list_profiles,
    load_profile,
    load_scan_reference,
    load_tone_override,
    rename_profile,
    resolve_profile_path,
    save_named_profile,
    save_profile,
    suggest_profile_name,
    update_profile,
    validate_profile_name,
)
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.scan_metadata import ScanSettings

PROFILE = DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78))


def test_save_and_load_roundtrips(tmp_path):
    path = tmp_path / "portra400.json"
    save_profile(PROFILE, path)
    loaded = load_profile(path)
    assert loaded.white_balance == PROFILE.white_balance
    assert loaded.density_scale == PROFILE.density_scale


def test_notes_roundtrips(tmp_path):
    profile = DensityProfile(
        white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78), notes="shot on a light table"
    )
    path = tmp_path / "with-notes.json"
    save_profile(profile, path)
    assert load_profile(path).notes == "shot on a light table"


def test_load_profile_without_notes_field_defaults_to_none(tmp_path):
    path = tmp_path / "no-notes.json"
    path.write_text('{"white_balance": [1.0, 1.0, 1.0], "density_scale": [1.0, 1.0, 1.0]}')
    assert load_profile(path).notes is None


def test_load_missing_field_raises_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"white_balance": [1.0, 1.0, 1.0]}')  # missing density_scale
    with pytest.raises(ValueError, match="density_scale"):
        load_profile(path)


def test_save_named_profile_stamps_name_and_created_at(tmp_path):
    path = save_named_profile(PROFILE, "portra400-c41-v600", profiles_dir=tmp_path)
    loaded = load_profile(path)
    assert loaded.name == "portra400-c41-v600"
    assert loaded.created_at is not None
    assert path == tmp_path / "portra400-c41-v600.json"


def test_resolve_profile_path_accepts_existing_file(tmp_path):
    path = tmp_path / "somewhere.json"
    save_profile(PROFILE, path)
    assert resolve_profile_path(path) == path


def test_resolve_profile_path_resolves_bare_name(tmp_path):
    save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path)
    resolved = resolve_profile_path("portra400", profiles_dir=tmp_path)
    assert resolved == tmp_path / "portra400.json"


def test_resolve_profile_path_raises_when_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no saved profile named"):
        resolve_profile_path("nonexistent", profiles_dir=tmp_path)


def test_list_profiles_returns_all_sorted(tmp_path):
    save_named_profile(PROFILE, "zzz", profiles_dir=tmp_path)
    save_named_profile(PROFILE, "aaa", profiles_dir=tmp_path)
    results = list_profiles(profiles_dir=tmp_path)
    assert [name for name, _, _ in results] == ["aaa", "zzz"]
    assert all(problem is None for _, _, problem in results)


def test_list_profiles_reports_damaged_json_without_crashing(tmp_path):
    save_named_profile(PROFILE, "good", profiles_dir=tmp_path)
    (tmp_path / "corrupt.json").write_text("not valid json{{{")
    results = list_profiles(profiles_dir=tmp_path)
    by_name = {name: (profile, problem) for name, profile, problem in results}
    assert by_name["good"][0] is not None and by_name["good"][1] is None
    assert by_name["corrupt"][0] is None
    assert by_name["corrupt"][1] is not None  # a plain-English reason, not a crash


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions, so chmod 000 has no effect")
def test_list_profiles_reports_unreadable_file_without_crashing(tmp_path):
    save_named_profile(PROFILE, "good", profiles_dir=tmp_path)
    blocked = tmp_path / "noperm.json"
    save_named_profile(PROFILE, "noperm", profiles_dir=tmp_path)
    blocked.chmod(0)
    try:
        results = list_profiles(profiles_dir=tmp_path)
    finally:
        blocked.chmod(0o644)  # tmp_path cleanup needs this back
    by_name = {name: (profile, problem) for name, profile, problem in results}
    assert by_name["good"][0] is not None  # one bad file doesn't take down the rest (2.2-4)
    assert by_name["noperm"][0] is None
    assert by_name["noperm"][1] is not None


def test_list_profiles_on_nonexistent_directory_returns_empty(tmp_path):
    assert list_profiles(profiles_dir=tmp_path / "does_not_exist") == []


def test_suggest_profile_name_excludes_damaged_profiles(tmp_path):
    save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path)
    (tmp_path / "portra40x.json").write_text("not valid json{{{")
    assert suggest_profile_name("portra40", profiles_dir=tmp_path) == "portra400"


def test_suggest_profile_name_none_when_nothing_close(tmp_path):
    save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path)
    assert suggest_profile_name("completely-different-xyz", profiles_dir=tmp_path) is None


def test_damaged_profile_message_names_recovery_steps():
    message = damaged_profile_message("myroll", ValueError("boom"))
    assert "myroll" in message
    assert "damaged" in message
    assert "halide calibrate --profile myroll" in message
    assert "halide profile delete myroll" in message


# --- validate_profile_name --------------------------------------------------------------------


def test_validate_profile_name_rejects_empty():
    with pytest.raises(ProfileNameError, match="can't be empty"):
        validate_profile_name("")


def test_validate_profile_name_rejects_whitespace_only():
    with pytest.raises(ProfileNameError, match="can't be empty"):
        validate_profile_name("   ")


def test_validate_profile_name_rejects_slash():
    with pytest.raises(ProfileNameError, match=r"can't contain / or \\"):
        validate_profile_name("a/b")


def test_validate_profile_name_rejects_backslash():
    with pytest.raises(ProfileNameError, match=r"can't contain / or \\"):
        validate_profile_name("a\\b")


def test_validate_profile_name_rejects_nul():
    with pytest.raises(ProfileNameError, match=r"can't contain / or \\"):
        validate_profile_name("a\0b")


def test_validate_profile_name_rejects_leading_dot():
    with pytest.raises(ProfileNameError, match="can't start with"):
        validate_profile_name(".hidden")


def test_validate_profile_name_rejects_dotdot():
    with pytest.raises(ProfileNameError, match="can't start with"):
        validate_profile_name("..")


def test_validate_profile_name_rejects_leading_dash():
    with pytest.raises(ProfileNameError, match="can't start with"):
        validate_profile_name("-weird")


def test_validate_profile_name_rejects_too_long():
    with pytest.raises(ProfileNameError, match="at most 64 characters"):
        validate_profile_name("a" * 65)


def test_validate_profile_name_accepts_64_characters():
    assert validate_profile_name("a" * 64) == "a" * 64


def test_validate_profile_name_strips_surrounding_whitespace():
    assert validate_profile_name("  portra400  ") == "portra400"


# --- traversal / find_profile ------------------------------------------------------------------


def test_dotdot_name_cannot_touch_files_outside_profiles_dir(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    sentinel = tmp_path / "sentinel-important.json"
    sentinel.write_text("do not touch")

    with pytest.raises(ProfileNameError):
        delete_profile("../sentinel-important", profiles_dir=profiles_dir)
    assert sentinel.exists()

    with pytest.raises(ProfileNameError):
        save_named_profile(PROFILE, "../sentinel-important", profiles_dir=profiles_dir)
    assert sentinel.read_text() == "do not touch"

    with pytest.raises(ProfileNameError):
        rename_profile("../sentinel-important", "whatever", profiles_dir=profiles_dir)
    with pytest.raises(ProfileNameError):
        update_profile("../sentinel-important", profiles_dir=profiles_dir, notes="x")


def test_find_profile_exact_match(tmp_path):
    save_named_profile(PROFILE, "Portra400", profiles_dir=tmp_path)
    assert find_profile("Portra400", tmp_path) == tmp_path / "Portra400.json"


def test_find_profile_case_insensitive_match(tmp_path):
    save_named_profile(PROFILE, "Portra400", profiles_dir=tmp_path)
    assert find_profile("portra400", tmp_path) == tmp_path / "Portra400.json"


def test_find_profile_returns_none_when_missing(tmp_path):
    assert find_profile("nonexistent", tmp_path) is None


def test_find_profile_on_nonexistent_directory_returns_none(tmp_path):
    assert find_profile("anything", tmp_path / "does_not_exist") is None


# --- save_named_profile overwrite protection ----------------------------------------------------


def test_saving_existing_name_raises_unless_overwrite(tmp_path):
    save_named_profile(PROFILE, "roll16", profiles_dir=tmp_path)
    with pytest.raises(ProfileExistsError):
        save_named_profile(replace(PROFILE, notes="second"), "roll16", profiles_dir=tmp_path)
    assert load_profile(tmp_path / "roll16.json").notes is None  # the original survives

    path = save_named_profile(replace(PROFILE, notes="second"), "roll16", profiles_dir=tmp_path, overwrite=True)
    assert load_profile(path).notes == "second"


def test_saving_name_differing_only_by_case_counts_as_existing(tmp_path):
    save_named_profile(PROFILE, "Portra400", profiles_dir=tmp_path)
    with pytest.raises(ProfileExistsError):
        save_named_profile(replace(PROFILE, notes="second"), "portra400", profiles_dir=tmp_path)

    # overwrite=True reuses the file's own on-disk name/casing rather than creating a second file
    path = save_named_profile(replace(PROFILE, notes="second"), "portra400", profiles_dir=tmp_path, overwrite=True)
    assert path == tmp_path / "Portra400.json"
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["Portra400.json"]


def test_rename_profile(tmp_path):
    save_named_profile(PROFILE, "old_name", profiles_dir=tmp_path)
    new_path = rename_profile("old_name", "new_name", profiles_dir=tmp_path)
    assert new_path == tmp_path / "new_name.json"
    assert not (tmp_path / "old_name.json").exists()
    assert load_profile(new_path).name == "new_name"


def test_rename_profile_raises_if_source_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        rename_profile("nonexistent", "new_name", profiles_dir=tmp_path)


def test_rename_profile_raises_if_target_exists(tmp_path):
    save_named_profile(PROFILE, "a", profiles_dir=tmp_path)
    save_named_profile(PROFILE, "b", profiles_dir=tmp_path)
    with pytest.raises(FileExistsError):
        rename_profile("a", "b", profiles_dir=tmp_path)


def test_rename_profile_raises_if_target_exists_case_insensitive(tmp_path):
    save_named_profile(PROFILE, "a", profiles_dir=tmp_path)
    save_named_profile(PROFILE, "B", profiles_dir=tmp_path)
    with pytest.raises(FileExistsError):
        rename_profile("a", "b", profiles_dir=tmp_path)


def test_delete_profile(tmp_path):
    save_named_profile(PROFILE, "throwaway", profiles_dir=tmp_path)
    delete_profile("throwaway", profiles_dir=tmp_path)
    assert not (tmp_path / "throwaway.json").exists()


def test_delete_profile_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_profile("nonexistent", profiles_dir=tmp_path)


def test_save_profile_with_tone_roundtrips(tmp_path):
    path = tmp_path / "with_tone.json"
    tone = ToneCurveParams(mode="paper", exposure=0.3, contrast=0.6)
    save_profile(PROFILE, path, tone=tone)
    loaded = load_profile(path)
    assert loaded.white_balance == PROFILE.white_balance  # tone doesn't leak into DensityProfile
    restored = load_tone_override(path)
    assert restored.exposure == 0.3
    assert restored.contrast == 0.6


def test_save_profile_without_tone_has_no_override(tmp_path):
    path = tmp_path / "no_tone.json"
    save_profile(PROFILE, path)
    assert load_tone_override(path) is None


def test_save_named_profile_with_tone(tmp_path):
    tone = ToneCurveParams(mode="paper", exposure=-0.5, contrast=0.4)
    path = save_named_profile(PROFILE, "with_tone", profiles_dir=tmp_path, tone=tone)
    restored = load_tone_override(path)
    assert restored.exposure == -0.5
    assert restored.contrast == 0.4


def test_load_tone_override_never_restores_linear_mode(tmp_path):
    # Even if a "tone" sidecar somehow had a mode field, load_tone_override should ignore it and
    # always return "paper" - linear-output is deliberately never inherited from a saved profile.
    path = tmp_path / "sneaky.json"
    save_profile(PROFILE, path, tone=ToneCurveParams(mode="linear", exposure=0.1, contrast=0.5))
    assert load_tone_override(path).mode == "paper"


def test_update_profile_sets_editable_fields(tmp_path):
    save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path)
    path = update_profile(
        "portra400", profiles_dir=tmp_path, film_stock="Kodak Portra 400", notes="anchored on IMG_0151"
    )
    updated = load_profile(path)
    assert updated.film_stock == "Kodak Portra 400"
    assert updated.notes == "anchored on IMG_0151"
    # untouched fields stay as they were
    assert updated.name == "portra400"
    assert updated.white_balance == PROFILE.white_balance


def test_update_profile_clears_a_field_with_none(tmp_path):
    save_named_profile(replace(PROFILE, notes="temporary"), "portra400", profiles_dir=tmp_path)
    path = update_profile("portra400", profiles_dir=tmp_path, notes=None)
    assert load_profile(path).notes is None


def test_update_profile_rejects_unknown_field(tmp_path):
    save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path)
    with pytest.raises(ValueError, match="white_balance"):
        update_profile("portra400", profiles_dir=tmp_path, white_balance=(1.0, 1.0, 1.0))


def test_update_profile_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_profile("nonexistent", profiles_dir=tmp_path, notes="x")


def test_update_profile_keeps_tone_and_scan_sidecars(tmp_path):
    tone = ToneCurveParams(mode="paper", exposure=0.2, contrast=0.7)
    scan = ScanSettings(exposure_time=1 / 50, f_number=8.0, iso=100)
    path = save_named_profile(PROFILE, "portra400", profiles_dir=tmp_path, tone=tone, scan=scan)
    update_profile("portra400", profiles_dir=tmp_path, film_stock="Kodak Portra 400")
    assert load_profile(path).film_stock == "Kodak Portra 400"
    assert load_tone_override(path).exposure == 0.2
    assert load_scan_reference(path) == scan


def test_interrupted_profile_save_keeps_old_profile(tmp_path, monkeypatch):
    """A save that dies partway through (disk full, kill -9) must not corrupt or truncate a
    profile that was already there (2.2-5)."""
    path = tmp_path / "portra400.json"
    save_profile(PROFILE, path)
    old_bytes = path.read_bytes()

    import json as json_module

    def _broken_dump(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(json_module, "dump", _broken_dump)

    with pytest.raises(RuntimeError, match="disk full"):
        save_profile(replace(PROFILE, notes="should not be saved"), path)

    assert path.read_bytes() == old_bytes  # the old profile survives untouched
    assert load_profile(path).notes is None
    assert list(tmp_path.iterdir()) == [path]  # no leftover temp file
