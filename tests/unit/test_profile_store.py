import pytest

from halide.calibration.profile_store import (
    delete_profile,
    list_profiles,
    load_profile,
    rename_profile,
    resolve_profile_path,
    save_named_profile,
    save_profile,
)
from halide.core.types import DensityProfile

PROFILE = DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78))


def test_save_and_load_roundtrips(tmp_path):
    path = tmp_path / "portra400.json"
    save_profile(PROFILE, path)
    loaded = load_profile(path)
    assert loaded.white_balance == PROFILE.white_balance
    assert loaded.density_scale == PROFILE.density_scale


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
    assert [name for name, _ in results] == ["aaa", "zzz"]


def test_list_profiles_skips_unreadable_files(tmp_path, capsys):
    save_named_profile(PROFILE, "good", profiles_dir=tmp_path)
    (tmp_path / "corrupt.json").write_text("not valid json{{{")
    results = list_profiles(profiles_dir=tmp_path)
    assert [name for name, _ in results] == ["good"]
    assert "skipping unreadable" in capsys.readouterr().out


def test_list_profiles_on_nonexistent_directory_returns_empty(tmp_path):
    assert list_profiles(profiles_dir=tmp_path / "does_not_exist") == []


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


def test_delete_profile(tmp_path):
    save_named_profile(PROFILE, "throwaway", profiles_dir=tmp_path)
    delete_profile("throwaway", profiles_dir=tmp_path)
    assert not (tmp_path / "throwaway.json").exists()


def test_delete_profile_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_profile("nonexistent", profiles_dir=tmp_path)
