"""End-to-end tests for `halide profile` and the --save-profile-as / bare-name --profile flags,
isolated from the real default profiles directory via $XDG_CONFIG_HOME."""

import numpy as np
import pytest

from halide.calibration.profile_store import load_profile
from halide.cli.main import main
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


@pytest.fixture(autouse=True)
def isolated_profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "halide" / "profiles"


@pytest.fixture
def negative_tiff(tmp_path):
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def test_invert_save_profile_as_then_reuse_by_name(negative_tiff, tmp_path, isolated_profiles_dir):
    output1 = tmp_path / "positive1.tiff"
    exit_code = main(
        [
            "invert", str(negative_tiff), str(output1),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "portra400-c41-v600",
        ]
    )
    assert exit_code == 0
    assert (isolated_profiles_dir / "portra400-c41-v600.json").exists()
    saved = load_profile(isolated_profiles_dir / "portra400-c41-v600.json")
    assert saved.name == "portra400-c41-v600"
    assert saved.white_balance == pytest.approx((2.28, 1.0, 1.47))

    # Reuse by bare name, no path needed.
    output2 = tmp_path / "positive2.tiff"
    exit_code = main(["invert", str(negative_tiff), str(output2), "--profile", "portra400-c41-v600"])
    assert exit_code == 0
    assert output2.exists()


def test_save_profile_as_errors_with_per_frame_auto_density(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="needs a single resolved profile"):
        main(
            [
                "invert", str(negative_tiff), str(output),
                "--auto-density", "--save-profile-as", "should-fail",
            ]
        )


def test_save_profile_as_errors_with_invert_only(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="needs a single resolved profile"):
        main(["invert", str(negative_tiff), str(output), "--invert-only", "--save-profile-as", "should-fail"])


def test_profile_list_show_rename_delete(negative_tiff, tmp_path, isolated_profiles_dir, capsys):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "ektar100",
        ]
    )
    capsys.readouterr()  # discard invert's own output

    assert main(["profile", "list"]) == 0
    assert "ektar100" in capsys.readouterr().out

    assert main(["profile", "show", "ektar100"]) == 0
    show_output = capsys.readouterr().out
    assert "2.28" in show_output

    assert main(["profile", "rename", "ektar100", "ektar100-renamed"]) == 0
    capsys.readouterr()
    assert not (isolated_profiles_dir / "ektar100.json").exists()
    assert (isolated_profiles_dir / "ektar100-renamed.json").exists()

    assert main(["profile", "delete", "ektar100-renamed"]) == 0
    assert not (isolated_profiles_dir / "ektar100-renamed.json").exists()


def test_profile_list_empty(isolated_profiles_dir, capsys):
    assert main(["profile", "list"]) == 0
    assert "No saved profiles" in capsys.readouterr().out


def test_profile_show_nonexistent_errors():
    with pytest.raises(SystemExit, match="no such profile file"):
        main(["profile", "show", "nonexistent"])


def test_batch_save_profile_as_with_auto_density_roll(tmp_path, isolated_profiles_dir):
    rng = np.random.default_rng(0)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
        img[8:16, :] = HIGHLIGHT_RGB
        img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
        write_tiff(in_dir / f"f{i}.tiff", img, icc_profile=build_icc(LINEAR_TAGS))

    out_dir = tmp_path / "out"
    exit_code = main(
        ["batch", str(in_dir), str(out_dir), "--auto-density-roll", "--quiet", "--save-profile-as", "roll-cal"]
    )
    assert exit_code == 0
    assert (isolated_profiles_dir / "roll-cal.json").exists()
