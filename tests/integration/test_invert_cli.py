"""End-to-end tests for `halide invert`, driving the real CLI entry point against synthetic
scans (same ICC-profile fixtures used by the ICC unit tests)."""

import numpy as np
import pytest

from halide.calibration.profile_store import save_profile
from halide.cli.main import main
from halide.core.types import DensityProfile
from halide.io.tiff import read_tiff, write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


@pytest.fixture
def negative_tiff(tmp_path):
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def test_invert_with_manual_calibration_produces_valid_output(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    exit_code = main(
        ["invert", str(negative_tiff), str(output), "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]
    )
    assert exit_code == 0
    result = read_tiff(output)
    assert np.all(np.isfinite(result.image))
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0 + 1e-6
    assert result.icc_profile is not None  # output ACEScg profile must be embedded


def test_invert_with_saved_profile(negative_tiff, tmp_path):
    profile_path = tmp_path / "profile.json"
    save_profile(
        DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78)), profile_path
    )
    output = tmp_path / "positive.tiff"
    exit_code = main(["invert", str(negative_tiff), str(output), "--profile", str(profile_path)])
    assert exit_code == 0


def test_linear_output_is_viewable_not_solid_white(negative_tiff, tmp_path):
    # Regression test for a real bug: the bare unbounded invert() output (typically tens in linear
    # value) is unusable directly — almost every pixel lands above 1.0 and a standard viewer (e.g.
    # darktable) shows solid white. --output flat must scale into a viewable range.
    output = tmp_path / "positive_linear.tiff"
    main(
        [
            "invert", str(negative_tiff), str(output),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--output", "flat",
        ]
    )
    result = read_tiff(output)
    assert result.image.max() <= 1.0 + 1e-6
    assert (result.image >= 1.0).mean() < 0.01


def test_invert_with_auto_density(tmp_path):
    # A roughly-balanced split between the two reference patches with a little per-pixel noise
    # (avoiding an exact bimodal tie, which is a synthetic-fixture artifact — see
    # tests/unit/test_auto_calibration.py for the detailed behavior of the neutral_fraction
    # heuristic) — this test only smoke-tests that --auto-density is wired up correctly end to end.
    rng = np.random.default_rng(0)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[9:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    path = tmp_path / "negative_balanced.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))

    output = tmp_path / "positive_auto.tiff"
    exit_code = main(["invert", str(path), str(output), "--auto-density"])
    assert exit_code == 0
    result = read_tiff(output)
    assert np.all(np.isfinite(result.image))


def test_invert_only_skips_density_balance(negative_tiff, tmp_path):
    output = tmp_path / "positive_invert_only.tiff"
    exit_code = main(["invert", str(negative_tiff), str(output), "--invert-only"])
    assert exit_code == 0


def test_missing_calibration_source_errors(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="calibration source"):
        main(["invert", str(negative_tiff), str(output)])


def test_missing_calibration_source_error_mentions_calibrate_and_pick(negative_tiff, tmp_path):
    # Real workflow gap found via use: nothing pointed a user toward `halide calibrate` or --pick
    # when they hit this error cold — the fix is specifically in the wording, so pin it.
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match=r"halide calibrate --save-profile-as NAME") as exc_info:
        main(["invert", str(negative_tiff), str(output)])
    assert "--pick" in str(exc_info.value)


def test_pick_uses_the_interactively_chosen_profile(negative_tiff, tmp_path, monkeypatch):
    picked = DensityProfile(white_balance=(1.5, 1.0, 0.7), density_scale=(1.1, 1.0, 0.9), source="anchor")
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: (picked, None))

    output = tmp_path / "positive.tiff"
    exit_code = main(["invert", str(negative_tiff), str(output), "--pick"])
    assert exit_code == 0
    result = read_tiff(output)
    assert np.all(np.isfinite(result.image))


def test_pick_cancelled_without_choosing_errors(negative_tiff, tmp_path, monkeypatch):
    # Closing the picker window without clicking "Use these values" returns None from
    # run_quick_pick — must fail clearly, not silently proceed with no calibration.
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: None)

    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="no calibration picked"):
        main(["invert", str(negative_tiff), str(output), "--pick"])


def test_pick_conflicts_with_other_calibration_sources(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: None)
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(["invert", str(negative_tiff), str(output), "--pick", "--auto-density"])


def test_profile_and_manual_override_conflict_errors(negative_tiff, tmp_path):
    profile_path = tmp_path / "profile.json"
    save_profile(DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0)), profile_path)
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(["invert", str(negative_tiff), str(output), "--profile", str(profile_path), "--rm", "2.0"])


def test_missing_input_file_errors_before_calibration_check(tmp_path):
    # Regression test: calibration-source resolution used to run before the input path was ever
    # checked, so a mistyped/missing input file reported "needs a calibration source" instead of
    # the actually-wrong thing. Deliberately passes no calibration flags to prove input-file
    # validation now happens first.
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="input file not found"):
        main(["invert", str(tmp_path / "does-not-exist.tiff"), str(output)])


def test_profile_typo_suggests_closest_match(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from halide.calibration.profile_store import save_named_profile
    from halide.core.types import DensityProfile

    save_named_profile(DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78)), "portra400")

    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="did you mean 'portra400'"):
        main(["invert", str(negative_tiff), str(output), "--profile", "portra40"])


def test_rejects_scan_with_no_embedded_icc_profile(tmp_path):
    path = tmp_path / "no_icc.tiff"
    write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))  # no icc_profile
    output = tmp_path / "out.tiff"
    # --invert-only needs no calibration source, so this isolates the ICC check specifically.
    with pytest.raises(SystemExit, match="no embedded ICC profile"):
        main(["invert", str(path), str(output), "--invert-only"])
