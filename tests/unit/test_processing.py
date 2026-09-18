import numpy as np
import pytest

from halide.io.tiff import write_tiff
from halide.processing import ScanColorError, estimate_roll_density_profile
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def _write_negative(path, seed):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


def test_estimate_roll_density_profile_skips_unreadable_files(tmp_path, capsys):
    good_paths = []
    for i in range(3):
        path = tmp_path / f"good_{i}.tiff"
        _write_negative(path, seed=i)
        good_paths.append(path)

    bad_path = tmp_path / "bad.tiff"
    write_tiff(bad_path, np.zeros((16, 16, 3), dtype=np.float32))  # no ICC profile

    # Must not raise, despite the unreadable file among otherwise-good ones.
    profile = estimate_roll_density_profile(good_paths + [bad_path])
    assert profile.source == "auto"
    assert "skipping" in capsys.readouterr().out


def test_estimate_roll_density_profile_raises_when_all_files_unreadable(tmp_path):
    bad_paths = []
    for i in range(2):
        path = tmp_path / f"bad_{i}.tiff"
        write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))
        bad_paths.append(path)

    with pytest.raises(ScanColorError, match="no readable frames"):
        estimate_roll_density_profile(bad_paths)
