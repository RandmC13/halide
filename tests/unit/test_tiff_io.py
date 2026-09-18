import numpy as np
import pytest

from halide.io.tiff import read_tiff, write_tiff


def test_write_then_read_roundtrips_float_values(tmp_path):
    path = tmp_path / "scan.tiff"
    original = np.random.default_rng(0).uniform(0.0, 1.0, size=(8, 8, 3)).astype(np.float32)
    write_tiff(path, original)
    result = read_tiff(path)
    assert result.image == pytest.approx(original, abs=1e-6)
    assert result.icc_profile is None


def test_write_then_read_roundtrips_icc_profile(tmp_path):
    path = tmp_path / "scan.tiff"
    fake_icc = b"not a real profile, just bytes to round-trip" + b"\x00" * 4
    write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32), icc_profile=fake_icc)
    result = read_tiff(path)
    assert result.icc_profile == fake_icc


@pytest.mark.parametrize(
    ("dtype", "max_value"),
    [(np.uint8, 255), (np.uint16, 65535)],
)
def test_read_normalizes_integer_dtypes_to_unit_range(tmp_path, dtype, max_value):
    import tifffile

    path = tmp_path / "scan.tiff"
    raw = np.array([[[0, max_value // 2, max_value]]], dtype=dtype)
    tifffile.imwrite(path, raw)
    result = read_tiff(path)
    assert result.image.min() == pytest.approx(0.0, abs=1e-6)
    assert result.image.max() == pytest.approx(1.0, abs=1e-4)
