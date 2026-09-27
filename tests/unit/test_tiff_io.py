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


def test_read_tiff_into_out_buffer_returns_the_same_object_for_float32(tmp_path):
    path = tmp_path / "scan.tiff"
    original = np.random.default_rng(0).uniform(0.0, 1.0, size=(6, 5, 3)).astype(np.float32)
    write_tiff(path, original)

    out = np.empty(original.shape, dtype=np.float32)
    result = read_tiff(path, out=out)

    assert result.image is out
    np.testing.assert_array_equal(out, original)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint32],
)
def test_read_tiff_into_out_buffer_matches_read_tiff_without_out_bit_for_bit(tmp_path, dtype):
    """The B1 task brief's ruling: `out=` must reproduce today's `raw.astype(np.float32) / scale`
    exactly, including its rounding — not `np.divide(raw, scale, out=out)`, which computes from the
    integer input and can round differently in the last bit. Exercises values that stress that
    rounding: the dtype's full range for uint8/uint16, and values right at the top of uint32's range
    (2**32-1), where the cast to float32 loses the most precision."""
    import tifffile

    path = tmp_path / "scan.tiff"
    info = np.iinfo(dtype)
    if dtype == np.uint32:
        # The full range (4 billion values) isn't practical to enumerate; exercise the extreme end
        # instead, where a uint32->float32 cast rounds the most.
        rng = np.random.default_rng(0)
        top = rng.integers(info.max - 100000, info.max, size=(8, 8, 2), dtype=np.uint64).astype(dtype)
        edges = np.array([0, 1, info.max // 2, info.max - 1, info.max], dtype=dtype)
        raw = np.concatenate([top.reshape(-1), edges])
    else:
        # Every representable value of the dtype, once each.
        raw = np.arange(int(info.max) + 1, dtype=np.int64).astype(dtype)
    side = int(np.ceil(len(raw) ** 0.5))
    padded = np.zeros(side * side, dtype=dtype)
    padded[: len(raw)] = raw
    raw_image = np.repeat(padded.reshape(side, side, 1), 3, axis=2)
    tifffile.imwrite(path, raw_image)

    expected = read_tiff(path)
    out = np.empty(expected.image.shape, dtype=np.float32)
    result = read_tiff(path, out=out)

    assert result.image is out
    np.testing.assert_array_equal(out, expected.image)


def test_read_tiff_out_must_match_the_files_shape(tmp_path):
    path = tmp_path / "scan.tiff"
    write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))
    out = np.empty((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        read_tiff(path, out=out)


def test_read_tiff_out_must_be_float32(tmp_path):
    path = tmp_path / "scan.tiff"
    write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))
    out = np.empty((4, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        read_tiff(path, out=out)


def test_read_tiff_out_must_be_c_contiguous(tmp_path):
    path = tmp_path / "scan.tiff"
    write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))
    out = np.empty((4, 4, 6), dtype=np.float32)[:, :, ::2]  # shape (4, 4, 3), non-contiguous
    with pytest.raises(ValueError):
        read_tiff(path, out=out)
