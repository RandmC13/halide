from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halide.core._xp import register_namespace
from halide.io.raster import encode_srgb, srgb_matrix, to_srgb_8bit, write_delivery_image
from tests.unit._fake_device import FakeDeviceArray, fake_xp, to_device, to_host


@pytest.fixture(autouse=True)
def _fake():
    register_namespace(FakeDeviceArray, fake_xp)


def test_to_srgb_8bit_shape_and_dtype():
    acescg = np.full((4, 4, 3), 0.18, dtype=np.float32)
    result = to_srgb_8bit(acescg)
    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8


def test_to_srgb_8bit_clips_out_of_range_values():
    # Very bright (highlight) and negative (out-of-gamut) values must clip into [0, 255], not
    # wrap or raise — this is the expected, deliberate final clip for a bounded delivery format.
    acescg = np.array([[[100.0, 100.0, 100.0], [-1.0, -1.0, -1.0]]], dtype=np.float32)
    result = to_srgb_8bit(acescg)
    assert result.min() >= 0
    assert result.max() <= 255


def test_to_srgb_8bit_neutral_gray_stays_neutral():
    acescg = np.full((2, 2, 3), 0.18, dtype=np.float32)
    result = to_srgb_8bit(acescg)
    assert result[..., 0] == pytest.approx(result[..., 1], abs=1)
    assert result[..., 2] == pytest.approx(result[..., 1], abs=1)


def test_write_delivery_image_png_roundtrips_and_embeds_icc(tmp_path):
    acescg = np.full((4, 4, 3), 0.18, dtype=np.float32)
    path = tmp_path / "out.png"
    write_delivery_image(path, acescg)

    assert path.exists()
    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.size == (4, 4)
        assert img.info.get("icc_profile") is not None


def test_write_delivery_image_jpeg(tmp_path):
    acescg = np.full((4, 4, 3), 0.18, dtype=np.float32)
    path = tmp_path / "out.jpg"
    write_delivery_image(path, acescg, quality=90)
    assert path.exists()
    with Image.open(path) as img:
        assert img.format == "JPEG"


def test_write_delivery_image_rejects_unsupported_extension(tmp_path):
    acescg = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported export format"):
        write_delivery_image(tmp_path / "out.gif", acescg)


# ---------------------------------------------------------------------------
# D1: srgb_matrix's BLAS matmul form vs colour-science's own per-pixel broadcast
# (see CLAUDE.md, "D1" — the user accepted "looks identical", not bit-identical).
# ---------------------------------------------------------------------------


def _to_srgb_8bit_with_colour(acescg_image: np.ndarray) -> np.ndarray:
    """Verbatim copy of to_srgb_8bit's body as of dff3cb2 — the D1 oracle."""
    import colour

    srgb_linear = colour.RGB_to_RGB(
        acescg_image,
        input_colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        output_colourspace=colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=True,
    )
    clipped = np.clip(srgb_linear, 0.0, 1.0)
    return (clipped * 255).round().astype(np.uint8)


def test_srgb_matrix_reproduces_colour_science():
    import colour

    expected = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES["ACEScg"], colour.RGB_COLOURSPACES["sRGB"], "Bradford"
    )
    np.testing.assert_allclose(srgb_matrix(), expected, rtol=1e-14, atol=0)


def test_to_srgb_8bit_matches_colour_science_on_random_values():
    n_pixels = 1_000_000
    n_values = n_pixels * 3
    rng = np.random.default_rng(7)
    # Includes values right at 8-bit rounding boundaries (k/255 +/- a hair) as well as out-of-range
    # highlight/negative values that must clip, per the module's own docstring.
    boundary = np.arange(256, dtype=np.float32) / 255.0
    boundary_jitter = np.concatenate([boundary - 1e-4, boundary, boundary + 1e-4]).astype(np.float32)
    random_vals = rng.uniform(-0.1, 1.5, n_values - boundary_jitter.size).astype(np.float32)
    values = np.concatenate([boundary_jitter, random_vals])
    rng.shuffle(values)
    acescg = values.reshape(n_pixels, 1, 3)

    actual = to_srgb_8bit(acescg)
    expected = _to_srgb_8bit_with_colour(acescg)
    n_diff = int(np.count_nonzero(actual != expected))
    assert n_diff == 0, f"{n_diff} of {actual.size} uint8 values differ from colour-science"


# ---------------------------------------------------------------------------
# Task 6: to_srgb_8bit / encode_srgb on the device (a strict fake here; the real GPU is checked in
# tests/gpu/test_gpu_parity.py against the D2 tolerance, not exact equality).
# ---------------------------------------------------------------------------


def _random_acescg(shape=(41, 7, 3), seed=11):
    rng = np.random.default_rng(seed)
    # Includes negative (out-of-gamut) and >1 (highlight) values, same reasoning as the
    # colour-science comparison above: encode_srgb's xp.where must pick the right branch for both.
    return rng.uniform(-0.2, 1.6, size=shape).astype(np.float32)


def test_encode_srgb_matches_colour_science_cctf_encoding():
    import colour

    linear = _random_acescg(shape=(1000,), seed=13)
    actual = encode_srgb(linear)
    expected = colour.RGB_COLOURSPACES["sRGB"].cctf_encoding(linear)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


def test_to_srgb_8bit_on_fake_device_matches_cpu_exactly():
    acescg = _random_acescg()
    cpu = to_srgb_8bit(acescg)
    gpu = to_host(to_srgb_8bit(to_device(acescg)))
    assert gpu.dtype == np.uint8 and gpu.shape == cpu.shape
    assert np.array_equal(gpu, cpu)


def test_to_srgb_8bit_on_fake_device_never_touches_numpy_directly():
    # The strict fake device (tests/unit/_fake_device.py) raises on any bare numpy leak — a stage
    # that quietly reached for colour.RGB_COLOURSPACES["sRGB"].cctf_encoding or plain np. functions
    # on a device array instead of encode_srgb/xp would fail here, not silently work.
    acescg = to_device(_random_acescg(shape=(3, 3, 3)))
    result = to_srgb_8bit(acescg)
    assert isinstance(result, FakeDeviceArray)


def test_to_srgb_8bit_on_fake_device_clips_out_of_range_values():
    acescg = to_device(np.array([[[100.0, 100.0, 100.0], [-1.0, -1.0, -1.0]]], dtype=np.float32))
    result = to_host(to_srgb_8bit(acescg))
    assert result.min() >= 0 and result.max() <= 255
