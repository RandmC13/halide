from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halide.io.raster import to_srgb_8bit, write_delivery_image


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
