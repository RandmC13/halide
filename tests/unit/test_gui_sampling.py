import numpy as np
import pytest

from halide.gui.sampling import (
    downsample_for_display,
    patch_chroma,
    preview_stretch,
    snap_to_representative_pixel,
)


def test_snap_to_representative_pixel_avoids_a_noise_outlier():
    image = np.full((30, 30, 3), 0.2, dtype=np.float32)
    image[15, 15] = (0.9, 0.9, 0.9)  # a stray bright dust-speck pixel exactly at the click point
    x, y, rgb = snap_to_representative_pixel(image, 15, 15, radius=5)
    assert rgb == pytest.approx((0.2, 0.2, 0.2))
    assert (x, y) != (15, 15)


def test_snap_to_representative_pixel_stays_put_on_uniform_patch():
    image = np.full((30, 30, 3), 0.4, dtype=np.float32)
    x, y, rgb = snap_to_representative_pixel(image, 15, 15, radius=5)
    assert rgb == pytest.approx((0.4, 0.4, 0.4))


def test_snap_to_representative_pixel_clamps_near_image_edge():
    image = np.full((10, 10, 3), 0.3, dtype=np.float32)
    x, y, rgb = snap_to_representative_pixel(image, 0, 0, radius=5)
    assert 0 <= x < 10 and 0 <= y < 10


def test_patch_chroma_zero_for_neutral():
    assert patch_chroma((0.3, 0.3, 0.3)) == pytest.approx(0.0)


def test_patch_chroma_matches_known_reference_values():
    # These are exactly the reference shadow/highlight patches used elsewhere in the test suite —
    # genuinely scene-neutral, but far from zero on this raw metric (the whole point of the caveat
    # documented in patch_chroma's docstring).
    assert patch_chroma((0.094, 0.131, 0.050)) == pytest.approx(0.618, abs=1e-3)
    assert patch_chroma((0.048, 0.054, 0.016)) == pytest.approx(0.704, abs=1e-3)


def test_downsample_for_display_keeps_small_images_unchanged():
    image = np.zeros((100, 200, 3), dtype=np.float32)
    result, stride = downsample_for_display(image, max_width=1400, max_height=1400)
    assert stride == 1
    assert result.shape == image.shape


def test_downsample_for_display_reduces_large_images():
    image = np.zeros((4000, 6000, 3), dtype=np.float32)
    result, stride = downsample_for_display(image, max_width=1400, max_height=1400)
    assert stride > 1
    assert max(result.shape[:2]) <= 1400


def test_downsample_for_display_constrains_height_independently_of_width():
    # The bug found via interactive testing: constraining only the longest side let a landscape
    # image's displayed *height* exceed a fixed-height container even though its width fit fine.
    image = np.zeros((3276, 4849, 3), dtype=np.float32)  # the real test scan's dimensions
    result, stride = downsample_for_display(image, max_width=950, max_height=580)
    assert result.shape[0] <= 580
    assert result.shape[1] <= 950


def test_preview_stretch_output_range():
    image = np.random.default_rng(0).uniform(0.0, 5.0, size=(10, 10, 3)).astype(np.float32)
    result = preview_stretch(image)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preview_stretch_handles_flat_image_without_crashing():
    image = np.full((5, 5, 3), 0.5, dtype=np.float32)
    result = preview_stretch(image)
    assert result.shape == image.shape
    assert np.all(np.isfinite(result))
