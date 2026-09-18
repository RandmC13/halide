import numpy as np
import pytest

from halide.gui.sampling import (
    apply_stretch,
    compute_stretch_bounds,
    display_to_full_res_coords,
    downsample_for_display,
    extract_magnifier_patch,
    full_res_to_display_coords,
    magnify_patch,
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


def test_preview_stretch_equals_compute_then_apply():
    # preview_stretch is a thin wrapper over the split-out pair — the split must not change
    # behavior, only let the hover magnifier reuse bounds computed once instead of per-patch.
    image = np.random.default_rng(1).uniform(0.0, 5.0, size=(10, 10, 3)).astype(np.float32)
    lo, hi = compute_stretch_bounds(image)
    assert apply_stretch(image, lo, hi) == pytest.approx(preview_stretch(image))


def test_apply_stretch_degenerate_bounds_returns_zeros():
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    result = apply_stretch(image, lo=0.5, hi=0.5)
    assert np.all(result == 0.0)


def test_display_to_full_res_coords_scales_by_stride():
    assert display_to_full_res_coords(10, 20, stride=4, image_width=1000, image_height=1000) == (40, 80)


def test_display_to_full_res_coords_clamps_to_image_bounds():
    # A display coordinate near the downsampled edge can map past the real image edge once
    # multiplied by stride (integer downsampling isn't exact) — must clamp, not go out of bounds.
    x, y = display_to_full_res_coords(999, 999, stride=4, image_width=100, image_height=50)
    assert x == 99 and y == 49


def test_full_res_to_display_coords_is_the_inverse_mapping():
    full_x, full_y = display_to_full_res_coords(10, 20, stride=4, image_width=1000, image_height=1000)
    assert full_res_to_display_coords(full_x, full_y, stride=4) == (10, 20)


def test_extract_magnifier_patch_shape_is_always_fixed_size():
    # The caller uploads this into a fixed-size texture on every hover event, so the shape must
    # never change — including right at an image corner, the most extreme clamping case.
    image = np.random.default_rng(2).uniform(size=(30, 30, 3)).astype(np.float32)
    for x, y in [(15, 15), (0, 0), (29, 29), (0, 29)]:
        patch = extract_magnifier_patch(image, x, y, radius=12)
        assert patch.shape == (25, 25, 3)


def test_extract_magnifier_patch_centers_on_the_requested_pixel():
    image = np.zeros((30, 30, 3), dtype=np.float32)
    image[15, 15] = (1.0, 1.0, 1.0)
    patch = extract_magnifier_patch(image, 15, 15, radius=5)
    assert patch.shape == (11, 11, 3)
    assert patch[5, 5] == pytest.approx((1.0, 1.0, 1.0))


def test_extract_magnifier_patch_edge_padding_repeats_the_edge_pixel():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    image[0, :] = (1.0, 1.0, 1.0)  # top row is distinct
    patch = extract_magnifier_patch(image, 0, 0, radius=3)
    assert patch.shape == (7, 7, 3)
    # Rows above the real top edge are padded by repeating the top row, not zero-filled.
    assert patch[0, 3] == pytest.approx((1.0, 1.0, 1.0))


def test_magnify_patch_scales_shape_by_zoom():
    patch = np.zeros((5, 5, 3), dtype=np.float32)
    result = magnify_patch(patch, zoom=8)
    assert result.shape == (40, 40, 3)


def test_magnify_patch_is_blocky_nearest_neighbor_not_interpolated():
    patch = np.zeros((2, 2, 3), dtype=np.float32)
    patch[0, 0] = (1.0, 0.0, 0.0)
    patch[0, 1] = (0.0, 1.0, 0.0)
    result = magnify_patch(patch, zoom=4)
    # A 4x4 block of exact source-pixel copies, not any blended/averaged value in between.
    assert result[0:4, 0:4] == pytest.approx(np.tile((1.0, 0.0, 0.0), (4, 4, 1)))
    assert result[0:4, 4:8] == pytest.approx(np.tile((0.0, 1.0, 0.0), (4, 4, 1)))
