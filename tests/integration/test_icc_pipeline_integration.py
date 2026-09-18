"""End-to-end: a synthetic TIFF with a real embedded linear ICC profile, read, color-managed, and
run through the full inversion pipeline — proving io.tiff + io.icc + core.pipeline compose
correctly, ahead of the CLI (Phase 3) that will wire them together for real usage."""

import numpy as np

from halide.core.density import solve_density_balance
from halide.core.pipeline import run_pipeline
from halide.io.icc import convert_to_working_space, parse_linear_rgb_profile
from halide.io.tiff import read_tiff, write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def test_synthetic_scan_with_embedded_icc_through_full_pipeline(tmp_path):
    negative = np.array([[SHADOW_RGB, HIGHLIGHT_RGB]], dtype=np.float32)
    icc_bytes = build_icc(LINEAR_TAGS)

    path = tmp_path / "negative.tiff"
    write_tiff(path, negative, icc_profile=icc_bytes)

    scan = read_tiff(path)
    assert scan.icc_profile == icc_bytes

    profile = parse_linear_rgb_profile(scan.icc_profile)
    working_space_image = convert_to_working_space(scan.image, profile)

    # Calibration points must be re-derived in the *working* color space, not the source space —
    # solving density balance against un-converted values would silently miscalibrate.
    shadow_working = working_space_image[0, 0]
    highlight_working = working_space_image[0, 1]
    density_profile = solve_density_balance(tuple(shadow_working), tuple(highlight_working))

    result = run_pipeline(working_space_image, density_profile)

    assert result.shape == negative.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0) and np.all(result <= 1.0 + 1e-6)
