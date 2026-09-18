import pytest

from halide.core.density import apply_density_balance, apply_white_balance, solve_density_balance

import numpy as np


# Ground truth from abpy/color-neg-resources' own `density levels.py` reference script, run
# verbatim with its built-in sample values (see tests/golden/luts/LICENSE-abpy.txt for license).
SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)
EXPECTED_WHITE_BALANCE = (2.277304130526006, 1.0, 1.4658041418862766)
EXPECTED_DENSITY_SCALE = (1.318585760497101, 1.0, 0.7777660280302214)


def test_solve_density_balance_matches_reference_script():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    assert profile.white_balance == pytest.approx(EXPECTED_WHITE_BALANCE)
    assert profile.density_scale == pytest.approx(EXPECTED_DENSITY_SCALE)


def test_green_channel_is_always_the_fixed_reference():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    assert profile.white_balance[1] == pytest.approx(1.0)
    assert profile.density_scale[1] == pytest.approx(1.0)


def test_solved_profile_makes_both_calibration_points_neutral():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    for patch in (SHADOW_RGB, HIGHLIGHT_RGB):
        img = np.asarray(patch, dtype=np.float64).reshape(1, 1, 3)
        corrected = apply_density_balance(apply_white_balance(img, profile), profile)
        r, g, b = corrected[0, 0]
        assert r == pytest.approx(g, rel=1e-6)
        assert b == pytest.approx(g, rel=1e-6)


def test_solve_density_balance_rejects_equal_density_patches():
    with pytest.raises(ValueError):
        solve_density_balance((0.05, 0.05, 0.05), (0.05, 0.05, 0.05))


def test_apply_white_balance_is_per_channel_multiply():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    img = np.ones((2, 2, 3), dtype=np.float64) * 0.2
    result = apply_white_balance(img, profile)
    assert result[0, 0, 0] == pytest.approx(0.2 * profile.white_balance[0])
    assert result[0, 0, 1] == pytest.approx(0.2 * profile.white_balance[1])
    assert result[0, 0, 2] == pytest.approx(0.2 * profile.white_balance[2])


def test_apply_density_balance_green_is_noop():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    img = np.full((2, 2, 3), 0.3, dtype=np.float64)
    result = apply_density_balance(img, profile)
    assert result[0, 0, 1] == pytest.approx(0.3)
