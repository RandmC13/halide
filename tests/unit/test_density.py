import pytest

from halide.core.density import (
    apply_density_balance,
    apply_white_balance,
    describe_cast,
    fit_density_balance,
    leave_one_out_residuals,
    neutral_residuals,
    solve_density_balance,
)

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


# --- multi-point fit -------------------------------------------------------------------------

# A film's neutral axis, in the form fit_density_balance models it: D_c = log10(wb_c) + D_G / s_c.
AXIS_SCALE = (1.12, 1.0, 0.78)
AXIS_WB = (0.73, 1.0, 1.05)


def _neutral_rgb(green_density, deviation=(0.0, 0.0, 0.0)):
    """Transmittance of a point on AXIS_* at `green_density`, optionally pushed off-axis by a
    per-channel negative density `deviation` (positive = more density in that channel)."""
    d = np.array([np.log10(AXIS_WB[c]) + green_density / AXIS_SCALE[c] for c in range(3)])
    d += np.asarray(deviation)
    return tuple(10.0 ** -d)


def test_fit_with_two_points_equals_solve_density_balance():
    fitted = fit_density_balance([SHADOW_RGB, HIGHLIGHT_RGB])
    solved = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    assert fitted.white_balance == pytest.approx(solved.white_balance, rel=1e-12)
    assert fitted.density_scale == pytest.approx(solved.density_scale, rel=1e-12)


def test_fit_is_order_independent_and_matches_reference_values():
    fitted = fit_density_balance([HIGHLIGHT_RGB, SHADOW_RGB])
    assert fitted.white_balance == pytest.approx(EXPECTED_WHITE_BALANCE)
    assert fitted.density_scale == pytest.approx(EXPECTED_DENSITY_SCALE)


def test_fit_recovers_a_known_axis_from_noisy_points():
    rng = np.random.default_rng(0)
    points = [
        _neutral_rgb(g, deviation=(rng.normal(0, 0.01), 0.0, rng.normal(0, 0.01)))
        for g in np.linspace(0.8, 1.6, 12)
    ]
    fitted = fit_density_balance(points)
    assert fitted.density_scale == pytest.approx(AXIS_SCALE, rel=0.03)
    assert fitted.white_balance == pytest.approx(AXIS_WB, rel=0.05)


def test_more_points_average_down_one_objects_error():
    # One non-neutral highlight (0.05 D too little blue density) among true neutrals: the fit lands
    # closer to the true axis than a two-point solve through that bad point does - but only ~2.4x
    # closer, not proportionally to the point count: a point at the end of the density range has the
    # most leverage on a fitted line. So averaging alone isn't the whole defence - see the next test.
    bad = _neutral_rgb(1.55, deviation=(0.0, 0.0, -0.05))
    good = [_neutral_rgb(g) for g in (0.8, 1.0, 1.2, 1.4, 1.6)]
    two_point_error = abs(solve_density_balance(good[0], bad).density_scale[2] - AXIS_SCALE[2])
    fit_error = abs(fit_density_balance(good + [bad]).density_scale[2] - AXIS_SCALE[2])
    assert fit_error < two_point_error / 2


def test_leave_one_out_exposes_a_bad_point_that_the_full_fit_hides():
    bad = _neutral_rgb(1.55, deviation=(0.0, 0.0, -0.05))  # truly 0.05 * 0.78 = CC 3.9 Y off
    good = [_neutral_rgb(g) for g in (0.8, 1.0, 1.2, 1.4, 1.6)]
    points = good + [bad]

    # Judged against the fit that includes it, the bad point looks closer than it is and the good
    # points take part of the blame...
    in_fit = [describe_cast(r)[0] for r in neutral_residuals(fit_density_balance(points), points)]
    assert in_fit[-1] < 3.0 and max(in_fit[:-1]) > 1.0

    # ...judged against the others, it shows its true size and direction and is the largest.
    loo = [describe_cast(r) for r in leave_one_out_residuals(points)]
    assert loo[-1][0] == pytest.approx(0.05 * AXIS_SCALE[2] * 100, rel=1e-6)
    assert loo[-1][1] == "Y"
    assert max(cc for cc, _ in loo[:-1]) < loo[-1][0]
    # Only one good point (D 1.6) shares its tone range, and without that point the bad one alone
    # defines the top of the line - so the neighbour reads as disagreeing in the *opposite*
    # direction. Two points disagreeing is the honest answer until a third object of similar tone
    # settles which one is off; points far from the disagreement stay small.
    assert loo[4][1] == "B" and loo[4][0] > 1.5
    assert max(cc for cc, _ in loo[:3]) < 1.0


def test_leave_one_out_is_undefined_below_three_points():
    loo = leave_one_out_residuals([SHADOW_RGB, HIGHLIGHT_RGB])
    assert np.isnan(loo).all()


def test_residuals_are_zero_for_points_on_the_fitted_axis():
    points = [_neutral_rgb(g) for g in (0.8, 1.2, 1.6)]
    residuals = neutral_residuals(fit_density_balance(points), points)
    assert np.abs(residuals).max() < 1e-9


def test_residuals_flag_an_off_axis_point_with_its_colour():
    # A cream wall: more red and less blue density than neutral on the negative -> prints warm.
    axis_points = [_neutral_rgb(g) for g in (0.8, 1.0, 1.3, 1.6)]
    wall = _neutral_rgb(1.15, deviation=(0.06, 0.0, -0.06))
    residuals = neutral_residuals(fit_density_balance(axis_points), axis_points + [wall])
    cc, direction = describe_cast(residuals[-1])
    # balanced deviation = scale * deviation: R +0.0672, B -0.0468 -> spread 0.114 -> CC 11.4
    assert cc == pytest.approx((0.06 * AXIS_SCALE[0] + 0.06 * AXIS_SCALE[2]) * 100, rel=1e-6)
    assert direction == "R"
    assert all(describe_cast(r)[0] < 1e-6 for r in residuals[:4])


@pytest.mark.parametrize(
    "deviation, expected",
    [
        ((0.10, -0.05, -0.05), (15.0, "R")),
        ((-0.05, 0.10, -0.05), (15.0, "G")),
        ((-0.05, -0.05, 0.10), (15.0, "B")),
        ((-0.10, 0.05, 0.05), (15.0, "C")),
        ((0.05, -0.10, 0.05), (15.0, "M")),
        ((0.05, 0.05, -0.10), (15.0, "Y")),
    ],
)
def test_describe_cast_names_filter_value_and_direction(deviation, expected):
    cc, direction = describe_cast(deviation)
    assert cc == pytest.approx(expected[0])
    assert direction == expected[1]


def test_fit_rejects_too_few_or_flat_points():
    with pytest.raises(ValueError):
        fit_density_balance([SHADOW_RGB])
    with pytest.raises(ValueError):
        fit_density_balance([(0.05, 0.05, 0.05), (0.06, 0.05, 0.04)])  # same green density
