import numpy as np
import pytest

from halide.core.invert import invert
from halide.core.tone_render import (
    _DEFAULT_CURVE_PATH,
    _load_curve,
    estimate_exposure,
    estimate_linear_scale,
    linear_passthrough,
    tone_render,
)
from halide.core.types import ToneCurveParams


def test_linear_passthrough_itself_is_identity():
    # The scaling happens in tone_render before calling this — the function itself stays a pure
    # passthrough, so it's still meaningfully named and testable in isolation.
    x = np.array([0.1, 1.0, 50.0, 1000.0])
    assert np.array_equal(linear_passthrough(x), x)


def test_linear_mode_is_a_pure_proportional_scale_not_a_curve():
    # Unlike "paper" mode, linear mode must not reshape the tonal relationship between pixels —
    # only rescale by a single constant, so ratios between values are preserved exactly.
    x = np.array([0.1, 1.0, 50.0, 1000.0])
    result = tone_render(x, ToneCurveParams(mode="linear"))
    ratios = result / x
    assert np.all(np.isclose(ratios, ratios[0]))


def test_linear_mode_brings_typical_scan_values_into_a_viewable_range():
    # The bug this fixes: a real scan's bare invert() output is typically in the tens (e.g. ~10 for
    # a density-balanced transmittance around 10%), so a naive identity passthrough put ~100% of
    # pixels above 1.0 — solid white in any standard viewer. Simulate that with a realistic spread.
    x = np.geomspace(0.5, 150.0, num=1000)  # spans a realistic scan's dynamic range
    result = tone_render(x, ToneCurveParams(mode="linear"))
    assert result.max() <= 1.0 + 1e-9
    assert (result <= 1.0).mean() > 0.99  # only the thin extreme tail may still exceed 1.0


def test_estimate_linear_scale_positions_highlight_percentile_at_target():
    x = np.geomspace(0.5, 150.0, num=1000)
    scale = estimate_linear_scale(x, highlight_percentile=99.9, target_value=0.8)
    positioned = np.percentile(x, 99.9) * scale
    assert positioned == pytest.approx(0.8, abs=1e-6)


def test_estimate_linear_scale_is_robust_to_a_single_extreme_outlier():
    # A single pathological pixel (e.g. the MIN_TRANSMITTANCE floor clamp on a near-zero pixel)
    # must not crush the rest of the image to near-black to accommodate it. A realistically large
    # sample (real images are ~10M+ pixels) so one outlier is a comparably tiny fraction of the
    # data, matching what was observed on the real test scans (a single 1e7 pixel out of ~15.8M).
    x = np.concatenate([np.geomspace(0.5, 100.0, num=99_999), [1e7]])
    scale = estimate_linear_scale(x)
    typical_pixel_after_scale = 10.0 * scale  # a representative mid-range value
    assert typical_pixel_after_scale > 0.01  # nowhere near crushed to black


def test_paper_mode_is_monotonic_increasing_and_bounded():
    # Positive-linear values spanning a very dense to very thin negative.
    x = np.geomspace(1e-2, 1e4, num=200)
    result = tone_render(x, ToneCurveParams())
    # A small negative tolerance absorbs float noise on the curve's flat toe/shoulder plateaus.
    assert np.all(np.diff(result) >= -1e-9)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0 + 1e-9)


def test_exposure_parameter_can_recover_a_saturated_point():
    # The old script had a single hardcoded constant (0.01) baked into the inversion itself, with
    # no way to adapt to a particular frame's density range — a clipped highlight was unfixable
    # short of editing the source. `exposure` is exactly that missing lever: repositioning where a
    # frame's density range falls on the curve, the way choosing enlarger exposure time does in a
    # real darkroom. The same input that saturates under one exposure must be recoverable under
    # another — proving the fix is a real, usable control, not just a differently-shaped constant.
    x = np.array([0.01])
    saturated = tone_render(invert(x), ToneCurveParams(exposure=0.43, contrast=1.0))
    recovered = tone_render(invert(x), ToneCurveParams(exposure=-0.5, contrast=1.0))
    assert saturated[0] >= 0.999
    assert recovered[0] < 0.999


def test_matches_vendored_curve_at_sample_density_points_at_full_contrast():
    # contrast=1.0 reproduces the untouched reference curve exactly (the default, <1.0, deliberately
    # does not — see ToneCurveParams' docstring for why the default is softer).
    curve = _load_curve(str(_DEFAULT_CURVE_PATH))
    exposure = 0.43
    densities = np.linspace(curve.domain_min + 0.1, curve.domain_max - 0.1, num=25)
    positive_linear = 10.0 ** (densities - exposure)
    result = tone_render(positive_linear, ToneCurveParams(exposure=exposure, contrast=1.0))
    expected = curve.lookup(densities)
    assert result == pytest.approx(expected, rel=1e-6)


def test_contrast_below_one_compresses_the_response_toward_the_pivot():
    curve = _load_curve(str(_DEFAULT_CURVE_PATH))
    pivot_density = (curve.domain_min + curve.domain_max) / 2.0
    pivot_value = curve.lookup(np.array([pivot_density]))[0]

    # positive_linear values chosen so density = log10(x) lands at 0.5 and 2.5 (exposure=0) —
    # inside the curve's domain at full contrast, so compression toward the pivot is observable
    # rather than both ends already being clamped to the domain edge regardless of contrast.
    x = 1.0 / np.array([10.0**0.5, 10.0**2.5])
    full = tone_render(invert(x), ToneCurveParams(exposure=0.0, contrast=1.0))
    softer = tone_render(invert(x), ToneCurveParams(exposure=0.0, contrast=0.3))

    # Softer contrast must pull both points closer to the pivot's own output value.
    assert abs(softer[0] - pivot_value) < abs(full[0] - pivot_value)
    assert abs(softer[1] - pivot_value) < abs(full[1] - pivot_value)


def test_contrast_zero_is_a_flat_constant_at_the_pivot():
    curve = _load_curve(str(_DEFAULT_CURVE_PATH))
    pivot_density = (curve.domain_min + curve.domain_max) / 2.0
    pivot_value = curve.lookup(np.array([pivot_density]))[0]

    x = np.geomspace(1e-3, 1e3, num=20)
    result = tone_render(invert(x), ToneCurveParams(exposure=0.0, contrast=0.0))
    assert result == pytest.approx(np.full_like(result, pivot_value), abs=1e-9)


def test_estimate_exposure_positions_the_shadow_percentile_at_the_target_density():
    # A synthetic "positive" whose 1st percentile density is known exactly (density = log10(x)).
    rng = np.random.default_rng(0)
    shadow_density = 0.6
    bulk = 10.0 ** rng.uniform(shadow_density + 0.5, shadow_density + 2.0, size=9700)
    # A 3% flat tail safely straddles the 1st percentile regardless of interpolation method.
    shadow_tail = np.full(300, 10.0**shadow_density)
    positive = np.concatenate([shadow_tail, bulk])

    exposure = estimate_exposure(positive, shadow_percentile=1.0, target_density=1.0)
    density_after = np.log10(np.percentile(positive, 1.0)) + exposure
    assert density_after == pytest.approx(1.0, abs=1e-6)


def test_estimate_exposure_differs_per_image_rather_than_using_one_constant():
    # Two images with genuinely different density ranges must get genuinely different exposures —
    # this is the entire point (see ToneCurveParams' docstring: a fixed constant does not correctly
    # position every scan).
    dense_image = np.full((10, 10), 10.0**0.5)  # a "denser" (lower-transmittance) shadow
    thin_image = np.full((10, 10), 10.0**1.5)  # a "thinner" (higher-transmittance) shadow
    assert estimate_exposure(dense_image) != pytest.approx(estimate_exposure(thin_image))


def test_tone_render_defaults_to_auto_exposure_when_none_given():
    x = np.geomspace(1e-2, 1e2, num=50)
    positive = invert(x)
    auto_result = tone_render(positive, ToneCurveParams())  # exposure=None by default
    explicit_result = tone_render(positive, ToneCurveParams(exposure=estimate_exposure(positive)))
    assert auto_result == pytest.approx(explicit_result)
