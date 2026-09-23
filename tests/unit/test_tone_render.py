import numpy as np
import pytest

from halide.core.invert import invert
from halide.core.tone_render import (
    _DEFAULT_CURVE_PATH,
    MAX_PRINT_CONTRAST,
    _load_curve,
    estimate_linear_scale,
    fit_print,
    linear_passthrough,
    negative_density_range,
    paper_exposure_range,
    resolve_tone,
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


def _curve():
    return _load_curve(str(_DEFAULT_CURVE_PATH))


def _synthetic_positive(d_lo: float, d_hi: float, n: int = 20_000) -> np.ndarray:
    """A neutral RGB "positive" whose luminance density (log10) spans exactly [d_lo, d_hi] — so
    its 0.1/99.9th percentiles sit a hair inside those, at known values."""
    density = np.linspace(d_lo, d_hi, n)
    return np.repeat((10.0**density)[:, None], 3, axis=1)


def test_paper_exposure_range_hits_the_iso_6846_densities_on_the_curve():
    # ISO 6846: highlight point = 0.04 above paper white, shadow point = 90% of D-max.
    curve = _curve()
    shadow, highlight = paper_exposure_range(curve)
    d_max = -np.log10(curve.values.min())
    assert -np.log10(curve.lookup(np.array([highlight]))[0]) == pytest.approx(0.04, abs=1e-6)
    assert -np.log10(curve.lookup(np.array([shadow]))[0]) == pytest.approx(0.9 * d_max, abs=1e-6)
    assert shadow < highlight


def test_fit_print_fills_the_paper_range_from_the_negatives_own_range():
    # A normal-contrast negative (range ~0.95, like the real test scans): the fitted grade must put
    # its robust highlight on the paper's highlight point and its robust shadow on the shadow point.
    curve = _curve()
    positive = _synthetic_positive(0.7, 1.65)
    exposure, contrast = fit_print(positive, curve)
    shadow, highlight = paper_exposure_range(curve)
    assert contrast < MAX_PRINT_CONTRAST  # this range is wide enough not to hit the cap

    d_lo, d_hi = negative_density_range(positive)
    rendered = tone_render(np.array([10.0**d_lo, 10.0**d_hi]), ToneCurveParams(exposure=exposure, contrast=contrast))
    expected = curve.lookup(np.array([shadow, highlight]))
    assert rendered == pytest.approx(expected, rel=1e-5)


def test_fit_print_caps_the_grade_at_the_real_paper_for_a_low_contrast_negative():
    # A foggy/overcast negative: filling the paper would need a harder grade than the real paper
    # has. It must print soft (as it really was) — highlights still placed, shadows NOT forced to black.
    curve = _curve()
    positive = _synthetic_positive(1.0, 1.3)
    exposure, contrast = fit_print(positive, curve)
    assert contrast == MAX_PRINT_CONTRAST
    shadow, highlight = paper_exposure_range(curve)
    d_lo, d_hi = negative_density_range(positive)
    rendered = tone_render(np.array([10.0**d_lo, 10.0**d_hi]), ToneCurveParams(exposure=exposure, contrast=contrast))
    assert rendered[1] == pytest.approx(curve.lookup(np.array([highlight]))[0], rel=1e-5)
    assert rendered[0] > curve.lookup(np.array([shadow]))[0] * 10  # well above paper black


def test_fitted_print_keeps_neutral_pixels_exactly_neutral():
    # The fit is two scalars applied identically to every channel before a channel-identical curve:
    # it can decide where on the paper the image sits, never what colour anything is.
    rng = np.random.default_rng(1)
    positive = 10.0 ** rng.uniform(0.6, 1.7, size=(64, 64, 1)).repeat(3, axis=2)
    result = tone_render(positive, ToneCurveParams())
    assert np.array_equal(result[..., 0], result[..., 1])
    assert np.array_equal(result[..., 1], result[..., 2])


def test_fitted_print_is_invariant_to_a_global_exposure_multiply():
    # What makes `halide print` on an exposure-adjusted flat positive well-defined: a multiply is a
    # constant density offset, which the fitted exposure absorbs exactly.
    rng = np.random.default_rng(2)
    positive = 10.0 ** rng.uniform(0.6, 1.7, size=(64, 64, 3))
    reference = tone_render(positive, ToneCurveParams())
    for factor in (0.013, 0.8, 37.0):
        assert tone_render(positive * factor, ToneCurveParams()) == pytest.approx(reference, rel=1e-6, abs=1e-9)


def test_pinned_contrast_fits_only_exposure_to_the_highlight_point():
    curve = _curve()
    positive = _synthetic_positive(0.7, 1.65)
    exposure, contrast = fit_print(positive, curve, contrast=0.5)
    assert contrast == 0.5
    _, highlight = paper_exposure_range(curve)
    _, d_hi = negative_density_range(positive)
    rendered = tone_render(np.array([10.0**d_hi]), ToneCurveParams(exposure=exposure, contrast=0.5))
    assert rendered[0] == pytest.approx(curve.lookup(np.array([highlight]))[0], rel=1e-5)


def test_fit_differs_per_image_rather_than_using_one_constant():
    curve = _curve()
    soft = fit_print(_synthetic_positive(0.7, 1.9), curve)
    hard = fit_print(_synthetic_positive(0.7, 1.5), curve)
    assert soft[1] < hard[1]  # the wider-range negative gets the softer grade


def test_resolve_tone_reports_what_tone_render_uses():
    rng = np.random.default_rng(3)
    positive = 10.0 ** rng.uniform(0.6, 1.7, size=(32, 32, 3))
    resolved = resolve_tone(positive, ToneCurveParams())
    explicit = tone_render(positive, ToneCurveParams(exposure=resolved.exposure, contrast=resolved.contrast))
    assert tone_render(positive, ToneCurveParams()) == pytest.approx(explicit)

    pinned = resolve_tone(positive, ToneCurveParams(exposure=0.3, contrast=0.7))
    assert (pinned.exposure, pinned.contrast) == (0.3, 0.7)

    linear = resolve_tone(positive, ToneCurveParams(mode="linear"))
    assert linear.linear_scale == pytest.approx(estimate_linear_scale(positive))
