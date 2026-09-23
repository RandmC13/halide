import numpy as np
import pytest

from halide.calibration.auto import (
    _density_local_saturation,
    _neutral_candidate_mask,
    _neutral_candidates,
    _saturation,
    auto_density_balance,
    roll_auto_density_balance,
)
from halide.core.density import solve_density_balance

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def _synthetic_image_with_decoys(n_neutral=1000, n_decoy=50):
    """A synthetic negative: mostly two genuinely scene-neutral patches (the same reference
    shadow/highlight values used elsewhere in the test suite) plus a minority of saturated
    "decoy" pixels far more extreme in raw transmittance than either real patch — exactly the
    situation that fooled the old script's raw-percentile approach."""
    shadow_patch = np.tile(SHADOW_RGB, (n_neutral, 1))
    highlight_patch = np.tile(HIGHLIGHT_RGB, (n_neutral, 1))
    decoy_low_red = np.tile([0.001, 0.5, 0.5], (n_decoy, 1))
    decoy_low_blue = np.tile([0.5, 0.5, 0.001], (n_decoy, 1))
    pixels = np.concatenate([shadow_patch, highlight_patch, decoy_low_red, decoy_low_blue], axis=0)
    return pixels.reshape(-1, 1, 3)


def test_naive_unfiltered_percentile_is_fooled_by_decoys():
    # Documents the actual bug being fixed: percentiles over *all* pixels, unfiltered, latch onto
    # the decoys rather than the real neutral patches.
    image = _synthetic_image_with_decoys()
    flat = image.reshape(-1, 3)
    naive_shadow = np.percentile(flat, 99.9, axis=0)
    naive_highlight = np.percentile(flat, 0.1, axis=0)
    assert not np.allclose(naive_shadow, SHADOW_RGB, atol=0.01)
    assert not np.allclose(naive_highlight, HIGHLIGHT_RGB, atol=0.01)


def test_auto_density_balance_recovers_true_profile_despite_decoys():
    image = _synthetic_image_with_decoys()
    expected = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    result = auto_density_balance(image)
    assert result.white_balance == pytest.approx(expected.white_balance, rel=1e-6)
    assert result.density_scale == pytest.approx(expected.density_scale, rel=1e-6)
    assert result.source == "auto"


def test_auto_density_balance_recovers_true_profile_even_with_a_narrow_fraction():
    # Used to fail here (a too-narrow fraction captured only one of the two genuinely-neutral
    # clusters, since they didn't share identical residual "saturation" against a single
    # frame-wide median). Fixed by _density_local_saturation's density-local reference — each cluster is
    # now judged against its own nearby density, not a frame-wide reference that could favor one
    # cluster over the other. Regression test for that fix: recovery now holds even far below the
    # generous DEFAULT_NEUTRAL_FRACTION default.
    image = _synthetic_image_with_decoys()
    expected = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    result = auto_density_balance(image, neutral_fraction=0.01)
    assert result.white_balance == pytest.approx(expected.white_balance, rel=1e-6)
    assert result.density_scale == pytest.approx(expected.density_scale, rel=1e-6)


def test_neutral_candidate_mask_matches_neutral_candidates():
    # The mask is the spatial counterpart to _neutral_candidates' flat array (used by the GUI
    # picker to overlay which regions the statistical method considers plausible) — it must select
    # exactly the same pixels, just without discarding their location.
    image = _synthetic_image_with_decoys()
    mask = _neutral_candidate_mask(image, neutral_fraction=0.5)
    candidates = _neutral_candidates(image, neutral_fraction=0.5)
    assert mask.shape == image.shape[:2]
    assert mask.sum() == len(candidates)


def test_auto_density_balance_raises_with_too_few_candidates():
    # Too few total pixels to ever produce 2 candidates, regardless of saturation — must fail
    # clearly rather than silently produce a bogus profile.
    image = np.array([[[0.05, 0.05, 0.05]]])  # a single pixel
    with pytest.raises(ValueError, match="not enough near-neutral pixels"):
        auto_density_balance(image, neutral_fraction=0.5)


def test_roll_auto_density_balance_combines_multiple_frames():
    # Split the same synthetic content across several "frames" — the roll-wide estimate should
    # match the single-image estimate almost exactly.
    single_image = _synthetic_image_with_decoys(n_neutral=1000, n_decoy=50)
    flat = single_image.reshape(-1, 3)
    frames = [flat[i::4].reshape(-1, 1, 3) for i in range(4)]

    expected = auto_density_balance(single_image)
    result = roll_auto_density_balance(frames)
    assert result.white_balance == pytest.approx(expected.white_balance, rel=1e-6)
    assert result.density_scale == pytest.approx(expected.density_scale, rel=1e-6)


def test_neutral_candidate_mask_is_not_fooled_by_film_toe_compression():
    # Real bug found via testing on a real scan (IMG_0151.tif): color negative film's toe (the
    # compressed underexposed end of its characteristic curve) makes near-black real-world content
    # of ANY hue converge toward nearly the same raw color, almost regardless of whether it's
    # genuinely neutral — confirmed on the real photo, a black traffic light (genuinely neutral)
    # and a head of dark hair (not neutral) had near-identical raw RGB. Against a single frame-wide
    # median (the old approach), that toe-converged majority sets the reference, and a properly-
    # exposed, genuinely neutral object at a different (moderate) density — the real photo's white
    # sign — scores as spuriously "saturated" and gets excluded, purely because it's being judged
    # against a reference from a completely different, unrelated density level. Mirrors the real
    # measured values (toe ~(0.29,0.17,0.09), moderate neutral ~(0.185,0.146,0.082)).
    TOE_RGB = (0.29, 0.17, 0.09)
    MODERATE_NEUTRAL_RGB = (0.185, 0.146, 0.082)
    toe = np.tile(TOE_RGB, (3000, 1))
    moderate = np.tile(MODERATE_NEUTRAL_RGB, (300, 1))
    image = np.concatenate([toe, moderate], axis=0).reshape(-1, 1, 3)

    # Document what the old single-global-median _saturation call would have scored: high enough
    # to be excluded by any reasonable neutral_fraction threshold, since the toe population (10x
    # more numerous) dominates the median and the moderate point sits far from it in ratio terms.
    old_style_saturation = _saturation(image.reshape(-1, 3), reference=None)
    assert old_style_saturation[3000] > 0.25  # the moderate-neutral point, judged against the (toe-dominated) global median

    # The actual (density-local) candidate mask must recover most of the moderate-neutral
    # population despite being a small minority overwhelmed by toe-converged content.
    mask = _neutral_candidate_mask(image, neutral_fraction=0.5).reshape(-1)
    assert mask[3000:].mean() > 0.5  # a healthy majority of the moderate-neutral pixels included


def test_auto_density_balance_warns_on_low_density_separation():
    # A cheap, narrow safety net (see _check_density_separation's docstring: it does NOT catch the
    # toe-compression bug above, only a near-degenerate near-zero-span pair) — verify it actually
    # fires rather than silently returning an unreliable profile.
    close_shadow = (0.10, 0.13, 0.05)
    close_highlight = (0.099, 0.129, 0.0498)  # deliberately almost identical density to close_shadow
    image = np.concatenate(
        [np.tile(close_shadow, (200, 1)), np.tile(close_highlight, (200, 1))], axis=0
    ).reshape(-1, 1, 3)
    with pytest.warns(UserWarning, match="unusually close in density"):
        auto_density_balance(image, neutral_fraction=0.9)


def test_check_density_separation_does_not_warn_on_the_reference_patches():
    # The existing, known-good reference shadow/highlight patches used throughout this test suite
    # must not trip the new warning — it's meant to catch genuinely degenerate cases, not flag
    # normal, legitimate calibration pairs.
    import warnings as warnings_module

    from halide.calibration.auto import _check_density_separation

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        _check_density_separation(np.array(SHADOW_RGB), np.array(HIGHLIGHT_RGB))


def test_roll_auto_density_balance_selects_neutral_candidates_per_frame_not_from_pooled_pixels(
    monkeypatch,
):
    # Real bug found via testing on two real scans (a warm-toned portrait and a daylight street
    # scene, --auto-density-roll): the original implementation concatenated raw pixels from every
    # frame *before* computing the per-channel median used to judge "how neutral is this pixel"
    # (see `_saturation`) — that median is only a valid proxy for the film's own systematic
    # per-channel imbalance when it's computed from one frame's own pixels; pooled across frames
    # with different scene content, it instead blends in each frame's *scene-color average*, which
    # is not shared across a roll the way the film base is. That skewed which pixels were selected
    # as "neutral" for whichever frame differed most from the blend — confirmed on the real photos
    # as a visible blue cast on a frame whose own calibration (per-frame --auto-density) was
    # correct. This pins the actual mechanism of the fix: candidate selection must happen once per
    # input image, on that image's own pixels, never on pixels pooled across images.
    calls = []
    from halide.calibration import auto as auto_module

    real_neutral_candidates = auto_module._neutral_candidates

    def spy(image, neutral_fraction):
        calls.append(image)
        return real_neutral_candidates(image, neutral_fraction)

    monkeypatch.setattr(auto_module, "_neutral_candidates", spy)

    frame_a = np.tile(SHADOW_RGB, (100, 1)).reshape(-1, 1, 3)
    frame_b = np.tile(HIGHLIGHT_RGB, (100, 1)).reshape(-1, 1, 3)
    auto_module.roll_auto_density_balance([frame_a, frame_b])

    assert len(calls) == 2  # called once per frame, never once on pooled/concatenated pixels
    assert calls[0] is frame_a
    assert calls[1] is frame_b


def _old_density_reference(pixels, target_bins=20, min_pixels_per_bin=200):
    """Verbatim copy of the pre-memory-diet `_density_reference` (a full per-pixel reference array
    built from an int64 np.arange split) — kept only as an oracle for the bin-by-bin rewrite."""
    n = len(pixels)
    n_bins = max(1, min(target_bins, n // min_pixels_per_bin))
    if n_bins == 1:
        return np.broadcast_to(np.median(pixels, axis=0), pixels.shape)
    order = np.argsort(pixels.mean(axis=1))
    reference = np.empty_like(pixels)
    for bin_positions in np.array_split(np.arange(n), n_bins):
        bin_indices = order[bin_positions]
        reference[bin_indices] = np.median(pixels[bin_indices], axis=0)
    return reference


def _old_neutral_candidate_mask(image, neutral_fraction):
    flat = image.reshape(-1, 3)
    saturation = _saturation(flat, _old_density_reference(flat))
    threshold = np.percentile(saturation, neutral_fraction * 100)
    return (saturation <= threshold).reshape(image.shape[:2])


def _noisy_frame(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.005, 0.4, size=shape).astype(dtype)
    image[: shape[0] // 3] = rng.choice([0.05, 0.1, 0.2], size=(shape[0] // 3, shape[1], 3)).astype(dtype)  # exact ties
    return image


@pytest.mark.parametrize(
    "image",
    [
        _noisy_frame((61, 97, 3), np.float32, 1),  # 5917 px -> 20 bins with a remainder
        _noisy_frame((41, 43, 3), np.float64, 2),  # 1763 px -> 8 bins
        _noisy_frame((9, 11, 3), np.float32, 3),  # 99 px -> the single-global-median path
        _synthetic_image_with_decoys(),
    ],
    ids=["float32-20-bins", "float64-8-bins", "one-bin", "decoys"],
)
@pytest.mark.parametrize("neutral_fraction", [0.01, 0.5, 0.9])
def test_neutral_candidate_mask_is_bit_identical_to_the_old_full_reference_form(image, neutral_fraction):
    # _density_local_saturation computes each bin's saturation directly instead of first building a
    # full-frame per-pixel reference array (~500 MiB of scratch on a real scan). Same bins, same
    # medians, same arithmetic — the saturation values must match exactly, not approximately, so
    # the selected candidates (and every auto-calibrated profile) are unchanged.
    flat = image.reshape(-1, 3)
    old = _saturation(flat, _old_density_reference(flat))
    new = _density_local_saturation(flat)
    assert new.dtype == old.dtype
    assert np.array_equal(new.view(np.uint8), np.ascontiguousarray(old).view(np.uint8))
    assert np.array_equal(_neutral_candidate_mask(image, neutral_fraction), _old_neutral_candidate_mask(image, neutral_fraction))
