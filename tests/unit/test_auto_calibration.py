import numpy as np
import pytest

from halide.calibration.auto import auto_density_balance, roll_auto_density_balance
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


def test_auto_density_balance_too_narrow_a_fraction_fails_on_bimodal_content():
    # Documents *why* the default is generous (0.5, not something smaller): a too-narrow fraction
    # can capture only one of the two genuinely-neutral clusters (they don't share identical
    # residual "saturation" after the simple per-channel-median normalization — see auto.py's
    # docstring), which here leaves only one real density level as a candidate.
    image = _synthetic_image_with_decoys()
    with pytest.raises(ValueError, match="must differ in density"):
        auto_density_balance(image, neutral_fraction=0.25)


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
