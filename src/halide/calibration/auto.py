"""Zero-effort, statistical density-balance estimation — the fallback tier for users without a
ColorChecker or a hand-picked anchor frame.

The old script's equivalent (`auto_density_balance`/`roll_analysis_density_balance`) took
percentiles of *every* pixel and assumed the single darkest/brightest pixel was neutral gray. The
darkest pixel in a color negative is far more likely to be a saturated shadow color than a neutral
one — that mismatch between what the statistic measures and what the math needs is a real
methodological gap, not just noise, and it is why per-frame results varied. This module instead
restricts candidate "neutral" pixels to the least-saturated fraction of the image *first*, then
takes percentiles only within that filtered set — a materially different, better-justified
algorithm, not just a tidied-up version of the old one.
"""

from __future__ import annotations

import warnings

import numpy as np

from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.density import solve_density_balance
from halide.core.types import DensityProfile

DEFAULT_NEUTRAL_FRACTION = 0.5  # fraction of least-saturated pixels kept as calibration candidates
# ^ This is a heuristic, not a derived constant, and needs to be generous for a specific reason:
# dividing by a single global per-channel median (see _saturation) only corrects the *white-balance*
# part of the raw channel imbalance — a constant per-channel multiplier. It does not correct the
# *density-balance* part (a per-channel power/contrast difference), which means genuinely neutral
# pixels at very different density levels (deep shadow vs. bright highlight) can still show
# noticeably different residual "saturation" after this normalization, even though both are equally
# neutral. Too low a fraction risks capturing only one end of the tonal range as candidates, which
# silently produces a *worse* estimate than using the full range would (verified in
# tests/unit/test_auto_calibration.py, where 0.25 systematically fails and 0.5 recovers exactly).
DEFAULT_SHADOW_PERCENTILE = 99.9  # highest transmittance among candidates -> shadow after invert
DEFAULT_HIGHLIGHT_PERCENTILE = 0.1  # lowest transmittance among candidates -> highlight after invert

DEFAULT_DENSITY_BINS = 20  # target number of density-local reference bins (see _density_reference)
MIN_PIXELS_PER_BIN = 200  # below this, a bin's own median is too noisy to trust as a local reference

# A single global per-channel median (the original approach) is only a valid stand-in for the
# film's systematic per-channel imbalance at ONE density level — the level the median itself
# happens to sit at. It is not a fixed ratio across the whole tonal range: color negative film's
# three dye layers have different characteristic-curve *shapes*, most dramatically at the
# underexposed "toe," where near-black content of any real hue converges toward nearly the same
# raw color (there is little image-forming density left to differentiate it). Confirmed via real
# testing (see CLAUDE.md): a genuinely neutral black object and a head of dark (non-neutral) hair
# had near-identical raw RGB and both passed the old global-median filter, while a genuinely
# neutral, properly-exposed white object was excluded. _density_reference fixes the "compared
# against the wrong density level" half of that bug by computing a LOCAL reference per density
# bin instead of one global one. It does not fix the separate, harder problem that
# _shadow_and_highlight_from_candidates still extracts from the extreme percentiles of whatever
# survives — investigated and found not fixable by better statistics on this same data (see
# CLAUDE.md); anchor-frame or ColorChecker calibration remain the reliable options when this
# matters.
MIN_DENSITY_SEPARATION = 0.1  # log10 density units; deliberately conservative, see _check_density_separation


def _density_reference(pixels: np.ndarray, target_bins: int = DEFAULT_DENSITY_BINS) -> np.ndarray:
    """For each pixel, the per-channel median of *other pixels at a similar density* (luminance-
    sorted, equal-population bins) — the local stand-in for "the film's systematic imbalance at
    this pixel's own density level," replacing a single frame-wide median. Bin count adapts down
    (to as low as 1, i.e. the original single-global-median behavior) when there aren't enough
    pixels to support MIN_PIXELS_PER_BIN per bin, so small inputs (a handful of pixels, as in some
    unit tests) degrade gracefully rather than producing noisy per-bin medians from a handful of
    pixels each.
    """
    n = len(pixels)
    n_bins = max(1, min(target_bins, n // MIN_PIXELS_PER_BIN))
    if n_bins == 1:
        return np.broadcast_to(np.median(pixels, axis=0), pixels.shape)

    order = np.argsort(pixels.mean(axis=1))
    reference = np.empty_like(pixels)
    for bin_positions in np.array_split(np.arange(n), n_bins):
        bin_indices = order[bin_positions]
        reference[bin_indices] = np.median(pixels[bin_indices], axis=0)
    return reference


def _saturation(pixels: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """A chroma proxy suitable for ranking pixels by "how likely is this to be scene-neutral,"
    computed on raw (uncorrected) transmittance.

    Naively taking (max-min)/max directly on raw values does not work: a genuinely scene-neutral
    patch can have large R/G/B spread in *raw* transmittance precisely because of the differential
    dye density that density balance exists to correct (e.g. the reference shadow/highlight
    calibration patches from abpy/color-neg-resources — (.094,.131,.050) and (.048,.054,.016) —
    score 0.62 and 0.70 on that naive metric, nowhere near "low saturation"). Dividing each channel
    by a reference (see `_density_reference` — per-pixel, density-local by default) first removes
    that *systematic* per-channel imbalance (the same thing density balance corrects) before
    measuring spread, so what survives is genuine hue variation relative to what's typical at that
    pixel's own density — not an artifact of the film's own dye imbalance being mistaken for "this
    pixel is colored." `reference=None` falls back to a single frame-wide median (broadcast to
    every pixel) for callers that don't need density-local behavior.
    """
    if reference is None:
        reference = np.median(pixels, axis=0)
    normalized = pixels / reference
    max_channel = normalized.max(axis=-1)
    min_channel = normalized.min(axis=-1)
    return np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )


def _check_density_separation(shadow_rgb: np.ndarray, highlight_rgb: np.ndarray) -> None:
    """Warn (not raise) when the solved shadow/highlight candidates are suspiciously close in
    density — a cheap, real, but narrow safety net: it only catches a near-degenerate (tiny-span)
    pair. It will NOT catch a wrong-but-well-separated pair, which is the specific failure mode
    found via real testing on IMG_0151 (see CLAUDE.md) — that failure needs a fundamentally
    different, external signal (anchor-frame or ColorChecker calibration), not a cheaper check.
    """
    shadow_density = np.log10(1.0 / np.maximum(shadow_rgb, MIN_TRANSMITTANCE))
    highlight_density = np.log10(1.0 / np.maximum(highlight_rgb, MIN_TRANSMITTANCE))
    span = np.abs(highlight_density - shadow_density)
    if np.any(span < MIN_DENSITY_SEPARATION):
        warnings.warn(
            "auto-detected shadow/highlight candidates are unusually close in density "
            f"(min span {span.min():.3f}, expected at least {MIN_DENSITY_SEPARATION}) — this "
            "estimate may be unreliable; consider anchor-frame or ColorChecker calibration instead",
            stacklevel=3,
        )


def _neutral_candidate_mask(image: np.ndarray, neutral_fraction: float) -> np.ndarray:
    """(H, W) boolean mask of pixels that pass the saturation filter — the spatial counterpart to
    `_neutral_candidates`' flat array. Exists so a caller can visualize *which regions* of an image
    the statistical method considers plausible (e.g. an overlay in the GUI calibration picker, to
    cross-check a manual pick against), which a flat array of just the surviving RGB values can't
    show — the location of each candidate pixel is exactly what a flat, reordered array discards.
    """
    flat = image.reshape(-1, 3)
    reference = _density_reference(flat)
    saturation = _saturation(flat, reference)
    threshold = np.percentile(saturation, neutral_fraction * 100)
    return (saturation <= threshold).reshape(image.shape[:2])


def _neutral_candidates(image: np.ndarray, neutral_fraction: float) -> np.ndarray:
    """The least-saturated `neutral_fraction` of a single image's own pixels — saturation judged
    relative to a reference local to *that pixel's own density*, within *that image's own* pixels
    (see `_saturation`/`_density_reference`).

    Deliberately scoped to one image at a time. Normalizing against a frame's own pixels is what
    makes `_saturation` measure "how neutral is this pixel relative to the film's systematic
    per-channel imbalance" rather than "relative to this pixel's absolute color" — that only holds
    when the reference is computed from pixels that share the same systematic imbalance, i.e. one
    frame. See `roll_auto_density_balance` for why this must run *before* pooling across frames,
    not after.
    """
    mask = _neutral_candidate_mask(image, neutral_fraction)
    return image.reshape(-1, 3)[mask.reshape(-1)]


def _shadow_and_highlight_from_candidates(
    candidates: np.ndarray,
    shadow_percentile: float,
    highlight_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(candidates) < 2:
        raise ValueError(
            "not enough near-neutral pixels found to estimate density balance automatically "
            "(try a higher neutral_fraction, or use a manual/ColorChecker calibration instead)"
        )
    shadow_rgb = np.percentile(candidates, shadow_percentile, axis=0)
    highlight_rgb = np.percentile(candidates, highlight_percentile, axis=0)
    _check_density_separation(shadow_rgb, highlight_rgb)
    return shadow_rgb, highlight_rgb


def auto_density_balance(
    image: np.ndarray,
    neutral_fraction: float = DEFAULT_NEUTRAL_FRACTION,
    shadow_percentile: float = DEFAULT_SHADOW_PERCENTILE,
    highlight_percentile: float = DEFAULT_HIGHLIGHT_PERCENTILE,
) -> DensityProfile:
    """Estimate a density-balance profile from a single frame (working-space image)."""
    candidates = _neutral_candidates(image, neutral_fraction)
    shadow_rgb, highlight_rgb = _shadow_and_highlight_from_candidates(
        candidates, shadow_percentile, highlight_percentile
    )
    profile = solve_density_balance(tuple(shadow_rgb), tuple(highlight_rgb))
    return DensityProfile(
        white_balance=profile.white_balance, density_scale=profile.density_scale, source="auto"
    )


def roll_auto_density_balance(
    images: list[np.ndarray],
    neutral_fraction: float = DEFAULT_NEUTRAL_FRACTION,
    shadow_percentile: float = DEFAULT_SHADOW_PERCENTILE,
    highlight_percentile: float = DEFAULT_HIGHLIGHT_PERCENTILE,
) -> DensityProfile:
    """Estimate one shared density-balance profile from several frames of the same roll —
    generally more robust than per-frame estimation, since a single shared film-base (shadow) and
    typical highlight density are measured across many frames' worth of candidate pixels rather
    than one frame's, at the cost of not adapting to a given frame's individual content.

    Selects neutral candidates *per frame first* (each judged against its own per-channel median),
    then pools only the resulting candidate pixels before the final shadow/highlight percentiles —
    not the other way around. Concatenating raw pixels from every frame before computing one
    shared median (the original implementation) was a real bug, found via testing on two real
    scans of genuinely different scenes (a warm-toned portrait and a daylight street scene): the
    combined median ends up a blend of both frames' average *scene color*, not just the film's
    systematic dye imbalance the normalization is meant to isolate (see `_saturation`), because
    that blend is not actually shared across frames the way the film-base imbalance is. That
    mis-selected candidates for whichever frame differed most from the blended average, producing
    a visible color cast (confirmed: the affected frame's neutral objects, e.g. a white car,
    rendered visibly blue). Selecting per-frame first keeps each frame's candidate selection
    correct on its own terms; only the final shadow/highlight measurement is roll-wide.
    """
    per_frame_candidates = [_neutral_candidates(image, neutral_fraction) for image in images]
    combined_candidates = np.concatenate(per_frame_candidates, axis=0)
    shadow_rgb, highlight_rgb = _shadow_and_highlight_from_candidates(
        combined_candidates, shadow_percentile, highlight_percentile
    )
    profile = solve_density_balance(tuple(shadow_rgb), tuple(highlight_rgb))
    return DensityProfile(
        white_balance=profile.white_balance, density_scale=profile.density_scale, source="auto"
    )
