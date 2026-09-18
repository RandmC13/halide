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

import numpy as np

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


def _saturation(pixels: np.ndarray) -> np.ndarray:
    """A chroma proxy suitable for ranking pixels by "how likely is this to be scene-neutral,"
    computed on raw (uncorrected) transmittance.

    Naively taking (max-min)/max directly on raw values does not work: a genuinely scene-neutral
    patch can have large R/G/B spread in *raw* transmittance precisely because of the differential
    dye density that density balance exists to correct (e.g. the reference shadow/highlight
    calibration patches from abpy/color-neg-resources — (.094,.131,.050) and (.048,.054,.016) —
    score 0.62 and 0.70 on that naive metric, nowhere near "low saturation"). Dividing each channel
    by its own image-wide median first removes that *systematic* per-channel imbalance (the same
    thing density balance corrects) before measuring spread, so what survives is genuine hue
    variation relative to the image's own typical response — not an artifact of the film's own
    dye imbalance being mistaken for "this pixel is colored."
    """
    channel_median = np.median(pixels, axis=0)
    normalized = pixels / channel_median
    max_channel = normalized.max(axis=-1)
    min_channel = normalized.min(axis=-1)
    return np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )


def _neutral_candidates(image: np.ndarray, neutral_fraction: float) -> np.ndarray:
    """The least-saturated `neutral_fraction` of a single image's own pixels — saturation judged
    relative to *that image's own* per-channel median (see `_saturation`).

    Deliberately scoped to one image at a time. Normalizing against a frame's own median is what
    makes `_saturation` measure "how neutral is this pixel relative to the film's systematic
    per-channel imbalance" rather than "relative to this pixel's absolute color" — that only holds
    when the median is computed from pixels that share the same systematic imbalance, i.e. one
    frame. See `roll_auto_density_balance` for why this must run *before* pooling across frames,
    not after.
    """
    flat = image.reshape(-1, 3)
    saturation = _saturation(flat)
    threshold = np.percentile(saturation, neutral_fraction * 100)
    return flat[saturation <= threshold]


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
