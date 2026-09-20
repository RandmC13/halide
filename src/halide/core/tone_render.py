"""The final pipeline stage: compress an unbounded, inverted-linear positive into a displayable
range without hard-clipping.

The default curve is vendored from abpy/color-neg-resources' `paper_a.cube` (MIT licensed, see
assets/tone_curves/LICENSE-paper_endura.txt) — an emulation of a real photographic paper's
characteristic response (Kodak Endura), including its natural toe/shoulder compression. This is
deliberately *not* an invented/hand-tuned analytic curve: photographic paper response is measured,
published photographic knowledge, which fits the goal of faithfulness over subjective looks.

mode="linear" is the escape hatch for users who want to grade fully in an external tool — flat
(no tone-curve compression/contrast shape), but still scaled into a viewable, headroom-preserving
range (see estimate_linear_scale) rather than the bare unbounded invert() output, which is
unusable directly: a real scan's reciprocal values are typically in the tens (e.g. a density-
balanced transmittance around 10% inverts to 10), so almost every pixel lands above 1.0 and a
standard viewer shows solid white.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.types import ToneCurveParams
from halide.io.lut import Cube1D, load_1d_cube

_DEFAULT_CURVE_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "tone_curves" / "paper_endura.cube"
)


@lru_cache(maxsize=8)
def _load_curve(path: str) -> Cube1D:
    cube = load_1d_cube(path)
    # The raw curve's peak is below 1.0 (real paper doesn't reach maximum theoretical reflectance);
    # normalizing to 1.0 here matches how the reference `neglut.py` uses this same curve.
    return Cube1D(values=cube.values / cube.values.max(), domain_min=cube.domain_min, domain_max=cube.domain_max)


_AUTO_EXPOSURE_SHADOW_PERCENTILE = 1.0
_AUTO_EXPOSURE_TARGET_DENSITY = 0.85  # see estimate_exposure's docstring for how this was chosen


def estimate_exposure(
    positive_linear: np.ndarray,
    shadow_percentile: float = _AUTO_EXPOSURE_SHADOW_PERCENTILE,
    target_density: float = _AUTO_EXPOSURE_TARGET_DENSITY,
) -> float:
    """Auto-exposure: position *this image's own* shadow-end density at a fixed point on the tone
    curve, rather than trusting one fixed constant for every negative.

    This exists because a single hardcoded exposure cannot correctly position every scan — found
    on a real test scan where a fixed exposure left even the darkest 1% of pixels stuck in the
    curve's flat near-black toe without ever reaching it, giving a washed-out result with no real
    blacks, while the same constant worked fine on a different scan. That's the same class of
    problem a single global calibration constant caused for color (see calibration/auto.py) — the
    fix is the same shape: measure *this* image's own statistics instead of assuming one constant
    fits every negative. Directly analogous to a darkroom printer determining enlarger exposure
    time from a test strip per negative rather than reusing one fixed time for a whole box of
    paper.

    `target_density` is itself a fixed constant, and was initially set to 1.0 (roughly where the
    reference curve's toe starts to meaningfully rise) from testing against two real scans — which
    turned out to be the *same* mistake this function exists to fix, just one level up: 1.0 was too
    bright specifically for an image whose darkest 1% is a small, distinct cluster (e.g. a subject
    wearing dark clothing against a much larger bright majority of frame), because it says nothing
    about where the rest of the tonal range lands, only the extreme shadow point. Lowered to 0.85
    after re-testing against the same two scans, which holds up better on both — but this is still
    a single global constant standing in for "how much of the curve's toe compression to use,"
    which is inherently a scene-dependent judgment (classic camera metering has the same limit).
    Don't tune this further without rendering and *looking at* more than one or two real images —
    see the project's memory/feedback notes on verifying against real data before calling a fix
    correct.
    """
    # `density` is a private temporary (never positive_linear itself, and not returned) — safe to
    # let both np.log10's `out=` and np.percentile's `overwrite_input=True` mutate/reorder it in
    # place rather than each making their own extra full-size copy of it.
    density = np.maximum(positive_linear, MIN_TRANSMITTANCE)
    np.log10(density, out=density)
    shadow_density = np.percentile(density, shadow_percentile, overwrite_input=True)
    return target_density - shadow_density


_LINEAR_HIGHLIGHT_PERCENTILE = 99.9
_LINEAR_HEADROOM_TARGET = 0.8  # leaves margin below 1.0 for the thin tail above this percentile


def estimate_linear_scale(
    positive_linear: np.ndarray,
    highlight_percentile: float = _LINEAR_HIGHLIGHT_PERCENTILE,
    target_value: float = _LINEAR_HEADROOM_TARGET,
) -> float:
    """A multiplicative scale (not a curve — preserves "flat") so a robust highlight reference
    lands at `target_value`, leaving headroom below 1.0, rather than writing the bare unbounded
    reciprocal with no inherent brightness reference at all.

    Uses a percentile, not the true maximum, for the same reason calibration/auto.py and
    estimate_exposure do: a handful of degenerate pixels (dust, or the MIN_TRANSMITTANCE floor
    clamp on a near-zero-transmittance pixel) can be many orders of magnitude brighter than any
    real image content — on both real test scans used during development, the true maximum was
    exactly 1/MIN_TRANSMITTANCE (10,000,000) while the 99.9th percentile was under 100, confirming
    the tail past that point is clamp artifacts, not photographed detail. Scaling to the true max
    would crush the entire rest of the image to near-black to accommodate one pathological pixel.
    This leaves a small fraction of the extreme tail (typically <0.05% of pixels on the scans
    tested) still above 1.0 after scaling — a deliberate trade-off, not an oversight: avoiding
    real highlight detail loss matters far more than a few dust-speck pixels.
    """
    highlight = np.percentile(positive_linear, highlight_percentile)
    if highlight <= 0:
        return 1.0
    return target_value / highlight


def tone_render(positive_linear: np.ndarray, params: ToneCurveParams) -> np.ndarray:
    if params.mode == "linear":
        return linear_passthrough(positive_linear * estimate_linear_scale(positive_linear))

    curve = _load_curve(params.curve_path or str(_DEFAULT_CURVE_PATH))
    exposure = params.exposure if params.exposure is not None else estimate_exposure(positive_linear)
    # `safe` is a fresh copy of positive_linear (never positive_linear itself), so every step below
    # reuses its buffer via `out=`/in-place ops instead of allocating a new full-size array at each
    # line — positive_linear itself is left untouched throughout (callers, e.g. auto-exposure above
    # and repeated tone_render calls on the same array in tests, still see the original values).
    safe = np.maximum(positive_linear, MIN_TRANSMITTANCE)
    # positive_linear == invert(negative) == 1/negative, so log10(positive_linear) is exactly the
    # negative's density (log10(1/negative)) — the domain this curve is defined over. `exposure`
    # positions that density range on the paper's response curve (the darkroom-printing analogue
    # of enlarger exposure time), matching the reference's `base_exposure` constant.
    density = np.log10(safe, out=safe)
    density += exposure
    # `contrast` compresses/expands density around the curve's own domain midpoint before lookup —
    # see ToneCurveParams' docstring for why this exists (contrast=1.0 is the untouched reference
    # curve).
    pivot = (curve.domain_min + curve.domain_max) / 2.0
    density -= pivot
    density *= params.contrast
    density += pivot
    return curve.lookup(density)


def linear_passthrough(positive_linear: np.ndarray) -> np.ndarray:
    return positive_linear
