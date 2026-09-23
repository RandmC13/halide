"""The final pipeline stage: compress an unbounded, inverted-linear positive into a displayable
range without hard-clipping.

The default curve is vendored from abpy/color-neg-resources' `paper_a.cube` (MIT licensed, see
assets/tone_curves/LICENSE-paper_endura.txt) — an emulation of a real photographic paper's
characteristic response (Kodak Endura), including its natural toe/shoulder compression. This is
deliberately *not* an invented/hand-tuned analytic curve: photographic paper response is measured,
published photographic knowledge, which fits the goal of faithfulness over subjective looks.

How a given negative is placed on that curve — exposure and paper grade — is fitted per image by
fit_print from the paper's own ISO 6846 range and the negative's own density range: the printer's
two real controls, chosen by measurement, both applied *before* the curve. There is deliberately no
post-curve levels/stretch step anywhere: that would be a second, non-physical tone curve.

mode="linear" is the escape hatch for users who want to grade fully in an external tool — flat
(no tone-curve compression/contrast shape), but still scaled into a viewable, headroom-preserving
range (see estimate_linear_scale) rather than the bare unbounded invert() output, which is
unusable directly: a real scan's reciprocal values are typically in the tens (e.g. a density-
balanced transmittance around 10% inverts to 10), so almost every pixel lands above 1.0 and a
standard viewer shows solid white.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# ACEScg (AP1) luminance weights: the Y row of the ACEScg RGB -> XYZ matrix (ACES spec, S-2014-004).
# Used only to *measure* a frame's density range for the print fit — never applied to the pixels.
ACESCG_LUMINANCE = (0.2722287168, 0.6740817658, 0.0536895174)

# ISO 6846's definition of a photographic paper's exposure range ("ISO range"): the log-exposure
# span between the exposure giving a density 0.04 above paper white (first visible highlight tone)
# and the exposure giving 90% of the paper's maximum density (last separable shadow tone). Written
# for B&W paper, but it only needs a characteristic curve — which is exactly what the vendored
# curve file is — so both points are computed from the curve itself, never hand-tuned.
_ISO_HIGHLIGHT_DENSITY_ABOVE_WHITE = 0.04
_ISO_SHADOW_FRACTION_OF_DMAX = 0.9

# Robust extremes of a frame's own negative density range, for the same dust/clamp-artifact reasons
# as estimate_linear_scale and calibration/auto.py: the true min/max are routinely single-pixel
# artifacts, not photographed detail. Anything beyond these goes into the curve's toe/shoulder —
# compressed, never clipped, since the curve is asymptotic at both ends.
_PRINT_SHADOW_PERCENTILE = 0.1
_PRINT_HIGHLIGHT_PERCENTILE = 99.9

# The one clamp on the fitted grade: never print harder than the real, measured paper the curve
# emulates (contrast=1.0 is that paper untouched). A genuinely low-contrast scene (fog, overcast)
# then prints as soft as it really was instead of being stretched to full black and white.
MAX_PRINT_CONTRAST = 1.0


@dataclass(frozen=True)
class ResolvedTone:
    """The concrete values one tone_render call actually used — auto-fitted or pinned — so callers
    can report/record the printing decision (see processing.py's provenance metadata) rather than
    it being hidden inside the render. `linear_scale` is set only in linear mode, `exposure`/
    `contrast` only in paper mode."""

    mode: str
    exposure: float | None = None
    contrast: float | None = None
    linear_scale: float | None = None


def _curve_input_at(curve: Cube1D, value: float) -> float:
    """The curve input (print log-exposure) at which the normalized curve first reaches `value`,
    linearly interpolated between LUT entries. The curve is monotonic non-decreasing but has flat
    plateaus at both ends, so this searches for the first crossing rather than using np.interp on
    the inverse (which requires strictly increasing sample points)."""
    values = curve.values
    step = (curve.domain_max - curve.domain_min) / (values.shape[0] - 1)
    index = int(np.argmax(values >= value))
    if index == 0:
        return curve.domain_min
    lo, hi = values[index - 1], values[index]
    fraction = (value - lo) / (hi - lo)
    return curve.domain_min + (index - 1 + fraction) * step


def paper_exposure_range(curve: Cube1D) -> tuple[float, float]:
    """(shadow_point, highlight_point) on the curve's input axis — the ISO 6846 paper range (see
    the constants above). The curve is normalized so paper white is 1.0 (density 0), so paper
    density is simply -log10(value)."""
    d_max = -np.log10(curve.values.min())
    highlight = _curve_input_at(curve, 10.0**-_ISO_HIGHLIGHT_DENSITY_ABOVE_WHITE)
    shadow = _curve_input_at(curve, 10.0 ** -(_ISO_SHADOW_FRACTION_OF_DMAX * d_max))
    return shadow, highlight


def negative_density_range(positive_linear: np.ndarray) -> tuple[float, float]:
    """(D_lo, D_hi): this frame's robust negative density range, measured on luminance so the fit
    responds to the image's overall tonal range rather than to any one channel. log10 of the
    inverted positive is exactly the negative's density (see tone_render)."""
    if positive_linear.ndim == 3 and positive_linear.shape[-1] == 3:
        luminance = positive_linear @ np.asarray(ACESCG_LUMINANCE, dtype=positive_linear.dtype)
    else:
        luminance = np.array(positive_linear, copy=True)
    # `luminance` is a private temporary — safe to mutate in place and let np.percentile reorder it.
    np.maximum(luminance, MIN_TRANSMITTANCE, out=luminance)
    np.log10(luminance, out=luminance)
    d_lo, d_hi = np.percentile(
        luminance, [_PRINT_SHADOW_PERCENTILE, _PRINT_HIGHLIGHT_PERCENTILE], overwrite_input=True
    )
    return float(d_lo), float(d_hi)


def _pivot(curve: Cube1D) -> float:
    return (curve.domain_min + curve.domain_max) / 2.0


def fit_print(
    positive_linear: np.ndarray, curve: Cube1D, contrast: float | None = None
) -> tuple[float, float]:
    """Fit (exposure, contrast) so this frame's negative density range fills the paper's range —
    the darkroom's "match the paper grade to the negative" choice, made by measurement.

    Two measurements (D_lo, D_hi), two unknowns, solved exactly:
      - contrast (paper grade) = paper range / negative range, capped at MAX_PRINT_CONTRAST;
      - exposure puts D_hi on the paper's highlight point ("print for the highlights", the normal
        rule for printing negatives). With an uncapped grade, D_lo then lands on the shadow point.

    A pinned `contrast` skips the grade fit and only solves exposure (highlight-anchored).

    Both results are single scalars applied identically to every channel before a channel-identical
    curve, so the fit can't change what colour anything is — only where on the paper the image sits.
    It is also invariant to a global multiply of the input (a multiply is a constant density offset,
    which the fitted exposure absorbs exactly) — what makes `halide print` on an externally
    exposure-adjusted flat positive well-defined.

    Replaces the old shadow-only `estimate_exposure` (one percentile pinned to one fixed target,
    whose own docstring admitted the target was a scene-dependent guess) and the fixed default
    contrast=0.5 — together those used only about half the paper on real scans (sRGB ~45 blacks,
    ~220-233 whites, against the paper's own ~11 / 255).
    """
    shadow_point, highlight_point = paper_exposure_range(curve)
    d_lo, d_hi = negative_density_range(positive_linear)
    if contrast is None:
        span = d_hi - d_lo
        contrast = MAX_PRINT_CONTRAST if span <= 0 else min((highlight_point - shadow_point) / span, MAX_PRINT_CONTRAST)
    pivot = _pivot(curve)
    if contrast == 0:
        # A zero grade maps everything to the pivot regardless of exposure — any exposure is valid.
        return pivot - d_hi, contrast
    exposure = (highlight_point - pivot) / contrast + pivot - d_hi
    return exposure, contrast


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
    fit_print do: a handful of degenerate pixels (dust, or the MIN_TRANSMITTANCE floor
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


def resolve_tone(positive_linear: np.ndarray, params: ToneCurveParams) -> ResolvedTone:
    """Work out the concrete values tone_render will use for this image: pinned values as given,
    anything left as None fitted from the image itself (see fit_print / estimate_linear_scale)."""
    if params.mode == "linear":
        return ResolvedTone(mode="linear", linear_scale=estimate_linear_scale(positive_linear))
    if params.exposure is not None and params.contrast is not None:
        return ResolvedTone(mode="paper", exposure=params.exposure, contrast=params.contrast)
    curve = _load_curve(params.curve_path or str(_DEFAULT_CURVE_PATH))
    fitted_exposure, fitted_contrast = fit_print(positive_linear, curve, contrast=params.contrast)
    exposure = params.exposure if params.exposure is not None else fitted_exposure
    return ResolvedTone(mode="paper", exposure=exposure, contrast=fitted_contrast)


def apply_tone(positive_linear: np.ndarray, resolved: ResolvedTone, curve_path: str | None = None) -> np.ndarray:
    if resolved.mode == "linear":
        return linear_passthrough(positive_linear * resolved.linear_scale)

    curve = _load_curve(curve_path or str(_DEFAULT_CURVE_PATH))
    # `safe` is a fresh copy of positive_linear (never positive_linear itself), so every step below
    # reuses its buffer via `out=`/in-place ops instead of allocating a new full-size array at each
    # line — positive_linear itself is left untouched throughout.
    safe = np.maximum(positive_linear, MIN_TRANSMITTANCE)
    # positive_linear == invert(negative) == 1/negative, so log10(positive_linear) is exactly the
    # negative's density (log10(1/negative)) — the domain this curve is defined over. `exposure`
    # positions that density range on the paper's response curve (the darkroom-printing analogue
    # of enlarger exposure time), matching the reference's `base_exposure` constant.
    density = np.log10(safe, out=safe)
    density += resolved.exposure
    # `contrast` (paper grade) compresses/expands density around the curve's own domain midpoint
    # before lookup — contrast=1.0 is the untouched reference curve.
    pivot = _pivot(curve)
    density -= pivot
    density *= resolved.contrast
    density += pivot
    return curve.lookup(density)


def tone_render(positive_linear: np.ndarray, params: ToneCurveParams) -> np.ndarray:
    return apply_tone(positive_linear, resolve_tone(positive_linear, params), params.curve_path)


def linear_passthrough(positive_linear: np.ndarray) -> np.ndarray:
    return positive_linear
