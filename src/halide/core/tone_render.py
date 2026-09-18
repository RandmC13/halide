"""The final pipeline stage: compress an unbounded, inverted-linear positive into a displayable
range without hard-clipping.

The default curve is vendored from abpy/color-neg-resources' `paper_a.cube` (MIT licensed, see
assets/tone_curves/LICENSE-paper_endura.txt) — an emulation of a real photographic paper's
characteristic response (Kodak Endura), including its natural toe/shoulder compression. This is
deliberately *not* an invented/hand-tuned analytic curve: photographic paper response is measured,
published photographic knowledge, which fits the goal of faithfulness over subjective looks.

mode="linear" is the escape hatch: identity passthrough of the unbounded invert() output, for
users who want to grade fully in an external tool.
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
_AUTO_EXPOSURE_TARGET_DENSITY = 1.0  # where the reference curve's toe starts to meaningfully rise


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
    """
    density = np.log10(np.maximum(positive_linear, MIN_TRANSMITTANCE))
    shadow_density = np.percentile(density, shadow_percentile)
    return target_density - shadow_density


def tone_render(positive_linear: np.ndarray, params: ToneCurveParams) -> np.ndarray:
    if params.mode == "linear":
        return linear_passthrough(positive_linear)

    curve = _load_curve(params.curve_path or str(_DEFAULT_CURVE_PATH))
    safe = np.maximum(positive_linear, MIN_TRANSMITTANCE)
    exposure = params.exposure if params.exposure is not None else estimate_exposure(positive_linear)
    # positive_linear == invert(negative) == 1/negative, so log10(positive_linear) is exactly the
    # negative's density (log10(1/negative)) — the domain this curve is defined over. `exposure`
    # positions that density range on the paper's response curve (the darkroom-printing analogue
    # of enlarger exposure time), matching the reference's `base_exposure` constant.
    density = np.log10(safe) + exposure
    # `contrast` compresses/expands density around the curve's own domain midpoint before lookup —
    # see ToneCurveParams' docstring for why this exists (contrast=1.0 is the untouched reference
    # curve).
    pivot = (curve.domain_min + curve.domain_max) / 2.0
    density = pivot + (density - pivot) * params.contrast
    return curve.lookup(density)


def linear_passthrough(positive_linear: np.ndarray) -> np.ndarray:
    return positive_linear
