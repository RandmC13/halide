"""The pipeline's stages composed in one place: negative_to_positive (white balance, density
balance, invert), then the print fit and paper curve (core.tone_render).

`develop`/`run_pipeline` chain them over a whole array; the GUI (gui/render.py) calls `develop`.
The CLI's full-resolution paths (processing.py) call the same stage functions band by band
(halide.banding) around the one whole-frame step, the print fit — bit-identical to `develop`
(pinned by tests/unit/test_banding.py). So there is one definition of each stage, not one of the
math per caller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.invert import invert
from halide.core.tone_render import ResolvedTone, apply_tone, resolve_tone
from halide.core.types import DensityProfile, ToneCurveParams

if TYPE_CHECKING:
    from halide.io.lut import Cube1D


def run_pipeline(
    negative_linear: np.ndarray,
    density_profile: DensityProfile,
    tone_params: ToneCurveParams | None = None,
    curve: Cube1D | None = None,
) -> np.ndarray:
    """white_balance -> density_balance -> invert -> tone_render.

    `negative_linear` must already be in the internal linear working color space (see
    halide.io.icc) — this function does no color management of its own.

    `tone_params=None` defaults to ToneCurveParams() (paper mode, exposure and grade fitted per
    image) — pass ToneCurveParams(mode="linear") explicitly for the flat linear output. `curve` is
    the loaded paper curve (halide.io.lut.load_paper_curve), needed for paper mode: core/ reads no
    files, so the caller loads it.
    """
    return develop(negative_linear, density_profile, tone_params, curve)[0]


def develop(
    negative_linear: np.ndarray,
    density_profile: DensityProfile,
    tone_params: ToneCurveParams | None = None,
    curve: Cube1D | None = None,
) -> tuple[np.ndarray, ResolvedTone]:
    """run_pipeline, also returning the tone values actually used (fitted or pinned) so a caller
    can report/record them — the same single chain, not a second implementation of it."""
    if tone_params is None:
        tone_params = ToneCurveParams()

    img = negative_to_positive(negative_linear, density_profile)
    resolved = resolve_tone(img, tone_params, curve)
    return apply_tone(img, resolved, curve), resolved


def negative_to_positive(negative_linear: np.ndarray, density_profile: DensityProfile) -> np.ndarray:
    """white_balance -> density_balance -> invert: every per-pixel stage before the print fit. The
    one definition of that order — develop() applies it to a whole array, processing.py's
    full-resolution paths to one band of rows at a time (see halide.banding)."""
    img = apply_white_balance(negative_linear, density_profile)
    img = apply_density_balance(img, density_profile)
    return invert(img)
