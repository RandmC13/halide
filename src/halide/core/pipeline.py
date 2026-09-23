"""The single composed pipeline entry point. Both the CLI and the GUI call this — neither
reimplements the math independently."""

from __future__ import annotations

import numpy as np

from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.invert import invert
from halide.core.tone_render import ResolvedTone, apply_tone, resolve_tone
from halide.core.types import DensityProfile, ToneCurveParams


def run_pipeline(
    negative_linear: np.ndarray,
    density_profile: DensityProfile,
    tone_params: ToneCurveParams | None = None,
) -> np.ndarray:
    """white_balance -> density_balance -> invert -> tone_render.

    `negative_linear` must already be in the internal linear working color space (see
    halide.io.icc) — this function does no color management of its own.

    `tone_params=None` defaults to ToneCurveParams() (paper mode, exposure and grade fitted per
    image) — pass ToneCurveParams(mode="linear") explicitly for the flat linear output.
    """
    return develop(negative_linear, density_profile, tone_params)[0]


def develop(
    negative_linear: np.ndarray,
    density_profile: DensityProfile,
    tone_params: ToneCurveParams | None = None,
) -> tuple[np.ndarray, ResolvedTone]:
    """run_pipeline, also returning the tone values actually used (fitted or pinned) so a caller
    can report/record them — the same single chain, not a second implementation of it."""
    if tone_params is None:
        tone_params = ToneCurveParams()

    img = apply_white_balance(negative_linear, density_profile)
    img = apply_density_balance(img, density_profile)
    img = invert(img)
    resolved = resolve_tone(img, tone_params)
    return apply_tone(img, resolved, tone_params.curve_path), resolved
