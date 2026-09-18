"""The single composed pipeline entry point. Both the CLI and the GUI call this — neither
reimplements the math independently."""

from __future__ import annotations

import numpy as np

from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.invert import invert
from halide.core.tone_render import tone_render
from halide.core.types import DensityProfile, ToneCurveParams


def run_pipeline(
    negative_linear: np.ndarray,
    density_profile: DensityProfile,
    tone_params: ToneCurveParams | None = None,
) -> np.ndarray:
    """white_balance -> density_balance -> invert -> tone_render.

    `negative_linear` must already be in the internal linear working color space (see
    halide.io.icc) — this function does no color management of its own.

    `tone_params=None` defaults to ToneCurveParams() (paper mode, tone-render on) — pass
    ToneCurveParams(mode="linear") explicitly for unbounded linear output.
    """
    if tone_params is None:
        tone_params = ToneCurveParams()

    img = apply_white_balance(negative_linear, density_profile)
    img = apply_density_balance(img, density_profile)
    img = invert(img)
    return tone_render(img, tone_params)
