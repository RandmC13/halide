"""White balance and density balance: the two per-channel corrections that make gray stay neutral
across the whole tonal range of a color negative scan.

The regression in solve_density_balance() is the one piece of real, working science already in the
old halide script — it appeared three times there (interactive_density_balance,
auto_density_balance, roll_analysis_density_balance in legacy/halide_v1.py), verbatim-identical
each time, differing only in how the two calibration points were gathered. It is reproduced here
exactly (verified against abpy/color-neg-resources' own `density levels.py` reference script,
see tests/unit/test_density.py) as a single function, decoupled from any particular calibration
strategy.
"""

from __future__ import annotations

import numpy as np

from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.types import DensityProfile

_REFERENCE_CHANNEL = 1  # green; density balance and white balance are solved relative to it


def solve_density_balance(
    shadow_rgb: tuple[float, float, float],
    highlight_rgb: tuple[float, float, float],
) -> DensityProfile:
    """Solve white balance + density balance from two neutral reference points.

    `shadow_rgb` and `highlight_rgb` are raw (linear, working-space) transmittance values sampled
    from two patches that are neutral gray in the *original scene* — one from a lightly-exposed,
    higher-transmittance area of the negative (which becomes a shadow after inversion) and one
    from a more heavily-exposed, lower-transmittance area (which becomes a highlight after
    inversion). This matches the "darker gray patch, lower density (light) negative" /
    "lighter gray patch, higher density (dark) negative" convention used by both reference
    implementations — a ColorChecker's two darkest gray patches, or any two neutral tones picked
    from a real frame, both work.

    Returns a DensityProfile with green fixed at 1.0 in both fields and red/blue solved so that,
    once applied, both reference points become perfectly neutral (equal R=G=B) at their correct
    relative densities.
    """
    shadow = np.maximum(np.asarray(shadow_rgb, dtype=np.float64), MIN_TRANSMITTANCE)
    highlight = np.maximum(np.asarray(highlight_rgb, dtype=np.float64), MIN_TRANSMITTANCE)

    # log10(1/transmittance) is optical density; density increases ~linearly with log exposure.
    shadow_density = np.log10(1.0 / shadow)
    highlight_density = np.log10(1.0 / highlight)

    # Density *slope* differs per channel (differing dye contrast) — the span between our two
    # known points, per channel, tells us how much steeper/flatter each channel's response is.
    span = highlight_density - shadow_density
    if np.any(span == 0):
        raise ValueError(
            "shadow and highlight patches must differ in density on every channel "
            "(got identical density on at least one channel — pick more separated patches)"
        )
    raw_scale = 1.0 / span

    # Normalize so density_scale[green] == 1.0: green becomes the fixed reference channel and
    # red/blue are expressed relative to it, matching both reference implementations.
    density_scale = raw_scale / raw_scale[_REFERENCE_CHANNEL]

    # With slopes equalized, find the per-channel density offset that makes the shadow point
    # neutral (equal density on every channel) — that offset, converted back to linear, is the
    # white-balance multiplier.
    scaled_shadow_density = shadow_density * density_scale
    reference_density = scaled_shadow_density[_REFERENCE_CHANNEL]
    density_offset = (reference_density - scaled_shadow_density) / density_scale
    white_balance = 1.0 / (10.0**density_offset)

    return DensityProfile(
        white_balance=tuple(white_balance.tolist()),
        density_scale=tuple(density_scale.tolist()),
    )


def apply_white_balance(img: np.ndarray, profile: DensityProfile) -> np.ndarray:
    """Per-channel linear multiply. Must run before apply_density_balance (order matters — the
    density-balance power function is not commutative with a subsequent multiply)."""
    multiplier = np.asarray(profile.white_balance, dtype=img.dtype)
    return img * multiplier


def apply_density_balance(img: np.ndarray, profile: DensityProfile) -> np.ndarray:
    """Per-channel power function on linear transmittance: x ** density_scale. Equivalent to
    scaling each channel's density (log10(1/x) * density_scale) and converting back — the power
    form avoids a log/exp round trip. Green's exponent is always 1.0 (no-op) by construction."""
    safe = np.maximum(img, MIN_TRANSMITTANCE)
    exponent = np.asarray(profile.density_scale, dtype=safe.dtype)
    return np.power(safe, exponent)
