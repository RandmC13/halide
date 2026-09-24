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

from collections.abc import Sequence

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


def fit_density_balance(neutral_rgbs: Sequence[tuple[float, float, float]]) -> DensityProfile:
    """Solve white balance + density balance from any number (>= 2) of neutral reference points.

    The same model as solve_density_balance: in log density, a scene-neutral point satisfies
    D_c = a_c + D_G / s_c for each non-green channel c, i.e. the film's neutral axis is a straight
    line per channel against green. With two points that line passes through both, exactly as
    solve_density_balance draws it; with more, it's the least-squares line (D_c regressed on D_G,
    green being the reference the axis is parameterised by), so one object that isn't quite neutral
    is averaged against the others instead of passed straight through. Exactly two points give
    solve_density_balance's result to float precision (tests/unit/test_density.py pins that).

    density_scale_c = 1 / slope_c and white_balance_c = 10 ** intercept_c - the per-channel
    multiply and power that make every point on that line equal R=G=B.
    """
    rgb = np.maximum(np.asarray(neutral_rgbs, dtype=np.float64).reshape(-1, 3), MIN_TRANSMITTANCE)
    if len(rgb) < 2:
        raise ValueError("need at least two neutral points to solve density balance")
    density = np.log10(1.0 / rgb)
    green = density[:, _REFERENCE_CHANNEL]
    if np.ptp(green) == 0:
        raise ValueError(
            "neutral points must differ in density (all of them have the same green density — "
            "pick objects of different brightness)"
        )

    white_balance = np.ones(3)
    density_scale = np.ones(3)
    design = np.column_stack([np.ones_like(green), green])
    for channel in (0, 2):
        (intercept, slope), *_ = np.linalg.lstsq(design, density[:, channel], rcond=None)
        if slope <= 0:
            raise ValueError(
                "these points don't describe a film response (a channel's density falls as green "
                "rises) — at least one of them isn't neutral"
            )
        density_scale[channel] = 1.0 / slope
        white_balance[channel] = 10.0**intercept
    return DensityProfile(white_balance=tuple(white_balance.tolist()), density_scale=tuple(density_scale.tolist()))


def neutral_residuals(profile: DensityProfile, rgbs: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """(N, 3) per-channel deviation of each point from neutral once `profile` is applied, in
    density, relative to the point's own mean over channels (so each row sums to zero).

    Measured on the density-balanced negative: balanced density = density_scale * (D - log10 wb).
    A channel with *more* balanced density prints *brighter* in that colour after inversion, so a
    positive entry means the point would print tinted toward that channel's colour.
    """
    rgb = np.maximum(np.asarray(rgbs, dtype=np.float64).reshape(-1, 3), MIN_TRANSMITTANCE)
    scale = np.asarray(profile.density_scale, dtype=np.float64)
    wb = np.asarray(profile.white_balance, dtype=np.float64)
    balanced = scale * (np.log10(1.0 / rgb) - np.log10(wb))
    return balanced - balanced.mean(axis=1, keepdims=True)


def leave_one_out_residuals(rgbs: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """(N, 3) like neutral_residuals, but each point is judged against the fit through all the
    *other* points - "how far is this object from what the rest agree neutral is".

    This, not neutral_residuals against the fit that includes the point, is what to show a user
    checking their picks: a least-squares line bends toward a bad point, so judged against a fit
    that includes it the bad point looks closer to neutral than it is and the good points pick up
    part of its error (tests/unit/test_density.py: a point truly CC 3.9 off read CC 2.6, with the
    good points at CC 1.5). Rows are NaN where the remaining points can't be fitted (fewer than
    three points in total, or the others all at one density)."""
    rgb = np.asarray(rgbs, dtype=np.float64).reshape(-1, 3)
    out = np.full(rgb.shape, np.nan)
    for i in range(len(rgb)):
        others = np.delete(rgb, i, axis=0)
        try:
            profile = fit_density_balance(others)
        except ValueError:
            continue
        out[i] = neutral_residuals(profile, rgb[i : i + 1])[0]
    return out


_PRIMARY = "RGB"
_COMPLEMENT = "CMY"  # cyan = minus red, magenta = minus green, yellow = minus blue


def describe_cast(deviation: Sequence[float]) -> tuple[float, str]:
    """A per-channel deviation row (see neutral_residuals) as a colour-printing filter value:
    (CC units, direction letter). CC is Kodak's Colour Compensating filter scale — density x 100,
    so CC10 = 0.10 — the unit of a dichroic enlarger head's filter dials. The value is the spread
    between the most and least dense channel; the direction is the channel that deviates most,
    named as a primary (R/G/B) if it's in excess or as its complement (C/M/Y) if it's lacking. A
    0.15 blue deficit reads "CC 15 Y", a 0.15 red excess "CC 15 R"."""
    d = np.asarray(deviation, dtype=np.float64)
    magnitude = float((d.max() - d.min()) * 100.0)
    channel = int(np.argmax(np.abs(d)))
    direction = _PRIMARY[channel] if d[channel] > 0 else _COMPLEMENT[channel]
    return magnitude, direction


def apply_white_balance(img: np.ndarray, profile: DensityProfile) -> np.ndarray:
    """Per-channel linear multiply. Must run before apply_density_balance (order matters — the
    density-balance power function is not commutative with a subsequent multiply)."""
    multiplier = np.asarray(profile.white_balance, dtype=img.dtype)
    return img * multiplier


def apply_density_balance(img: np.ndarray, profile: DensityProfile) -> np.ndarray:
    """Per-channel power function on linear transmittance: x ** density_scale. Equivalent to
    scaling each channel's density (log10(1/x) * density_scale) and converting back — the power
    form avoids a log/exp round trip. Green's exponent is always 1.0 (no-op) by construction."""
    # `safe` is a fresh copy of img (np.maximum never returns its input in place), so writing the
    # power result back into it via `out=` is safe — img itself is never mutated — and avoids a
    # second full-size allocation for what used to be a separate return value.
    safe = np.maximum(img, MIN_TRANSMITTANCE)
    exponent = np.asarray(profile.density_scale, dtype=safe.dtype)
    return np.power(safe, exponent, out=safe)
