"""Pure data structures shared across the core pipeline, calibration, and I/O layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CalibrationSource = Literal["colorchecker", "anchor", "auto", "manual"]


@dataclass(frozen=True)
class DensityProfile:
    """A solved white-balance + density-balance calibration for one film stock/process/scanner.

    Both tuples are (red, green, blue). By convention the green channel is always 1.0 in both
    tuples — density balance and white balance are solved relative to green, matching the
    reference implementation (abpy/color-neg-resources).
    """

    white_balance: tuple[float, float, float]
    density_scale: tuple[float, float, float]

    name: str | None = None
    film_stock: str | None = None
    process: str | None = None
    scanner: str | None = None
    created_at: str | None = None
    source: CalibrationSource = "manual"
    notes: str | None = None
    """Free-form user text about how this profile was produced (e.g. light source, scanner
    settings, which frame(s) were used to anchor it) — not read by any pipeline code, purely for
    the user's own future reference. Editable after the fact via `halide profile edit`."""


@dataclass(frozen=True)
class ToneCurveParams:
    """Parameters for the final tone-render stage.

    mode="paper" (the default, the "print" output) runs the inverted, density-balanced image
    through a print-emulating response curve with a smooth toe/shoulder, replacing the old hard
    np.clip(0, 1) that caused inconsistent clipping. mode="linear" (the "flat" output) is the
    minimal-bias escape hatch: white balance + density balance + invert + one global exposure
    scale, for users who want to grade elsewhere — per the reference blog, exposure and white
    balance are the only adjustments that keep a flat positive faithful to the negative.

    `exposure` positions the negative's density range along the paper curve (the darkroom analogue
    of enlarger exposure time); `contrast` scales how much of the curve that range covers, pivoted
    around the curve's domain midpoint (the analogue of paper grade — 1.0 is the vendored reference
    paper untouched, lower is softer).

    Both default to None = fitted per image (core.tone_render.fit_print): the grade that makes this
    negative's own density range fill the paper's ISO 6846 exposure range, capped at 1.0, with
    exposure placing the negative's highlights on the paper's highlight point — so the print uses
    the paper's real black and white through the curve's own toe/shoulder, never a post-curve
    stretch. History, so neither is "simplified" back to a constant: a fixed exposure was found to
    leave true blacks unreachable on one real scan while working on another; the replacement
    shadow-only auto-exposure plus a fixed contrast=0.5 (chosen because the raw 1.0 curve amplified
    small calibration residuals into visible casts) then used only about half the paper on every
    real scan — flat, lifted blacks (sRGB ~45) and dim whites (~220-233) against the paper's own
    ~11/255. The fitted grade typically lands around 0.8-0.9 on real scans, so a good calibration
    matters more than it did at 0.5: a residual cast is ~1.7x more visible. Pin either value
    explicitly (CLI flag, or a saved profile's Fine-tune override) to take it out of the fit.
    """

    mode: Literal["paper", "linear"] = "paper"
    exposure: float | None = None
    contrast: float | None = None
    curve_path: str | None = None  # override the bundled default curve asset
