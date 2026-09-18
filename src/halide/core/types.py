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


@dataclass(frozen=True)
class ToneCurveParams:
    """Parameters for the final tone-render stage.

    mode="paper" (the default) runs the inverted, density-balanced image through a print-emulating
    response curve with a smooth toe/shoulder, replacing the old hard np.clip(0, 1) that caused
    inconsistent clipping. mode="linear" is the escape hatch for users who want unclipped linear
    output to grade elsewhere.

    `contrast` scales how much of the paper curve's density range a given image engages, pivoted
    around the curve's domain midpoint — contrast=1.0 reproduces the vendored reference curve
    (a real, but extremely high-contrast, commercial paper) exactly; lower values are the digital
    equivalent of printing on a lower contrast-grade paper. This exists because the raw reference
    curve was found (on a real test scan, see project history) to amplify small, otherwise
    negligible residual calibration imperfections into a visible color cast in midtones — the same
    problem a real printer would reach for a softer paper grade to solve, not a different chemistry.

    `exposure=None` (the default) auto-computes a per-image exposure from the image's own shadow
    statistics (core.tone_render.estimate_exposure) rather than trusting one fixed constant for
    every negative — found necessary on a real test scan where a fixed exposure left true blacks
    unreachable on one scan while working fine on another. Pass an explicit float to pin it (e.g.
    once you've dialed in a look in the preview GUI and want every frame of a roll to match).
    """

    mode: Literal["paper", "linear"] = "paper"
    exposure: float | None = None
    contrast: float = 0.5
    curve_path: str | None = None  # override the bundled default curve asset
