"""What the calibration picker draws: a frame as the raw negative or as a positive, and a positive
that stays consistent between the image, the magnifier and the filmstrip. Pure numpy/core calls, no
Qt - the same split as gui/sampling.py.

The positive is one of two things (see gui/roll.py::CalibrationSession.positive_source):
  - the frame's own auto-density estimate (calibration/auto.py) - a rough, statistics-only guess,
    shown until the user has picked enough points to fit, and labelled as such;
  - the user's live fit, applied at the roll's reference scan exposure: a frame digitized brighter
    or darker than the reference is scaled by its scan gain first, exactly as
    `batch --match-scan-exposure` develops it (calibration/scan_consistency.py).
Either way the frame is printed through the real print stage (core.pipeline.develop): the paper
curve, with exposure and grade fitted per frame unless the Print drawer pins them.
"""

from __future__ import annotations

import numpy as np

from halide.core.pipeline import develop, negative_to_positive
from halide.core.tone_render import ResolvedTone, apply_tone
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui.sampling import apply_stretch
from halide.io.raster import to_srgb_8bit


def negative_display(image: np.ndarray, stretch_bounds: tuple[float, float]) -> np.ndarray:
    """The raw negative, percentile-stretched for viewing (float32 in [0, 1]). Not colour science."""
    return apply_stretch(image, *stretch_bounds)


def positive_display(
    image: np.ndarray, profile: DensityProfile, tone: ToneCurveParams, scan_gain: float = 1.0
) -> tuple[np.ndarray, ResolvedTone]:
    """The frame printed with `profile` (uint8 sRGB) and the print decision used, so the magnifier
    can print its patch with the very same exposure/grade (print_patch) instead of re-fitting a
    tiny patch on its own."""
    frame = image * np.float32(scan_gain) if scan_gain != 1.0 else image
    printed, resolved = develop(frame, profile, tone)
    return to_srgb_8bit(printed), resolved


def print_patch(
    patch: np.ndarray, profile: DensityProfile, resolved: ResolvedTone, scan_gain: float = 1.0
) -> np.ndarray:
    """A magnifier patch printed exactly as the displayed frame around it was (uint8 sRGB)."""
    frame = patch * np.float32(scan_gain) if scan_gain != 1.0 else patch
    return to_srgb_8bit(apply_tone(negative_to_positive(frame, profile), resolved))
