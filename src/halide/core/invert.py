"""The reciprocal inversion step.

This is deliberately just the bare, unbounded reciprocal — no exposure constant, no clipping. The
old script's `final_array = 0.01 / img_safe; np.clip(final_array, 0, 1)` conflated two different
concerns: the mathematical inversion (unbounded) and an ad hoc exposure/display-range guess (a
fixed 0.01 that cannot fit every negative's density range). That conflation is what caused
inconsistent clipping — a magic constant tuned for one frame's density range will clip a denser or
thinner frame differently. Bringing the result into a displayable range is tone_render's job.
"""

from __future__ import annotations

import numpy as np

from halide.core._constants import MIN_TRANSMITTANCE


def invert(negative_linear: np.ndarray) -> np.ndarray:
    """positive = 1 / negative. Unbounded: values are only guaranteed to be >= 1 for a
    well-exposed (density-balanced, transmittance <= 1) input, and are not clipped here."""
    # `safe` is a fresh copy (never negative_linear itself) — reusing it as the reciprocal's `out`
    # avoids a second full-size allocation.
    safe = np.maximum(negative_linear, MIN_TRANSMITTANCE)
    return np.divide(1.0, safe, out=safe)
