"""Pure, GUI-framework-independent helpers for the anchor-frame calibration picker: patch
sampling, outlier-robust snapping, and preview stretching. No dearpygui import here, so this is
unit-testable without a display — only gui/calibrate_screen.py touches dearpygui itself.
"""

from __future__ import annotations

import numpy as np


def snap_to_representative_pixel(
    image: np.ndarray, x: int, y: int, radius: int = 10
) -> tuple[int, int, np.ndarray]:
    """Within a small neighborhood around (x, y), find the pixel closest to the neighborhood's
    median RGB — avoids a click landing on a stray noise/dust-speck pixel rather than the patch's
    actual representative color. Returns (snapped_x, snapped_y, rgb)."""
    h, w = image.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    neighborhood = image[y0:y1, x0:x1]

    cols = x1 - x0
    flat = neighborhood.reshape(-1, 3)
    median = np.median(flat, axis=0)
    distances = np.linalg.norm(flat - median, axis=1)
    best_idx = int(np.argmin(distances))
    row, col = divmod(best_idx, cols)
    return x0 + col, y0 + row, flat[best_idx]


def patch_chroma(rgb: np.ndarray) -> float:
    """A simple (max-min)/max chroma metric for a single sampled point.

    Shown to the user as a rough guide only — NOT a reliable neutrality detector on its own. A
    genuinely scene-neutral point on a raw negative can still show substantial raw channel
    imbalance purely from the film's own dye density differences (verified on a real scan: the
    reference calibration patches from abpy/color-neg-resources score 0.6-0.7 on this metric
    despite being genuinely neutral — see calibration/auto.py's docstring for the same finding).
    The picker UI must pair this number with that caveat, not present it as pass/fail.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    channel_max, channel_min = rgb.max(), rgb.min()
    return float((channel_max - channel_min) / channel_max) if channel_max > 0 else 0.0


def downsample_for_display(
    image: np.ndarray, max_width: int = 1400, max_height: int = 1400
) -> tuple[np.ndarray, int]:
    """Downsample by an integer stride for on-screen display. Returns (downsampled_image, stride)
    — callers map a click on the displayed image back to full-resolution coordinates by
    multiplying by `stride`.

    Both width AND height are constrained independently (not just the longer side) — found via
    interactive testing that constraining only the longest side let a landscape photo's displayed
    height exceed a fixed-height scrollable container, silently making the bottom portion
    unclickable (clicks there land outside the hovered/visible widget and are correctly ignored,
    which looks like "clicks near the bottom don't work" rather than an obvious crash).
    """
    h, w = image.shape[:2]
    stride = max(1, -(-h // max_height), -(-w // max_width))  # ceil division on both axes
    return image[::stride, ::stride], stride


def preview_stretch(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """A simple percentile-stretch + gamma for viewing a raw (un-inverted) negative on screen —
    purely a display convenience, not part of the color pipeline. Returns float32 in [0, 1]."""
    lo, hi = np.percentile(image, [low_percentile, high_percentile])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    stretched = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (stretched ** (1.0 / 2.2)).astype(np.float32)
