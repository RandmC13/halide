"""Pure, GUI-framework-independent helpers for the anchor-frame calibration picker: patch
sampling, outlier-robust snapping, and preview stretching. No Qt import here, so this is
unit-testable without a display — only the widget modules (gui/main_window.py and friends) touch Qt.
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


def compute_stretch_bounds(
    image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0
) -> tuple[float, float]:
    """The percentile computation half of preview_stretch, split out so the calibration picker's
    hover magnifier can reuse bounds computed ONCE from the full downsampled preview rather than
    recomputing percentiles from its own tiny (~25x25) patch on every hover event — a patch-local
    percentile would make the magnifier's brightness flicker/jump as the mouse moves, since a
    small patch's own percentiles are unstable from one hover position to the next."""
    lo, hi = np.percentile(image, [low_percentile, high_percentile])
    return float(lo), float(hi)


def apply_stretch(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The clip+gamma half of preview_stretch, taking precomputed bounds (see
    compute_stretch_bounds) instead of computing them from `image` itself."""
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    stretched = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (stretched ** (1.0 / 2.2)).astype(np.float32)


def preview_stretch(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """A simple percentile-stretch + gamma for viewing a raw (un-inverted) negative on screen —
    purely a display convenience, not part of the color pipeline. Returns float32 in [0, 1]."""
    lo, hi = compute_stretch_bounds(image, low_percentile, high_percentile)
    return apply_stretch(image, lo, hi)


def display_to_full_res_coords(
    display_x: int, display_y: int, stride: int, image_width: int, image_height: int
) -> tuple[int, int]:
    """Map a click/hover position on the downsampled display back to clamped full-resolution
    pixel coordinates. Extracted from main_window's on-click handler math so the hover
    handler can reuse the exact same, tested mapping instead of duplicating it."""
    full_x = max(0, min(image_width - 1, display_x * stride))
    full_y = max(0, min(image_height - 1, display_y * stride))
    return full_x, full_y


def full_res_to_display_coords(full_x: int, full_y: int, stride: int) -> tuple[int, int]:
    """The inverse of display_to_full_res_coords — where a full-res point (e.g. a picked shadow/
    highlight point) lands on the downsampled display, for drawing a marker at the right spot."""
    return full_x // stride, full_y // stride


def extract_magnifier_patch(image: np.ndarray, x: int, y: int, radius: int = 12) -> np.ndarray:
    """Crop a fixed-size (2*radius+1, 2*radius+1, 3) patch from `image` centered at (x, y),
    edge-padding near the image border rather than shrinking — the caller uploads this into a
    FIXED-SIZE dpg dynamic texture on every hover event, so the output shape must never change or
    the update degrades into a full texture recreate."""
    h, w = image.shape[:2]
    y0, y1 = y - radius, y + radius + 1
    x0, x1 = x - radius, x + radius + 1
    pad_top, pad_bottom = max(0, -y0), max(0, y1 - h)
    pad_left, pad_right = max(0, -x0), max(0, x1 - w)
    cropped = image[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return cropped


def magnify_patch(patch: np.ndarray, zoom: int = 8) -> np.ndarray:
    """Nearest-neighbor upscale via np.repeat on both axes — deliberately blocky, not smoothed, so
    individual source pixels stay identifiable for precise picking."""
    return np.repeat(np.repeat(patch, zoom, axis=0), zoom, axis=1)
