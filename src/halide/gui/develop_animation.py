"""The develop animation's pictures, frame by frame, as a print is made: the lamp up through the
negative, blank paper into the tray, the print's density coming up under the safelight (shadows
first, highlights last), then the lights on. Pure numpy on the strip's 8-bit thumbnails; no Qt."""

from __future__ import annotations

import numpy as np

EXPOSE_MS = 800
TRAY_MS = 3200
LIGHTS_MS = 450
TRAY_HOLD = 0.92  # how far the tray gets while the frame's worker is still busy: just short of full

_SAFE = np.array([1.0, 0.47, 0.26], dtype=np.float32)  # white paper under an amber safelight
_PAPER = 150.0  # how bright that paper looks, of 255
_LAMP = np.array([255.0, 236.0, 205.0], dtype=np.float32)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _linear(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) / 255.0) ** 2.2


def _encode(v: np.ndarray) -> np.ndarray:
    return 255.0 * np.clip(v, 0.0, 1.0) ** (1 / 2.2)


class PrintFrames:
    """One frame's animation: built once from its negative and positive thumbnails (the same
    shape, HxWx3 uint8), then each step is one array expression."""

    def __init__(self, negative: np.ndarray, positive: np.ndarray):
        self.negative = negative.astype(np.float32)
        self.positive = positive.astype(np.float32)
        luminance = _linear(positive) @ _LUMA
        self.density = -np.log10(np.maximum(0.003, luminance))
        # The lighter a tone ends up, the later it starts to show.
        self.lag = 0.62 * (1 - np.minimum(1.0, self.density / 1.6))
        self._paper = _PAPER * _SAFE

    def expose(self, t: float) -> np.ndarray:
        """0 -> 1 over EXPOSE_MS: the lamp glows through the negative, then the paper goes in."""
        glow = np.sin(np.pi * min(1.0, t / 0.75)) * 0.7
        paper = (t - 0.75) / 0.25 if t > 0.75 else 0.0
        lit = self.negative + (_LAMP - self.negative) * glow
        return _u8(lit + (self._paper - lit) * paper)

    def tray(self, t: float) -> np.ndarray:
        """0 -> 1 over TRAY_MS: each pixel's density rises after its lag, levelling off."""
        g = 0.0 if t < 0.1 else (t - 0.1) / 0.9
        q = np.clip((g - self.lag) / (1 - self.lag), 0.0, 1.0)
        f = 1 - (1 - q) ** 2
        return self._safelit(f * self.density)

    def lights(self, t: float, start: np.ndarray | None = None) -> np.ndarray:
        """0 -> 1 over LIGHTS_MS: the safelit print (or `start`, wherever the frame had got to when
        its worker finished) cross-fades to the colour print."""
        e = t * t * (3 - 2 * t)
        start = self._safelit(self.density) if start is None else start
        start = start.astype(np.float32)
        return _u8(start + (self.positive - start) * e)

    def _safelit(self, density: np.ndarray) -> np.ndarray:
        v = _encode(10.0 ** -density) / 255.0 * _PAPER
        return _u8(v[..., None] * _SAFE)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)
