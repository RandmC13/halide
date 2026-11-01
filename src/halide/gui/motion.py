"""Small eased animations for the picker, all switched off by HALIDE_MOTION=0 (tests set it, so
they are instant and deterministic; anyone who dislikes motion can too)."""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QObject, QVariantAnimation


def duration(ms: int) -> int:
    return 0 if os.environ.get("HALIDE_MOTION") == "0" else ms


def animate(
    owner: QObject, start: float, end: float, ms: int, on_value: Callable[[float], None],
    on_done: Callable[[], None] | None = None,
) -> QVariantAnimation | None:
    """Ease `start` -> `end` over `ms`, calling `on_value` for each step. The animation is parented
    to `owner` (so it dies with it) and started at once. With motion off it jumps: `on_value(end)`
    and `on_done` run synchronously and the return value is None."""
    ms = duration(ms)
    if ms == 0:
        on_value(end)
        if on_done is not None:
            on_done()
        return None
    anim = QVariantAnimation(owner)
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.setDuration(ms)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.valueChanged.connect(lambda value: on_value(float(value)))
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim
