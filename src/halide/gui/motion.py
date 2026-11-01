"""Small eased animations for the picker, all switched off by HALIDE_MOTION=0 (tests set it, so
they are instant and deterministic; anyone who dislikes motion can too)."""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QObject, Qt, QVariantAnimation


def duration(ms: int) -> int:
    return 0 if os.environ.get("HALIDE_MOTION") == "0" else ms


def stop(owner: QObject, key: str) -> None:
    """Stop `owner`'s running animation named `key`, if there is one (its on_done doesn't run)."""
    anim = owner.findChild(QVariantAnimation, key, Qt.FindChildOption.FindDirectChildrenOnly)
    if anim is not None:
        anim.setObjectName("")  # deleted only later: it must not be found again as the running one
        anim.stop()


def animate(
    owner: QObject, key: str, start: float, end: float, ms: int, on_value: Callable[[float], None],
    on_done: Callable[[], None] | None = None, curve: QEasingCurve.Type = QEasingCurve.Type.OutCubic,
) -> None:
    """Ease `start` -> `end` over `ms`, calling `on_value` for each step. The animation is parented
    to `owner` (so it dies with it), named `key`, and started at once; a running animation of
    `owner`'s with the same name is stopped first, so two never fight over one value. Callers keep
    no reference: a finished animation deletes itself, and a kept one would be a dangling pointer
    (stop() finds the live one by name instead). With motion off it jumps: `on_value(end)` and
    `on_done` run synchronously."""
    stop(owner, key)
    ms = duration(ms)
    if ms == 0:
        on_value(end)
        if on_done is not None:
            on_done()
        return
    anim = QVariantAnimation(owner)
    anim.setObjectName(key)
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.setDuration(ms)
    anim.setEasingCurve(curve)
    anim.valueChanged.connect(lambda value: on_value(float(value)))
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
