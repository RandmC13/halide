"""The coverage bar: a printed step wedge spanning this roll's own density range - film base (prints
black) to the highlight end the print fit anchors to (prints white) - with a numbered tick for each
neutral point and a bracket over the range they cover. Its job is the variety nudge: points spread
across the tones tell the fit much more than points bunched at one tone.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from halide.gui import theme

_STEPS = 11
_BAR_TOP = 14
_BAR_HEIGHT = 14
_TICK_ROW = _BAR_TOP + _BAR_HEIGHT + 3
_LABEL_H = 13


@dataclass(frozen=True)
class WedgeTick:
    number: int  # 1-based point number, as the list and markers show it
    density: float
    colour: str
    selected: bool = False


class StepWedge(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._range: tuple[float, float] | None = None
        self._ticks: list[WedgeTick] = []
        self.setFixedHeight(_TICK_ROW + 2 * _LABEL_H + 18)
        self.setToolTip(
            "The roll's density range as a printed step wedge: the film-base end prints black, the "
            "highlight end white. Each tick is one of your neutral points. Points spread across the "
            "tones give the fit more to work with than several at the same tone."
        )

    def set_state(self, density_range: tuple[float, float] | None, ticks: list[WedgeTick]) -> None:
        self._range = density_range
        self._ticks = list(ticks)
        self.update()

    def _x(self, density: float, left: float, width: float) -> float:
        low, high = self._range
        t = 0.0 if high <= low else (density - low) / (high - low)
        return left + max(0.0, min(1.0, t)) * width

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        small = QFont(self.font())
        small.setPixelSize(10)
        painter.setFont(small)

        left, width = 4.0, float(self.width() - 8)
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.drawText(QRectF(left, 0, width, _BAR_TOP - 2), Qt.AlignmentFlag.AlignLeft, "film base")
        painter.drawText(QRectF(left, 0, width, _BAR_TOP - 2), Qt.AlignmentFlag.AlignRight, "roll highlights")

        step_w = width / _STEPS
        for i in range(_STEPS):
            level = round(255 * (i / (_STEPS - 1)) ** (1 / 2.2))
            painter.fillRect(QRectF(left + i * step_w, _BAR_TOP, step_w + 0.5, _BAR_HEIGHT), QColor(level, level, level))
        painter.setPen(QColor(theme.BORDER))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(left, _BAR_TOP, width, _BAR_HEIGHT))

        if self._range is None:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(QRectF(left, _TICK_ROW, width, _LABEL_H * 2), Qt.AlignmentFlag.AlignCenter, "measuring the roll…")
            return

        if self._ticks:
            xs = [self._x(t.density, left, width) for t in self._ticks]
            painter.setPen(QPen(QColor(theme.AMBER_ACTIVE), 2))
            lo, hi = min(xs), max(xs)
            painter.drawLine(int(lo), _BAR_TOP - 3, int(hi), _BAR_TOP - 3)

            # Labels in two alternating rows by position, so neighbouring ticks don't overprint.
            order = sorted(range(len(self._ticks)), key=lambda i: xs[i])
            for rank, i in enumerate(order):
                tick, x = self._ticks[i], xs[i]
                colour = QColor(tick.colour)
                painter.setPen(QPen(colour, 3 if tick.selected else 1.5))
                painter.drawLine(int(x), _BAR_TOP, int(x), _BAR_TOP + _BAR_HEIGHT + 2)
                row_y = _TICK_ROW + (rank % 2) * _LABEL_H
                label = QRectF(x - 12, row_y, 24, _LABEL_H)
                if tick.selected:
                    bold = QFont(small)
                    bold.setBold(True)
                    painter.setFont(bold)
                painter.drawText(label, Qt.AlignmentFlag.AlignCenter, str(tick.number))
                painter.setFont(small)

        low, high = self._range
        painter.setPen(QColor(theme.TEXT_DIM))
        caption_y = _TICK_ROW + 2 * _LABEL_H + 2
        if self._ticks:
            densities = [t.density for t in self._ticks]
            caption = f"points cover {min(densities):.2f}–{max(densities):.2f} D  ·  roll {low:.2f}–{high:.2f} D"
        else:
            caption = f"roll spans {low:.2f}–{high:.2f} D"
        painter.drawText(QRectF(left, caption_y, width, 14), Qt.AlignmentFlag.AlignLeft, caption)
