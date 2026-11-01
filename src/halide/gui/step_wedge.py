"""The coverage bar: a printed step wedge spanning this roll's own density range - film base (prints
black) to the highlight end the print fit anchors to (prints white) - with a numbered tick for each
neutral point and a bracket over the range they cover. Its job is the variety nudge: points spread
across the tones tell the fit much more than points bunched at one tone. When the fit through them
can't be trusted over the whole roll (calibration/anchors.py::fit_reliability), an amber bracket
under the labels marks the span it can be.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from halide.gui import motion, theme

_STEPS = 11
_DROP_PX = 6
_BAR_TOP = 14
_BAR_HEIGHT = 18  # tall enough for the density labels inside the bands
_TICK_ROW = _BAR_TOP + _BAR_HEIGHT + 3
_LABEL_H = 13
_RELIABLE_H = 8
_STUB = 4  # length of a point's tick inside the bar, from each edge  # the reliable-range bracket's row, between the tick labels and the caption


def band_labels(density_range: tuple[float, float], steps: int) -> list[str]:
    """The density at the centre of each band, to 2 decimals."""
    low, high = density_range
    return [f"{low + (i + 0.5) / steps * (high - low):.2f}" for i in range(steps)]


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
        self._reliable: tuple[float, float] | None = None
        self._measuring = True
        self._drop_number: int | None = None  # the tick still settling into place
        self._drop = 0.0  # px it sits above its final place
        self.setFixedHeight(_TICK_ROW + 2 * _LABEL_H + _RELIABLE_H + 18)
        self.setToolTip(
            "The roll's density range as a printed step wedge: the film-base end prints black, the "
            "highlight end white. Each tick is one of your neutral points. Points spread across the "
            "tones give the fit more to work with than several at the same tone."
        )

    def set_state(
        self,
        density_range: tuple[float, float] | None,
        ticks: list[WedgeTick],
        reliable: tuple[float, float] | None = None,
        measuring: bool = True,
    ) -> None:
        """`reliable`: the span the fit can be trusted over, drawn only when it's short of the roll.
        `measuring`: with no density range yet, whether previews are still on their way (else there
        is no roll to measure - "Continue without" - and the caption must not promise one)."""
        self._range = density_range
        self._ticks = list(ticks)
        self._reliable = reliable
        self._measuring = measuring
        self.update()

    def drop_in(self, number: int) -> None:
        """Let the tick for point `number` drop onto the bar from just above it."""
        def set_drop(progress: float) -> None:
            self._drop = _DROP_PX * (1 - progress)
            self.update()

        def done() -> None:
            self._drop_number = None

        self._drop_number = number
        motion.animate(self, "drop", 0.0, 1.0, 200, set_drop, done)

    def _x(self, density: float, left: float, width: float) -> float:
        low, high = self._range
        t = 0.0 if high <= low else (density - low) / (high - low)
        return left + max(0.0, min(1.0, t)) * width

    def _paint_band_labels(self, painter: QPainter, left: float, step_w: float) -> None:
        painter.save()
        font = theme.edge_font(9)
        painter.setFont(font)
        labels = band_labels(self._range, _STEPS)
        text_w = QFontMetrics(font).horizontalAdvance(labels[0])
        # Every other band: a label in all of them crowds the bar and runs under the point ticks.
        stride = 2 if 2 * step_w >= text_w + 4 else 0
        for i in range(0, _STEPS if stride else 0, stride or 1):
            dark = round(255 * (i / (_STEPS - 1)) ** (1 / 2.2)) < 128
            painter.setPen(QColor(theme.TEXT if dark else theme.BACKGROUND))
            painter.drawText(QRectF(left + i * step_w, _BAR_TOP, step_w, _BAR_HEIGHT), Qt.AlignmentFlag.AlignCenter, labels[i])
        painter.restore()

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
            painter.drawText(QRectF(left, _TICK_ROW, width, _LABEL_H * 2), Qt.AlignmentFlag.AlignCenter,
                "measuring the roll…" if self._measuring else "no roll loaded - no density range to show",
            )
            return

        self._paint_band_labels(painter, left, step_w)

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
                dy = -self._drop if tick.number == self._drop_number else 0.0
                painter.setPen(QPen(colour, 3 if tick.selected else 1.5))
                # Two stubs on the bar's edges, not a line across it: the band's density label sits in the middle.
                top, bottom = _BAR_TOP + dy, _BAR_TOP + _BAR_HEIGHT + dy
                painter.drawLine(int(x), int(top), int(x), int(top + _STUB))
                painter.drawLine(int(x), int(bottom - _STUB), int(x), int(bottom + 2))
                row_y = _TICK_ROW + (rank % 2) * _LABEL_H + dy
                label = QRectF(x - 12, row_y, 24, _LABEL_H)
                if tick.selected:
                    bold = QFont(small)
                    bold.setBold(True)
                    painter.setFont(bold)
                painter.drawText(label, Qt.AlignmentFlag.AlignCenter, str(tick.number))
                painter.setFont(small)

        bracket_y = _TICK_ROW + 2 * _LABEL_H + _RELIABLE_H - 2
        if self._reliable is not None:
            lo, hi = (self._x(d, left, width) for d in self._reliable)
            painter.setPen(QPen(QColor(theme.TEXT_WARNING), 1.5))
            painter.drawLine(int(lo), bracket_y, int(hi), bracket_y)
            painter.drawLine(int(lo), bracket_y, int(lo), bracket_y - _RELIABLE_H + 3)
            painter.drawLine(int(hi), bracket_y, int(hi), bracket_y - _RELIABLE_H + 3)

        low, high = self._range
        painter.setPen(QColor(theme.TEXT_DIM))
        caption_y = bracket_y + 2
        if self._ticks:
            densities = [t.density for t in self._ticks]
            caption = f"points cover {min(densities):.2f}–{max(densities):.2f} D  ·  roll {low:.2f}–{high:.2f} D"
        else:
            caption = f"roll spans {low:.2f}–{high:.2f} D"
        painter.drawText(QRectF(left, caption_y, width, 14), Qt.AlignmentFlag.AlignLeft, caption)
