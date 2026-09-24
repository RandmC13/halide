"""The roll as a strip of film across the top of the picker: every frame in order, in the same
visual language as halide's contact sheets and terminal progress display - black rebate, sprocket
holes, orange edge-print frame numbers. The current frame is outlined; frames with neutral points on
them carry a count in the lower rebate. Frames "develop" in as their previews load.

Thumbnails are handed in as QPixmaps (negative or positive, following the main view's switch); this
widget only lays them out, paints the film and reports clicks.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget

from halide.gui import theme

FRAME_HEIGHT = 56
FRAME_WIDTH = round(FRAME_HEIGHT * 3 / 2)  # a 35mm frame's cell; other shapes fitted inside it
GAP = 6
REBATE = 17  # above and below the frames
STRIP_HEIGHT = FRAME_HEIGHT + 2 * REBATE
_HOLES_PER_FRAME = 4  # halved from film's eight so they stay legible at this size
_HOLE_W, _HOLE_H = 7, 6


class _Strip(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmaps: list[QPixmap | None] = []
        self._failed: set[int] = set()
        self._numbers: list[int] = []
        self._counts: list[int] = []
        self._current: int | None = None
        self.setFixedHeight(STRIP_HEIGHT)
        self._resize()

    def _resize(self) -> None:
        count = len(self._pixmaps)
        self.setFixedWidth(max(1, GAP + count * (FRAME_WIDTH + GAP)))

    def frame_rect(self, index: int) -> QRect:
        return QRect(GAP + index * (FRAME_WIDTH + GAP), REBATE, FRAME_WIDTH, FRAME_HEIGHT)

    def set_frames(self, numbers: list[int]) -> None:
        self._numbers = list(numbers)
        self._pixmaps = [None] * len(numbers)
        self._counts = [0] * len(numbers)
        self._failed = set()
        self._current = None
        self._resize()
        self.update()

    def set_pixmap(self, index: int, pixmap: QPixmap | None, failed: bool = False) -> None:
        self._pixmaps[index] = pixmap
        if failed:
            self._failed.add(index)
        self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def set_counts(self, counts: list[int]) -> None:
        self._counts = list(counts)
        self.update()

    def set_current(self, index: int | None) -> None:
        self._current = index
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.FILM_REBATE))

        # sprocket holes along both edges
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.FILM_SPROCKET))
        pitch = (FRAME_WIDTH + GAP) / _HOLES_PER_FRAME
        x = GAP / 2 + (pitch - _HOLE_W) / 2
        while x < self.width():
            for y in (3, STRIP_HEIGHT - 3 - _HOLE_H):
                painter.drawRoundedRect(QRectF(x, y, _HOLE_W, _HOLE_H), 1.5, 1.5)
            x += pitch

        edge_font = QFont(self.font())
        edge_font.setPixelSize(9)
        edge_font.setBold(True)
        painter.setFont(edge_font)
        for i, number in enumerate(self._numbers):
            rect = self.frame_rect(i)
            pixmap = self._pixmaps[i]
            if pixmap is not None:
                scaled = pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(
                    rect.x() + (rect.width() - scaled.width()) // 2, rect.y() + (rect.height() - scaled.height()) // 2, scaled
                )
            else:
                # undeveloped: loading, or a frame that failed to load
                painter.fillRect(rect, QColor("#3a1a18" if i in self._failed else "#1c1916"))

            painter.setPen(QColor(theme.EDGE_PRINT))
            painter.drawText(QRect(rect.x(), 0, rect.width(), REBATE), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f" {number}")
            if self._counts and self._counts[i]:
                painter.drawText(
                    QRect(rect.x(), rect.bottom() + 1, rect.width(), REBATE),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"● {self._counts[i]} ",
                )
            if i == self._current:
                painter.setPen(QPen(QColor(theme.AMBER_ACTIVE), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect.adjusted(-1, -1, 1, 1))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(theme.FILM_SPROCKET))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        for i in range(len(self._numbers)):
            if self.frame_rect(i).adjusted(-GAP // 2, -REBATE, GAP // 2, REBATE).contains(pos):
                self.frameClicked.emit(i)
                return


class Filmstrip(QScrollArea):
    """The strip in a horizontal scroll area; the mouse wheel scrolls it sideways."""

    frameClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.strip = _Strip()
        self.strip.frameClicked.connect(self.frameClicked)
        self.setWidget(self.strip)
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFixedHeight(STRIP_HEIGHT + 12)  # room for the thin horizontal scrollbar
        self.setStyleSheet(f"QScrollArea {{ background: {theme.FILM_REBATE}; border-radius: 4px; }}")

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)

    def show_frame(self, index: int) -> None:
        self.strip.set_current(index)
        self.ensureVisible(self.strip.frame_rect(index).center().x(), STRIP_HEIGHT // 2, FRAME_WIDTH, 0)
