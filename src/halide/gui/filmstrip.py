"""The roll as a strip of film across the top of the picker: every frame in order, in the same
visual language as halide's contact sheets and terminal progress display - black rebate, sprocket
holes, orange edge-print frame numbers. The current frame is outlined; frames with neutral points on
them carry a count in the lower rebate. Frames "develop" in as their previews load.

Thumbnails are handed in as QPixmaps (negative or positive, following the main view's switch); this
widget only lays them out, paints the film and reports clicks.
"""

from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QEvent, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QScrollArea, QToolTip, QWidget

from halide.gui import motion, theme

FRAME_HEIGHT = 56
FRAME_WIDTH = round(FRAME_HEIGHT * 3 / 2)  # a 35mm frame's cell; other shapes fitted inside it
GAP = 6
REBATE = 17  # above and below the frames
STRIP_HEIGHT = FRAME_HEIGHT + 2 * REBATE
_HOLES_PER_FRAME = 4  # halved from film's eight so they stay legible at this size
_HOLE_W, _HOLE_H = 7, 6
_FEED_OFFSET = 14  # px a frame slides in from the left as it develops
_FEED_STAGGER_MS = 45
_FEED_MAX_DELAY_MS = 1000
_FEED_MS = 260


class _Strip(QWidget):
    frameClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmaps: list[QPixmap | None] = []
        self._failed: dict[int, str] = {}  # frame index -> why it didn't load
        self._numbers: list[int] = []
        self._counts: list[int] = []
        self._current: int | None = None
        self._appear: list[float] = []  # per frame 0 -> 1 as its first thumbnail feeds in
        self._generation = 0  # bumped per roll, so a late feed-in of a replaced roll does nothing
        self._since_roll = QElapsedTimer()
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
        self._failed = {}
        self._current = None
        self._appear = [1.0] * len(numbers)
        self._generation += 1
        self._since_roll.start()
        self._resize()
        self.update()

    def set_pixmap(self, index: int, pixmap: QPixmap | None, failed: bool = False, reason: str = "") -> None:
        first = self._pixmaps[index] is None and pixmap is not None
        self._pixmaps[index] = pixmap
        if first:  # only the first print feeds in; later re-prints (view switch) stay put
            self._feed_in(index)
        if failed:
            self._failed[index] = reason or "couldn't load"
        self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def _feed_in(self, index: int) -> None:
        """Slide and fade a frame in, staggered by its place in the roll (measured from the roll's start)."""
        generation = self._generation

        def set_progress(value: float) -> None:
            if generation == self._generation:
                self._appear[index] = value
                self.update(self.frame_rect(index).adjusted(-_FEED_OFFSET - 2, -2, 2, 2))

        def start() -> None:
            if generation == self._generation:
                motion.animate(self, 0.0, 1.0, _FEED_MS, set_progress)

        delay = max(0, min(_FEED_MAX_DELAY_MS, _FEED_STAGGER_MS * index) - self._since_roll.elapsed())
        if motion.duration(_FEED_MS) == 0:
            set_progress(1.0)
        else:
            self._appear[index] = 0.0
            QTimer.singleShot(delay, self, start)

    def mark_failed(self, index: int, reason: str) -> None:
        """Put a warning over a frame (keeping any thumbnail it has), with `reason` as its tooltip."""
        if 0 <= index < len(self._numbers):
            self._failed[index] = reason or "couldn't load"
            self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def clear_failure(self, index: int) -> None:
        if self._failed.pop(index, None) is not None:
            self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def failure_at(self, pos) -> str | None:
        for i in self._failed:
            if self.frame_rect(i).contains(pos):
                return self._failed[i]
        return None

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            reason = self.failure_at(event.pos())
            if reason is not None:
                QToolTip.showText(event.globalPos(), f"Couldn't load this frame: {reason}", self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)

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

        edge_font = theme.edge_font(9)
        painter.setFont(edge_font)
        for i, number in enumerate(self._numbers):
            rect = self.frame_rect(i)
            pixmap = self._pixmaps[i]
            if pixmap is not None:
                appear = self._appear[i]
                if appear < 1.0:
                    painter.fillRect(rect, QColor(theme.FILM_LOADING))
                painter.setOpacity(appear)
                # Scaled to device pixels (and tagged with the ratio), so a HiDPI screen gets a sharp
                # thumbnail rather than a stretched one.
                ratio = self.devicePixelRatioF()
                scaled = pixmap.scaled(
                    round(rect.width() * ratio), round(rect.height() * ratio),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
                )
                scaled.setDevicePixelRatio(ratio)
                painter.drawPixmap(
                    rect.x() + round((rect.width() - scaled.width() / ratio) / 2 - _FEED_OFFSET * (1 - appear)),
                    rect.y() + round((rect.height() - scaled.height() / ratio) / 2),
                    scaled,
                )
                painter.setOpacity(1.0)
            else:
                # undeveloped: loading, or a frame that failed to load
                painter.fillRect(rect, QColor(theme.FILM_FAILED if i in self._failed else theme.FILM_LOADING))

            if i in self._failed:
                warn_font = QFont(self.font())
                warn_font.setPixelSize(22)
                painter.setFont(warn_font)
                painter.setPen(QColor(theme.TEXT_WARNING))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "⚠")
                painter.setFont(edge_font)
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
        if event.button() != Qt.MouseButton.LeftButton:
            return
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

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Left/Right step through the frames (handled by the window), not scroll the strip's own view.
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def show_frame(self, index: int) -> None:
        self.strip.set_current(index)
        self.ensureVisible(self.strip.frame_rect(index).center().x(), STRIP_HEIGHT // 2, FRAME_WIDTH, 0)
