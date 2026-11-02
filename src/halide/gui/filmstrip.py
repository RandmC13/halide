"""The roll as a strip of film across the top of the picker: every frame in order, in the same
visual language as halide's contact sheets and terminal progress display - black rebate, sprocket
holes, orange edge-print frame numbers. The current frame is outlined; frames with neutral points on
them carry a count in the lower rebate, frames scanned differently from the rest of the roll a ⚠
there, and frames whose colour runs away from the rest of the roll a word in its centre (WARM, COOL,
GREEN, MAGENTA). Frames "develop" in as their previews load.

Thumbnails are handed in as QPixmaps (negative or positive, following the main view's switch); this
widget only lays them out, paints the film and reports clicks.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEasingCurve, QElapsedTimer, QEvent, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QScrollArea, QToolTip, QWidget

from halide.gui import motion, theme
from halide.gui.develop_animation import EXPOSE_MS, LIGHTS_MS, TRAY_HOLD, TRAY_MS, PrintFrames

FRAME_HEIGHT = 56
FRAME_WIDTH = round(FRAME_HEIGHT * 3 / 2)  # a 35mm frame's cell; other shapes fitted inside it
GAP = 6
ROLL_GAP = 18  # extra space between one roll's last frame and the next roll's first
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
        self._flags: dict[int, str] = {}  # frame index -> how it was scanned differently
        self._marks: dict[int, str] = {}  # frame index -> the way its colour runs ("warm")
        self._numbers: list[int] = []
        self._roll_starts: list[tuple[int, str]] = []  # (first frame index, roll name), several rolls only
        self._counts: list[int] = []
        self._current: int | None = None
        self._appear: list[float] = []  # per frame 0 -> 1 as its first thumbnail feeds in
        self._generation = 0  # bumped per roll, so a late feed-in of a replaced roll does nothing
        self._since_roll = QElapsedTimer()
        # Frames being printed while the roll develops: index -> [phase, PrintFrames, the picture now
        # as a pixmap, and as an array].
        # A printed frame keeps its print until clear_prints().
        self._prints: dict[int, list] = {}
        self.setFixedHeight(STRIP_HEIGHT)
        self._resize()

    def _resize(self) -> None:
        count = len(self._pixmaps)
        self.setFixedWidth(max(1, self._x(count)))

    def _x(self, index: int) -> int:
        """Left edge of frame `index`, counting the extra gap before each roll after the first."""
        gaps = sum(start <= index for start, _ in self._roll_starts[1:])
        return GAP + index * (FRAME_WIDTH + GAP) + gaps * ROLL_GAP

    def frame_rect(self, index: int) -> QRect:
        return QRect(self._x(index), REBATE, FRAME_WIDTH, FRAME_HEIGHT)

    def set_frames(self, numbers: list[int], roll_starts: list[tuple[int, str]] | None = None) -> None:
        self._numbers = list(numbers)
        self._roll_starts = list(roll_starts or [])
        self._pixmaps = [None] * len(numbers)
        self._counts = [0] * len(numbers)
        self._failed = {}
        self._flags = {}
        self._marks = {}
        self._current = None
        self._appear = [1.0] * len(numbers)
        self._generation += 1
        self._since_roll.start()
        self.clear_prints()
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
                motion.animate(self, f"feed-{index}", 0.0, 1.0, _FEED_MS, set_progress)

        delay = max(0, min(_FEED_MAX_DELAY_MS, _FEED_STAGGER_MS * index) - self._since_roll.elapsed())
        if motion.duration(_FEED_MS) == 0:
            set_progress(1.0)
        else:
            self._appear[index] = 0.0
            QTimer.singleShot(delay, self, start)

    # --- developing: each frame printed as its worker goes ------------------------------------

    def print_phase(self, index: int) -> str | None:
        """expose, tray, lights or printed while the roll develops; None for a queued negative."""
        entry = self._prints.get(index)
        return entry[0] if entry else None

    def _show_print(self, index: int, rgb: np.ndarray) -> None:
        if index in self._prints:
            self._prints[index][2:] = [_pixmap(rgb), rgb]
            self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def print_start(self, index: int, negative: np.ndarray, positive: np.ndarray) -> None:
        """The frame's worker has started: the lamp comes on through the negative, then the paper
        goes into the tray and develops until print_done. Both thumbnails are HxWx3 uint8."""
        if not 0 <= index < len(self._numbers):
            return
        self.clear_failure(index)
        frames = PrintFrames(negative, positive)
        self._prints[index] = ["expose", frames, _pixmap(negative), negative]
        key = f"print-{index}"

        def tray() -> None:
            if self.print_phase(index) == "expose":
                self._prints[index][0] = "tray"
                # Linear: the tray's own curve is in PrintFrames.tray; it holds at TRAY_HOLD until done.
                motion.animate(self, key, 0.0, TRAY_HOLD, round(TRAY_MS * TRAY_HOLD),
                               lambda t: self._show_print(index, frames.tray(t)), curve=QEasingCurve.Type.Linear)

        motion.animate(self, key, 0.0, 1.0, EXPOSE_MS, lambda t: self._show_print(index, frames.expose(t)),
                       on_done=tray, curve=QEasingCurve.Type.Linear)

    def print_done(self, index: int, ok: bool, reason: str = "") -> None:
        """The frame's worker is done: lights on, from wherever the print had got to, to the colour
        print; or, if it failed, back to the negative with a warning."""
        entry = self._prints.get(index)
        if entry is None:
            if not ok:
                self.mark_failed(index, reason)
            return
        key = f"print-{index}"
        motion.stop(self, key)
        if not ok:
            del self._prints[index]
            self.mark_failed(index, reason)
            return
        entry[0] = "lights"
        frames, start = entry[1], entry[3]

        def lights(t: float) -> None:
            self._show_print(index, frames.lights(t, start))

        def printed() -> None:
            if index in self._prints:
                self._prints[index][0] = "printed"

        motion.animate(self, key, 0.0, 1.0, LIGHTS_MS, lights, on_done=printed, curve=QEasingCurve.Type.Linear)

    def drop_unfinished(self) -> None:
        """The run ended (cancelled, or the child stopped): frames still being printed are negatives again."""
        for index in [i for i, entry in self._prints.items() if entry[0] in ("expose", "tray")]:
            motion.stop(self, f"print-{index}")
            del self._prints[index]
        self.update()

    def clear_prints(self) -> None:
        for index in list(self._prints):
            motion.stop(self, f"print-{index}")
        self._prints = {}
        self.update()

    def mark_failed(self, index: int, reason: str) -> None:
        """Put a warning over a frame (keeping any thumbnail it has), with `reason` as its tooltip."""
        if 0 <= index < len(self._numbers):
            self._failed[index] = reason or "couldn't load"
            self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def clear_failure(self, index: int) -> None:
        if self._failed.pop(index, None) is not None:
            self.update(self.frame_rect(index).adjusted(-2, -2, 2, 2))

    def set_flags(self, flags: dict[int, str]) -> None:
        """Mark frames scanned differently from the rest of the roll: frame index -> the reason."""
        self._flags = dict(flags)
        self.update()

    def set_marks(self, marks: dict[int, str]) -> None:
        """Name frames whose colour differs from the rest of the roll: frame index -> its direction."""
        self._marks = dict(marks)
        self.update()

    def failure_at(self, pos) -> str | None:
        for i in self._failed:
            if self.frame_rect(i).contains(pos):
                return self._failed[i]
        return None

    def tooltip_at(self, pos) -> str | None:
        reason = self.failure_at(pos)
        if reason is not None:
            return f"Couldn't load this frame: {reason}"
        for i, flag in self._flags.items():
            if self.frame_rect(i).adjusted(0, -REBATE, 0, REBATE).contains(pos):
                return f"Scanned differently: {flag}"
        return None

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            text = self.tooltip_at(event.pos())
            if text is not None:
                QToolTip.showText(event.globalPos(), text, self)
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
            printing = self._prints.get(i)
            pixmap = printing[2] if printing else self._pixmaps[i]
            if pixmap is not None:
                appear = 1.0 if printing else self._appear[i]
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
            name = next((n for start, n in self._roll_starts if start == i), None)
            if name:  # the roll's name follows its first frame number, in the same edge print
                x = rect.x() + painter.fontMetrics().horizontalAdvance(f" {number}") + 8
                # Cut to its first frame, so it never runs into the next frame's number.
                name = painter.fontMetrics().elidedText(name.upper(), Qt.TextElideMode.ElideRight, rect.right() - x)
                painter.drawText(QRect(x, 0, rect.right() - x, REBATE), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
            if self._counts and self._counts[i]:
                painter.drawText(
                    QRect(rect.x(), rect.bottom() + 1, rect.width(), REBATE),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"● {self._counts[i]} ",
                )
            if i in self._marks:  # centred, clear of the flag on the left and the count on the right
                below = QRect(rect.x() + 16, rect.bottom() + 1, rect.width() - 32, REBATE)
                word = painter.fontMetrics().elidedText(self._marks[i].upper(), Qt.TextElideMode.ElideRight, below.width())
                painter.drawText(below, Qt.AlignmentFlag.AlignCenter, word)
            if i in self._flags:
                _paint_flag(painter, QPointF(rect.x() + 8, rect.bottom() + 1 + REBATE / 2))
                painter.setFont(edge_font)
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


def _pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def _paint_flag(painter: QPainter, centre: QPointF) -> None:
    """A small filled warning triangle with a "!": the font's ⚠ is a thin outline at rebate size."""
    w, h = 11.0, 10.0
    path = QPainterPath()
    path.moveTo(centre.x(), centre.y() - h / 2)
    path.lineTo(centre.x() + w / 2, centre.y() + h / 2)
    path.lineTo(centre.x() - w / 2, centre.y() + h / 2)
    path.closeSubpath()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(theme.TEXT_WARNING))
    painter.drawPath(path)
    painter.setPen(QPen(QColor(theme.FILM_REBATE), 1.4))
    painter.drawLine(QPointF(centre.x(), centre.y() - 1.5), QPointF(centre.x(), centre.y() + 2))
    painter.drawPoint(QPointF(centre.x(), centre.y() + 3.7))
    painter.setBrush(Qt.BrushStyle.NoBrush)


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
