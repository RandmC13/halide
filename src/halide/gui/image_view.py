"""The picker's image view: the frame in its film carrier with edge print, numbered point markers,
and the loupe painted over the image."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from halide.gui import motion, theme

# The frame sits in a strip of film: more rebate above and below (where the edge print runs) than at the sides.
_CARRIER_MARGIN_X = 16
_CARRIER_MARGIN_Y = 22
_MARKER_RADIUS = 7
_LOUPE = 120  # the lens's diameter
_LOUPE_OFFSET = QPointF(22, -150)  # from the cursor to the lens's top-left: above and to the right
_READOUT_SIZE = QSizeF(80, 22)


class ImageView(QWidget):
    """Paints the current frame scaled to fit its area, surrounded by film rebate carrying the edge
    print, plus numbered point markers; reports clicks and hovers in the *displayed image's* pixel coordinates
    (MainWindow maps those to full resolution via gui/sampling.py)."""

    clicked = Signal(int, int)
    hovered = Signal(int, int)
    left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        self._placeholder = "Drop a roll folder here, or Load roll…"
        self._edge_print = ""
        self.markers: list[tuple[int, int, int, str, bool]] = []  # (x, y, number, colour, selected)
        self._flash: int | None = None  # number of a marker currently flashing
        self._flash_on = False
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._flash_step)
        self._flash_steps = 0
        self._old_pixmap: QPixmap | None = None  # the picture being faded out, if a fade is running
        self._blend = 1.0  # how much of the new picture shows
        # The loupe is painted here, over the image, rather than in a window of its own: a separate
        # window can't be placed exactly or made round on Wayland, and showing and hiding one on every
        # mouse move made Qt's Wayland text-input warn in the terminal.
        self._loupe: tuple[QPixmap, float, QPointF] | None = None  # (patch, density, cursor)
        self._probe_number: int | None = None  # the marker still closing in on its spot
        self._probe = 1.0

    def set_image(self, pixmap: QPixmap | None, placeholder: str = "", fade: bool = False) -> None:
        """`fade`: cross-fade from the picture shown now (the Negative/Positive switch)."""
        motion.stop(self, "fade")
        self._old_pixmap = self._pixmap if fade and pixmap is not None else None
        self._pixmap = pixmap
        if placeholder:
            self._placeholder = placeholder

        def set_blend(value: float) -> None:
            self._blend = value
            self.update()

        def done() -> None:
            self._old_pixmap = None

        self._blend = 0.0 if self._old_pixmap is not None else 1.0
        if self._old_pixmap is not None:
            motion.animate(self, "fade", 0.0, 1.0, 250, set_blend, done)
        self.update()

    def set_loupe(self, pixmap: QPixmap, density: float, cursor: QPointF) -> None:
        self._loupe = (pixmap, density, cursor)
        self.update()

    def clear_loupe(self) -> None:
        if self._loupe is not None:
            self._loupe = None
            self.update()

    def loupe_rect(self) -> QRectF | None:
        return None if self._loupe is None else loupe_rect(self._loupe[2], QRectF(self.rect()))

    def probe(self, number: int) -> None:
        """Close marker `number` in on its spot, as a probe lands (a newly added point)."""
        def set_probe(value: float) -> None:
            self._probe = value
            self.update()

        def done() -> None:
            self._probe_number = None

        self._probe_number = number
        motion.animate(self, "probe", 0.0, 1.0, 200, set_probe, done)

    def set_edge_print(self, text: str) -> None:
        self._edge_print = text
        self.update()

    def flash(self, number: int) -> None:
        self._flash, self._flash_steps = number, 6
        self._flash_timer.start(140)

    def _flash_step(self) -> None:
        self._flash_on = not self._flash_on
        self._flash_steps -= 1
        if self._flash_steps <= 0:
            self._flash_timer.stop()
            self._flash, self._flash_on = None, False
        self.update()

    def _image_rect(self) -> QRectF | None:
        """Where the image is drawn: scaled to fit inside the rebate, centred."""
        return None if self._pixmap is None else self._fit_rect(self._pixmap)

    def _fit_rect(self, pixmap: QPixmap) -> QRectF:
        avail_w = self.width() - 2 * _CARRIER_MARGIN_X
        avail_h = self.height() - 2 * _CARRIER_MARGIN_Y
        scale = min(avail_w / pixmap.width(), avail_h / pixmap.height())
        w, h = pixmap.width() * scale, pixmap.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _scale(self) -> float:
        rect = self._image_rect()
        return rect.width() / self._pixmap.width() if rect is not None else 1.0

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._image_rect()
        if rect is None:
            painter.fillRect(self.rect(), QColor(theme.FILM_REBATE))
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        painter.fillRect(self.rect(), QColor(theme.FILM_REBATE))
        self._paint_edge_print(painter)
        if self._old_pixmap is not None and self._blend < 1.0:
            # In its own fitted rect: the outgoing picture may be another shape (a frame still loading).
            painter.drawPixmap(self._fit_rect(self._old_pixmap), self._old_pixmap, QRectF(self._old_pixmap.rect()))
        painter.setOpacity(self._blend)
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setOpacity(1.0)

        scale = self._scale()
        label_font = QFont(self.font())
        label_font.setPixelSize(10)
        label_font.setBold(True)
        painter.setFont(label_font)
        for x, y, number, colour, selected in self.markers:
            if number == self._flash and not self._flash_on:
                continue
            centre = QPointF(rect.x() + (x + 0.5) * scale, rect.y() + (y + 0.5) * scale)
            landing = number == self._probe_number
            radius = _MARKER_RADIUS * (1 + 1.6 * (1 - self._probe)) if landing else _MARKER_RADIUS
            painter.setOpacity(self._probe if landing else 1.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 170), 4 if selected else 3))
            painter.drawEllipse(centre, radius, radius)
            painter.setPen(QPen(QColor(colour), 2.5 if selected else 1.5))
            painter.drawEllipse(centre, radius, radius)
            label = QRectF(centre.x() + _MARKER_RADIUS, centre.y() - _MARKER_RADIUS - 12, 22, 14)
            painter.setPen(QColor(0, 0, 0, 200))
            painter.drawText(label.translated(1, 1), Qt.AlignmentFlag.AlignLeft, str(number))
            painter.setPen(QColor(colour))
            painter.drawText(label, Qt.AlignmentFlag.AlignLeft, str(number))
            painter.setOpacity(1.0)
        if self._loupe is not None:
            pixmap, density, _ = self._loupe
            paint_loupe(painter, self.loupe_rect(), pixmap, density)

    def _paint_edge_print(self, painter: QPainter) -> None:
        if not self._edge_print:
            return
        painter.setFont(theme.edge_font(10))
        colour = QColor(theme.EDGE_PRINT)
        painter.setPen(colour)
        strip = self.height() - _CARRIER_MARGIN_Y
        painter.drawText(
            QRectF(_CARRIER_MARGIN_X, strip, self.width(), _CARRIER_MARGIN_Y),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._edge_print,
        )
        colour.setAlphaF(0.8)
        painter.setPen(colour)
        painter.drawText(
            QRectF(0, 0, self.width() - _CARRIER_MARGIN_X, _CARRIER_MARGIN_Y),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "HALIDE",
        )

    def _to_image(self, event) -> tuple[int, int] | None:
        rect = self._image_rect()
        if rect is None:
            return None
        pos = event.position()
        if not rect.contains(pos):
            return None
        scale = self._scale()
        return int((pos.x() - rect.x()) / scale), int((pos.y() - rect.y()) / scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        point = self._to_image(event) if event.button() == Qt.MouseButton.LeftButton else None
        if point is not None:
            self.clicked.emit(*point)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = self._to_image(event)
        if point is not None:
            self.hovered.emit(*point)
        else:
            self.left.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.left.emit()

    def display_scale(self) -> float:
        return self._scale()


def loupe_rect(cursor: QPointF, bounds: QRectF) -> QRectF:
    """Where the lens goes for a cursor at `cursor`: above and to the right of it, flipped below or to
    the left where that would leave `bounds` (the image view), so it never hides under an edge."""
    x, y = cursor.x() + _LOUPE_OFFSET.x(), cursor.y() + _LOUPE_OFFSET.y()
    if x + _LOUPE > bounds.right():
        x = cursor.x() - _LOUPE_OFFSET.x() - _LOUPE
    if y < bounds.top():
        y = cursor.y() + _LOUPE_OFFSET.x()
    return QRectF(x, y, _LOUPE, _LOUPE)


def paint_loupe(painter: QPainter, lens: QRectF, pixmap: QPixmap, density: float) -> None:
    """A round lens of `pixmap` with a crosshair and a ring on the pixel a click would sample, and
    the density that click would read in edge print underneath."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setOpacity(1.0)
    path = QPainterPath()
    path.addEllipse(lens)
    painter.save()
    painter.setClipPath(path)
    painter.drawPixmap(lens, pixmap, QRectF(pixmap.rect()))
    painter.restore()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(0, 0, 0, 120), 6))  # a soft dark edge, so the ring reads on light frames
    painter.drawEllipse(lens.adjusted(-2, -2, 2, 2))
    painter.setPen(QPen(QColor(theme.LOUPE_RING), 3))
    painter.drawEllipse(lens.adjusted(1.5, 1.5, -1.5, -1.5))
    centre = lens.center()
    painter.setPen(QPen(QColor(255, 255, 255, 128), 1))
    painter.drawLine(QPointF(lens.left() + 6, centre.y()), QPointF(centre.x() - 8, centre.y()))
    painter.drawLine(QPointF(centre.x() + 8, centre.y()), QPointF(lens.right() - 6, centre.y()))
    painter.drawLine(QPointF(centre.x(), lens.top() + 6), QPointF(centre.x(), centre.y() - 8))
    painter.drawLine(QPointF(centre.x(), centre.y() + 8), QPointF(centre.x(), lens.bottom() - 6))
    painter.drawEllipse(centre, 8, 8)
    readout = QRectF(centre.x() - _READOUT_SIZE.width() / 2, lens.bottom() + 4, _READOUT_SIZE.width(), _READOUT_SIZE.height())
    painter.setBrush(QColor(theme.READOUT_BG))
    painter.setPen(QColor(theme.BORDER))
    painter.drawRoundedRect(readout, 3, 3)
    painter.setFont(theme.edge_font(12))
    painter.setPen(QColor(theme.EDGE_PRINT))  # not a status colour: a density isn't good or bad
    painter.drawText(readout, Qt.AlignmentFlag.AlignCenter, loupe_text(density))
    painter.restore()


def loupe_text(density: float) -> str:
    return f"D {density:.2f}"
