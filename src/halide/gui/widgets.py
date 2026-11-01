"""Small widgets of the picker's window: the eliding status label, the Negative | Positive switch and
the saved-profile stamp."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from halide.gui import motion, theme

# Names in the top bar are cut beyond this width (full name in the tooltip).
PROFILE_NAME_MAX_W = 140
_PILL_PAD = 2  # the switch's recess around its two buttons


class ElidedLabel(QLabel):
    """A one-line label that elides with "…" to the width it's given instead of forcing the window
    wider, and carries its full text as the tooltip (the status line: long paths and errors)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._full = ""
        self.set_full_text(text)

    def full_text(self) -> str:
        return self._full

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def _elide(self) -> None:
        one_line = self._full.replace("\n", " ")
        self.setText(QFontMetrics(self.font()).elidedText(one_line, Qt.TextElideMode.ElideRight, max(0, self.width())))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()


class ViewSwitch(QWidget):
    """The Negative | Positive switch: one recessed pill holding two checkable buttons (exclusive, in
    a QButtonGroup), with an amber highlight painted behind the checked one that slides across when
    the choice changes. The buttons are bare text; the pill, highlight and focus mark are painted
    here."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.negative_button = QPushButton("Negative")
        self.positive_button = QPushButton("Positive")
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        row = QHBoxLayout(self)
        row.setContentsMargins(_PILL_PAD, _PILL_PAD, _PILL_PAD, _PILL_PAD)
        row.setSpacing(0)
        for button in (self.negative_button, self.positive_button):
            button.setProperty("role", "segment")
            button.setCheckable(True)
            self.group.addButton(button)
            button.installEventFilter(self)  # repaint the focus mark as focus moves
            row.addWidget(button)
        self.negative_button.setChecked(True)
        self._from: QRectF | None = None  # where a slide in progress began
        self._progress = 1.0
        self.group.buttonToggled.connect(lambda button, checked: checked and self._slide_from_other(button))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            self.update()
        return False

    def _checked_rect(self) -> QRectF:
        button = self.group.checkedButton()
        return QRectF(button.geometry()) if button is not None else QRectF()

    def highlight_rect(self) -> QRectF:
        target = self._checked_rect()
        if self._from is None or self._progress >= 1.0:
            return target
        t = self._progress
        return QRectF(
            self._from.x() + (target.x() - self._from.x()) * t, target.y(),
            self._from.width() + (target.width() - self._from.width()) * t, target.height(),
        )

    def _slide_from_other(self, checked) -> None:
        other = self.positive_button if checked is self.negative_button else self.negative_button
        self._from = QRectF(other.geometry())

        def set_progress(value: float) -> None:
            self._progress = value
            self.update()

        self._progress = 0.0
        motion.animate(self, "slide", 0.0, 1.0, 220, set_progress)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(theme.BORDER))
        painter.setBrush(QColor(theme.BACKGROUND_ALT))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.AMBER))
        painter.drawRoundedRect(self.highlight_rect(), 3, 3)
        focused = next((b for b in (self.negative_button, self.positive_button) if b.hasFocus()), None)
        if focused is not None:  # keyboard focus: the bare-text buttons draw none of their own
            painter.setPen(QPen(QColor(theme.TEXT), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(focused.geometry()).adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)


class SaveStamp(QWidget):
    """"<NAME> → SAVED" stamped in edge print over a strip of rebate, settling from wide letter
    spacing to tight as it fades in."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = ""
        self._progress = 1.0
        self.setFixedHeight(22)
        self.hide()

    def text(self) -> str:
        return self._text

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        font = theme.edge_font(11)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        return QSize(QFontMetrics(font).horizontalAdvance(self._text) + 20, 22)  # 4 spare: Qt elides text that fits exactly

    def stamp(self, name: str) -> None:
        font = theme.edge_font(11)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        short = QFontMetrics(font).elidedText(name.upper(), Qt.TextElideMode.ElideMiddle, PROFILE_NAME_MAX_W)
        self._text = f"{short} \u2192 SAVED"
        self.setToolTip(name)
        self.setMinimumWidth(self.sizeHint().width())  # the top bar's stretch must not cut it further
        self.show()

        def set_progress(value: float) -> None:
            self._progress = value
            self.update()

        self._progress = 0.0
        motion.animate(self, "stamp", 0.0, 1.0, 350, set_progress)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.FILM_REBATE))
        font = theme.edge_font(11)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4 - 2.8 * self._progress)
        painter.setFont(font)
        painter.setOpacity(self._progress)
        painter.setPen(QColor(theme.EDGE_PRINT))
        text = QFontMetrics(font).elidedText(self._text, Qt.TextElideMode.ElideRight, self.width() - 16)
        painter.drawText(self.rect().adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
