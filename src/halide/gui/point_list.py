"""The neutral-point list in the picker's side panel: one compact row per point - number, frame,
density, agreement with the other points, remove. Fixed height: past a handful of points it scrolls
rather than growing the window (the window is a fixed size, see main_window.py). Clicking a row
selects that point and jumps to its frame; the markers on the image and the step wedge's ticks use
the same numbers and colours.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from halide.gui import theme
from halide.gui.roll import PointView

ROW_HEIGHT = 24

CC_EXPLANATION = (
    "How far each point is from what the other points agree neutral is, as a colour-printing filter "
    "pack: CC is Kodak's Colour Compensating scale (density × 100, so CC 10 = 0.10), the unit of a "
    "dichroic enlarger head's filter dials. The pack is the cast the point would print, in Y/M/C "
    "dials as you'd set them on the head - a red cast reads as M + Y, and CC 20Y + 10M is an "
    "orange-yellow. Measured on the density-balanced negative, before the paper curve. Shown from 3 points: with 2, the line passes through both.\n\n"
    "Around CC 5 or more, check the object really was neutral (a cream wall, a sunlit cloud edge, "
    "skin...). Deal with the worst point first: one bad point makes the others read a few CC off too."
)


def no_wrap(text: str) -> str:
    """Non-breaking spaces, so a filter pack ("CC 10Y + 6M") never wraps or elides mid-pack."""
    return text.replace(" ", "\u00a0")


def agreement_colour(view: PointView) -> str:
    if view.agreement is None:
        return theme.AGREE_NONE
    return {"calm": theme.AGREE_CALM, "amber": theme.AGREE_AMBER, "red": theme.AGREE_RED}[view.agreement.band]


class _ElidedNameLabel(QLabel):
    """The frame name, cut with "…" when the row is narrower than it (the tooltip has it in full)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._elide()

    def _elide(self) -> None:
        self.setText(QFontMetrics(self.font()).elidedText(self._full, Qt.TextElideMode.ElideRight, max(0, self.width())))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()


class _Row(QFrame):
    clicked = Signal(int)
    removeClicked = Signal(int)

    def __init__(self, view: PointView, selected: bool) -> None:
        super().__init__()
        self._index = view.index
        self.setFixedHeight(ROW_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        background = theme.BORDER if selected else "transparent"
        self.setStyleSheet(f"QFrame {{ background: {background}; border-radius: 4px; }}")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 2, 0)
        layout.setSpacing(8)

        colour = agreement_colour(view)
        number = QLabel(f"{view.index + 1}")
        number.setFixedWidth(18)
        number.setStyleSheet(f"color: {colour}; font-weight: bold;")
        layout.addWidget(number)

        frame_text = f"#{view.frame_number}  {view.point.frame.stem}" if view.frame_number else f"{view.point.frame.stem} (missing)"
        if view.note:
            frame_text += "  ⚠"
        frame = _ElidedNameLabel(frame_text)
        frame.setToolTip(f"{view.point.frame}\n{view.note}" if view.note else str(view.point.frame))
        layout.addWidget(frame, stretch=1)

        density = QLabel(f"D {view.green_density:.2f}")
        density.setProperty("role", "dim")
        layout.addWidget(density)

        agreement = QLabel(f"● {no_wrap(view.agreement.label())}" if view.agreement else "·")
        agreement.setStyleSheet(f"color: {colour};")
        agreement.setFixedWidth(104)  # "● CC 20Y + 10M"
        layout.addWidget(agreement)

        remove = QPushButton("✕")
        remove.setProperty("role", "remove")
        remove.setToolTip("Remove this point")
        remove.setFixedWidth(22)
        remove.clicked.connect(lambda: self.removeClicked.emit(self._index))
        layout.addWidget(remove)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self._index)


class PointList(QScrollArea):
    rowClicked = Signal(int)
    removeClicked = Signal(int)

    def __init__(self, visible_rows: int = 5, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # One row: the list is what gives up height first when the panel is squeezed (a drawer open
        # with the reliability note showing, smallest window) so nothing below it clips.
        self.setMinimumHeight(ROW_HEIGHT + 4)
        self.setMaximumHeight(ROW_HEIGHT * visible_rows + 12)
        self._body = QWidget()
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(0, 0, 4, 0)
        self._layout.setSpacing(2)
        self.setWidget(self._body)
        self._rows: list[_Row] = []
        self.set_rows([], None)

    def set_rows(self, views: list[PointView], selected: int | None) -> None:
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                # Detach now: deleteLater() alone leaves the old row visible (and painted under the
                # new one) until the event loop gets round to deleting it.
                widget.setParent(None)
                widget.deleteLater()
        self._rows = []
        if not views:
            empty = QLabel("No points yet — click something that was white, grey or black in the real scene.")
            empty.setWordWrap(True)
            empty.setProperty("role", "dim")
            self._layout.addWidget(empty)
        for view in views:
            row = _Row(view, selected=(view.index == selected))
            row.clicked.connect(self.rowClicked)
            row.removeClicked.connect(self.removeClicked)
            self._layout.addWidget(row)
            self._rows.append(row)
        self._layout.addStretch(1)
        if selected is not None and selected < len(self._rows):
            self.ensureWidgetVisible(self._rows[selected])
