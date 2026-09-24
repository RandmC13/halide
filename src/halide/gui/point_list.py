"""The neutral-point list in the picker's side panel: one compact row per point - number, frame,
density, agreement with the other points, remove. Fixed height: past a handful of points it scrolls
rather than growing the window (the window is a fixed size, see main_window.py). Clicking a row
selects that point and jumps to its frame; the markers on the image and the step wedge's ticks use
the same numbers and colours.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from halide.gui import theme
from halide.gui.roll import PointView

ROW_HEIGHT = 24

CC_EXPLANATION = (
    "How far each point is from what the other points agree neutral is, as a colour-printing filter "
    "value: CC is Kodak's Colour Compensating scale (density × 100, so CC 10 = 0.10), the unit of a "
    "dichroic enlarger head's filter dials. The letter is the colour the point would print "
    "(R/G/B, or C/M/Y for a lack of red/green/blue). Measured on the density-balanced negative, "
    "before the paper curve. Shown from 3 points: with 2, the line passes through both.\n\n"
    "Around CC 5 or more, check the object really was neutral (a cream wall, a sunlit cloud edge, "
    "skin...). Deal with the worst point first: one bad point makes the others read a few CC off too."
)


def agreement_colour(view: PointView) -> str:
    if view.agreement is None:
        return theme.AGREE_NONE
    return {"calm": theme.AGREE_CALM, "amber": theme.AGREE_AMBER, "red": theme.AGREE_RED}[view.agreement.band]


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
        frame = QLabel(frame_text)
        frame.setToolTip(str(view.point.frame))
        layout.addWidget(frame, stretch=1)

        density = QLabel(f"D {view.green_density:.2f}")
        density.setProperty("role", "dim")
        layout.addWidget(density)

        agreement = QLabel(f"● {view.agreement.label()}" if view.agreement else "·")
        agreement.setStyleSheet(f"color: {colour};")
        agreement.setFixedWidth(76)
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
        self.setMinimumHeight(ROW_HEIGHT * 3 + 4)
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
