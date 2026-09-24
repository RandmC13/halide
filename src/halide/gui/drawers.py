"""The side panel's collapsible drawers - Extra information, Print, Details - as an accordion: opening
one closes the others, and each opens to a bounded height, so the fixed-size window never has to
grow or scroll (the same reason the original Details panel became a fixed-height scroll area).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from halide.core.types import ToneCurveParams
from halide.gui.roll import DETAIL_FIELDS

_EXPOSURE_RANGE = (-200, 200)  # hundredths, i.e. -2.00..2.00
_GRADE_RANGE = (0, 100)  # hundredths, i.e. 0.00..1.00


class Drawer(QWidget):
    """A titled, collapsible section. A fixed drawer opens to exactly its content's height (capped);
    an expanding one (`expanding=True`, for long reading like Details) opens to take the panel's
    spare height - the panel lets it (MainWindow._on_drawer_opened) - down to `max_height` as its
    minimum, scrolling only beyond that."""

    toggled = Signal(object, bool)  # (drawer, open)

    def __init__(self, title: str, content: QWidget, max_height: int, expanding: bool = False) -> None:
        super().__init__()
        self._title = title
        self.expanding = expanding
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = QToolButton()
        self.header.setCheckable(True)
        self.header.clicked.connect(lambda checked: self.toggled.emit(self, checked))
        layout.addWidget(self.header)
        self.body = QScrollArea()
        self.body.setWidget(content)
        self.body.setWidgetResizable(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if expanding:
            self.body.setMinimumHeight(max_height)
            self.body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        else:
            # Open to exactly the content's own height (capped): a drawer that opens shorter than its
            # content hides fields behind an inner scrollbar, which reads as broken, not compact.
            self.body.setFixedHeight(min(max_height, content.sizeHint().height() + 4))
        layout.addWidget(self.body)
        self.set_open(False)

    def set_open(self, open_: bool) -> None:
        self.header.setChecked(open_)
        self.header.setText(("▾ " if open_ else "▸ ") + self._title)
        self.body.setVisible(open_)
        # Closed, or a fixed drawer: never take more height than it needs - spare panel height
        # otherwise got shared out as gaps around the drawer headers.
        vertical = QSizePolicy.Policy.Expanding if (open_ and self.expanding) else QSizePolicy.Policy.Maximum
        self.setSizePolicy(QSizePolicy.Policy.Preferred, vertical)


class Accordion(QWidget):
    """Opening one drawer closes the others. `openChanged` reports the open drawer (or None), so
    the panel can hand spare height to an expanding one."""

    openChanged = Signal(object)

    def __init__(self, drawers: list[Drawer]) -> None:
        super().__init__()
        self.drawers = drawers
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for drawer in drawers:
            drawer.toggled.connect(self._on_toggled)
            layout.addWidget(drawer)

    def _on_toggled(self, drawer: Drawer, open_: bool) -> None:
        for other in self.drawers:
            other.set_open(open_ and other is drawer)
        opened = drawer if open_ else None
        expanding = opened is not None and opened.expanding
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding if expanding else QSizePolicy.Policy.Maximum
        )
        self.openChanged.emit(opened)


class RollDetailsForm(QWidget):
    """Film stock, process, scanner, notes - saved with the profile, and the film stock is what the
    proof sheet's edge print says ("INVERTED BY HALIDE" when unset)."""

    changed = Signal(dict)

    _LABELS = {"film_stock": "Film stock", "process": "Process", "scanner": "Scanner", "notes": "Notes"}
    _PLACEHOLDERS = {
        "film_stock": "e.g. Kodak Portra 400",
        "process": "e.g. C-41, home kit",
        "scanner": "e.g. camera + macro lens",
        "notes": "anything worth remembering",
    }

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        form.setContentsMargins(4, 2, 4, 2)
        self.fields: dict[str, QLineEdit] = {}
        for key in DETAIL_FIELDS:
            edit = QLineEdit()
            edit.setPlaceholderText(self._PLACEHOLDERS[key])
            edit.textChanged.connect(lambda _text: self.changed.emit(self.values()))
            form.addRow(self._LABELS[key], edit)
            self.fields[key] = edit

    def values(self) -> dict[str, str]:
        return {k: e.text().strip() for k, e in self.fields.items()}

    def set_values(self, values: dict[str, str]) -> None:
        for key, edit in self.fields.items():
            edit.blockSignals(True)
            edit.setText(values.get(key, ""))
            edit.blockSignals(False)


class PrintControls(QWidget):
    """Enlarger exposure and paper grade for the Positive view. Left alone they're fitted per frame
    (the project default, see core/tone_render.py::fit_print) and the sliders just follow the current
    frame's fit; moving either pins both, and that pinned pair is saved with the profile as its print
    override. Flat output is a preview-only look at the linear positive, never saved."""

    changed = Signal(object, bool)  # (ToneCurveParams override or None, flat preview)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.status = QLabel("Fitted per frame")
        self.status.setProperty("role", "dim")
        layout.addWidget(self.status)

        self.exposure, self.exposure_value = self._slider_row(layout, "Exposure", _EXPOSURE_RANGE)
        self.grade, self.grade_value = self._slider_row(layout, "Grade", _GRADE_RANGE)

        self.flat = QCheckBox("Flat output (preview only)")
        self.flat.toggled.connect(lambda _checked: self._emit())
        layout.addWidget(self.flat)
        self.reset_button = QPushButton("Back to fitted")
        self.reset_button.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_button)
        self._pinned = False

    def _slider_row(self, layout: QVBoxLayout, label: str, value_range: tuple[int, int]) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        name = QLabel(label)
        name.setFixedWidth(62)
        row.addWidget(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(*value_range)
        slider.valueChanged.connect(self._on_slider)
        row.addWidget(slider)
        value = QLabel("")
        value.setFixedWidth(40)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        layout.addLayout(row)
        return slider, value

    def _set_quietly(self, exposure: float, grade: float) -> None:
        for slider, value in ((self.exposure, exposure), (self.grade, grade)):
            slider.blockSignals(True)
            slider.setValue(round(value * 100))
            slider.blockSignals(False)
        self.exposure_value.setText(f"{exposure:+.2f}")
        self.grade_value.setText(f"{grade:.2f}")

    def show_fitted(self, exposure: float, grade: float) -> None:
        """The current frame's own fit - shown on the sliders unless the user has pinned values."""
        if not self._pinned:
            self._set_quietly(exposure, grade)

    def set_override(self, tone: ToneCurveParams | None) -> None:
        self._pinned = tone is not None
        if tone is not None:
            self._set_quietly(tone.exposure, tone.contrast)
        self.status.setText("Pinned for every frame (saved with the profile)" if self._pinned else "Fitted per frame")

    def _on_slider(self, _value: int) -> None:
        self._pinned = True
        self.exposure_value.setText(f"{self.exposure.value() / 100:+.2f}")
        self.grade_value.setText(f"{self.grade.value() / 100:.2f}")
        self.status.setText("Pinned for every frame (saved with the profile)")
        self._emit()

    def _on_reset(self) -> None:
        self._pinned = False
        self.status.setText("Fitted per frame")
        self._emit()

    def _emit(self) -> None:
        tone = (
            ToneCurveParams(mode="paper", exposure=self.exposure.value() / 100, contrast=self.grade.value() / 100)
            if self._pinned
            else None
        )
        self.changed.emit(tone, self.flat.isChecked())
