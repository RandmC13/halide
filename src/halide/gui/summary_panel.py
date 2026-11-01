"""The summary side of the right panel: a developed roll's end-of-run rows, as the terminal prints
them, and what to do next."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from halide.gui import theme


class SummaryPanel(QWidget):
    backClicked = Signal()
    contactClicked = Signal()
    shareClicked = Signal()
    openClicked = Signal()
    carryOnClicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(9, 0, 9, 0)
        col.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("Developed")
        title.setProperty("role", "section")
        self.title = title
        head.addWidget(title)
        head.addStretch(1)
        self.back_button = QPushButton("← Back to points")
        self.back_button.clicked.connect(self.backClicked)
        head.addWidget(self.back_button)
        col.addLayout(head)

        self.end_line = QLabel("")
        self.end_line.setWordWrap(True)
        col.addWidget(self.end_line)
        self._rows = QGridLayout()
        self._rows.setHorizontalSpacing(10)
        self._rows.setVerticalSpacing(4)
        self._rows.setColumnStretch(1, 1)
        col.addLayout(self._rows)
        col.addStretch(1)

        self.contact_button = QPushButton("Contact sheet")
        self.contact_button.setToolTip("A contact sheet of the developed prints, saved beside them")
        self.contact_button.clicked.connect(self.contactClicked)
        self.share_button = QPushButton("Export JPEG or PNG…")
        self.share_button.setToolTip("sRGB JPEG or PNG copies of the prints, in the sharing folder")
        self.share_button.clicked.connect(self.shareClicked)
        # One per row: side by side, a wider desktop font cut the second label.
        col.addWidget(self.contact_button)
        col.addWidget(self.share_button)
        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self.openClicked)
        self.carry_on_button = QPushButton("Carry on")
        self.carry_on_button.setToolTip("Develop the frames that aren't done yet (--skip-existing)")
        self.carry_on_button.clicked.connect(self.carryOnClicked)
        last = QHBoxLayout()
        last.addWidget(self.open_button)
        last.addWidget(self.carry_on_button)
        col.addLayout(last)

    def show_result(self, end_line: str, rows: list[tuple[str, str, str]], cancelled: bool) -> None:
        """`rows` are ("row" or "warn", label, value) as the run printed them. A cancelled run leads
        with its cancelled line and offers Carry on as the one red button."""
        self.title.setText("Cancelled" if cancelled else "Developed")
        self.end_line.setText(end_line)
        while self._rows.count():
            self._rows.takeAt(0).widget().deleteLater()
        last = None
        for r, (kind, label, value) in enumerate(rows):
            warn = kind == "warn"
            name = QLabel("" if label == last else label)  # a label once per run of equal labels, as the sheet
            last = label
            name.setProperty("role", "dim")
            text = QLabel(value)
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if warn:
                text.setStyleSheet(f"color: {theme.TEXT_WARNING};")
            self._rows.addWidget(name, r, 0, Qt.AlignmentFlag.AlignTop)
            self._rows.addWidget(text, r, 1)
        self.carry_on_button.setVisible(cancelled)
        for button, red in ((self.open_button, not cancelled), (self.carry_on_button, cancelled)):
            button.setProperty("role", "primary" if red else None)
            button.style().unpolish(button)
            button.style().polish(button)
