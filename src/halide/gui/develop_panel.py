"""The develop side of the right panel: where, how and under what names the roll is developed, and
the exact `halide batch` command that will run (gui/develop.py builds it). Nothing here runs it."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase, QFontMetrics, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from halide.calibration.profile_store import ProfileNameError, default_profiles_dir, find_profile, validate_profile_name
from halide.cli import console
from halide.gui.develop import DEFAULT_NAME, DevelopSettings, example_name, frame_count, shown_command
from halide.gui.widgets import ElidedLabel, Segmented

OUTPUTS = ("print", "flat")
EXISTING = ("ask", "replace", "skip")
SHARING = ("none", "jpg", "png")
_NAME_MAX_W = 150  # a profile's name inside a choice is cut beyond this (the full name is the tooltip)
_LABEL_W = 62


class DevelopPanel(QWidget):
    """What the person decides before developing. `prepare()` says which calibration is in hand:
    - "profile": a saved profile, unchanged; nothing to decide;
    - "new": unsaved points; "save" them as a profile (named here) or "once" (use, don't save);
    - "refined": a profile in use with changed points; "add" them to it or use them for "roll" only.
    Develop stays off until that choice is made, and says why."""

    backClicked = Signal()
    developClicked = Signal(object, object)  # (DevelopSettings, the calibration choice)
    cancelClicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roll: Path | None = None
        self._case = "profile"
        self._profile_path: Path | None = None
        self._signature: tuple | None = None
        self._name_edited = False
        self._device: str | None = None

        col = QVBoxLayout(self)
        col.setContentsMargins(9, 0, 9, 0)  # the points side's left edge, so the panel doesn't jump when it turns
        col.setSpacing(6)

        # The way back shares the first title's row: the panel is fixed-size, and this is its tightest state.
        self.back_button = QPushButton("\u2190 Back to points")
        self.back_button.clicked.connect(self.backClicked)
        head = QHBoxLayout()
        head.addWidget(self._title("Calibration"))
        head.addStretch(1)
        head.addWidget(self.back_button)
        col.addLayout(head)
        self.profile_label = QLabel("")
        self.profile_label.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(self.profile_label)
        self.choice = Segmented(["", ""], checked=None, vertical=True)
        self.choice.changed.connect(lambda _i: self._refresh())
        col.addWidget(self.choice)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Profile name")
        self.name_input.textEdited.connect(self._on_name_edited)
        self.name_input.textChanged.connect(lambda _t: self._refresh())
        col.addWidget(self.name_input)
        self.name_error = ElidedLabel("")
        self.name_error.setProperty("role", "warning")
        col.addWidget(self.name_error)

        self._section(col, "Prints")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)
        self.into_input = QLineEdit()
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose_into)
        into_row = QHBoxLayout()
        into_row.setSpacing(6)
        into_row.addWidget(self.into_input, 1)
        into_row.addWidget(choose)
        self.output = Segmented(["Print", "Flat"])
        self.output.setToolTip("Print: through the paper curve. Flat: the film's own contrast, for editing elsewhere")
        self.names = QLineEdit(DEFAULT_NAME)
        self.names.setToolTip("{stem} the scan's name · {roll} · {n} or {n:02} the frame's number · {stock}")
        self.existing = Segmented(["Ask", "Replace", "Skip"])
        self.existing.setToolTip("When a print is already in the folder: ask once, replace it, or leave it")
        self.example = ElidedLabel("")
        self.example.setProperty("role", "dim")
        self.match_exposure = QCheckBox("Even out (--match-scan-exposure)")
        self.match_exposure.setToolTip("These scans were digitized at different exposures: correct for it from their EXIF")
        rows = [
            ("Into", into_row), ("Output", self._left(self.output)), ("Names", self.names),
            ("", self.example), ("Existing", self._left(self.existing)),
        ]
        for r, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(widget, r, 1) if isinstance(widget, QWidget) else grid.addLayout(widget, r, 1)
        self.exposure_label = QLabel("Exposure")
        grid.addWidget(self.exposure_label, len(rows), 0)
        grid.addWidget(self.match_exposure, len(rows), 1)
        grid.setColumnMinimumWidth(0, _LABEL_W)
        col.addLayout(grid)

        self._section(col, "For sharing")
        share_row = QHBoxLayout()
        share_row.addWidget(QLabel("Also save"))
        self.sharing = Segmented(["Nothing", "JPEG", "PNG"])
        share_row.addWidget(self.sharing)
        share_row.addStretch(1)
        col.addLayout(share_row)
        self.sharing_note = ElidedLabel("")
        self.sharing_note.setProperty("role", "dim")
        col.addWidget(self.sharing_note)

        self._section(col, "The command")
        self.command = QPlainTextEdit()
        self.command.setReadOnly(True)
        self.command.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.command.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.command.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(8)
        self.command.setFont(mono)
        self.command.setFixedHeight(QFontMetrics(mono).lineSpacing() * 4 + 10)
        col.addWidget(self.command)

        col.addStretch(1)
        self.error = ElidedLabel("")
        self.error.setProperty("role", "warning")
        col.addWidget(self.error)
        self.develop_button = QPushButton("Develop")
        self.develop_button.setProperty("role", "primary")
        self.develop_button.clicked.connect(self._on_develop)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop after the frames in hand, as Ctrl-C does")
        self.cancel_button.clicked.connect(self.cancelClicked)
        self.cancel_button.hide()
        bottom = QHBoxLayout()
        bottom.addWidget(self.develop_button, 1)
        bottom.addWidget(self.cancel_button)
        col.addLayout(bottom)
        self._running = False

        for edit in (self.names, self.into_input):
            edit.textChanged.connect(lambda _t: self._refresh())
        for seg in (self.output, self.existing, self.sharing):
            seg.changed.connect(lambda _i: self._refresh())
        self.match_exposure.toggled.connect(lambda _c: self._refresh())

    # --- building -------------------------------------------------------------------------------

    @staticmethod
    def _title(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "section")
        return label

    def _section(self, col: QVBoxLayout, text: str) -> None:
        col.addSpacing(4)
        col.addWidget(self._title(text))

    @staticmethod
    def _left(widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addStretch(1)
        return row

    # --- state in -------------------------------------------------------------------------------

    def prepare(
        self, roll: Path, case: str, *, profile_path: Path | None = None, name: str = "", new_points: int = 0,
        suggested_name: str = "", exposure_differs: bool = False, device: str | None = None,
    ) -> None:
        """Show the panel for `roll` with the calibration in hand. The choice is asked afresh when the
        calibration is not the one last shown; Into is remembered while the roll stays the same."""
        if roll != self._roll:
            self._roll = roll  # a new roll gets its own default; the same roll keeps what was typed
            self.into_input.setText(str(roll / "developed"))
        self._case, self._profile_path, self._device = case, profile_path, device
        signature = (case, name, new_points)
        fresh = signature != self._signature
        self._signature = signature
        shown = QFontMetrics(self.font()).elidedText(name, Qt.TextElideMode.ElideMiddle, _NAME_MAX_W)
        self.profile_label.setVisible(case == "profile")
        self.profile_label.setText(f"Profile <b>{html.escape(shown)}</b> ✓ saved")
        self.profile_label.setToolTip(name)
        self.choice.setVisible(case != "profile")
        if case == "new":
            labels = ["Save as a profile", "Use once, don't save"]
        else:
            if new_points == 1:
                first, second = f"Add this point to '{shown}'", "Use it for this roll only"
            elif new_points:
                first, second = f"Add these {new_points} points to '{shown}'", "Use them for this roll only"
            else:
                first, second = f"Save these changes to '{shown}'", "Use them for this roll only"
            labels = [first, second]
        for button, label in zip(self.choice.buttons, labels):
            button.setText(label)
        if fresh:
            self.choice.clear()
        if case == "new" and (not self._name_edited or fresh):
            self.name_input.setText(suggested_name)
            self._name_edited = False
        self.exposure_label.setVisible(exposure_differs)
        self.match_exposure.setVisible(exposure_differs)
        if not exposure_differs:
            self.match_exposure.setChecked(False)
        self._refresh()

    def choice_value(self) -> str | None:
        """"profile", or the person's choice ("save"/"once", "add"/"roll"), or None until made."""
        if self._case == "profile":
            return "profile"
        index = self.choice.index()
        if index is None:
            return None
        return (("save", "once") if self._case == "new" else ("add", "roll"))[index]

    # --- state out ------------------------------------------------------------------------------

    def into(self) -> Path:
        return self._roll / Path(self.into_input.text().strip() or "developed").expanduser()

    def _shown_profile(self, choice: str | None) -> Path:
        """The profile file the command names. Who writes it is decided when developing starts: a
        saved profile is used where it is, a calibration used once goes beside the prints."""
        if choice == "save":
            text = self.name_input.text().strip() or "profile"
            return find_profile(text) or default_profiles_dir() / f"{text}.json"
        if choice in ("profile", "add") and self._profile_path is not None:
            return self._profile_path
        return self.into() / "halide-calibration.json"

    def settings(self) -> DevelopSettings:
        return DevelopSettings(
            roll=self._roll, into=self.into(), profile=self._shown_profile(self.choice_value()),
            output=OUTPUTS[self.output.index()], name=self.names.text().strip() or DEFAULT_NAME,
            existing=EXISTING[self.existing.index()], match_scan_exposure=self.match_exposure.isChecked(),
            sharing=SHARING[self.sharing.index()], device=self._device,
        )

    # --- behaviour ------------------------------------------------------------------------------

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def _choose_into(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Develop into", str(self.into()))
        if folder:
            self.into_input.setText(folder)

    def _name_problem(self) -> str:
        if self.choice_value() != "save":
            return ""
        text = self.name_input.text()
        if not text.strip():
            return "Name the profile"
        try:
            name = validate_profile_name(text)
        except ProfileNameError as exc:
            return str(exc)
        return f"A profile named '{name}' already exists: choose another name" if find_profile(name) else ""

    def _refresh(self) -> None:
        if self._roll is None:
            return
        s = self.settings()
        self.command.setPlainText(shown_command(s))
        self.command.setToolTip(self.command.toPlainText())
        self.sharing_note.set_full_text(f"sRGB, full size, in {s.sharing_dir}")
        self.sharing_note.setVisible(s.sharing != "none")
        self.name_input.setVisible(self._case == "new" and self.choice_value() == "save")
        problem = self._name_problem()
        self.name_error.set_full_text(problem)
        self.name_error.setVisible(bool(problem))
        count, refusal = None, ""
        try:
            count = frame_count(s)
            self.example.set_full_text(example_name(s))
        except ValueError as exc:
            refusal = str(exc)
            self.example.set_full_text("")
        self.error.set_full_text(refusal)
        self.error.setVisible(bool(refusal))
        reason = ""
        if self.choice_value() is None:
            reason = "Choose what to do with these points first"
        elif problem:
            reason = problem
        elif refusal:
            reason = refusal
        elif not count:
            reason = "There are no scans to develop in this folder"
        self.develop_button.setText(f"Develop {console.plural(count, 'frame')}" if count else "Develop")
        self.develop_button.setEnabled(not reason)
        self.develop_button.setToolTip(reason)

    def _on_develop(self) -> None:
        if not self._running:
            self.developClicked.emit(self.settings(), self.choice_value())

    def set_running(self, text: str | None) -> None:
        """While a roll develops: every control off, the red button showing `text`, Cancel beside
        it. None puts the panel back as it was."""
        self._running = text is not None
        for child in self.findChildren(QWidget):
            if child not in (self.develop_button, self.cancel_button) and child.parent() is self:
                child.setEnabled(not self._running)
        self.cancel_button.setVisible(self._running)
        self.cancel_button.setEnabled(True)
        if self._running:
            self.develop_button.setText(text)
            self.develop_button.setToolTip("")
        else:
            self._refresh()

    def skip_existing(self) -> None:
        self.existing.set_index(EXISTING.index("skip"))

