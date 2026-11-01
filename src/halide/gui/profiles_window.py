"""The Profiles window: every saved profile, newest first, with what can be done to one - use it on
the open roll, reopen its points, rename or delete it - and a way in for any calibration file, such
as the halide-calibration.json written beside a developed roll. Small, fixed-size and non-modal, so
the main window stays usable beside it."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from halide.calibration.profile_store import (
    ProfileError,
    ProfileNameError,
    default_profiles_dir,
    delete_profile,
    find_profile,
    list_profiles,
    load_anchors,
    rename_profile,
    validate_profile_name,
)
from halide.cli import console
from halide.core.types import DensityProfile
from halide.gui import theme

SIZE = QSize(720, 380)


def made_text(created_at: str | None, today: date | None = None) -> str:
    """The Made column: "today", else the local date the profile was saved, else blank."""
    if not created_at:
        return ""
    try:
        made = datetime.fromisoformat(created_at).astimezone().date()
    except ValueError:
        return created_at[:10]
    return "today" if made == (today or date.today()) else made.isoformat()


def points_text(path: Path) -> str:
    """The Points column, from the profile's recorded picks: "6 on 3 frames", or a dash for a profile
    made without the window (hand-entered or automatic)."""
    try:
        records = load_anchors(path)
    except (ValueError, OSError):
        return "—"
    if not records:
        return "—"
    frames = len({r.get("frame") for r in records})
    return f"{console.plural(len(records), 'point')} on {console.plural(frames, 'frame')}"


class ProfilesWindow(QWidget):
    useRequested = Signal(object)  # Path: put this profile on the open roll
    editRequested = Signal(object)  # Path: reopen its points (and their rolls)
    fileOpened = Signal(object)  # Path: a calibration file from anywhere, opened like Edit points

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("halide · profiles")
        self.setFixedSize(SIZE)
        self._rows: list[tuple[str, DensityProfile | None, str | None]] = []
        self._roll_open = False
        self._renaming = ""  # the profile whose name is being edited

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title = QHBoxLayout()
        heading = QLabel("PROFILES")
        heading.setFont(theme.edge_font(15))
        heading.setStyleSheet(f"color: {theme.EDGE_PRINT};")
        title.addWidget(heading)
        title.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setProperty("role", "dim")
        title.addWidget(self.count_label)
        layout.addLayout(title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Film stock", "Made", "Points"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # no dotted focus box over the selected row
        # Selected as the point list shows it, not in the desktop's own highlight colour.
        self.table.setStyleSheet(
            f"QTableWidget {{ background: {theme.BACKGROUND}; border: 1px solid {theme.RULE}; }}"
            f"QTableWidget::item {{ padding: 4px 6px; border-bottom: 1px solid {theme.RULE}; }}"
            f"QTableWidget::item:selected {{ background: {theme.BORDER}; color: {theme.TEXT}; }}"
            f"QHeaderView::section {{ background: {theme.BACKGROUND}; color: {theme.TEXT_DIM}; border: none;"
            f" border-bottom: 1px solid {theme.BORDER}; padding: 4px 6px; font-weight: 600; }}"
        )
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda _item: self._on_edit())
        layout.addWidget(self.table, 1)

        # The bottom row is either the buttons or, while renaming, the new name and why it can't be used.
        self._bar = QWidget()
        bar = QHBoxLayout(self._bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        self.open_file_button = QPushButton("Open a calibration file…")
        self.open_file_button.setToolTip("Any profile file, such as the halide-calibration.json beside a developed roll")
        self.open_file_button.clicked.connect(self._on_open_file)
        bar.addWidget(self.open_file_button)
        bar.addStretch(1)
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self._start_rename)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete)
        self.edit_button = QPushButton("Edit points")
        self.edit_button.setToolTip("Reopen the profile's neutral points on the frames they were picked on")
        self.edit_button.clicked.connect(self._on_edit)
        self.use_button = QPushButton("Use for this roll")
        self.use_button.setProperty("role", "primary")
        self.use_button.clicked.connect(self._on_use)
        for button in (self.rename_button, self.delete_button, self.edit_button, self.use_button):
            bar.addWidget(button)
        layout.addWidget(self._bar)

        self._rename_bar = QWidget()
        rename = QHBoxLayout(self._rename_bar)
        rename.setContentsMargins(0, 0, 0, 0)
        rename.setSpacing(8)
        rename.addWidget(QLabel("New name:"))
        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self._validate_name)
        self.name_input.returnPressed.connect(self._finish_rename)
        rename.addWidget(self.name_input, 1)
        self.name_error = QLabel("")
        self.name_error.setProperty("role", "warning")
        rename.addWidget(self.name_error)
        self.rename_confirm = QPushButton("Rename")
        self.rename_confirm.clicked.connect(self._finish_rename)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self._end_rename)
        rename.addWidget(self.rename_confirm)
        rename.addWidget(cancel)
        self._rename_bar.hide()
        layout.addWidget(self._rename_bar)
        self.refresh()

    # --- the list -------------------------------------------------------------------------------

    def set_roll_open(self, open_: bool) -> None:
        self._roll_open = open_
        self._update_buttons()

    def refresh(self, select: str | None = None) -> None:
        """Re-read the profiles folder, keeping (or moving to) the selected profile."""
        keep = select or self.selected_name()
        rows = list_profiles()
        rows.sort(key=lambda row: (row[1].created_at or "") if row[1] is not None else "", reverse=True)
        self._rows = rows
        self.table.setRowCount(len(rows))
        for i, (name, profile, problem) in enumerate(rows):
            if profile is None:
                cells = [name, f"damaged — {problem}", "", ""]
            else:
                cells = [name, profile.film_stock or "", made_text(profile.created_at), points_text(self.path_of(name))]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                # The name reads first, the rest dim; a damaged profile's problem in the warning colour.
                colour = theme.TEXT if column == 0 else theme.TEXT_WARNING if profile is None else theme.TEXT_DIM
                item.setForeground(QColor(colour))
                self.table.setItem(i, column, item)
        self.count_label.setText(f"{len(rows)} saved · newest first" if rows else "none saved yet")
        names = [row[0] for row in rows]
        self.table.clearSelection()
        if keep in names:
            self.table.selectRow(names.index(keep))
        elif rows:
            self.table.selectRow(0)
        self._update_buttons()

    def path_of(self, name: str) -> Path:
        return default_profiles_dir() / f"{name}.json"

    def selected_name(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._rows[rows[0].row()][0] if rows else None

    def _selected_readable(self) -> bool:
        rows = self.table.selectionModel().selectedRows()
        return bool(rows) and self._rows[rows[0].row()][1] is not None

    def _update_buttons(self) -> None:
        chosen = self.selected_name() is not None
        readable = self._selected_readable()
        self.delete_button.setEnabled(chosen)
        self.rename_button.setEnabled(readable)
        self.edit_button.setEnabled(readable)
        self.use_button.setEnabled(readable and self._roll_open)
        self.use_button.setToolTip("" if self._roll_open else "Load a roll first, then use the profile on it")

    # --- actions --------------------------------------------------------------------------------

    def _on_use(self) -> None:
        name = self.selected_name()
        if name is not None and self._selected_readable():
            self.useRequested.emit(self.path_of(name))

    def _on_edit(self) -> None:
        name = self.selected_name()
        if name is not None and self._selected_readable():
            self.editRequested.emit(self.path_of(name))

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a calibration file", str(Path.home()), "halide calibrations (*.json)")
        if path:
            self.fileOpened.emit(Path(path))

    def _on_delete(self) -> None:
        name = self.selected_name()
        if name is None:
            return
        answer = QMessageBox.question(
            self, "Delete profile", f"Delete the profile '{name}'? This can't be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_profile(name)
        except (ProfileError, OSError) as exc:
            QMessageBox.warning(self, "Couldn't delete profile", str(exc))
        self.refresh()

    def _start_rename(self) -> None:
        name = self.selected_name()
        if name is None:
            return
        self._renaming = name
        self._bar.hide()
        self._rename_bar.show()
        self.name_input.setText(name)
        self.name_input.selectAll()
        self.name_input.setFocus()
        self._validate_name(name)

    def _end_rename(self) -> None:
        self._rename_bar.hide()
        self._bar.show()

    def _name_problem(self, text: str) -> str | None:
        try:
            name = validate_profile_name(text)
        except ProfileNameError as exc:
            return str(exc)
        clash = find_profile(name)
        if clash is not None and clash.stem.casefold() != self._renaming.casefold():
            return f"'{clash.stem}' already exists"
        return None

    def _validate_name(self, text: str) -> None:
        problem = self._name_problem(text)
        self.name_error.setText(problem or "")
        self.name_error.setToolTip(problem or "")
        self.rename_confirm.setEnabled(problem is None and text.strip() != self._renaming)

    def _finish_rename(self) -> None:
        if not self.rename_confirm.isEnabled():
            return
        new = self.name_input.text().strip()
        try:
            rename_profile(self._renaming, new)
        except (ProfileError, OSError) as exc:
            self.name_error.setText(str(exc))
            return
        self._end_rename()
        self.refresh(select=new)
