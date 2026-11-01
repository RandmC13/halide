"""The dialog that names a calibration and saves it as a profile."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from halide.calibration.profile_store import (
    ProfileError,
    ProfileNameError,
    find_profile,
    save_named_profile,
    validate_profile_name,
)
from halide.core.types import DensityProfile


class SaveProfileDialog(QDialog):
    """Name the calibration and save it. On a name collision (checked via find_profile, so a name
    differing only by case counts too) the same dialog asks again, with its button
    relabelled "Overwrite", rather than stacking a second dialog. The name is validated live as it's
    typed: an invalid name disables Save and shows why, rather than only failing once
    clicked, and any disk error is shown in the dialog rather than escaping it."""

    saved = Signal(str, str)  # (name, message)

    def __init__(
        self, parent: QWidget, profile: DensityProfile, sidecars: dict, suggested_name: str = "",
        update_name: str | None = None,
    ) -> None:
        """`update_name`: a profile in use that has new points - offered as "Update '<name>'" beside
        saving under a new name, which then leaves it exactly as it was."""
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle("Save profile")
        self._profile = profile
        self._sidecars = sidecars
        self._pending_overwrite_name: str | None = None
        self._save_label = "Save as a new profile" if update_name else "Save"
        self._prompt = "Or save as a new profile, named:" if update_name else "Name this profile:"

        layout = QVBoxLayout(self)
        self.update_button: QPushButton | None = None
        if update_name:
            self.update_button = QPushButton(f"Update '{update_name}'")
            self.update_button.setProperty("role", "primary")
            self.update_button.setToolTip(f"Add the new points to '{update_name}', for every later roll")
            self.update_button.clicked.connect(lambda: self._save(update_name))
            layout.addWidget(self.update_button)
        self._message = QLabel(self._prompt)
        self._message.setWordWrap(True)
        layout.addWidget(self._message)
        self._name_input = QLineEdit(suggested_name)
        self._name_input.returnPressed.connect(self._on_save_clicked)
        self._name_input.textChanged.connect(self._validate_name)
        layout.addWidget(self._name_input)
        self._name_error = QLabel("")
        self._name_error.setProperty("role", "warning")
        self._name_error.setWordWrap(True)
        self._name_error.hide()
        layout.addWidget(self._name_error)
        row = QHBoxLayout()
        self._save_button = QPushButton(self._save_label)
        if not update_name:
            self._save_button.setProperty("role", "primary")
        self._save_button.clicked.connect(self._on_save_clicked)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(self._save_button)
        row.addWidget(cancel)
        layout.addLayout(row)
        self._name_input.setFocus()
        self._name_input.selectAll()
        self._validate_name(self._name_input.text())

    def _validate_name(self, text: str) -> None:
        if text != self._pending_overwrite_name:
            # The name changed since an "already exists" prompt (if any) - back to a plain Save.
            self._pending_overwrite_name = None
            self._save_button.setText(self._save_label)
            self._message.setText(self._prompt)
        if not text.strip():  # nothing typed yet: nothing to save, but nothing wrong either
            self._name_error.hide()
            self._save_button.setEnabled(False)
            return
        try:
            validate_profile_name(text)
        except ProfileNameError as exc:
            self._name_error.setText(str(exc))
            self._name_error.show()
            self._save_button.setEnabled(False)
            return
        self._name_error.hide()
        self._save_button.setEnabled(True)

    def _on_save_clicked(self) -> None:
        if not self._save_button.isEnabled():
            return  # Enter can still reach here with an invalid name typed but not yet re-validated
        try:
            name = validate_profile_name(self._name_input.text())
        except ProfileNameError as exc:
            self._name_error.setText(str(exc))
            self._name_error.show()
            return
        if find_profile(name) is not None and self._pending_overwrite_name != name:
            self._pending_overwrite_name = name
            self._message.setText(f"A profile named '{name}' already exists - save again to overwrite it.")
            self._save_button.setText("Overwrite")
            return
        self._save(name)

    def _save(self, name: str) -> None:
        try:
            path = save_named_profile(self._profile, name, overwrite=True, **self._sidecars)
        except (ProfileError, OSError) as exc:
            QMessageBox.warning(self, "Couldn't save profile", str(exc))
            return
        self.saved.emit(name, f"Saved profile '{name}' to {path}")
        self.accept()
