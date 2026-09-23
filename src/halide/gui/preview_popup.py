"""The live-preview popup: shows what the current shadow/highlight picks look like inverted,
downsampled for interactive speed. Non-modal and opened fresh each time the main window's Preview
button is clicked (see main_window.py) — it's a snapshot of the picks at that moment, not
live-synced to further picking, matching the redesign's "keep it simple" intent.

By default it shows only the render. An opt-in "Fine-tune" checkbox reveals exposure/contrast/
linear-output controls beneath the image (the same controls the old standalone Preview tab had).
Exposure/contrast tweaks, if made, are reported back to the main window via `toneChanged` so they
can be saved into the profile later (see main_window.py's pending_tone_override) - linear-output
mode is deliberately excluded from that signal and never persisted, see the redesign plan.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from halide.core.density import apply_density_balance, apply_white_balance, solve_density_balance
from halide.core.invert import invert
from halide.core.pipeline import run_pipeline
from halide.core.tone_render import estimate_exposure
from halide.core.types import ToneCurveParams
from halide.io.raster import to_srgb_8bit

_EXPOSURE_RANGE = (-200, 200)  # hundredths, i.e. -2.00..2.00
_CONTRAST_RANGE = (0, 100)  # hundredths, i.e. 0.00..1.00
_DEFAULT_CONTRAST = 0.5


def _to_qimage_uint8(rgb_uint8: np.ndarray) -> QImage:
    rgb_uint8 = np.ascontiguousarray(rgb_uint8)
    height, width = rgb_uint8.shape[:2]
    return QImage(rgb_uint8.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class PreviewPopup(QDialog):
    toneChanged = Signal(object)  # ToneCurveParams | None

    def __init__(
        self,
        parent: QWidget | None,
        preview_working: np.ndarray,
        shadow_rgb: tuple[float, float, float],
        highlight_rgb: tuple[float, float, float],
        initial_tone: ToneCurveParams | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("halide · preview")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.preview_working = preview_working
        self.live_profile = solve_density_balance(shadow_rgb, highlight_rgb)
        self._auto_exposure = self._compute_auto_exposure()

        self._build_ui()

        if initial_tone is not None:
            self.finetune_checkbox.setChecked(True)
            self.exposure_slider.setValue(round(initial_tone.exposure * 100))
            self.contrast_slider.setValue(round(initial_tone.contrast * 100))
        self._render()

    def update_points(self, shadow_rgb: tuple[float, float, float], highlight_rgb: tuple[float, float, float]) -> None:
        """Re-solve and re-render for a new shadow/highlight pick, without closing/reopening the
        popup - called by main_window.py whenever a point changes while this popup is still open,
        so the preview actually tracks what you're picking instead of going stale. Any active
        Fine-tune slider values are left exactly as the user set them; only the underlying density
        calibration (and, if Fine-tune is off, the auto-exposure seed) changes."""
        self.live_profile = solve_density_balance(shadow_rgb, highlight_rgb)
        self._auto_exposure = self._compute_auto_exposure()
        self._render()

    def _compute_auto_exposure(self) -> float:
        wb = apply_white_balance(self.preview_working, self.live_profile)
        db = apply_density_balance(wb, self.live_profile)
        positive = invert(db)
        return estimate_exposure(positive)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.finetune_checkbox = QCheckBox("Fine-tune")
        self.finetune_checkbox.toggled.connect(self._toggle_finetune)
        layout.addWidget(self.finetune_checkbox)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        exposure_row = QHBoxLayout()
        exposure_row.addWidget(QLabel("Exposure"))
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(*_EXPOSURE_RANGE)
        self.exposure_slider.setValue(round(self._auto_exposure * 100))
        self.exposure_slider.valueChanged.connect(self._render)
        exposure_row.addWidget(self.exposure_slider)
        controls_layout.addLayout(exposure_row)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(*_CONTRAST_RANGE)
        self.contrast_slider.setValue(round(_DEFAULT_CONTRAST * 100))
        self.contrast_slider.valueChanged.connect(self._render)
        contrast_row.addWidget(self.contrast_slider)
        controls_layout.addLayout(contrast_row)

        self.linear_checkbox = QCheckBox("Linear output (skip tone curve, preview only)")
        self.linear_checkbox.toggled.connect(self._render)
        controls_layout.addWidget(self.linear_checkbox)

        self.controls.setVisible(False)
        layout.addWidget(self.controls)

    def _toggle_finetune(self, checked: bool) -> None:
        self.controls.setVisible(checked)
        self._render()

    def _current_tone_params(self) -> ToneCurveParams:
        if not self.finetune_checkbox.isChecked():
            return ToneCurveParams(mode="paper", exposure=self._auto_exposure, contrast=_DEFAULT_CONTRAST)
        exposure = self.exposure_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        mode = "linear" if self.linear_checkbox.isChecked() else "paper"
        return ToneCurveParams(mode=mode, exposure=exposure, contrast=contrast)

    def _render(self) -> None:
        tone_params = self._current_tone_params()
        result = run_pipeline(self.preview_working, self.live_profile, tone_params)
        display_8bit = to_srgb_8bit(result)
        self.image_label.setPixmap(QPixmap.fromImage(_to_qimage_uint8(display_8bit)))

        if self.finetune_checkbox.isChecked():
            # Never carries linear-output - that stays a preview-only toggle, not a saved override.
            self.toneChanged.emit(ToneCurveParams(mode="paper", exposure=tone_params.exposure, contrast=tone_params.contrast))
        else:
            self.toneChanged.emit(None)
