"""The main halide GUI window: load a negative, pick a shadow and a highlight point on it. One
window class serves both the persistent `halide calibrate` app and the one-shot `--pick` picker
(`quick_pick.py`) — the two differ only in whether the load controls are shown and what the primary
button at the bottom says/does (wired up in later steps), not in the picking interaction itself.

Step 2 scope: top bar (filename + load), the image box (paint + click/hover picking), and the
floating mouse-anchored magnifier. Control row, Details disclosure, status bar polish, sizing, the
Preview popup, and the Save/Develop button all arrive in later steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from halide.calibration.auto import DEFAULT_NEUTRAL_FRACTION, _neutral_candidate_mask, auto_density_balance
from halide.calibration.profile_store import default_profiles_dir, save_named_profile
from halide.core.density import solve_density_balance
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui import theme
from halide.gui.preview_popup import PreviewPopup
from halide.gui.sampling import (
    apply_stretch,
    compute_stretch_bounds,
    display_to_full_res_coords,
    downsample_for_display,
    extract_magnifier_patch,
    full_res_to_display_coords,
    magnify_patch,
    patch_chroma,
    snap_to_representative_pixel,
)
from halide.processing import ScanColorError, load_working_space_image

# "~25% of screen space at most," per the redesign - but on a typical screen 25% of the *height*
# is too short to hold this layout without the scrolling the redesign was explicitly meant to
# eliminate, so height is clamped to a floor well above a literal 25% on most real displays. Width
# mostly does land at the literal ~25% on an ordinary monitor.
_WINDOW_MIN = QSize(380, 680)
_WINDOW_MAX = QSize(560, 860)

# Pinned explicitly (not left to the platform style's default) so the window's real content width
# is known exactly - see MainWindow._compute_display_budget's docstring for why the image box's
# own size can't be a module-level constant at all (it depends on the *real*, per-machine computed
# window size, not the theoretical minimum, and on the actual loaded image's own shape).
_LAYOUT_MARGIN = 14
_DISPLAY_ASPECT_RATIO = 3 / 2  # a real 35mm film frame's own aspect ratio (36x24mm) - a budget cap
# only now, not the box's own shape (see MainWindow._compute_display_budget).
_IMAGE_PADDING = 10  # uniform gap on all four sides between the image and its drawn border
_DETAILS_HEIGHT = 100  # fixed height for the scrollable Details panel - see its own comment

_MAGNIFIER_RADIUS = 10
_MAGNIFIER_ZOOM = 6

_SHADOW_COLOR = QColor(theme.SHADOW_POINT_COLOR)
_HIGHLIGHT_COLOR = QColor(theme.HIGHLIGHT_POINT_COLOR)
_OVERLAY_TINT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_MARKER_RADIUS = 5  # a location marker, not a depiction of the sampling neighborhood - see
# sampling.snap_to_representative_pixel's own radius (in full-res pixels, unrelated to this)
_MARKER_CENTER_DOT_RADIUS = 1

_SHADOW_CAPTION = "Shadow point: a scene shadow (dark in real life); looks LIGHT/thin on this raw negative"
_HIGHLIGHT_CAPTION = "Highlight point: a scene highlight (bright in real life); looks DARK/dense on this raw negative"


def _compute_window_size() -> QSize:
    screen = QGuiApplication.primaryScreen()
    available = screen.availableGeometry() if screen is not None else None
    if available is None:
        return QSize(_WINDOW_MIN)
    width = min(max(int(available.width() * 0.25), _WINDOW_MIN.width()), _WINDOW_MAX.width())
    height = min(max(int(available.height() * 0.25), _WINDOW_MIN.height()), _WINDOW_MAX.height())
    return QSize(width, height)


def _to_qimage(rgb_float: np.ndarray) -> QImage:
    """A (H, W, 3) float array in [0, 1] -> a QImage. Copies the buffer so the QImage stays valid
    after the source numpy array is garbage collected (QImage doesn't own external data by default)."""
    rgb_uint8 = np.ascontiguousarray((np.clip(rgb_float, 0.0, 1.0) * 255).astype(np.uint8))
    height, width = rgb_uint8.shape[:2]
    image = QImage(rgb_uint8.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()


class ImageView(QWidget):
    """Paints the current downsampled negative and emits raw widget-local pixel coordinates on
    click/hover — deliberately dumb about color science/coordinate mapping, which stays in
    MainWindow (backed by the framework-independent gui/sampling.py)."""

    clicked = Signal(int, int)
    hovered = Signal(int, int)
    left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        # Real size is set right after construction via set_view_size() (the display budget while
        # empty, then shrink-wrapped to whatever image is actually loaded, plus _IMAGE_PADDING on
        # each side - see MainWindow.load_image) - this placeholder just needs *a* size so the
        # widget is valid before that call.
        self.setFixedSize(300, 200)
        self._pixmap: QPixmap | None = None
        self.shadow_marker: QPoint | None = None
        self.highlight_marker: QPoint | None = None

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        # Defensively re-fit rather than trust the caller's size always matches this box exactly:
        # a pixmap taller/wider than the box would make paintEvent's centering offset go negative,
        # which Qt silently clips - invisible cropping plus asymmetric-looking padding, exactly the
        # bug class this guards against. KeepAspectRatio means this is a no-op whenever the pixmap
        # already fits (the normal case), so it never re-crops or upscales anything either.
        if pixmap is not None and (pixmap.width() > self.width() or pixmap.height() > self.height()):
            pixmap = pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        self._pixmap = pixmap
        self.update()

    def set_view_size(self, width: int, height: int) -> None:
        self.setFixedSize(width, height)

    def clear_markers(self) -> None:
        self.shadow_marker = None
        self.highlight_marker = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BACKGROUND_ALT))
        painter.setPen(QColor(theme.BORDER))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._pixmap is None:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a negative to begin")
            return
        x = (self.width() - self._pixmap.width()) // 2
        y = (self.height() - self._pixmap.height()) // 2
        painter.drawPixmap(x, y, self._pixmap)
        for marker, color in ((self.shadow_marker, _SHADOW_COLOR), (self.highlight_marker, _HIGHLIGHT_COLOR)):
            if marker is not None:
                center = marker + QPoint(x, y)
                painter.setPen(color)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, _MARKER_RADIUS, _MARKER_RADIUS)
                painter.setBrush(color)
                painter.drawEllipse(center, _MARKER_CENTER_DOT_RADIUS, _MARKER_CENTER_DOT_RADIUS)

    def _offset(self) -> QPoint:
        if self._pixmap is None:
            return QPoint(0, 0)
        return QPoint((self.width() - self._pixmap.width()) // 2, (self.height() - self._pixmap.height()) // 2)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        pos = event.position().toPoint() - self._offset()
        if self._pixmap.rect().contains(pos):
            self.clicked.emit(pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        pos = event.position().toPoint() - self._offset()
        if self._pixmap.rect().contains(pos):
            self.hovered.emit(pos.x(), pos.y())
        else:
            self.left.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.left.emit()


class Magnifier(QWidget):
    """A frameless, always-on-top widget that follows the cursor over the image, showing a zoomed
    patch — the "dynamic zoomed preview anchored to the mouse" the redesign asked for. Qt's
    tooltip-style window flags are a native fit for exactly this.

    Must be constructed with a real parent, not None: Wayland (unlike X11) refuses to place a
    parentless popup-type window at all and floods the terminal with "Failed to create popup ...
    has a transientParent set" on every single move - X11 tolerated a parentless popup, Wayland
    doesn't, so this needs a real transient parent to work on both."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(
            parent,
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._label = QLabel(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addWidget(self._label)
        self.setStyleSheet(f"background: {theme.BACKGROUND_ALT}; border: 1px solid {theme.BORDER};")

    def show_patch(self, image: QImage, global_pos: QPoint) -> None:
        self._label.setPixmap(QPixmap.fromImage(image))
        self.adjustSize()
        self.move(global_pos + QPoint(18, 18))
        self.show()


class SaveProfileDialog(QDialog):
    """"Just one big button that says save calibration profile that, when clicked, asks the user
    what they would like to call the profile" - this is that ask. On a name collision, the same
    dialog swaps its own message and relabels its button to "Overwrite" for a second click, rather
    than stacking a second confirmation dialog on top."""

    saved = Signal(str)

    def __init__(self, parent: QWidget, profile: DensityProfile, tone: ToneCurveParams | None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle("Save calibration profile")
        self._profile = profile
        self._tone = tone
        self._pending_overwrite_name: str | None = None

        layout = QVBoxLayout(self)
        self._message = QLabel("Name this calibration:")
        self._message.setWordWrap(True)
        layout.addWidget(self._message)
        self._name_input = QLineEdit()
        self._name_input.returnPressed.connect(self._on_save_clicked)
        layout.addWidget(self._name_input)

        button_row = QHBoxLayout()
        self._save_button = QPushButton("Save")
        self._save_button.setProperty("role", "primary")
        self._save_button.clicked.connect(self._on_save_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self._save_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        self._name_input.setFocus()

    def _on_save_clicked(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            self._message.setText("Enter a profile name.")
            return

        if (default_profiles_dir() / f"{name}.json").exists() and self._pending_overwrite_name != name:
            self._pending_overwrite_name = name
            self._message.setText(f"A profile named '{name}' already exists - save again to overwrite it.")
            self._save_button.setText("Overwrite")
            return

        path = save_named_profile(self._profile, name, tone=self._tone)
        self.saved.emit(f"Saved calibration profile '{name}' to {path}")
        self.accept()


class MainWindow(QWidget):
    """`is_pick_session=True` is the one-shot `--pick` variant (see quick_pick.py): the primary
    button reads "Develop" instead of "Save calibration profile," and clicking it (or closing the
    window) emits `pickCompleted` with the picked (DensityProfile, tone_override) or None, rather
    than opening the named-profile save dialog. Everything else about the picking interaction is
    identical - one window class serves both flows."""

    pickCompleted = Signal(object)  # tuple[DensityProfile, ToneCurveParams | None] | None

    def __init__(self, *, show_load_controls: bool = True, is_pick_session: bool = False) -> None:
        super().__init__()
        self.show_load_controls = show_load_controls
        self.is_pick_session = is_pick_session
        self._pick_result_emitted = False
        self._window_size = _compute_window_size()
        self._max_display_width, self._max_display_height = self._compute_display_budget()
        self.negative_path: Path | None = None
        self.full_image: np.ndarray | None = None  # working-space (ACEScg), full resolution
        self.preview_working: np.ndarray | None = None  # downsampled, pre-stretch
        self.stride: int = 1
        self._stretch_bounds: tuple[float, float] = (0.0, 1.0)
        self.pick_mode: str = "shadow"
        self.shadow_point: tuple[int, int, np.ndarray] | None = None
        self.highlight_point: tuple[int, int, np.ndarray] | None = None
        self.auto_profile = None
        self.auto_candidate_mask: np.ndarray | None = None
        self.pending_tone_override: ToneCurveParams | None = None
        self._preview_popup: PreviewPopup | None = None

        self._magnifier = Magnifier(self)

        self._build_ui()
        self.setFixedSize(self._window_size)

    def _compute_display_budget(self) -> tuple[int, int]:
        """The *maximum* pixel budget for the downsampled preview - passed to
        sampling.downsample_for_display as max_width/max_height. This is not the image box's own
        shape: the box now always shrink-wraps whatever the actual downsampled image turns out to
        be, with a fixed uniform margin (see _IMAGE_PADDING and _on-load sizing in load_image()) -
        fitting a fixed-shape box around an image of a different aspect ratio was a real bug
        (unequal padding: whichever axis didn't match the box's assumed shape got most of the
        slack, e.g. a wide gap left/right with almost none top/bottom). The budget width is derived
        from the *real*, per-machine computed window size (`_compute_window_size()`), not a fixed
        constant - hardcoding it from the theoretical minimum window size was an earlier bug (the
        box stayed pinned to the minimum-window size and sat left-anchored on a real, wider screen).
        The budget height is a generous cap (same 3:2 relationship, now just an upper bound, not a
        shape requirement) so a very tall/portrait scan can't grow the box enough to push other
        controls into needing to scroll.

        Reserves 2*_IMAGE_PADDING off the content width up front - the box adds that padding back
        on top of the downsampled image size (see load_image), so without reserving it here the
        padded box could end up wider than the actual available content width, overflowing past the
        window's edge the same way the box itself once did before it accounted for layout margins."""
        width = self._window_size.width() - 2 * _LAYOUT_MARGIN - 2 * _IMAGE_PADDING
        height = round(width / _DISPLAY_ASPECT_RATIO)
        return width, height

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(_LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN)

        top_bar = QHBoxLayout()
        self.filename_label = QLabel("No negative loaded")
        self.filename_label.setProperty("role", "dim")
        top_bar.addWidget(self.filename_label, stretch=1)
        if self.show_load_controls:
            load_button = QPushButton("Load negative...")
            load_button.clicked.connect(self._on_load_clicked)
            top_bar.addWidget(load_button)
        root.addLayout(top_bar)

        self.image_view = ImageView()
        self.image_view.set_view_size(self._max_display_width, self._max_display_height)
        self.image_view.clicked.connect(self._on_image_clicked)
        self.image_view.hovered.connect(self._on_image_hovered)
        self.image_view.left.connect(self._magnifier.hide)
        # AlignHCenter matters once an image is loaded: the box then shrink-wraps that image's own
        # shape (see load_image) and is usually narrower than the window, unlike every full-width
        # sibling here - without this it would sit left-anchored again, the same bug as before.
        root.addWidget(self.image_view, alignment=Qt.AlignmentFlag.AlignHCenter)

        neutral_points_frame = QFrame()
        neutral_points_frame.setProperty("role", "panel")
        frame_layout = QVBoxLayout(neutral_points_frame)
        neutral_points_title = QLabel("NEUTRAL POINTS")
        neutral_points_title.setProperty("role", "dim")
        frame_layout.addWidget(neutral_points_title)

        toggle_row = QHBoxLayout()
        self.shadow_button = QPushButton("Shadow")
        self.shadow_button.setProperty("role", "shadow")
        self.shadow_button.setCheckable(True)
        self.shadow_button.setChecked(True)
        self.highlight_button = QPushButton("Highlight")
        self.highlight_button.setProperty("role", "highlight")
        self.highlight_button.setCheckable(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.shadow_button)
        self.mode_group.addButton(self.highlight_button)
        self.shadow_button.toggled.connect(lambda checked: checked and self._set_pick_mode("shadow"))
        self.highlight_button.toggled.connect(lambda checked: checked and self._set_pick_mode("highlight"))
        toggle_row.addWidget(self.shadow_button)
        toggle_row.addWidget(self.highlight_button)
        frame_layout.addLayout(toggle_row)
        root.addWidget(neutral_points_frame)

        self.preview_button = QPushButton("Preview")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._on_preview_clicked)
        root.addWidget(self.preview_button)

        self.safety_caption = QLabel(_SHADOW_CAPTION)
        self.safety_caption.setProperty("role", "dim")
        self.safety_caption.setWordWrap(True)
        root.addWidget(self.safety_caption)

        details_header = QHBoxLayout()
        self.details_toggle = QToolButton()
        self.details_toggle.setText("▸ Details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.clicked.connect(self._toggle_details)
        details_header.addWidget(self.details_toggle)
        details_header.addStretch(1)
        root.addLayout(details_header)

        self.details_panel = QFrame()
        self.details_panel.setProperty("role", "panel")
        details_layout = QVBoxLayout(self.details_panel)
        self.shadow_text = QLabel("Shadow point: not picked")
        self.shadow_text.setWordWrap(True)
        self.shadow_text.setProperty("role", "dim")
        self.highlight_text = QLabel("Highlight point: not picked")
        self.highlight_text.setWordWrap(True)
        self.highlight_text.setProperty("role", "dim")
        details_layout.addWidget(self.shadow_text)
        details_layout.addWidget(self.highlight_text)
        self.auto_overlay_checkbox = QCheckBox("Show auto-detected neutral-candidate overlay")
        self.auto_overlay_checkbox.toggled.connect(self._on_toggle_auto_overlay)
        details_layout.addWidget(self.auto_overlay_checkbox)
        self.auto_comparison_text = QLabel("")
        self.auto_comparison_text.setWordWrap(True)
        self.auto_comparison_text.setProperty("role", "dim")
        details_layout.addWidget(self.auto_comparison_text)

        # A fixed-height scroll area, not a widget that grows with its content: expanding Details
        # used to change the whole window's required layout height, and since the window itself is
        # a fixed size, that overflow didn't clip cleanly - Qt overlapped the Neutral Points panel
        # on top of the image instead (a real, reported bug). Capping this at a known height means
        # toggling Details can never change the layout's total height at all, whatever the content.
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidget(self.details_panel)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.details_scroll.setFixedHeight(_DETAILS_HEIGHT)
        self.details_scroll.setVisible(False)
        root.addWidget(self.details_scroll)

        root.addStretch(1)

        self.primary_button = QPushButton("Develop" if self.is_pick_session else "Save calibration profile")
        self.primary_button.setProperty("role", "primary")
        self.primary_button.setEnabled(False)
        self.primary_button.clicked.connect(self._on_primary_clicked)
        root.addWidget(self.primary_button)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "dim")
        root.addWidget(self.status_label)

    def _status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_pick_mode(self, mode: str) -> None:
        self.pick_mode = mode
        self.safety_caption.setText(_SHADOW_CAPTION if mode == "shadow" else _HIGHLIGHT_CAPTION)

    def _toggle_details(self) -> None:
        visible = self.details_toggle.isChecked()
        self.details_scroll.setVisible(visible)
        self.details_toggle.setText(("▾" if visible else "▸") + " Details")

    def _on_preview_clicked(self) -> None:
        if self.shadow_point is None or self.highlight_point is None or self.preview_working is None:
            return
        shadow_rgb = tuple(float(v) for v in self.shadow_point[2])
        highlight_rgb = tuple(float(v) for v in self.highlight_point[2])
        self._preview_popup = PreviewPopup(
            self, self.preview_working, shadow_rgb, highlight_rgb, initial_tone=self.pending_tone_override
        )
        self._preview_popup.toneChanged.connect(self._on_tone_changed)
        self._preview_popup.show()

    def _on_tone_changed(self, tone: ToneCurveParams | None) -> None:
        self.pending_tone_override = tone

    def _on_primary_clicked(self) -> None:
        if self.shadow_point is None or self.highlight_point is None:
            return
        shadow_rgb = tuple(float(v) for v in self.shadow_point[2])
        highlight_rgb = tuple(float(v) for v in self.highlight_point[2])
        profile = solve_density_balance(shadow_rgb, highlight_rgb)

        if self.is_pick_session:
            self._pick_result_emitted = True
            self.pickCompleted.emit((profile, self.pending_tone_override))
            self.close()
            return

        dialog = SaveProfileDialog(self, profile, self.pending_tone_override)
        dialog.saved.connect(self._status)
        dialog.exec()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.is_pick_session and not self._pick_result_emitted:
            self._pick_result_emitted = True
            self.pickCompleted.emit(None)
        super().closeEvent(event)

    def _refresh_point_labels(self) -> None:
        for label_widget, point, name in (
            (self.shadow_text, self.shadow_point, "Shadow point"),
            (self.highlight_text, self.highlight_point, "Highlight point"),
        ):
            if point is None:
                label_widget.setText(f"{name}: not picked")
                continue
            x, y, rgb = point
            chroma = patch_chroma(rgb)
            label_widget.setText(
                f"{name}: pixel ({x},{y})  RGB=({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})  chroma={chroma:.2f}"
            )
        both_picked = self.shadow_point is not None and self.highlight_point is not None
        self.preview_button.setEnabled(both_picked)
        self.primary_button.setEnabled(both_picked)

    def _refresh_auto_comparison_text(self) -> None:
        if self.auto_profile is None:
            self.auto_comparison_text.setText("Auto-detected estimate: unavailable for this image.")
            return
        self.auto_comparison_text.setText(
            "Auto-detected estimate (comparison only): "
            f"white_balance={tuple(round(v, 4) for v in self.auto_profile.white_balance)}  "
            f"density_scale={tuple(round(v, 4) for v in self.auto_profile.density_scale)}"
        )

    def _build_main_pixmap(self) -> QPixmap:
        stretched = apply_stretch(self.preview_working, *self._stretch_bounds)
        if self.auto_overlay_checkbox.isChecked() and self.auto_candidate_mask is not None:
            stretched = stretched.copy()
            mask = self.auto_candidate_mask
            stretched[mask] = stretched[mask] * 0.6 + _OVERLAY_TINT * 0.4
        return QPixmap.fromImage(_to_qimage(stretched))

    def _on_toggle_auto_overlay(self, checked: bool) -> None:  # noqa: ARG002 - Qt callback signature
        if self.preview_working is None:
            return
        self.image_view.set_pixmap(self._build_main_pixmap())

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load negative", "", "TIFF negatives (*.tif *.tiff)")
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        self._status(f"Loading {Path(path).name}...")
        self.repaint()
        try:
            working = load_working_space_image(path)
        except ScanColorError as exc:
            self._status(f"Error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface any I/O error in the UI, don't crash it
            self._status(f"Error loading {path}: {exc}")
            return

        self.negative_path = Path(path)
        self.full_image = working
        self.shadow_point = None
        self.highlight_point = None
        self.pending_tone_override = None
        self.image_view.clear_markers()

        display, stride = downsample_for_display(
            working, max_width=self._max_display_width, max_height=self._max_display_height
        )
        self.stride = stride
        self.preview_working = display
        self._stretch_bounds = compute_stretch_bounds(display)

        # Shrink-wrap the box to this specific image's own downsampled shape, plus a fixed uniform
        # margin on every side - guarantees equal padding on all four sides for any image's aspect
        # ratio, rather than fitting into a box of some other fixed shape (which only looked right
        # for an image that happened to already match that shape, and put all the leftover slack on
        # whichever axis didn't match otherwise - a real, reported bug).
        display_height, display_width = display.shape[:2]
        self.image_view.set_view_size(display_width + 2 * _IMAGE_PADDING, display_height + 2 * _IMAGE_PADDING)

        try:
            self.auto_profile = auto_density_balance(display)
            self.auto_candidate_mask = _neutral_candidate_mask(display, DEFAULT_NEUTRAL_FRACTION)
        except ValueError:
            # Auto-detection can legitimately fail (too few candidates) on an unusual image - this
            # is a comparison aid, not a required part of picking, so degrade gracefully.
            self.auto_profile = None
            self.auto_candidate_mask = None

        self.image_view.set_pixmap(self._build_main_pixmap())
        self._refresh_point_labels()
        self._refresh_auto_comparison_text()

        self.filename_label.setText(self.negative_path.name)
        self.filename_label.setProperty("role", None)
        self.filename_label.style().unpolish(self.filename_label)
        self.filename_label.style().polish(self.filename_label)
        self._status(f"Loaded {working.shape[1]}x{working.shape[0]}, displayed at 1/{stride} scale")

    def _on_image_clicked(self, disp_x: int, disp_y: int) -> None:
        if self.full_image is None:
            return
        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(disp_x, disp_y, self.stride, w, h)
        x, y, rgb = snap_to_representative_pixel(self.full_image, full_x, full_y)

        display_point = QPoint(*full_res_to_display_coords(x, y, self.stride))
        if self.pick_mode == "shadow":
            self.shadow_point = (x, y, rgb)
            self.image_view.shadow_marker = display_point
        else:
            self.highlight_point = (x, y, rgb)
            self.image_view.highlight_marker = display_point
        self.image_view.update()
        self._refresh_point_labels()
        self._sync_preview_popup()

    def _sync_preview_popup(self) -> None:
        """If the Preview popup is currently open, keep it tracking the live picks instead of
        going stale - re-picking a point while the popup is open should update what it shows, not
        require closing and reopening it."""
        if self._preview_popup is None or not self._preview_popup.isVisible():
            return
        if self.shadow_point is None or self.highlight_point is None:
            return
        shadow_rgb = tuple(float(v) for v in self.shadow_point[2])
        highlight_rgb = tuple(float(v) for v in self.highlight_point[2])
        self._preview_popup.update_points(shadow_rgb, highlight_rgb)

    def _on_image_hovered(self, disp_x: int, disp_y: int) -> None:
        if self.full_image is None:
            return
        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(disp_x, disp_y, self.stride, w, h)
        patch = extract_magnifier_patch(self.full_image, full_x, full_y, radius=_MAGNIFIER_RADIUS)
        magnified = magnify_patch(apply_stretch(patch, *self._stretch_bounds), zoom=_MAGNIFIER_ZOOM)
        self._magnifier.show_patch(_to_qimage(magnified), self.image_view.mapToGlobal(QPoint(disp_x, disp_y)))
