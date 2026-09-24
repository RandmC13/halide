"""The calibration picker: pick neutral points - things that were white, grey or black in the real
scene - on any frames of a roll, and fit the film's density balance through all of them.

Layout (landscape, fixed size, never scrolls - see _compute_window_size):
  top bar         roll name · Open profile… · Load roll…
  filmstrip       the whole roll, current frame outlined, point counts under frames (gui/filmstrip.py)
  left column     Negative | Positive switch + what the positive is · the image with numbered point
                  markers · the "what to click" caption (states which way brightness is reversed on
                  the negative - see CLAUDE.md, never soften it) · status line
  right panel     step wedge (gui/step_wedge.py) · neutral points list (gui/point_list.py) · notice
                  line · Proof roll… · Roll details / Print / Details drawers (gui/drawers.py,
                  accordion) · the one red primary button: Save calibration profile (Develop in the
                  one-shot `invert --pick` flow)

All picking state and logic is in gui/roll.py::CalibrationSession (no Qt); this module lays out the
widgets and keeps them in step with it. Picks always sample the raw full-resolution negative
(gui/sampling.py), whichever view is shown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from halide.batch.orchestrator import TIFF_SUFFIXES
from halide.calibration.anchors import NeutralPoint
from halide.calibration.auto import DEFAULT_NEUTRAL_FRACTION, _neutral_candidate_mask, auto_density_balance
from halide.calibration.profile_store import default_profiles_dir, save_named_profile
from halide.core.tone_render import ResolvedTone
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui import theme
from halide.gui.drawers import Accordion, Drawer, PrintControls, RollDetailsForm
from halide.gui.filmstrip import Filmstrip
from halide.gui.loaders import FrameLoader, PreviewLoader
from halide.gui.point_list import CC_EXPLANATION, PointList, agreement_colour
from halide.gui.render import negative_display, positive_display, print_patch
from halide.gui.roll import PREVIEW_LONG_EDGE, CalibrationSession, Frame, PointView
from halide.gui.sampling import (
    apply_stretch,
    compute_stretch_bounds,
    display_to_full_res_coords,
    downsample_for_display,
    extract_magnifier_patch,
    magnify_patch,
    patch_chroma,
    snap_to_representative_pixel,
)
from halide.gui.step_wedge import StepWedge, WedgeTick
from halide.io.contact_sheet import downsample_linear
from halide.processing import read_roll_scan_metadata

# Fixed, landscape, sized from the real screen: roughly 55% x 70% of the available area, clamped so
# it's never tiny on a small laptop nor sprawling on a big monitor. Fixed-size and non-resizable as
# before - a small purpose-built tool, not a document window.
_WINDOW_MIN = QSize(900, 700)
_WINDOW_MAX = QSize(1400, 900)
_PANEL_WIDTH = 340
_LAYOUT_MARGIN = 14
_IMAGE_PADDING = 10  # uniform gap between the image and its drawn border, on all four sides
_DISPLAY_BUDGET = (1600, 1100)  # downsampled display resolution; painted scaled to fit the image area

_MAGNIFIER_RADIUS = 10
_MAGNIFIER_ZOOM = 6
_MARKER_RADIUS = 7
_MARKER_HIT_RADIUS = 10  # display pixels: a click this close to a marker selects it instead of adding
_OVERLAY_TINT = np.array([0.0, 1.0, 0.0], dtype=np.float32)

_CAPTION_NEGATIVE = (
    "Click things that were white, grey or black in the real scene. This is the raw negative, so "
    "brightness is reversed: real whites look DARK here, real shadows look LIGHT."
)
_CAPTION_POSITIVE = (
    "Click things that were white, grey or black in the real scene. Points are always measured on "
    "the negative, whichever view you click on."
)


def _compute_window_size() -> QSize:
    screen = QGuiApplication.primaryScreen()
    available = screen.availableGeometry() if screen is not None else None
    if available is None:
        return QSize(_WINDOW_MIN)
    width = min(max(int(available.width() * 0.55), _WINDOW_MIN.width()), _WINDOW_MAX.width(), available.width())
    height = min(max(int(available.height() * 0.70), _WINDOW_MIN.height()), _WINDOW_MAX.height(), available.height())
    return QSize(width, height)


def _qimage(rgb: np.ndarray) -> QImage:
    """float [0, 1] or uint8 (H, W, 3) -> an owned QImage copy."""
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """Paints the current frame scaled to fit its area, inside a border with uniform padding, plus
    numbered point markers; reports clicks and hovers in the *displayed image's* pixel coordinates
    (MainWindow maps those to full resolution via gui/sampling.py)."""

    clicked = Signal(int, int)
    hovered = Signal(int, int)
    left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        self._placeholder = "Load a roll to begin"
        self.markers: list[tuple[int, int, int, str, bool]] = []  # (x, y, number, colour, selected)
        self._flash: int | None = None  # number of a marker currently flashing
        self._flash_on = False
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._flash_step)
        self._flash_steps = 0

    def set_image(self, pixmap: QPixmap | None, placeholder: str = "") -> None:
        self._pixmap = pixmap
        if placeholder:
            self._placeholder = placeholder
        self.update()

    def flash(self, number: int) -> None:
        self._flash, self._flash_steps = number, 6
        self._flash_timer.start(140)

    def _flash_step(self) -> None:
        self._flash_on = not self._flash_on
        self._flash_steps -= 1
        if self._flash_steps <= 0:
            self._flash_timer.stop()
            self._flash, self._flash_on = None, False
        self.update()

    def _image_rect(self) -> QRectF | None:
        """Where the image is drawn: scaled to fit inside the padding, centred."""
        if self._pixmap is None:
            return None
        avail_w = self.width() - 2 * _IMAGE_PADDING - 2
        avail_h = self.height() - 2 * _IMAGE_PADDING - 2
        scale = min(avail_w / self._pixmap.width(), avail_h / self._pixmap.height())
        w, h = self._pixmap.width() * scale, self._pixmap.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _scale(self) -> float:
        rect = self._image_rect()
        return rect.width() / self._pixmap.width() if rect is not None else 1.0

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._image_rect()
        if rect is None:
            painter.fillRect(self.rect(), QColor(theme.BACKGROUND_ALT))
            painter.setPen(QColor(theme.BORDER))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        # The border shrink-wraps the image with uniform padding, whatever its shape.
        box = rect.adjusted(-_IMAGE_PADDING, -_IMAGE_PADDING, _IMAGE_PADDING, _IMAGE_PADDING)
        painter.setBrush(QColor(theme.BACKGROUND_ALT))
        painter.setPen(QColor(theme.BORDER))
        painter.drawRoundedRect(box, 6, 6)
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        scale = self._scale()
        label_font = QFont(self.font())
        label_font.setPixelSize(10)
        label_font.setBold(True)
        painter.setFont(label_font)
        for x, y, number, colour, selected in self.markers:
            if number == self._flash and not self._flash_on:
                continue
            centre = QPointF(rect.x() + (x + 0.5) * scale, rect.y() + (y + 0.5) * scale)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 170), 4 if selected else 3))
            painter.drawEllipse(centre, _MARKER_RADIUS, _MARKER_RADIUS)
            painter.setPen(QPen(QColor(colour), 2.5 if selected else 1.5))
            painter.drawEllipse(centre, _MARKER_RADIUS, _MARKER_RADIUS)
            label = QRectF(centre.x() + _MARKER_RADIUS, centre.y() - _MARKER_RADIUS - 12, 22, 14)
            painter.setPen(QColor(0, 0, 0, 200))
            painter.drawText(label.translated(1, 1), Qt.AlignmentFlag.AlignLeft, str(number))
            painter.setPen(QColor(colour))
            painter.drawText(label, Qt.AlignmentFlag.AlignLeft, str(number))

    def _to_image(self, event) -> tuple[int, int] | None:
        rect = self._image_rect()
        if rect is None:
            return None
        pos = event.position()
        if not rect.contains(pos):
            return None
        scale = self._scale()
        return int((pos.x() - rect.x()) / scale), int((pos.y() - rect.y()) / scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        point = self._to_image(event)
        if point is not None:
            self.clicked.emit(*point)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = self._to_image(event)
        if point is not None:
            self.hovered.emit(*point)
        else:
            self.left.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.left.emit()

    def display_scale(self) -> float:
        return self._scale()


class Magnifier(QWidget):
    """A frameless, always-on-top zoomed patch following the cursor over the image. Needs a real
    parent: Wayland refuses to place a parentless popup-type window (see the Qt rewrite's notes)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(
            parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
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
    """Name the calibration and save it. On a name collision the same dialog asks again, with its
    button relabelled "Overwrite", rather than stacking a second dialog."""

    saved = Signal(str, str)  # (name, message)

    def __init__(self, parent: QWidget, profile: DensityProfile, sidecars: dict, suggested_name: str = "") -> None:
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle("Save calibration profile")
        self._profile = profile
        self._sidecars = sidecars
        self._pending_overwrite_name: str | None = None

        layout = QVBoxLayout(self)
        self._message = QLabel("Name this calibration:")
        self._message.setWordWrap(True)
        layout.addWidget(self._message)
        self._name_input = QLineEdit(suggested_name)
        self._name_input.returnPressed.connect(self._on_save_clicked)
        layout.addWidget(self._name_input)
        row = QHBoxLayout()
        self._save_button = QPushButton("Save")
        self._save_button.setProperty("role", "primary")
        self._save_button.clicked.connect(self._on_save_clicked)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(self._save_button)
        row.addWidget(cancel)
        layout.addLayout(row)
        self._name_input.setFocus()
        self._name_input.selectAll()

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
        path = save_named_profile(self._profile, name, **self._sidecars)
        self.saved.emit(name, f"Saved calibration profile '{name}' to {path}")
        self.accept()


class MainWindow(QWidget):
    """`is_pick_session=True` is the one-shot `invert --pick` variant (see quick_pick.py): one frame,
    so no filmstrip, roll loading or proof sheet; the red button reads "Develop" and emits
    `pickCompleted` with (DensityProfile, tone override) - or None if the window is closed first."""

    pickCompleted = Signal(object)

    def __init__(self, *, show_load_controls: bool = True, is_pick_session: bool = False) -> None:
        super().__init__()
        self.show_load_controls = show_load_controls and not is_pick_session
        self.is_pick_session = is_pick_session
        self._pick_result_emitted = False
        self.session = CalibrationSession()
        self._profile_name = ""  # a reopened profile's name, offered again on save

        self._current: int | None = None  # index into session.frames
        self.full_image: np.ndarray | None = None  # the current frame, working space, full resolution
        self.display: np.ndarray | None = None  # downsampled for display
        self.stride = 1
        self._stretch = (0.0, 1.0)
        self._display_estimate: DensityProfile | None = None  # auto estimate at display resolution
        self._auto_mask: np.ndarray | None = None
        self._resolved: ResolvedTone | None = None  # print decision of the displayed positive
        self._positive_profile: DensityProfile | None = None
        self._positive_gain = 1.0
        self._flat_preview = False
        self._nudge: str | None = None
        self._pending_flash: int | None = None
        self._preview_loader: PreviewLoader | None = None
        self._frame_loader: FrameLoader | None = None
        self._thumbs_timer = QTimer(self)
        self._thumbs_timer.setSingleShot(True)
        self._thumbs_timer.timeout.connect(self._refresh_filmstrip_thumbnails)

        self._magnifier = Magnifier(self)
        self._build_ui()
        self.setFixedSize(_compute_window_size())
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._remove_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self._remove_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._deselect)
        self._refresh_points()

    # --- layout ---------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(_LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.roll_label = QLabel("No roll loaded")
        self.roll_label.setProperty("role", "dim")
        top.addWidget(self.roll_label, stretch=1)
        if self.show_load_controls:
            open_button = QPushButton("Open profile…")
            open_button.setToolTip("Reopen a saved profile's neutral points and roll details to add to or change them")
            open_button.clicked.connect(self._on_open_profile)
            top.addWidget(open_button)
            load_button = QPushButton("Load roll…")
            load_button.setToolTip("Choose a folder of scans (one roll)")
            load_button.clicked.connect(self._on_load_roll)
            top.addWidget(load_button)
        root.addLayout(top)

        self.filmstrip = Filmstrip()
        self.filmstrip.frameClicked.connect(self._select_frame)
        self.filmstrip.setVisible(not self.is_pick_session)
        root.addWidget(self.filmstrip)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        # left column: view switch, image, caption, status
        left = QVBoxLayout()
        left.setSpacing(6)
        switch_row = QHBoxLayout()
        self.negative_button = QPushButton("Negative")
        self.positive_button = QPushButton("Positive")
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        for button in (self.negative_button, self.positive_button):
            button.setProperty("role", "segment")
            button.setCheckable(True)
            self.view_group.addButton(button)
            switch_row.addWidget(button)
        self.negative_button.setChecked(True)
        self.view_group.buttonToggled.connect(lambda _b, checked: checked and self._on_view_changed())
        self.source_note = QLabel("")
        self.source_note.setProperty("role", "dim")
        switch_row.addSpacing(8)
        switch_row.addWidget(self.source_note, stretch=1)
        left.addLayout(switch_row)

        self.image_view = ImageView()
        self.image_view.clicked.connect(self._on_image_clicked)
        self.image_view.hovered.connect(self._on_image_hovered)
        self.image_view.left.connect(self._magnifier.hide)
        left.addWidget(self.image_view, stretch=1)

        self.caption = QLabel(_CAPTION_NEGATIVE)
        self.caption.setWordWrap(True)
        self.caption.setProperty("role", "dim")
        left.addWidget(self.caption)
        self.status_label = QLabel("")
        self.status_label.setProperty("role", "dim")
        left.addWidget(self.status_label)
        body.addLayout(left, stretch=1)

        # right panel: wedge, points, notice, proof, drawers, primary
        panel = QVBoxLayout()
        panel.setSpacing(6)
        panel_widget = QWidget()
        panel_widget.setFixedWidth(_PANEL_WIDTH)
        panel_widget.setLayout(panel)

        coverage_title = QLabel("COVERAGE")
        coverage_title.setProperty("role", "section")
        panel.addWidget(coverage_title)
        self.wedge = StepWedge()
        panel.addWidget(self.wedge)

        points_header = QHBoxLayout()
        points_title = QLabel("NEUTRAL POINTS  (?)")
        points_title.setProperty("role", "section")
        points_title.setToolTip(CC_EXPLANATION)
        points_header.addWidget(points_title)
        points_header.addStretch(1)
        self.point_count = QLabel("")
        self.point_count.setProperty("role", "section")
        points_header.addWidget(self.point_count)
        panel.addLayout(points_header)
        self.point_list = PointList()
        self.point_list.rowClicked.connect(self._on_row_clicked)
        self.point_list.removeClicked.connect(self._remove_point)
        panel.addWidget(self.point_list, stretch=1)

        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setMinimumHeight(30)
        panel.addWidget(self.notice)

        self.proof_button = QPushButton("Proof roll…")
        self.proof_button.setToolTip("A zoomable contact sheet of the whole roll printed with this calibration")
        self.proof_button.setEnabled(False)
        self.proof_button.setVisible(not self.is_pick_session)
        panel.addWidget(self.proof_button)

        self.details_form = RollDetailsForm()
        self.details_form.changed.connect(self._on_details_changed)
        self.print_controls = PrintControls()
        self.print_controls.changed.connect(self._on_print_changed)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setProperty("role", "dim")
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.auto_overlay = QCheckBox("Show auto-detected neutral candidates")
        self.auto_overlay.toggled.connect(lambda _c: self._refresh_image())
        details_body = QWidget()
        details_layout = QVBoxLayout(details_body)
        details_layout.setContentsMargins(4, 2, 4, 2)
        details_layout.addWidget(self.auto_overlay)
        details_layout.addWidget(self.details_label)
        self.print_drawer = Drawer("Print", self.print_controls, max_height=150)
        drawers = [Drawer("Roll details", self.details_form, max_height=140), self.print_drawer]
        drawers.append(Drawer("Details", details_body, max_height=150))
        if self.is_pick_session:
            drawers = drawers[1:]  # one-shot develop: nothing to save roll details into
        panel.addWidget(Accordion(drawers))

        self.primary_button = QPushButton("Develop" if self.is_pick_session else "Save calibration profile")
        self.primary_button.setProperty("role", "primary")
        self.primary_button.setEnabled(False)
        self.primary_button.clicked.connect(self._on_primary_clicked)
        panel.addWidget(self.primary_button)
        body.addWidget(panel_widget)

    # --- loading --------------------------------------------------------------------------------

    def _on_load_roll(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Load roll (a folder of scans)")
        if folder:
            self.load_roll(Path(folder))

    def _confirm_new_calibration(self) -> bool:
        if not self.session.points:
            return True
        answer = QMessageBox.question(
            self,
            "Start a new calibration?",
            f"Loading a different roll starts a new calibration. Clear the {len(self.session.points)} "
            "neutral point(s) picked so far?",
        )
        return answer == QMessageBox.StandardButton.Yes

    def load_roll(self, folder: Path, keep_points: bool = False) -> None:
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
        if not paths:
            self._status(f"No TIFF scans in {folder}")
            return
        if not keep_points:
            if not self._confirm_new_calibration():
                return
            self.session.reset()
            self._profile_name = ""
            self.details_form.set_values(self.session.details)
            self.print_controls.set_override(None)
        self._start_roll(paths, folder)

    def load_files(self, paths: list[Path]) -> None:
        """Individual scans as a roll of their own (`halide calibrate a.tif b.tif`, `invert --pick`)."""
        self.session.reset()
        self._start_roll(sorted(paths) if len(paths) > 1 else list(paths), paths[0].parent if len(paths) > 1 else None)

    def _start_roll(self, paths: list[Path], folder: Path | None) -> None:
        self._stop_loaders()
        metadata = read_roll_scan_metadata(paths)
        scans = [metadata[str(p)][0] for p in paths]
        self.session.set_roll(paths, scans, folder)
        self.roll_label.setText(
            f"{folder.name} · {len(paths)} frames" if folder is not None else paths[0].name
        )
        self.roll_label.setProperty("role", None)
        self.roll_label.style().unpolish(self.roll_label)
        self.roll_label.style().polish(self.roll_label)
        self.filmstrip.strip.set_frames([f.number for f in self.session.frames])
        self._current = None
        self.full_image = None
        self.display = None
        self.image_view.set_image(None, "Loading…")
        if not self.is_pick_session:
            self._preview_loader = PreviewLoader(paths, self)
            self._preview_loader.previewReady.connect(self._on_preview_ready)
            self._preview_loader.start()
        self._select_frame(0)
        self._refresh_points()

    def _on_open_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open calibration profile", str(default_profiles_dir()), "halide profiles (*.json)"
        )
        if path:
            self.open_profile(Path(path))

    def open_profile(self, path: Path) -> None:
        if not self._confirm_new_calibration():
            return
        self.session.reset()
        try:
            folder = self.session.restore(path)
        except (ValueError, KeyError, OSError) as exc:
            self._status(f"Couldn't open {path.name}: {exc}")
            return
        self._profile_name = path.stem
        self.details_form.set_values(self.session.details)
        self.print_controls.set_override(self.session.tone_override)
        if folder is not None:
            self.load_roll(folder, keep_points=True)
            self._status(f"Reopened '{path.stem}': {len(self.session.points)} point(s)")
        else:
            self._refresh_points()
            self._status(
                f"Reopened '{path.stem}': {len(self.session.points)} point(s). Its roll folder wasn't "
                "found - load the roll to see them on their frames (they still count in the fit)."
            )

    def _on_preview_ready(self, index: int, preview, estimate, error) -> None:
        if index >= len(self.session.frames):
            return
        frame = self.session.frames[index]
        frame.preview, frame.estimate, frame.error = preview, estimate, error
        if error:
            self.filmstrip.strip.set_pixmap(index, None, failed=True)
        else:
            self.filmstrip.strip.set_pixmap(index, QPixmap.fromImage(_qimage(self._thumbnail(frame))))
        if index == self._current and self.display is None and preview is not None:
            self._show_placeholder_preview(frame)
        if all(f.loaded for f in self.session.frames):
            self._status(f"Roll loaded: {len(self.session.frames)} frames")
        self._refresh_wedge()
        self.proof_button.setEnabled(False)  # wired up in a later step

    def _select_frame(self, index: int) -> None:
        if not (0 <= index < len(self.session.frames)) or index == self._current:
            return
        self._current = index
        self.filmstrip.show_frame(index)
        frame = self.session.frames[index]
        self.full_image = None
        self.display = None
        self._magnifier.hide()
        if frame.preview is not None:
            self._show_placeholder_preview(frame)
        else:
            self.image_view.set_image(None, f"Loading {frame.path.name}…")
        self._status(f"Loading {frame.path.name}…")
        if self._frame_loader is not None:
            self._frame_loader.loaded.disconnect()
        self._frame_loader = FrameLoader(frame.path, self)
        self._frame_loader.loaded.connect(self._on_frame_loaded)
        self._frame_loader.start()

    def _show_placeholder_preview(self, frame: Frame) -> None:
        """While the full-resolution frame loads, show its small preview (not clickable yet)."""
        self.image_view.markers = []
        self.image_view.set_image(QPixmap.fromImage(_qimage(self._thumbnail(frame))))

    def _on_frame_loaded(self, path: Path, image, error) -> None:
        if self._current is None or self.session.frames[self._current].path != path:
            return  # the user has moved on to another frame
        if error:
            self.image_view.set_image(None, f"Couldn't load {path.name}")
            self._status(f"Error: {error}")
            return
        self.full_image = image
        frame = self.session.frames[self._current]
        if frame.preview is None:
            # No background preview for this frame (the one-shot --pick flow loads none): derive it
            # from the full frame, so the step wedge and filmstrip have it too.
            frame.preview = downsample_linear(image, PREVIEW_LONG_EDGE)
            self._refresh_wedge()
        self.display, self.stride = downsample_for_display(image, *_DISPLAY_BUDGET)
        self._stretch = compute_stretch_bounds(self.display)
        try:
            self._display_estimate = auto_density_balance(self.display)
            self._auto_mask = _neutral_candidate_mask(self.display, DEFAULT_NEUTRAL_FRACTION)
        except ValueError:
            self._display_estimate, self._auto_mask = None, None
        if self.status_label.text().startswith("Loading"):
            self._status("")  # don't overwrite anything more useful (e.g. "Reopened …")
        self._refresh_image()
        if self._pending_flash is not None:
            self.image_view.flash(self._pending_flash)
            self._pending_flash = None

    def _stop_loaders(self, wait: bool = False) -> None:
        if self._preview_loader is not None:
            self._preview_loader.requestInterruption()
            self._preview_loader.previewReady.disconnect()
            if wait:
                self._preview_loader.wait(3000)  # its loop checks for interruption every 0.2 s
            self._preview_loader = None

    # --- rendering ------------------------------------------------------------------------------

    def _positive_mode(self) -> bool:
        return self.positive_button.isChecked()

    def _tone(self) -> ToneCurveParams:
        return ToneCurveParams(mode="linear") if self._flat_preview else self.session.print_tone()

    def _thumbnail(self, frame: Frame) -> np.ndarray:
        """A frame's filmstrip thumbnail in the current view (negative, or positive by the same rule
        as the main image)."""
        if frame.preview is None:
            return np.zeros((2, 3, 3), dtype=np.uint8)
        if self._positive_mode():
            profile, _is_fit, gain = self.session.positive_source(frame)
            if profile is not None:
                return positive_display(frame.preview, profile, self._tone(), gain)[0]
        return negative_display(frame.preview, compute_stretch_bounds(frame.preview))

    def _refresh_filmstrip_thumbnails(self) -> None:
        for i, frame in enumerate(self.session.frames):
            if frame.preview is not None:
                self.filmstrip.strip.set_pixmap(i, QPixmap.fromImage(_qimage(self._thumbnail(frame))))

    def _refresh_image(self) -> None:
        """Re-render the current frame in the current view, and the view's labels."""
        positive = self._positive_mode()
        self.caption.setText(_CAPTION_POSITIVE if positive else _CAPTION_NEGATIVE)
        frame = self.session.frames[self._current] if self._current is not None else None
        self._resolved = None
        self._positive_profile = None
        if positive and frame is not None:
            profile, is_fit, gain = self.session.positive_source(frame, self._display_estimate)
            if is_fit:
                self.source_note.setText(f"Your calibration · {len(self.session.points)} points")
            elif profile is not None:
                self.source_note.setText("Rough auto estimate — not your final output")
            else:
                self.source_note.setText("No auto estimate for this frame — showing the negative")
            self._positive_profile, self._positive_gain = profile, gain
        else:
            self.source_note.setText("Raw negative" if frame is not None else "")
        self.print_drawer.setEnabled(positive)

        if self.display is None:
            self._refresh_markers()
            return
        if self._positive_profile is not None:
            rgb, self._resolved = positive_display(self.display, self._positive_profile, self._tone(), self._positive_gain)
            if self._resolved.mode == "paper":
                self.print_controls.show_fitted(self._resolved.exposure, self._resolved.contrast)
            shown = rgb.astype(np.float32) / 255.0
        else:
            shown = negative_display(self.display, self._stretch)
        if self.auto_overlay.isChecked() and self._auto_mask is not None:
            shown = shown.copy()
            shown[self._auto_mask] = shown[self._auto_mask] * 0.6 + _OVERLAY_TINT * 0.4
        self.image_view.set_image(QPixmap.fromImage(_qimage(shown)))
        self._refresh_markers()

    def _refresh_markers(self) -> None:
        frame = self.session.frames[self._current] if self._current is not None else None
        markers = []
        if frame is not None and self.display is not None:
            for view in self.session.views():
                if view.point.frame == frame.path:
                    markers.append(
                        (view.point.x // self.stride, view.point.y // self.stride, view.index + 1,
                         agreement_colour(view), view.index == self.session.selected)
                    )
        self.image_view.markers = markers
        self.image_view.update()

    # --- points ---------------------------------------------------------------------------------

    def _on_image_clicked(self, disp_x: int, disp_y: int) -> None:
        if self.full_image is None or self._current is None:
            return
        frame = self.session.frames[self._current]
        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(disp_x, disp_y, self.stride, w, h)
        hit_radius = max(1, round(_MARKER_HIT_RADIUS * self.stride / max(self.image_view.display_scale(), 1e-6)))
        existing = self.session.point_at(frame.path, full_x, full_y, hit_radius)
        if existing is not None:
            self.session.selected = existing
            self._nudge = None
            self._refresh_points()
            return
        x, y, rgb = snap_to_representative_pixel(self.full_image, full_x, full_y)
        point = NeutralPoint(frame=frame.path, x=int(x), y=int(y), rgb=tuple(float(v) for v in rgb), scan=frame.scan)
        self._nudge = self.session.add_point(point)
        self._points_changed()

    def _on_row_clicked(self, index: int) -> None:
        self.session.selected = index
        self._nudge = None
        point = self.session.points[index]
        target = next((i for i, f in enumerate(self.session.frames) if f.path == point.frame), None)
        self._refresh_points()
        if target is not None and target != self._current:
            self._pending_flash = index + 1  # flashed once that frame has loaded and shows markers
            self._select_frame(target)
        else:
            self.image_view.flash(index + 1)

    def _remove_point(self, index: int) -> None:
        self.session.remove_point(index)
        self._nudge = None
        self._points_changed()

    def _remove_selected(self) -> None:
        if self.session.selected is not None:
            self._remove_point(self.session.selected)

    def _deselect(self) -> None:
        self.session.selected = None
        self._refresh_points()

    def _points_changed(self) -> None:
        """The fit may have changed: everything that shows it follows."""
        self._refresh_points()
        if self._positive_mode():
            self._refresh_image()
            self._thumbs_timer.start(0)

    def _refresh_points(self) -> None:
        views = self.session.views()
        self.point_list.set_rows(views, self.session.selected)
        self.point_count.setText(str(len(views)) if views else "")
        self._refresh_wedge(views)
        counts = self.session.point_counts()
        self.filmstrip.strip.set_counts([counts.get(f.path, 0) for f in self.session.frames])
        self._refresh_markers()
        self._refresh_notice(views)
        self._refresh_details(views)
        self.primary_button.setEnabled(self.session.can_fit())

    def _refresh_wedge(self, views: list[PointView] | None = None) -> None:
        views = self.session.views() if views is None else views
        ticks = [
            WedgeTick(v.index + 1, v.green_density, agreement_colour(v), v.index == self.session.selected)
            for v in views
        ]
        self.wedge.set_state(self.session.wedge_range(), ticks)

    def _refresh_notice(self, views: list[PointView]) -> None:
        worst = self.session.worst()
        if self._nudge:
            text, colour = f"› {self._nudge}", theme.TEXT_DIM
        elif worst is not None:
            a = views[worst].agreement
            text = f"⚠ Point {worst + 1} is furthest from the others ({a.label()}). Was that object really neutral?"
            colour = agreement_colour(views[worst])
        elif len(views) < 2:
            text, colour = "Pick neutral objects of different tones, on any frames.", theme.TEXT_DIM
        elif len(views) == 2:
            text, colour = "Add a third point to see how well your points agree.", theme.TEXT_DIM
        else:
            text, colour = "", theme.TEXT_DIM
        self.notice.setText(text)
        self.notice.setStyleSheet(f"color: {colour};")

    def _refresh_details(self, views: list[PointView]) -> None:
        lines = []
        if self.session.selected is not None and self.session.selected < len(views):
            v = views[self.session.selected]
            rgb = v.point.rgb
            lines.append(
                f"Point {v.index + 1}: {v.point.frame.name} pixel ({v.point.x}, {v.point.y})  "
                f"RGB ({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})  raw chroma {patch_chroma(np.array(rgb)):.2f}"
            )
        fitted = self.session.profile()
        if fitted is not None:
            wb = ", ".join(f"{x:.4f}" for x in fitted.white_balance)
            ds = ", ".join(f"{x:.4f}" for x in fitted.density_scale)
            lines.append(f"Fit: white balance ({wb})  density scale ({ds})")
        reference = self.session.reference
        if reference is not None:
            lines.append(f"Scan exposure reference: {reference.describe()}")
        if self._display_estimate is not None:
            wb = ", ".join(f"{x:.4f}" for x in self._display_estimate.white_balance)
            ds = ", ".join(f"{x:.4f}" for x in self._display_estimate.density_scale)
            lines.append(f"This frame's auto estimate (comparison only): ({wb}) / ({ds})")
        lines.append(CC_EXPLANATION)
        self.details_label.setText("\n\n".join(lines))

    # --- drawers --------------------------------------------------------------------------------

    def _on_view_changed(self) -> None:
        self._refresh_image()
        self._thumbs_timer.start(0)

    def _on_details_changed(self, values: dict) -> None:
        self.session.details = values

    def _on_print_changed(self, tone: ToneCurveParams | None, flat: bool) -> None:
        self.session.tone_override = tone
        self._flat_preview = flat
        if self._positive_mode():
            self._refresh_image()
            self._thumbs_timer.start(0)

    # --- magnifier ------------------------------------------------------------------------------

    def _on_image_hovered(self, disp_x: int, disp_y: int) -> None:
        if self.full_image is None:
            return
        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(disp_x, disp_y, self.stride, w, h)
        patch = extract_magnifier_patch(self.full_image, full_x, full_y, radius=_MAGNIFIER_RADIUS)
        if self._positive_profile is not None and self._resolved is not None:
            shown = print_patch(patch, self._positive_profile, self._resolved, self._positive_gain)
        else:
            shown = apply_stretch(patch, *self._stretch)
        magnified = magnify_patch(shown, zoom=_MAGNIFIER_ZOOM)
        anchor = self.image_view.mapToGlobal(QPoint(0, 0))
        rect = self.image_view._image_rect()
        scale = self.image_view.display_scale()
        global_pos = anchor + QPoint(int(rect.x() + disp_x * scale), int(rect.y() + disp_y * scale))
        self._magnifier.show_patch(_qimage(magnified), global_pos)

    # --- save / develop -------------------------------------------------------------------------

    def _on_primary_clicked(self) -> None:
        if not self.session.can_fit():
            return
        profile = self.session.profile()
        if self.is_pick_session:
            self._pick_result_emitted = True
            self.pickCompleted.emit((profile, self.session.tone_override))
            self.close()
            return
        dialog = SaveProfileDialog(self, self.session.profile_to_save(), self.session.sidecars(), self._profile_name)
        dialog.saved.connect(self._on_saved)
        dialog.exec()

    def _on_saved(self, name: str, message: str) -> None:
        self._profile_name = name
        self._status(message)

    def _status(self, message: str) -> None:
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._stop_loaders(wait=True)
        if self.is_pick_session and not self._pick_result_emitted:
            self._pick_result_emitted = True
            self.pickCompleted.emit(None)
        super().closeEvent(event)
