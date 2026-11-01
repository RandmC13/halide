"""The calibration picker: pick neutral points - things that were white, grey or black in the real
scene - on any frames of a roll, and fit the film's density balance through all of them.

Layout (landscape, fixed size, never scrolls - see _compute_window_size):
  top bar         roll name in edge print · FRAME n / N · profile in use · Contact sheet · Profiles ·
                  Load roll…
  filmstrip       the whole roll, current frame outlined, point counts under frames (gui/filmstrip.py)
  left column     Negative | Positive switch + what the positive is · the image with numbered point
                  markers · the "what to click" caption (states which way brightness is reversed on
                  the negative; never soften it) · status line
  right panel     step wedge (gui/step_wedge.py) · neutral points list (gui/point_list.py) · notice
                  line · Extra information / Print / Details drawers (gui/drawers.py,
                  accordion) · Save profile (quiet) and Develop roll… (the one red button), which turns the
                  panel over to the develop side (gui/develop_panel.py) and its summary; in the
                  one-shot `invert --pick` flow a single red Develop

All picking state and logic is in gui/roll.py::CalibrationSession (no Qt); this module lays out the
widgets and keeps them in step with it. Picks always sample the raw full-resolution negative
(gui/sampling.py), whichever view is shown.
"""

from __future__ import annotations

import html
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QProcess, QRect, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QDesktopServices,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from halide.io.roll import list_scans
from halide.calibration.anchors import NeutralPoint, point_rolls
from halide.calibration.auto import DEFAULT_NEUTRAL_FRACTION, _neutral_candidate_mask
from halide.calibration.scan_consistency import assess_roll, frame_flags
from halide.calibration.profile_store import (
    ProfileError,
    default_profiles_dir,
    find_profile,
    load_tone_override,
    save_named_profile,
    save_profile,
    validate_profile_name,
)
from halide.cli import console
from halide.core.tone_render import ResolvedTone
from halide.core.types import DensityProfile, ToneCurveParams
from halide.device import ComputeDevice
from halide.gui import motion, theme
from halide.gui.develop import DevelopSettings, existing_outputs, export_command, batch_command, frame_count
from halide.gui.develop_panel import DevelopPanel
from halide.gui.develop_runner import DevelopRunner
from halide.gui.drawers import Accordion, Drawer, PrintControls, RollDetailsForm
from halide.gui.filmstrip import Filmstrip
from halide.gui.image_view import ImageView
from halide.gui.loaders import FrameLoader, PreviewLoader
from halide.gui.notices import Notice, NoticeBar
from halide.gui.point_list import CC_EXPLANATION, PointList, agreement_colour, no_wrap
from halide.gui.profiles_window import ProfilesWindow
from halide.gui.proof_window import ProofWindow, wait_for_stopping_renderers
from halide.gui.render import negative_display, positive_display, print_patch
from halide.gui.roll import PREVIEW_LONG_EDGE, CalibrationSession, Frame, PointView, quiet_auto_estimate
from halide.gui.save_dialog import SaveProfileDialog
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
from halide.gui.summary_panel import SummaryPanel
from halide.gui.widgets import PROFILE_NAME_MAX_W, SaveStamp, ViewSwitch
from halide.io.contact_sheet import downsample_linear
from halide.processing import read_roll_scan_metadata

# Fixed, landscape, sized from the real screen: roughly 55% x 70% of the available area, clamped so
# it's never tiny on a small laptop nor sprawling on a big monitor. Fixed-size and non-resizable as
# before - a small purpose-built tool, not a document window.
_WINDOW_MIN = QSize(900, 700)
_WINDOW_MAX = QSize(1400, 900)
_PANEL_WIDTH = 340
# Names in the top bar are cut beyond these widths (full name in the tooltip), so at the 900 px floor
# the roll name, counter, profile stamp and three buttons all fit on one row.
_ROLL_NAME_MAX_W = 160
_LAYOUT_MARGIN = 14
_DISPLAY_BUDGET = (1600, 1100)  # downsampled display resolution; painted scaled to fit the image area

# Re-printing all the filmstrip thumbnails takes ~0.4 s on a 37-frame roll, so a slider drag or a run of
# clicks waits for a pause and prints once at the end.
_THUMBS_DELAY_MS = 150

_MAGNIFIER_RADIUS = 10
_MAGNIFIER_ZOOM = 6
_MARKER_HIT_RADIUS = 10  # display pixels: a click this close to a marker selects it instead of adding
_OVERLAY_TINT = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# The guide under the picture. On the negative the reversal is the one thing a newcomer must not miss,
# so it's picked out in the warning colour (never softened).
_CAPTION_NEGATIVE = (
    "Click things that were white, grey or black in the scene. Brightness is reversed: "
    f"<b style='color:{theme.TEXT_WARNING}'>real whites look DARK here</b>."
)
_CAPTION_POSITIVE = (
    "Click things that were white, grey or black in the scene. Points are measured on the negative "
    "either way."
)


def window_size_for(available: QRect | None) -> QSize:
    """The picker's fixed size for a screen's available area: 55% x 70%, never below _WINDOW_MIN
    (unless the screen itself is smaller) and capped at _WINDOW_MAX - on large screens the picker
    stays a comfortable working size rather than filling the screen."""
    if available is None:
        return QSize(_WINDOW_MIN)
    width = min(max(int(available.width() * 0.55), _WINDOW_MIN.width()), _WINDOW_MAX.width(), available.width())
    height = min(max(int(available.height() * 0.70), _WINDOW_MIN.height()), _WINDOW_MAX.height(), available.height())
    return QSize(width, height)


def _compute_window_size() -> QSize:
    screen = QGuiApplication.primaryScreen()
    return window_size_for(screen.availableGeometry() if screen is not None else None)


def _qimage(rgb: np.ndarray) -> QImage:
    """float [0, 1] or uint8 (H, W, 3) -> an owned QImage copy."""
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


_FORMAT_NAMES = {"jpg": "JPEG", "png": "PNG"}  # as the develop panel's sharing choice names them


class MainWindow(QWidget):
    """`is_pick_session=True` is the one-shot `invert --pick` variant (see quick_pick.py): one frame,
    so no filmstrip, roll loading or contact sheet; the red button reads "Develop" and emits
    `pickCompleted` with (DensityProfile, tone override, reliability warning or None) - or None if the
    window is closed first."""

    pickCompleted = Signal(object)

    def __init__(
        self, *, show_load_controls: bool = True, is_pick_session: bool = False, device: ComputeDevice | None = None
    ) -> None:
        super().__init__()
        self.setWindowTitle("halide")
        self.device = device  # what the contact sheet window develops frames on (None = CPU)
        self.show_load_controls = show_load_controls and not is_pick_session
        self.is_pick_session = is_pick_session
        self._pick_result_emitted = False
        self.session = CalibrationSession()
        self._saved_points: list = []  # the points as last saved or reopened; close asks if they differ
        self._profile_name = ""  # a reopened profile's name, offered again on save
        self.last_saved: tuple[str, Path | None] | None = None  # (profile name, its roll): the last save here
        self._profiles: ProfilesWindow | None = None

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
        self._preview_tone: ToneCurveParams | None = None  # the Print sliders' test strip, this frame only
        self._nudge: str | None = None
        self._shown_positive = False  # the view the image last rendered in
        self._pending_flash: int | None = None
        # A status worth keeping ("Reopened …", "Skipped …"): loading doesn't replace it until the
        # person moves to another frame themselves.
        self._sticky_status: str | None = None
        self._scan_flags_by_index: dict[int, str] = {}
        self._loose = False  # the open frames are loose scans, not a roll folder
        self._hold_side = False  # refreshing the points mustn't turn the panel (a save as part of developing)
        self._dev: dict | None = None  # the develop run in hand: its settings, progress and summary
        self._runner = DevelopRunner(self)
        self._runner.frameStarted.connect(self._on_frame_started)
        self._runner.frameDone.connect(self._on_frame_done)
        self._runner.summaryRow.connect(self._on_summary_row)
        self._runner.finished.connect(self._on_develop_finished)
        self._runner.failed.connect(self._on_develop_failed)
        self._exposure_differs = False  # the scan check found frames digitized at different exposures
        self._proof: ProofWindow | None = None
        self._preview_loader: PreviewLoader | None = None
        self._frame_loader: FrameLoader | None = None
        # Every loader thread until it finishes, including ones replaced by a newer load: Qt aborts
        # the whole process ("QThread: Destroyed while thread is still running") if the window is
        # destroyed while any of them runs, so closing waits for them all (closeEvent).
        self._threads: set[QThread] = set()
        # Bumped whenever a new roll starts; a loader's late signals carry the generation they were
        # started under and are dropped if it's no longer current (see _start_roll).
        self._generation = 0
        self._closing = False
        self._loading: tuple[Path, int] | None = None  # (frame, generation) of the last full-resolution load started
        self._thumbs_timer = QTimer(self)
        self._thumbs_timer.setSingleShot(True)
        self._thumbs_timer.timeout.connect(self._refresh_filmstrip_thumbnails)

        self._build_ui()
        size = _compute_window_size()
        self.setFixedSize(size)
        if size.height() <= _WINDOW_MIN.height():
            # The smallest window only: with the reliability note showing and a drawer open the panel
            # is otherwise ~50 px too tall, so it gives up its 9 px bottom margin and 2 px of spacing per gap (the point list
            # gives up rows first, see point_list.py). Bigger windows keep the default margin.
            margins = self._panel.contentsMargins()
            self._panel.setContentsMargins(margins.left(), margins.top(), margins.right(), 0)
            self._panel.setSpacing(4)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._remove_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self._remove_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._deselect)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self, activated=self._undo)
        self.setAcceptDrops(self.show_load_controls)  # not in the one-frame `invert --pick` window
        self._refresh_points()

    # --- layout ---------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(_LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN, _LAYOUT_MARGIN)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.roll_label = QLabel("NO ROLL LOADED")
        self.roll_label.setFont(theme.edge_font(15))
        self.roll_label.setStyleSheet(f"color: {theme.EDGE_PRINT};")
        top.addWidget(self.roll_label)
        self.frame_counter = QLabel("")
        self.frame_counter.setFont(theme.edge_font(13))
        self.frame_counter.setProperty("role", "dim")
        top.addWidget(self.frame_counter)
        self.profile_chip = QLabel("New calibration")
        self.profile_chip.setProperty("role", "dim")
        top.addWidget(self.profile_chip)
        # A save stamps the profile's name where the chip was, rather than taking a row of the panel,
        # which can't spare the height in the smallest window.
        self.save_stamp = SaveStamp()
        top.addWidget(self.save_stamp)
        top.addStretch(1)
        self.proof_button = QPushButton("Contact sheet")
        self.proof_button.setToolTip("A zoomable contact sheet of the whole roll printed with this calibration")
        self.proof_button.setEnabled(False)
        self.proof_button.setVisible(not self.is_pick_session)
        self.proof_button.clicked.connect(self._on_proof)
        top.addWidget(self.proof_button)
        if self.show_load_controls:
            self.profiles_button = QPushButton("Profiles")
            self.profiles_button.setToolTip("Your saved profiles: use one on this roll, edit its points, rename or delete it")
            self.profiles_button.clicked.connect(self.show_profiles)
            top.addWidget(self.profiles_button)
            self.load_button = QPushButton("Load roll…")
            self.load_button.setToolTip("Choose a folder of scans (one roll)")
            self.load_button.clicked.connect(self._on_load_roll)
            top.addWidget(self.load_button)
        root.addLayout(top)

        self.filmstrip = Filmstrip()
        self.filmstrip.frameClicked.connect(self._go_to_frame)
        self.filmstrip.setVisible(not self.is_pick_session)
        root.addWidget(self.filmstrip)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        # left column: view switch, image, caption, status
        left = QVBoxLayout()
        left.setSpacing(6)
        switch_row = QHBoxLayout()
        self.view_switch = ViewSwitch()
        self.negative_button = self.view_switch.negative_button
        self.positive_button = self.view_switch.positive_button
        self.view_group = self.view_switch.group
        switch_row.addWidget(self.view_switch)
        self.view_group.buttonToggled.connect(lambda _b, checked: checked and self._on_view_changed())
        self.source_note = QLabel("")
        self.source_note.setProperty("role", "dim")
        switch_row.addSpacing(8)
        switch_row.addWidget(self.source_note, stretch=1)
        self.auto_overlay = QCheckBox("Show likely neutrals")
        self.auto_overlay.setToolTip(
            "Marks the pixels the automatic estimate counts as neutral candidates. "
            "A hint for where to click, not a guarantee they really were neutral."
        )
        self.auto_overlay.toggled.connect(lambda _c: self._refresh_image())
        switch_row.addWidget(self.auto_overlay)
        left.addLayout(switch_row)

        self.image_view = ImageView()
        self.image_view.clicked.connect(self._on_image_clicked)
        self.image_view.hovered.connect(self._on_image_hovered)
        self.image_view.left.connect(self.image_view.clear_loupe)
        left.addWidget(self.image_view, stretch=1)

        guide = QHBoxLayout()
        guide.setSpacing(8)
        self.view_mark = QLabel("NEGATIVE")
        self.view_mark.setFont(theme.edge_font(11))
        self.view_mark.setStyleSheet(
            f"color: {theme.FILM_REBATE}; background: {theme.EDGE_PRINT}; border-radius: 2px; padding: 0 4px;"
        )
        guide.addWidget(self.view_mark, 0, Qt.AlignmentFlag.AlignTop)
        self.caption = QLabel(_CAPTION_NEGATIVE)
        self.caption.setWordWrap(True)
        guide.addWidget(self.caption, 1)
        left.addLayout(guide)
        # What's true of the roll as chips that stay; what's happening now as text that fades.
        self.notice_bar = NoticeBar()
        self.notice_bar.set_host(self.image_view)
        self.notice_bar.frameRequested.connect(self._go_to_frame)
        self.status_label = self.notice_bar.activity
        left.addWidget(self.notice_bar)
        body.addLayout(left, stretch=1)

        # right panel: wedge, points, notice, proof, drawers, primary
        panel = QVBoxLayout()
        panel.setSpacing(6)
        panel_widget = QWidget()
        panel_widget.setLayout(panel)

        coverage_title = QLabel("Coverage of the roll's densities")
        coverage_title.setProperty("role", "section")
        panel.addWidget(coverage_title)
        self.wedge = StepWedge()
        panel.addWidget(self.wedge)
        # Only while the fit can't be trusted over the whole roll: Save stays allowed.
        self.reliability_note = QLabel("")
        self.reliability_note.setWordWrap(True)
        self.reliability_note.setStyleSheet(f"color: {theme.TEXT_WARNING};")
        self.reliability_note.setVisible(False)
        self.reliability_note.setContentsMargins(0, 0, 0, 4)  # a gap above the Neutral points title
        panel.addWidget(self.reliability_note)

        points_header = QHBoxLayout()
        self.points_title = QLabel("Neutral points")
        self.points_title.setProperty("role", "section")
        points_header.addWidget(self.points_title)
        points_header.addStretch(1)
        cc_help = QLabel("what\u2019s CC?")
        cc_help.setStyleSheet(f"color: {theme.TEXT_DIM}; text-decoration: underline dotted;")
        cc_help.setToolTip(CC_EXPLANATION)
        cc_help.setCursor(Qt.CursorShape.WhatsThisCursor)
        points_header.addWidget(cc_help)
        panel.addLayout(points_header)
        self.point_list = PointList()
        self.point_list.rowClicked.connect(self._on_row_clicked)
        self.point_list.removeClicked.connect(self._remove_point)
        panel.addWidget(self.point_list, stretch=1)

        # Under the list rather than in its header, which they crowded until the title was clipped.
        clear_row = QHBoxLayout()
        clear_row.addStretch(1)
        self.clear_frame_button = QPushButton("Clear frame")
        self.clear_frame_button.setToolTip("Remove the points on the frame you're looking at")
        self.clear_frame_button.clicked.connect(self._on_clear_frame)
        self.clear_all_button = QPushButton("Clear all")
        self.clear_all_button.setToolTip("Remove every point and start the calibration again")
        self.clear_all_button.clicked.connect(self._on_clear_all)
        for button in (self.clear_frame_button, self.clear_all_button):
            button.setProperty("role", "remove")
            clear_row.addWidget(button)
        panel.addLayout(clear_row)

        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setMinimumHeight(30)
        panel.addWidget(self.notice)

        self.details_form = RollDetailsForm()
        self.details_form.changed.connect(self._on_details_changed)
        self.print_controls = PrintControls()
        self.print_controls.changed.connect(self._on_print_changed)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setProperty("role", "dim")
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_body = QWidget()
        details_layout = QVBoxLayout(details_body)
        details_layout.setContentsMargins(4, 2, 4, 2)
        details_layout.addWidget(self.details_label)
        self.print_drawer = Drawer("Print", self.print_controls, max_height=150)
        self.extra_drawer = Drawer("Extra information", self.details_form, max_height=140)
        self.details_drawer = Drawer("Details", details_body, max_height=140, expanding=True)
        drawers = [self.extra_drawer, self.print_drawer, self.details_drawer]
        if self.is_pick_session:
            drawers = drawers[1:]  # one-shot develop: nothing to save extra information into
        self.details_drawer.set_summary("numbers behind the fit")
        self._panel = panel
        self._accordion = Accordion(drawers)
        self._accordion.openChanged.connect(self._on_drawer_opened)
        panel.addWidget(self._accordion)

        # The points state ends in Save profile (quiet) and Develop roll… (the one red button); the
        # one-shot `invert --pick` window has just a red Develop.
        self.primary_button = QPushButton("Develop" if self.is_pick_session else "Develop roll\u2026")
        self.primary_button.setProperty("role", "primary")
        self.primary_button.setEnabled(False)
        if self.is_pick_session:
            self.primary_button.clicked.connect(self._on_primary_clicked)
            panel.addWidget(self.primary_button)
        else:
            self.primary_button.clicked.connect(self._open_develop)
            self.save_button = QPushButton("Save profile")
            self.save_button.setEnabled(False)
            self.save_button.clicked.connect(self._on_save_clicked)
            buttons = QHBoxLayout()
            buttons.addWidget(self.save_button)
            buttons.addWidget(self.primary_button)
            panel.addLayout(buttons)
        # The right panel is a stack of three states: points, develop, summary.
        self._stack = QStackedWidget()
        self._stack.setFixedWidth(_PANEL_WIDTH)
        self._stack.addWidget(panel_widget)
        if not self.is_pick_session:
            self.develop_panel = DevelopPanel()
            self.develop_panel.backClicked.connect(lambda: self._turn_to(0))
            self.develop_panel.developClicked.connect(self._start_develop)
            self.develop_panel.cancelClicked.connect(self._cancel_develop)
            self._stack.addWidget(self.develop_panel)
            self.summary_panel = SummaryPanel()
            self.summary_panel.backClicked.connect(lambda: self._turn_to(0))
            self.summary_panel.contactClicked.connect(self._developed_contact_sheet)
            self.summary_panel.shareClicked.connect(self._save_more_for_sharing)
            self.summary_panel.openClicked.connect(self._open_developed_folder)
            self.summary_panel.carryOnClicked.connect(self._carry_on)
            self._stack.addWidget(self.summary_panel)
        body.addWidget(self._stack)
        self._refresh_drawer_summaries()

    # --- loading --------------------------------------------------------------------------------

    def _on_load_roll(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Load roll (a folder of scans)")
        if folder:
            self.load_roll(Path(folder))

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        event.acceptProposedAction()
        if len(paths) == 1 and paths[0].is_dir():
            self.load_roll(paths[0])
        elif paths and all(p.is_file() and p.suffix.lower() in (".tif", ".tiff") for p in paths):
            if self._confirm_new_calibration():
                self.load_files(paths)
        else:
            self._status("Drop a folder of scans, or TIFF scans")

    def _confirm_new_calibration(self) -> bool:
        if not self.session.points or self.session.points == self._saved_points:
            return True  # nothing to lose: no points, or just saved or reopened
        answer = QMessageBox.question(
            self,
            "Start a new calibration?",
            f"Loading a different roll starts a new calibration. Clear the {len(self.session.points)} "
            "neutral point(s) picked so far?",
        )
        self._repaint_all()
        return answer == QMessageBox.StandardButton.Yes

    def load_roll(self, folder: Path, keep_points: bool = False) -> None:
        if self._busy():
            return
        paths, left_out = list_scans(folder)
        if not paths:
            self._error("No TIFF scans", f"No TIFF scans in {folder}")
            return
        if not keep_points:
            if not self._confirm_new_calibration():
                return
            self._new_calibration()
        self._start_roll(paths, folder)
        if left_out:
            count = sum((left_out.hidden, left_out.contact_sheets, left_out.folders, left_out.positives, *left_out.other.values()))
            self.notice_bar.set_notice("skipped", Notice(
                chip=f"{count} skipped", title=f"Left out of the roll, not being scans: {left_out.describe()}",
            ))

    def _new_calibration(self) -> None:
        """Forget the calibration in hand: points, profile name, extra information, print pin, and
        the top bar's profile chip or save stamp, which would otherwise name a profile this isn't."""
        self._sticky_status = None  # the last roll's message is about the last roll
        self.notice_bar.clear_all()
        self.session.reset()
        self._profile_name = ""
        self.details_form.set_values(self.session.details)
        self.print_controls.set_override(None)
        self._refresh_drawer_summaries()
        self.save_stamp.hide()
        self.profile_chip.setText("New calibration")
        self.profile_chip.show()

    def load_files(self, paths: list[Path], keep_points: bool = False) -> None:
        """Individual scans as a roll of their own (`halide open a.tif b.tif`, `invert --pick`)."""
        if self._busy():
            return
        if not keep_points:
            self._new_calibration()
        self._start_roll(sorted(paths) if len(paths) > 1 else list(paths), paths[0].parent if len(paths) > 1 else None, loose=True)

    def _start_roll(self, paths: list[Path], folder: Path | None, loose: bool = False) -> None:
        self._start_rolls([(folder, paths)], loose)

    def _start_rolls(self, rolls: list[tuple[Path | None, list[Path]]], loose: bool = False) -> None:
        """Load one or more rolls side by side in the filmstrip. Each roll's scan check is its own:
        rolls may legitimately differ from one another."""
        self._stop_loaders()
        self._generation += 1
        generation = self._generation
        self._loose = loose
        self._pending_flash = None
        self._turn_to(0, animate=False)  # the roll the develop side was set up for is gone
        if self._proof is not None:  # its frames belong to the roll being replaced
            old, self._proof = self._proof, None
            old.close()
        paths = [p for _, roll_paths in rolls for p in roll_paths]
        metadata = read_roll_scan_metadata(paths)
        self.session.set_rolls([(folder, roll_paths, [metadata[str(p)][0] for p in roll_paths]) for folder, roll_paths in rolls])
        if self._profiles is not None:
            self._profiles.set_roll_open(self.session.roll_folder is not None)
        # The counter carries the frame count. A long name is cut, with the full one as the tooltip,
        # so the top bar's buttons always fit the fixed-width window.
        names = [(folder.name if folder is not None else roll_paths[0].stem) for folder, roll_paths in rolls]
        if len(names) == 1:
            label, tip = names[0].upper(), names[0].upper()
        else:
            label, tip = f"{len(names)} ROLLS", "\n".join(n.upper() for n in names)
        self.roll_label.setText(QFontMetrics(self.roll_label.font()).elidedText(label, Qt.TextElideMode.ElideMiddle, _ROLL_NAME_MAX_W))
        self.roll_label.setToolTip(tip)
        if not self.is_pick_session:
            self.setWindowTitle(f"halide · {names[0] if len(names) == 1 else f'{len(names)} rolls'}")
        starts, at = [], 0
        for name, (_, roll_paths) in zip(names, rolls):
            starts.append((at, name))
            at += len(roll_paths)
        self.filmstrip.strip.set_frames([f.number for f in self.session.frames], starts if len(rolls) > 1 else None)
        flags: dict[str, str] = {}
        self._exposure_differs = False
        for _, roll_paths in rolls:
            report = assess_roll({str(p): metadata[str(p)] for p in roll_paths})
            flags.update(frame_flags(report))
            self._exposure_differs = self._exposure_differs or report.exposure_inconsistent
        # Keyed by the paths as given (relative when typed so on the command line), in frame order.
        self._scan_flags_by_index = {i: flags[str(p)] for i, p in enumerate(paths) if str(p) in flags}
        self.filmstrip.strip.set_flags(self._scan_flags_by_index)
        self.notice_bar.clear_notice("skipped")  # the caller sets it again for this roll
        self._current = None
        self.full_image = None
        self.display = None
        self.image_view.set_image(None, "Loading…")
        if not self.is_pick_session:
            self._preview_loader = PreviewLoader(paths, generation, self)
            self._preview_loader.previewReady.connect(self._on_preview_ready)
            self._start_thread(self._preview_loader)
        self._select_frame(0)
        self._refresh_points()
        self._show_scan_check(flags)

    def _show_scan_check(self, flags: dict[str, str]) -> None:
        """The roll's scan check as a chip; its list names each frame and why, and goes to it."""
        if not flags:
            self.notice_bar.clear_notice("scan")
            return
        several = len(self.session.rolls) > 1
        rows = []
        for i, frame in enumerate(self.session.frames):
            reason = self._scan_flags_by_index.get(i)
            if reason:
                label = f"FRAME {frame.number:02d}" + (f" {frame.roll.name.upper()}" if several else "")
                rows.append((label, reason, i))
        n = len(rows)
        self.notice_bar.set_notice("scan", Notice(
            chip=f"⚠ {n} scanned differently",
            title=f"{console.plural(n, 'frame')} {'was' if n == 1 else 'were'} scanned differently from most of the roll",
            rows=rows,
            footer="One profile shifts colour between them; scan a roll at one manual exposure and one white "
                   "balance (halide batch --match-scan-exposure evens out exposure)",
        ))

    def show_profiles(self) -> ProfilesWindow:
        if self._profiles is None:
            self._profiles = ProfilesWindow(self)
            self._profiles.editRequested.connect(self.open_profile)
            self._profiles.useRequested.connect(self.use_profile)
            self._profiles.renamed.connect(self._on_profile_renamed)
            self._profiles.deleted.connect(self._on_profile_deleted)
            self._profiles.fileOpened.connect(self.open_profile)
        self._profiles.set_roll_open(self.session.roll_folder is not None)
        self._profiles.refresh()
        self._profiles.show()
        self._profiles.raise_()
        self._profiles.activateWindow()
        return self._profiles

    def _is_current_profile(self, name: str) -> bool:
        in_use = self.session.profile_in_use
        current = in_use.stem if in_use is not None else self._profile_name
        return bool(current) and current.casefold() == name.casefold()

    def _on_profile_renamed(self, old: str, new: str) -> None:
        """The profile on screen was renamed in the Profiles window: follow it, so Save updates it."""
        if not self._is_current_profile(old):
            return
        self._profile_name = new
        if self.session.profile_in_use is not None:
            self.session.profile_in_use = self.session.profile_in_use.with_name(f"{new}.json")
            self._refresh_using_chip()
        else:
            self._show_profile_name(new)

    def _on_profile_deleted(self, name: str) -> None:
        """The profile on screen was deleted: its points stay, as an unsaved calibration, so a save
        can't quietly bring the deleted file back."""
        if not self._is_current_profile(name):
            return
        self.session.forget_profile()
        self._profile_name = ""
        self._saved_points = []
        self.save_stamp.hide()
        self.profile_chip.setText("New calibration")
        self.profile_chip.show()
        self._refresh_points()

    def use_profile(self, path: Path) -> None:
        """Put a saved profile on the open roll ("Use for this roll"): its fit drives the positive at
        once, and points clicked now refine it."""
        if self.session.roll_folder is None or self._busy():
            return
        unsaved = len(self.session.points) if self.session.points != self._saved_points else 0
        if unsaved:
            answer = QMessageBox.question(
                self, f"Use '{path.stem}'?",
                f"The {console.plural(unsaved, 'point')} on screen {'isn' if unsaved == 1 else 'aren'}'t saved, "
                "and the profile's points will replace them.",
            )
            self._repaint_all()
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.session.use_profile(path)
        except (ValueError, KeyError, OSError) as exc:
            self._error(f"Couldn't use {path.stem}", f"Couldn't use {path.stem}: {exc}")
            return
        self._profile_name = path.stem
        self._saved_points = list(self.session.points)
        self.details_form.set_values(self.session.details)
        self.print_controls.set_override(self.session.tone_override)
        self._refresh_drawer_summaries()
        self._nudge = None
        self._points_changed()
        self._status(f"Using '{path.stem}' on this roll: {console.plural(len(self.session.points), 'point')}")
        self._close_profiles()

    def open_profile(self, path: Path) -> None:
        if self._busy():
            return
        try:
            CalibrationSession().restore(path)  # read it first: an unreadable file changes nothing
        except (ValueError, KeyError, OSError) as exc:
            self._error(f"Couldn't open {path.name}", f"Couldn't open {path.name}: {exc}")
            return
        if not self._confirm_new_calibration():
            return
        self.notice_bar.clear_all()
        self.session.reset()
        try:
            self.session.restore(path)
        except (ValueError, KeyError, OSError) as exc:
            self._error(f"Couldn't open {path.name}", f"Couldn't open {path.name}: {exc}")
            return
        if path.parent.resolve() == default_profiles_dir().resolve():
            self._profile_name = path.stem
            self._show_profile_name(path.stem)
        else:  # a calibration file from elsewhere: Save keeps it as a profile, under a name asked for then
            self._profile_name = ""
            self._show_profile_name(path.name, "Profile file")
        self.details_form.set_values(self.session.details)
        self.print_controls.set_override(self.session.tone_override)
        self._refresh_drawer_summaries()
        self._sticky(f"Reopened '{path.stem}': {console.plural(len(self.session.points), 'point')}")
        loaded: list[tuple[Path, list[Path]]] = []
        for folder in point_rolls(self.session.points):
            scans = list_scans(folder)[0] if folder.is_dir() else []
            if not scans:
                folder = self._roll_not_found(path.stem, folder)
                scans = list_scans(folder)[0] if folder is not None else []
                if folder is not None and not scans:
                    self._error("No TIFF scans", f"No TIFF scans in {folder} - that roll's points still count in the fit")
            if scans and folder not in [f for f, _ in loaded]:  # Find roll… may name a roll already open
                loaded.append((folder, scans))
        if loaded:
            self._start_rolls(loaded)
        else:
            self._refresh_points()
        self._warn_missing_frames()
        self._saved_points = list(self.session.points)
        self._close_profiles()

    def _close_profiles(self) -> None:
        """A profile was put to use: the Profiles window has done its job and steps aside."""
        if self._profiles is not None:
            self._profiles.close()

    def _roll_not_found(self, profile_name: str, missing: Path) -> Path | None:
        """One of the reopened profile's rolls has gone: say so, and offer to point at where it went
        (points re-attach to the moved scans by file name - CalibrationSession.set_rolls). Returns the
        folder chosen to take its place, or None to continue without it."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Roll not found")
        box.setText(f"'{profile_name}' was made from\n{missing}\nwhich has been moved or deleted.")
        box.setInformativeText(
            "Its points on that roll still count in the fit. Point to the roll's new "
            "folder to see them on their frames again."
        )
        box.addButton("Continue without", QMessageBox.ButtonRole.RejectRole)
        find = box.addButton("Find roll…", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(find)
        box.exec()
        self._repaint_all()
        chosen = None
        if box.clickedButton() is find:
            start = next((str(q) for q in missing.parents if q.is_dir()), str(Path.home()))
            chosen = QFileDialog.getExistingDirectory(self, "Find the roll (its folder of scans)", start)
        if not chosen:
            self.notice_bar.set_notice(f"missing {missing}", Notice(
                chip=f"⚠ {missing.name} not found",
                title=f"Roll not found: {missing}. Its points still count in the fit; reopen the profile and "
                      "use Find roll… to show them on their frames",
            ))
            return None
        return Path(chosen)

    def _warn_missing_frames(self) -> None:
        """Frames points were picked on that aren't in the loaded roll or on disk any more."""
        missing = self.session.missing_frames()
        if not missing:
            return
        names = "\n".join(f"  {p.name}" for p in missing[:8]) + (f"\n  …and {len(missing) - 8} more" if len(missing) > 8 else "")
        count = sum(p.frame in set(missing) for p in self.session.points)
        QMessageBox.warning(
            self,
            "Frames not found",
            f"{len(missing)} frame(s) you picked on have been moved, renamed or deleted:\n{names}\n\n"
            f"Their {count} point(s) still count in the fit, but can't be shown on their frames.",
        )

    def _on_preview_ready(self, generation: int, index: int, preview, estimate, error) -> None:
        if generation != self._generation or index >= len(self.session.frames):
            return  # a late signal from the roll that was replaced
        frame = self.session.frames[index]
        frame.preview, frame.estimate, frame.error = preview, estimate, error
        if error:
            self.filmstrip.strip.set_pixmap(index, None, failed=True, reason=str(error))
        else:
            self.filmstrip.strip.set_pixmap(index, QPixmap.fromImage(_qimage(self._thumbnail(frame))))
        if index == self._current and self.display is None and preview is not None:
            self._show_placeholder_preview(frame)
        if all(f.loaded for f in self.session.frames):
            self._loading_progress(f"Roll loaded: {len(self.session.frames)} frames")
        self._refresh_wedge()

    def _loading_progress(self, message: str) -> None:
        """Loading's own status line: it replaces anything but a sticky status that's still showing."""
        if self.status_label.full_text() != self._sticky_status:
            self.notice_bar.show_activity(message, lasting=message.startswith("Loading"))

    def _sticky(self, message: str) -> None:
        self._status(message)
        self._sticky_status = message

    def _go_to_frame(self, index: int) -> None:
        """A frame the person chose (filmstrip click, arrow key): their own move ends any sticky status."""
        self._sticky_status = None
        self._select_frame(index)

    def _select_frame(self, index: int) -> None:
        if not (0 <= index < len(self.session.frames)):
            return
        frame = self.session.frames[index]
        retry = index == self._current and frame.error is not None and self.full_image is None
        if index == self._current and not retry:
            return
        frame.error = None
        previous, self._current = self._current, index
        if previous is not None and previous != index and self.print_controls.previewing:
            self.print_controls.drop_preview()  # a test strip belongs to the frame it was made on
            if self._positive_mode():
                self._refresh_thumbnail(previous)
        self._pending_flash = None
        self.filmstrip.show_frame(index)
        in_roll = sum(f.roll == frame.roll for f in self.session.frames)
        digits = 3 if in_roll > 99 else 2
        rolls = self.session.rolls
        which = f" · ROLL {rolls.index(frame.roll) + 1} OF {len(rolls)}" if len(rolls) > 1 else ""
        self.frame_counter.setText(
            f"FRAME <span style='color:{theme.TEXT}'>{frame.number:0{digits}d}</span> / {in_roll:0{digits}d}{which}"
        )
        number = 1 if self.is_pick_session else frame.number
        self.image_view.set_edge_print(f"{number}  →{number}A")  # the contact sheet's own arrow
        self.full_image = None
        self.display = None
        self._display_estimate = None
        self._auto_mask = None
        self._stretch = (0.0, 1.0)
        self.image_view.clear_loupe()
        if frame.preview is not None:
            self._show_placeholder_preview(frame)
        else:
            self.image_view.set_image(None, f"Loading {frame.path.name}…")
        self._loading_progress(f"Loading {frame.path.name}…")
        self._load_current_frame()

    def _load_current_frame(self) -> None:
        """Load the current frame at full resolution - one load in flight, and while the filmstrip is
        scrubbed only the frame the user finally lands on waits behind it (latest wins): each
        finished load starts the current frame's if that is a different one."""
        if self._closing or self._frame_loader is not None or self._current is None:
            return
        path = self.session.frames[self._current].path
        loader = FrameLoader(path, self._generation, self)
        loader.loaded.connect(self._on_frame_loaded)
        loader.finished.connect(self._frame_load_finished)
        self._frame_loader = loader
        self._loading = (path, self._generation)
        self._start_thread(loader)

    def _frame_load_finished(self) -> None:
        # Only one full-resolution load is ever in flight, so this is that one's end.
        self._frame_loader = None
        if self._current is None:
            return
        if (self.session.frames[self._current].path, self._generation) != self._loading:
            self._load_current_frame()

    def _show_placeholder_preview(self, frame: Frame) -> None:
        """While the full-resolution frame loads, show its small preview (not clickable yet)."""
        self.image_view.markers = []
        self.image_view.set_image(QPixmap.fromImage(_qimage(self._thumbnail(frame))))

    def _on_frame_loaded(self, generation: int, path: Path, image, error) -> None:
        if generation != self._generation:
            return  # a late signal from the roll that was replaced
        if self._current is None or self.session.frames[self._current].path != path:
            return  # the user has moved on to another frame
        if error:
            self.session.frames[self._current].error = str(error)
            self.image_view.set_image(None, f"Couldn't load {path.name}")
            self.filmstrip.strip.mark_failed(self._current, str(error))
            self._error(f"Couldn't load {path.name}", f"Couldn't load {path.name}: {error}")
            return
        self.full_image = image
        frame = self.session.frames[self._current]
        self.filmstrip.strip.clear_failure(self._current)  # a retry that worked
        if frame.preview is None:
            # No background preview for this frame (the one-shot --pick flow loads none): derive it
            # from the full frame, so the step wedge and filmstrip have it too.
            frame.preview = downsample_linear(image, PREVIEW_LONG_EDGE)
            self._refresh_wedge()
        self.display, self.stride = downsample_for_display(image, *_DISPLAY_BUDGET)
        self._stretch = compute_stretch_bounds(self.display)
        self._display_estimate = quiet_auto_estimate(self.display)
        self._auto_mask = _neutral_candidate_mask(self.display, DEFAULT_NEUTRAL_FRACTION)
        if self.status_label.full_text().startswith("Loading"):
            self._status("")  # don't overwrite anything more useful (e.g. "Reopened …")
        self._refresh_image()
        if self._pending_flash is not None:
            self.image_view.flash(self._pending_flash)
            self._pending_flash = None

    def _start_thread(self, thread: QThread) -> None:
        self._threads.add(thread)
        thread.finished.connect(lambda: self._threads.discard(thread))
        thread.start()

    def _stop_loaders(self, wait: bool = False) -> None:
        """Stop the roll's preview loading (a new roll, or the window closing). With `wait`, also
        wait for every loader thread - a full-resolution frame load can't be interrupted
        mid-decode, so that means waiting out the current one (about a second on a real scan)."""
        if self._preview_loader is not None:
            self._preview_loader.requestInterruption()
            self._preview_loader.previewReady.disconnect()
            self._preview_loader = None
        if wait:
            self._frame_loader = None
            for thread in list(self._threads):
                thread.requestInterruption()
                thread.wait()
            self._threads.clear()

    # --- rendering ------------------------------------------------------------------------------

    def _positive_mode(self) -> bool:
        return self.positive_button.isChecked()

    def _tone(self, current: bool = True) -> ToneCurveParams:
        """The print for the frame on screen, or (`current=False`) for any other frame, which never
        shows the sliders' preview."""
        if self._flat_preview:
            return ToneCurveParams(mode="linear")
        return (self._preview_tone if current else None) or self.session.print_tone()

    def _thumbnail(self, frame: Frame) -> np.ndarray:
        """A frame's filmstrip thumbnail in the current view (negative, or positive by the same rule
        as the main image)."""
        if frame.preview is None:
            return np.zeros((2, 3, 3), dtype=np.uint8)
        if self._positive_mode():
            profile, _is_fit, gain = self.session.positive_source(frame)
            if profile is not None:
                current = self._current is not None and frame is self.session.frames[self._current]
                return positive_display(frame.preview, profile, self._tone(current), gain)[0]
        return negative_display(frame.preview, compute_stretch_bounds(frame.preview))

    def _refresh_thumbnail(self, index: int) -> None:
        frame = self.session.frames[index]
        if frame.preview is not None:
            self.filmstrip.strip.set_pixmap(index, QPixmap.fromImage(_qimage(self._thumbnail(frame))))

    def _refresh_filmstrip_thumbnails(self) -> None:
        for i, frame in enumerate(self.session.frames):
            if frame.preview is not None:
                self.filmstrip.strip.set_pixmap(i, QPixmap.fromImage(_qimage(self._thumbnail(frame))))

    def _refresh_image(self) -> None:
        """Re-render the current frame in the current view, and the view's labels."""
        positive = self._positive_mode()
        self.caption.setText(_CAPTION_POSITIVE if positive else _CAPTION_NEGATIVE)
        self.view_mark.setText("POSITIVE" if positive else "NEGATIVE")
        frame = self.session.frames[self._current] if self._current is not None else None
        self._resolved = None
        self._positive_profile = None
        if positive and frame is not None:
            profile, is_fit, gain = self.session.positive_source(frame, self._display_estimate)
            in_use = self.session.profile_in_use
            if is_fit and in_use is not None:
                self.source_note.setText(f"Profile in use · {len(self.session.points)} points")  # the top bar names it
            elif is_fit:
                self.source_note.setText(f"Your calibration · {len(self.session.points)} points")
            elif profile is not None:
                self.source_note.setText("Rough auto estimate — not your final output")
            else:
                self.source_note.setText("No auto estimate for this frame — showing the negative")
            self._positive_profile, self._positive_gain = profile, gain
        else:
            self.source_note.setText("Raw negative" if frame is not None else "")
        self.print_drawer.setEnabled(positive)
        # Only a changed view fades; adding a point re-renders the same view in place.
        fade = positive != self._shown_positive
        self._shown_positive = positive

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
        self.image_view.set_image(QPixmap.fromImage(_qimage(shown)), fade=fade)
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
        if self.full_image is None or self._current is None or self._developing():
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
        number = len(self.session.points)
        self.image_view.probe(number)
        self.wedge.drop_in(number)

    def _on_row_clicked(self, index: int) -> None:
        self.session.selected = index
        self._nudge = None
        point = self.session.points[index]
        target = next((i for i, f in enumerate(self.session.frames) if f.path == point.frame), None)
        self._refresh_points()
        if target is not None and target != self._current:
            self._go_to_frame(target)
            self._pending_flash = index + 1  # flashed once that frame has loaded and shows markers
        else:
            self.image_view.flash(index + 1)

    def _remove_point(self, index: int) -> None:
        self.session.remove_point(index)
        self._nudge = None
        self._points_changed()

    def _remove_selected(self) -> None:
        if self.session.selected is not None and not self._developing():
            self._remove_point(self.session.selected)

    def _undo(self) -> None:
        if self._developing():
            return
        change = self.session.undo()
        if change is None:
            return
        self._nudge = None
        self._points_changed()
        if change > 0:
            self._status(f"Restored {console.plural(change, 'point')}")
        elif change < 0:
            self._status(f"Took back {console.plural(-change, 'point')}")

    def _deselect(self) -> None:
        self.session.selected = None
        self._refresh_points()

    def _points_changed(self) -> None:
        """The fit may have changed: everything that shows it follows."""
        self.save_stamp.hide()
        self.profile_chip.show()
        self._refresh_points()
        self._proof_is_stale()
        if self._positive_mode():
            self._refresh_image()
            self._thumbs_timer.start(_THUMBS_DELAY_MS)

    # --- proof sheet ----------------------------------------------------------------------------

    def _on_proof(self) -> None:
        profile = self.session.profile()
        if profile is None or not self.session.frames:
            return
        if self._proof is not None:
            old, self._proof = self._proof, None
            old.close()  # returns at once: its renderer stops in the background
        reference = self.session.reference
        tone = self.session.print_tone()
        bits = [f"{len(self.session.points)} neutral points"]
        bits.append("print exposure/grade pinned" if self.session.tone_override else "print fitted per frame")
        if reference is not None:
            bits.append(f"scan exposure matched to {reference.describe()}")
        frames = [(f.path, f.preview, self.session.gain(f.scan)) for f in self.session.frames]
        title = self.session.roll_folder.name if self.session.roll_folder else frames[0][0].stem
        self._proof = ProofWindow(
            self, title, frames, profile, tone, self.session.details.get("film_stock") or None, " · ".join(bits),
            save_dir=self.session.roll_folder, device=self.device,
        )
        # `destroyed` arrives late (deferred delete): by then a rebuilt window may already be the
        # current one, so only forget the window that actually went.
        proof = self._proof
        proof.destroyed.connect(lambda *_: self._on_proof_closed(proof))
        proof.show()

    def _on_proof_closed(self, window: ProofWindow) -> None:
        if self._proof is window:
            self._proof = None

    def _proof_is_stale(self) -> None:
        if self._proof is not None:
            self._proof.mark_stale(self._on_proof)

    def _refresh_points(self) -> None:
        if not self._developing() and not self._hold_side:
            self._turn_to(0)  # the develop side was set up for the calibration as it was
        views = self.session.views()
        self.point_list.set_rows(views, self.session.selected, self._profile_card())
        self._refresh_using_chip()
        self.points_title.setText(f"Neutral points ({len(views)})" if views else "Neutral points")
        self._refresh_wedge(views)
        counts = self.session.point_counts()
        self.filmstrip.strip.set_counts([counts.get(f.path, 0) for f in self.session.frames])
        self._refresh_markers()
        self._refresh_notice(views)
        self._refresh_details(views)
        fitted = self.session.profile() is not None
        one_roll = len(self.session.rolls) <= 1
        if self.is_pick_session:
            self.primary_button.setEnabled(fitted)
        else:
            self.save_button.setEnabled(fitted)
            folder = self._develop_folder()
            self.primary_button.setEnabled(fitted and folder is not None)
            self.primary_button.setToolTip(
                "Develop the whole roll with this calibration" if fitted and folder is not None
                else "Develop works on one roll folder: open a single roll" if fitted
                else "Click two neutral points of different densities first, or use a profile"
            )
        self.proof_button.setEnabled(fitted and bool(self.session.frames) and one_roll)
        self.proof_button.setToolTip(
            "A zoomable contact sheet of the whole roll printed with this calibration" if one_roll
            else "The contact sheet shows one roll: open a single roll to print one"
        )

    def _profile_card(self) -> list[str] | None:
        """A profile in use with nothing added yet shows as a card instead of its points' rows."""
        session = self.session
        if session.profile_in_use is None or session.changed_from_profile():
            return None
        stock = session.details.get("film_stock", "").strip()
        rolls = session.profile_rolls()
        where = f" on {', '.join(rolls)}" if rolls else ""
        return [
            session.profile_in_use.stem,
            *([stock] if stock else []),
            f"{console.plural(len(session.points), 'point')}{where}",
            "Click points on this roll to refine it",
        ]

    def _refresh_using_chip(self) -> None:
        if self.session.profile_in_use is None:
            return
        added = len(self.session.new_points())
        extra = f" + {console.plural(added, 'new point')}" if added else " (edited)" if self.session.changed_from_profile() else ""
        self._show_profile_name(self.session.profile_in_use.stem, "Using", extra, keep_stamp=True)

    def _refresh_wedge(self, views: list[PointView] | None = None) -> None:
        views = self.session.views() if views is None else views
        ticks = [
            WedgeTick(v.index + 1, v.green_density, agreement_colour(v), v.index == self.session.selected)
            for v in views
        ]
        wedge = self.session.wedge_range()
        reliability = self.session.reliability(wedge)
        note = reliability.warning() if reliability is not None else None
        shown_range = reliability.reliable_range if note and reliability.reliable_anywhere else None
        measuring = any(not f.loaded for f in self.session.frames)
        self.wedge.set_state(wedge, ticks, shown_range, measuring=measuring)
        self.reliability_note.setText(note or "")
        self.reliability_note.setVisible(note is not None)

    def _refresh_notice(self, views: list[PointView]) -> None:
        worst = self.session.worst()
        problem = self.session.fit_problem()
        if problem:
            text, colour = f"⚠ {problem}", theme.TEXT_WARNING
        elif self._nudge:
            text, colour = f"› {self._nudge}", theme.TEXT_DIM
        elif worst is not None:
            a = views[worst].agreement
            text = f"⚠ Point {worst + 1} is furthest from the others ({no_wrap(a.label())}). Was that object really neutral?"
            colour = agreement_colour(views[worst])
        elif len(views) < 2:
            text, colour = "Pick neutral objects of different tones, on any frames.", theme.TEXT_DIM
        elif len(views) == 2:
            text, colour = "Add a third point to see how well your points agree.", theme.TEXT_DIM
        else:
            text, colour = "", theme.TEXT_DIM
        self.notice.setText(text)
        self.notice.setStyleSheet(f"color: {colour};")
        self.notice.setVisible(bool(text))  # an empty notice keeps no height from the point list

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
        lines.extend(v.note for v in views if v.note)
        lines.append(CC_EXPLANATION)
        self.details_label.setText("\n\n".join(lines))

    # --- drawers --------------------------------------------------------------------------------

    def _on_view_changed(self) -> None:
        self._refresh_image()
        self._thumbs_timer.start(0)

    def _on_drawer_opened(self, drawer: Drawer | None) -> None:
        """The point list takes the panel's spare height; an expanding drawer (Details) takes it
        instead while it's open, and the list shrinks to its minimum."""
        expanding = drawer is not None and drawer.expanding
        self._panel.setStretchFactor(self.point_list, 0 if expanding else 1)
        self._panel.setStretchFactor(self._accordion, 1 if expanding else 0)

    def _on_clear_frame(self) -> None:
        if self._current is None:
            return
        frame = self.session.frames[self._current]
        count = sum(1 for p in self.session.points if p.frame == frame.path)
        if count and self._confirm(f"Remove the {count} point(s) on {frame.path.name}?"):
            self.session.clear_points(frame.path)
            self._nudge = None
            self._points_changed()

    def _on_clear_all(self) -> None:
        if self.session.profile_in_use is not None:
            count = len(self.session.new_points())
            question = f"Remove the {console.plural(count, 'new point')}? The profile's own points stay."
        else:
            count = len(self.session.points)
            question = f"Remove all {count} point(s) and start the calibration again?"
        if count and self._confirm(question):
            self.session.clear_points()
            self._nudge = None
            self._points_changed()

    def _confirm(self, question: str) -> bool:
        answer = QMessageBox.question(self, "Remove points", question)
        self._repaint_all()
        return answer == QMessageBox.StandardButton.Yes

    def _on_details_changed(self, values: dict) -> None:
        self.session.details = values
        self._refresh_drawer_summaries()

    def _refresh_drawer_summaries(self) -> None:
        stock = self.session.details.get("film_stock", "").strip()
        self.extra_drawer.set_summary(stock or "film stock, process, scanner")
        controls = self.print_controls
        self.print_drawer.set_summary(
            "previewing on this frame" if controls.previewing else "pinned for every frame" if controls.pinned else "fitted per frame"
        )

    def _on_print_changed(self, preview: ToneCurveParams | None, pin: ToneCurveParams | None, flat: bool) -> None:
        pin_changed = pin != self.session.tone_override
        everywhere = pin_changed or flat != self._flat_preview  # changes every thumbnail, not just this frame's
        self.session.tone_override = pin
        self._preview_tone = preview
        self._flat_preview = flat
        self._refresh_drawer_summaries()
        if pin_changed:
            self._proof_is_stale()
        if self._positive_mode():
            self._refresh_image()
            if everywhere:
                self._thumbs_timer.start(_THUMBS_DELAY_MS)
            elif self._current is not None:
                self._refresh_thumbnail(self._current)

    # --- magnifier ------------------------------------------------------------------------------

    def _on_image_hovered(self, disp_x: int, disp_y: int) -> None:
        if self.full_image is None:
            return
        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(disp_x, disp_y, self.stride, w, h)
        # Snap as a click would, so the density shown is the one the new point would read.
        _, _, rgb = snap_to_representative_pixel(self.full_image, full_x, full_y)
        density = self.session.density_at(self.session.frames[self._current], rgb)
        patch = extract_magnifier_patch(self.full_image, full_x, full_y, radius=_MAGNIFIER_RADIUS)
        if self._positive_profile is not None and self._resolved is not None:
            shown = print_patch(patch, self._positive_profile, self._resolved, self._positive_gain)
        else:
            shown = apply_stretch(patch, *self._stretch)
        magnified = magnify_patch(shown, zoom=_MAGNIFIER_ZOOM)
        rect = self.image_view._image_rect()
        scale = self.image_view.display_scale()
        cursor = QPointF(rect.x() + (disp_x + 0.5) * scale, rect.y() + (disp_y + 0.5) * scale)
        pixmap = QPixmap.fromImage(_qimage(magnified))
        ratio = self.image_view.devicePixelRatioF()
        if ratio != 1.0:  # a crisp patch at the screen's real resolution, not a stretched one
            pixmap = pixmap.scaled(round(pixmap.width() * ratio), round(pixmap.height() * ratio),
                                   Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
            pixmap.setDevicePixelRatio(ratio)
        self.image_view.set_loupe(pixmap, density, cursor)

    # --- save / develop -------------------------------------------------------------------------

    def _print_view_notes(self) -> None:
        for view in self.session.views():
            if view.note:
                print(console.warning(view.note))

    def _on_primary_clicked(self) -> None:
        """The one-shot pick window's Develop: hand the profile back and close."""
        profile = self.session.profile()
        if profile is None:
            return
        self._print_view_notes()
        self._pick_result_emitted = True
        reliability = self.session.reliability()
        note = reliability.warning() if reliability is not None else None
        self.pickCompleted.emit((profile, self.session.tone_override, note))
        self.close()

    def _develop_folder(self) -> Path | None:
        """The one roll folder Develop works on; None for several rolls or loose scans."""
        return None if self._loose or len(self.session.rolls) != 1 else self.session.roll_folder

    def _turn_to(self, index: int, animate: bool = True) -> None:
        """Turn the right panel over to another state (0 points, 1 develop, 2 summary): the new one
        swings in from the side it lies on."""
        if self.is_pick_session or index == self._stack.currentIndex():
            return
        forward = index > self._stack.currentIndex()
        self._stack.setCurrentIndex(index)
        page = self._stack.currentWidget()
        if not animate:
            return
        shift = _PANEL_WIDTH // 3 * (1 if forward else -1)

        def swing(value: float) -> None:
            page.move(round(shift * (1 - value)), 0)

        motion.animate(self._stack, "turn", 0.0, 1.0, 280, swing, lambda: page.setGeometry(self._stack.rect()))

    def _open_develop(self) -> None:
        folder = self._develop_folder()
        session = self.session
        if folder is None or session.profile() is None:
            return
        in_use = session.profile_in_use
        path = None
        if in_use is not None:
            changed = session.changed_from_profile() or not self._tone_as_saved(in_use)
            case, path, name = ("refined" if changed else "profile"), in_use, in_use.stem
        elif (self._profile_name and session.points == self._saved_points
              and (path := find_profile(self._profile_name)) and self._tone_as_saved(path)):
            case, name = "profile", self._profile_name
        else:
            case, name = "new", ""
        self.develop_panel.prepare(
            folder, case, profile_path=path, name=name, new_points=len(session.new_points()),
            suggested_name="" if in_use is not None else self._profile_name, exposure_differs=self._exposure_differs,
            device=self.device.kind if self.device is not None else None,
        )
        self._turn_to(1)

    # --- developing the roll --------------------------------------------------------------------
    # The real `halide batch` runs as a child (gui/develop_runner.py), so the prints are exactly the
    # terminal's; this window only follows its progress.

    def _developing(self) -> bool:
        return self._runner.is_running()

    def _tone_as_saved(self, path: Path) -> bool:
        """Whether the print pin in hand is the one saved in the profile at `path`: batch prints with
        the file's, so a pin made since must be saved (or used once) first."""
        try:
            return load_tone_override(path) == self.session.tone_override
        except (OSError, ValueError):
            return False

    def _busy(self) -> bool:
        """Developing: nothing may replace the roll or the calibration until it ends."""
        if self._developing():
            self._status("Developing — wait for it to finish, or cancel it first")
        return self._developing()

    def _start_develop(self, settings: DevelopSettings, choice: str) -> None:
        """Develop pressed: settle the calibration file the run will name (saving a profile first if
        that was the choice), settle what to do with prints already there, then start the run."""
        try:
            frame_count(settings)
        except ValueError as exc:
            self._error("Can't develop", str(exc))
            return
        self._hold_side = True
        try:
            settings = self._calibration_for_develop(settings, choice)
        except (ProfileError, OSError) as exc:
            self._error("Couldn't save the calibration", f"Nothing was developed: {exc}")
            return
        finally:
            self._hold_side = False
        if settings.existing == "ask":
            try:
                existing, total = existing_outputs(settings), frame_count(settings)
            except ValueError as exc:
                self._error("Can't develop", str(exc))
                return
            if existing:
                answer = self._ask_existing(len(existing), total, settings.into)
                if answer is None:
                    return
                settings = replace(settings, existing=answer)
        self._run_develop(settings)

    def _calibration_for_develop(self, settings: DevelopSettings, choice: str) -> DevelopSettings:
        session = self.session
        if choice == "profile":
            return settings
        if choice in ("save", "add"):
            if choice == "save":
                name = validate_profile_name(self.develop_panel.name_input.text())
                path = save_named_profile(session.profile_to_save(), name, **session.sidecars())
            else:
                name = session.profile_in_use.stem
                path = save_named_profile(session.profile_to_save(), name, overwrite=True, **session.sidecars())
            self._on_saved(name, f"Saved profile '{name}' to {path}")
            return replace(settings, profile=path)
        # Used once, or for this roll only: beside the prints, never in the profiles folder.
        settings.into.mkdir(parents=True, exist_ok=True)
        path = settings.into / "halide-calibration.json"
        save_profile(session.profile_to_save(), path, **session.sidecars())
        return replace(settings, profile=path)

    def _ask_existing(self, count: int, total: int, into: Path) -> str | None:
        """The terminal's overwrite question, once for the roll: "replace", "skip" or None (cancel)."""
        box = QMessageBox(
            QMessageBox.Icon.Question, "Prints already there",
            f"{count} of {console.plural(total, 'output')} already exist in {into} — overwrite them?",
            parent=self,
        )
        replace_button = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
        skip_button = box.addButton("Skip existing", QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(skip_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        self._repaint_all()
        return {replace_button: "replace", skip_button: "skip"}.get(box.clickedButton())

    def _run_develop(self, settings: DevelopSettings) -> None:
        try:
            total = frame_count(settings)
            if settings.existing == "skip":
                total -= len(existing_outputs(settings))
        except ValueError as exc:
            self._error("Can't develop", str(exc))
            return
        if total == 0:  # batch would only say so, without a run to follow
            self.notice_bar.show_activity(f"Nothing to do — every print in {settings.into} is already there")
            return
        self._dev = {
            "settings": settings, "total": total, "done": 0, "failed": 0, "names": {}, "rows": [],
            "end": "", "cancelled": False, "started": time.monotonic(), "sharing": None, "phase": "develop",
        }
        self.filmstrip.strip.clear_prints()
        self._lock_for_develop(True)
        self._show_develop_progress()
        self._runner.run(batch_command(settings))
        if settings.sharing != "none":
            self._runner.run_after(export_command(settings))

    def _lock_for_develop(self, locked: bool) -> None:
        self.develop_panel.set_running(f"Developing… 0 of {self._dev['total']}" if locked else None)
        for button in (getattr(self, "load_button", None), getattr(self, "profiles_button", None), self.proof_button):
            if button is not None:
                button.setEnabled(not locked)
        self.setAcceptDrops(not locked and self.show_load_controls)
        if not locked:
            self._hold_side = True
            try:
                self._refresh_points()  # the buttons as the calibration allows
            finally:
                self._hold_side = False

    def _show_develop_progress(self) -> None:
        dev = self._dev
        text = f"Developing… {dev['done']} of {dev['total']}"
        self.develop_panel.set_running(text)
        roll = self._develop_folder()
        self.notice_bar.show_activity(f"Developing {roll.name if roll else 'the roll'} — {dev['done']} of {dev['total']}", lasting=True)

    def _frame_index(self, name: str) -> int | None:
        return next((i for i, f in enumerate(self.session.frames) if f.path.name == name), None)

    def _print_thumbnails(self, frame: Frame) -> tuple[np.ndarray, np.ndarray] | None:
        """The frame's negative and its print with this run's calibration, as the strip shows them."""
        profile, _is_fit, gain = self.session.positive_source(frame)
        if frame.preview is None or profile is None:
            return None
        negative = negative_display(frame.preview, compute_stretch_bounds(frame.preview))
        positive = positive_display(frame.preview, profile, self._tone(False), gain)[0]
        return (negative, positive) if negative.shape == positive.shape else None

    def _on_frame_started(self, index: int, name: str) -> None:
        if self._dev["phase"] != "develop":
            return  # sharing copies: activity text only, the strip already shows the prints
        self._dev["names"][index] = name
        frame_index = self._frame_index(name)
        thumbs = self._print_thumbnails(self.session.frames[frame_index]) if frame_index is not None else None
        if thumbs is not None:
            self.filmstrip.strip.print_start(frame_index, *thumbs)

    def _on_frame_done(self, index: int, ok: bool, detail: str) -> None:
        dev = self._dev
        if dev["phase"] != "develop":
            dev["sharing_done"] = dev.get("sharing_done", 0) + 1
            self.notice_bar.show_activity(f"Saving for sharing — {dev['sharing_done']} of {dev['total']}", lasting=True)
            return
        dev["done"] += 1
        dev["failed"] += not ok
        frame_index = self._frame_index(dev["names"].get(index, ""))
        if frame_index is not None:
            self.filmstrip.strip.print_done(frame_index, ok, detail)
        self._show_develop_progress()

    def _on_summary_row(self, label: str, value: str) -> None:
        if self._dev["phase"] == "develop":
            self._dev["rows"].append((label, value))

    def _on_develop_finished(self, developed: int, total: int, cancelled: bool) -> None:
        dev = self._dev
        if dev["phase"] == "develop":
            dev["cancelled"] = cancelled
            if cancelled:
                dev["end"] = console.cancelled_message("invert", developed, total).split("\n")[0]
            else:
                dev["end"] = console.finish_message(
                    "invert", developed, total, dev["failed"], time.monotonic() - dev["started"])
            if self._runner.pending():  # the sharing copies come next, of every print in the folder
                dev.update(phase="sharing", sharing_done=0, total=self._prints_in(dev["settings"].into))
                return
            if dev["settings"].sharing != "none" and not cancelled:
                why = "some frames failed" if dev["failed"] else "developing was stopped first"
                dev["sharing"] = f"not saved — {why}; Save more for sharing… makes them"
            self._end_develop()
            return
        where = f"{_FORMAT_NAMES[dev['settings'].sharing]}, in {dev['settings'].sharing_dir}"
        dev["sharing"] = f"{console.plural(developed, 'file')}, {where}" if total else f"already saved, {where}"
        self._end_develop()

    def _on_develop_failed(self, message: str) -> None:
        if self._dev is not None and self._dev["phase"] == "sharing":
            self._dev["sharing"] = f"not saved — {message.splitlines()[-1]}"
            self._end_develop()
            return
        self._error("Developing didn't start", message)
        self.filmstrip.strip.drop_unfinished()
        self._lock_for_develop(False)
        self.notice_bar.show_activity("")

    def _end_develop(self) -> None:
        dev = self._dev
        dev["phase"] = "ended"
        self.filmstrip.strip.drop_unfinished()
        self._lock_for_develop(False)
        self.summary_panel.setEnabled(True)
        rows = list(dev["rows"])
        if dev["sharing"]:
            rows.append(("Sharing", dev["sharing"]))
        self.summary_panel.show_result(dev["end"], rows, dev["cancelled"])
        self.notice_bar.show_activity(dev["end"])
        self._turn_to(2)

    def _cancel_develop(self) -> None:
        self.develop_panel.cancel_button.setEnabled(False)
        self.notice_bar.show_activity("Cancelling — the frames in hand finish first", lasting=True)
        self._runner.cancel()

    def _carry_on(self) -> None:
        self.develop_panel.skip_existing()
        self._run_develop(replace(self._dev["settings"], existing="skip"))
        self._turn_to(1)

    def _save_more_for_sharing(self) -> None:
        box = QMessageBox(QMessageBox.Icon.Question, "Save for sharing",
                          "sRGB copies of the prints, full size. Which format?", parent=self)
        jpeg = box.addButton("JPEG", QMessageBox.ButtonRole.AcceptRole)
        png = box.addButton("PNG", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setEscapeButton(cancel)
        box.exec()
        self._repaint_all()
        fmt = {jpeg: "jpg", png: "png"}.get(box.clickedButton())
        if fmt is None:
            return
        settings = replace(self._dev["settings"], sharing=fmt, existing="skip")
        self._dev.update(settings=settings, sharing=None, sharing_done=0, total=self._prints_in(settings.into),
                         phase="sharing")
        self._lock_for_develop(True)
        self.summary_panel.setEnabled(False)
        self.notice_bar.show_activity("Saving for sharing…", lasting=True)
        self._runner.run(export_command(settings))

    @staticmethod
    def _prints_in(folder: Path) -> int:
        try:
            return len(list_scans(folder)[0])
        except OSError:
            return 0

    def _open_developed_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dev["settings"].into)))

    def _developed_contact_sheet(self) -> None:
        """`halide contact` on the developed folder, saved beside the prints, then shown."""
        into = self._dev["settings"].into
        sheet = into / f"{into.name}-contact-sheet.jpg"
        process = QProcess(self)
        self.summary_panel.contact_button.setEnabled(False)
        self.notice_bar.show_activity("Printing the contact sheet…", lasting=True)

        def done(code: int, _status) -> None:
            process.deleteLater()
            self.summary_panel.contact_button.setEnabled(True)
            if code == 0:
                self.notice_bar.show_activity(f"Contact sheet saved to {sheet}")
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(sheet)))
            else:
                tail = bytes(process.readAllStandardError()).decode("utf-8", "replace").strip()
                self._error("No contact sheet", tail or f"halide contact stopped (exit code {code})")

        process.finished.connect(done)
        process.start(sys.executable, ["-m", "halide.cli.main", "contact", str(into), str(sheet), "--overwrite"])

    def _on_save_clicked(self) -> None:
        if self.session.profile() is None:
            return
        self._print_view_notes()
        in_use = self.session.profile_in_use
        dialog = SaveProfileDialog(
            self, self.session.profile_to_save(), self.session.sidecars(),
            "" if in_use is not None else self._profile_name,
            update_name=in_use.stem if self.session.changed_from_profile() else None,
        )
        dialog.saved.connect(self._on_saved)
        dialog.exec()
        self._repaint_all()

    def _show_profile_name(self, name: str, kind: str = "Profile", extra: str = "", keep_stamp: bool = False) -> None:
        bold = QFont(self.profile_chip.font())
        bold.setBold(True)
        short = QFontMetrics(bold).elidedText(name, Qt.TextElideMode.ElideMiddle, PROFILE_NAME_MAX_W)
        self.profile_chip.setText(f"{kind} <b>{html.escape(short)}</b>{html.escape(extra)}")
        self.profile_chip.setToolTip(name)
        if not (keep_stamp and self.save_stamp.isVisible()):
            self.save_stamp.hide()
            self.profile_chip.show()

    def _on_saved(self, name: str, message: str) -> None:
        self._profile_name = name
        self.session.adopt_points(find_profile(name) or default_profiles_dir() / f"{name}.json")  # its real file, case and all
        self.last_saved = (name, self.session.roll_folder)
        if self._profiles is not None:
            self._profiles.refresh(select=name)
        self._show_profile_name(name)
        self._saved_points = list(self.session.points)
        self.profile_chip.hide()
        self.save_stamp.stamp(name)
        self._status(message)

    def _repaint_all(self) -> None:
        """Repaint every widget. A modal dialog closing over the window can leave stale pixels
        behind it (a half-drawn section title, seen as "JTRAL POINTS"), so the window redraws
        itself when it gets the focus back."""
        self.repaint()
        for widget in self.findChildren(QWidget):
            if widget.isVisible():
                widget.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Here rather than a window-wide shortcut, so a focused slider or text field keeps its arrows.
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(event.key())
        if step is not None and self._current is not None:
            self._go_to_frame(self._current + step)
            return
        super().keyPressEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._repaint_all()

    def _status(self, message: str) -> None:
        self.notice_bar.show_activity(message)

    def _error(self, chip: str, message: str) -> None:
        """Something went wrong: a red chip that stays until it's read, not a line the next message
        replaces."""
        self.notice_bar.set_notice("error", Notice(chip=f"✕ {chip}", title=message, error=True))

    def _confirm_close(self, count: int) -> bool:
        box = QMessageBox(
            QMessageBox.Icon.Question,
            "Close the window?",
            f"The {console.plural(count, 'neutral point')} you picked "
            f"{'isn' if count == 1 else 'aren'}'t saved in a profile and will be lost.",
            parent=self,
        )
        close = box.addButton("Close", QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton("Keep picking", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        self._repaint_all()
        return box.clickedButton() is close

    def _confirm_stop_developing(self) -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, "Stop developing?",
                          "Developing is still running. Stop it and close?", parent=self)
        stop = box.addButton("Stop and close", QMessageBox.ButtonRole.DestructiveRole)
        keep = box.addButton("Keep developing", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        self._repaint_all()
        return box.clickedButton() is stop

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._developing() and not self._confirm_stop_developing():
            event.ignore()
            return
        points = self.session.points
        if points and points != self._saved_points and not self._pick_result_emitted and not self._confirm_close(len(points)):
            event.ignore()
            return
        if self._developing():
            self._runner.cancel()  # Ctrl-C for the child: the frames in hand finish whole
            self._runner.wait()
        self.hide()  # closing feels instant even while a frame finishes loading
        self._closing = True
        self._stop_loaders(wait=True)
        if self._proof is not None:
            self._proof.close()  # stops its full-quality workers too
        if self._profiles is not None:
            self._profiles.close()
        wait_for_stopping_renderers()
        if self.is_pick_session and not self._pick_result_emitted:
            self._pick_result_emitted = True
            self.pickCompleted.emit(None)
        super().closeEvent(event)
