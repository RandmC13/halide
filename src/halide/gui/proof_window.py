"""The contact sheet window ("Build contact sheet…" in the picker): the whole roll printed with the
session's current calibration, zoomable and pannable - the check that the neutral points picked on
a few frames hold up across every frame of the roll (the way the A/B contact sheets revealed
Roll16-Profile1's warmth). Once every frame has developed it can be saved as a file.

Where the full-quality frames live meanwhile: each worker writes its frame's thumbnail PNG into a
temporary folder (`halide-proof-…` in the system temp directory); each is read into memory as it
finishes and the folder is deleted when the run ends or the window closes. The assembled sheet only
ever exists in memory - on disk only if the user saves it.

It opens at once with a *draft*: every frame's small filmstrip preview printed with the current
fit. Meanwhile the real thing develops in the background - each frame at full resolution through
the very worker `halide batch --contact-sheet` uses (batch/orchestrator.py::_worker, so the same
per-frame print fit and the same result as a real run) - and replaces its draft as it finishes,
with a progress bar. Closing the window stops the workers rather than waiting on them, and the
temporary thumbnails are always deleted.

The sheet is io/contact_sheet.py's own (the same renderer as `halide contact`), so a proof here
looks exactly like the sheet the CLI would write.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from halide.batch.orchestrator import BatchJob, _pool_context, _worker, default_worker_count
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui.render import positive_display
from halide.io.contact_sheet import SheetLayout, Tile, caption_from_provenance, load_thumbnail, render_sheet, write_sheet
from halide.processing import Stage

PROOF_FRAME_WIDTH = 800  # px per frame: plenty to zoom into, and a ~37-frame sheet stays ~5000 px wide
_RERENDER_INTERVAL_MS = 1200  # batch frames arriving together into one redraw
_PLACEHOLDER = np.full((2, 3, 3), 28, dtype=np.uint8)  # a frame with nothing to show yet


class ProofRenderer(QThread):
    """Develops every frame at full resolution with batch's own worker; emits
    frameDone(index, thumbnail, record, error) as each finishes."""

    frameDone = Signal(int, object, object, object)

    def __init__(self, paths: list[Path], gains: list[float], profile: DensityProfile, tone: ToneCurveParams, parent=None) -> None:
        super().__init__(parent)
        self._paths, self._gains = list(paths), list(gains)
        self._profile, self._tone = profile, tone

    def run(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="halide-proof-"))
        jobs = [
            BatchJob(input_path=p, output_path=None, scan_gain=g, thumbnail_path=folder / f"{i:04d}.png")
            for i, (p, g) in enumerate(zip(self._paths, self._gains))
        ]
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        executor = ProcessPoolExecutor(max_workers=default_worker_count(jobs), mp_context=_pool_context())
        try:
            pending = {
                executor.submit(_worker, job, Stage.FULL, self._profile, self._tone, PROOF_FRAME_WIDTH): i
                for i, job in enumerate(jobs)
            }
            while pending and not self.isInterruptionRequested():
                done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        result = future.result()
                        error = result.error
                    except Exception as exc:  # noqa: BLE001 - a crashed worker is one bad frame
                        error = str(exc)
                    if error is None and jobs[index].thumbnail_path.exists():
                        image, record = load_thumbnail(jobs[index].thumbnail_path)
                        self.frameDone.emit(index, image, record, None)
                    else:
                        self.frameDone.emit(index, None, None, error or "no thumbnail")
        finally:
            if self.isInterruptionRequested():  # closed: stop, don't wait (see gui/loaders.py)
                for process in list(getattr(executor, "_processes", {}).values()):
                    process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            shutil.rmtree(folder, ignore_errors=True)


class SheetView(QGraphicsView):
    """Wheel to zoom about the cursor, drag to pan, double-click to fit; hovering a frame names it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setMouseTracking(True)
        self.layout_: SheetLayout | None = None
        self.labels: list[str] = []
        self._user_zoomed = False

    def set_sheet(self, pixmap: QPixmap, layout: SheetLayout, labels: list[str]) -> None:
        self._item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.layout_, self.labels = layout, labels
        if not self._user_zoomed:
            self.fit()

    def fit(self) -> None:
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Keep the whole sheet in view as the window takes its real size (it's first fitted before
        # the window is shown, at a placeholder size) - until the user zooms in themselves.
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._user_zoomed = True
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current = self.transform().m11()
        fit_scale = min(self.viewport().width() / max(1, self.sceneRect().width()), self.viewport().height() / max(1, self.sceneRect().height()))
        if (factor < 1 and current * factor < fit_scale * 0.9) or (factor > 1 and current * factor > 4):
            return  # no smaller than the whole sheet, no bigger than 4x the sheet's own pixels
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        if self.layout_ is None:
            return
        pos = self.mapToScene(event.position().toPoint())
        index = self.layout_.frame_at(pos.x(), pos.y())
        if index is not None and index < len(self.labels):
            QToolTip.showText(event.globalPosition().toPoint(), self.labels[index], self)
        else:
            QToolTip.hideText()


class ProofWindow(QDialog):
    """`frames`: (path, preview, scan gain) for every frame of the roll, in order."""

    def __init__(
        self,
        parent: QWidget,
        title: str,
        frames: list[tuple[Path, np.ndarray | None, float]],
        profile: DensityProfile,
        tone: ToneCurveParams,
        film_stock: str | None,
        settings: str,
        save_dir: Path | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("halide · contact sheet")
        self._title, self._film_stock, self._settings = title, film_stock, settings
        self._save_dir = save_dir
        self._sheet = None
        self._names = [path.stem for path, _, _ in frames]
        self._captions = [""] * len(frames)
        self._tiles: list[Tile] = []
        self._done = 0
        self._failed = 0
        self._reproof_connected = False

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.status = QLabel("")
        self.status.setProperty("role", "dim")
        top.addWidget(self.status, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(frames))
        self.progress.setFixedWidth(260)
        self.progress.setTextVisible(False)
        top.addWidget(self.progress)
        self.stale_note = QLabel("Points or print settings have changed since this sheet was built.")
        self.stale_note.setProperty("role", "warning")
        self.stale_note.setVisible(False)
        top.addWidget(self.stale_note)
        self.reproof_button = QPushButton("Rebuild contact sheet")
        self.reproof_button.setVisible(False)
        top.addWidget(self.reproof_button)
        self.save_button = QPushButton("Save contact sheet…")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Available once every frame has developed at full quality")
        self.save_button.clicked.connect(self._on_save)
        top.addWidget(self.save_button)
        layout.addLayout(top)
        self.view = SheetView()
        layout.addWidget(self.view, stretch=1)
        hint = QLabel("Scroll to zoom · drag to pan · double-click to fit · hover a frame for its file")
        hint.setProperty("role", "dim")
        layout.addWidget(hint)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(int(area.width() * 0.7), int(area.height() * 0.85))

        # Draft first: the filmstrip previews printed with the current fit.
        for i, (path, preview, gain) in enumerate(frames):
            if preview is None:
                self._tiles.append(Tile(path.stem, _PLACEHOLDER, "developing…", number=i + 1))
                continue
            rgb, resolved = positive_display(preview, profile, tone, gain)
            record = {"output": "print", "exposure": resolved.exposure, "contrast": resolved.contrast, "scan_gain": gain} \
                if resolved.mode == "paper" else {"output": "flat"}
            self._captions[i] = caption_from_provenance(record)
            self._tiles.append(Tile(path.stem, rgb, f"{self._captions[i]}   (draft)", number=i + 1))
        self._render()
        self._update_status()

        self._rerender = QTimer(self)
        self._rerender.setSingleShot(True)
        self._rerender.timeout.connect(self._render)
        self._renderer = ProofRenderer([p for p, _, _ in frames], [g for _, _, g in frames], profile, tone, self)
        self._renderer.frameDone.connect(self._on_frame_done)
        self._renderer.start()

    # --- rendering ------------------------------------------------------------------------------

    def _render(self) -> None:
        subtitle_bits = [f"{len(self._tiles)} frame(s)", self._settings]
        if self._done < len(self._tiles):
            subtitle_bits.append(f"draft - developing full quality {self._done}/{len(self._tiles)}")
        sheet = render_sheet(self._tiles, self._title, "  ·  ".join(b for b in subtitle_bits if b),
                             frame_width=PROOF_FRAME_WIDTH, film_stock=self._film_stock)
        self._sheet = sheet
        rgb = np.ascontiguousarray(np.asarray(sheet))
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        labels = [f"{name}   {caption}".strip() for name, caption in zip(self._names, self._captions)]
        self.view.set_sheet(QPixmap.fromImage(image), SheetLayout(len(self._tiles), PROOF_FRAME_WIDTH), labels)

    def _on_frame_done(self, index: int, image, record, error) -> None:
        self._done += 1
        tile = self._tiles[index]
        if error:
            self._failed += 1
            self._tiles[index] = Tile(tile.name, None, f"failed: {error}", number=tile.number)
        else:
            self._captions[index] = caption_from_provenance(record)
            self._tiles[index] = Tile(tile.name, image, self._captions[index], number=tile.number)
        self.progress.setValue(self._done)
        self._update_status()
        if self._done == len(self._tiles):
            self._rerender.stop()
            self._render()
        elif not self._rerender.isActive():
            self._rerender.start(_RERENDER_INTERVAL_MS)

    def _update_status(self) -> None:
        total = len(self._tiles)
        if self._done < total:
            self.status.setText(f"Draft shown · developing full quality: {self._done} of {total} frames")
        else:
            failed = f" · {self._failed} failed" if self._failed else ""
            self.status.setText(f"Full quality · {total} frames{failed}")
            self.progress.setVisible(False)
            self.save_button.setEnabled(True)
            self.save_button.setToolTip("Save this sheet as a JPEG or PNG (sRGB, like `halide contact` writes)")

    def _on_save(self) -> None:
        suggested = str((self._save_dir or Path.home()) / f"{self._title}-contact-sheet.jpg")
        path, _ = QFileDialog.getSaveFileName(self, "Save contact sheet", suggested, "JPEG (*.jpg *.jpeg);;PNG (*.png)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            path += ".jpg"
        try:
            write_sheet(path, self._sheet)  # sRGB-tagged and marked, so `halide contact` skips it
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Couldn't save", str(exc))
            return
        self.status.setText(f"Saved {Path(path).name} ({self._sheet.width}×{self._sheet.height} px)")

    def mark_stale(self, reproof) -> None:
        """The session changed after this proof was made: say so, and offer a new one."""
        self.stale_note.setVisible(True)
        self.reproof_button.setVisible(True)
        if not self._reproof_connected:
            self.reproof_button.clicked.connect(reproof)
            self._reproof_connected = True

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._renderer.requestInterruption()
        self._renderer.wait(3000)
        super().closeEvent(event)
