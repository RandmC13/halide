"""Background loading for the calibration picker, so the window stays responsive while a roll's
frames "develop" into the filmstrip one by one and while a full-resolution frame loads for picking.

Previews load in worker processes through the same forkserver pool setup and memory-aware worker
count batch uses (batch/orchestrator.py) - decoding and colour-managing ~37 full-resolution scans is
real work (~1 s and a few hundred MiB each), the same shape as `halide contact`'s thumbnail pool.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from halide.batch.orchestrator import BatchJob, _pool_context, default_export_worker_count
from halide.gui.roll import load_frame_preview


class PreviewLoader(QThread):
    """Loads every frame's small preview in parallel; emits previewReady(index, preview, estimate,
    error) as each finishes (any order). Stops early, without waiting on queued frames, if
    interrupted (a new roll loaded, or the window closed)."""

    previewReady = Signal(int, object, object, object)

    def __init__(self, paths: list[Path], parent=None) -> None:
        super().__init__(parent)
        self._paths = list(paths)

    def run(self) -> None:
        if not self._paths:
            return
        # As batch/orchestrator.py::_run_pool: one BLAS thread per worker process, set before the
        # pool (and its forkserver) exists.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        workers = default_export_worker_count([BatchJob(input_path=p, output_path=None) for p in self._paths])
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context())
        try:
            pending = {executor.submit(load_frame_preview, str(p)): i for i, p in enumerate(self._paths)}
            while pending and not self.isInterruptionRequested():
                done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        preview, estimate, error = future.result()
                    except Exception as exc:  # noqa: BLE001 - a crashed worker is one bad frame, not a crash
                        preview, estimate, error = None, None, f"couldn't load: {exc}"
                    self.previewReady.emit(index, preview, estimate, error)
        finally:
            # Interrupted (window closed, another roll loaded): stop the frames still decoding
            # rather than letting them finish - otherwise the interpreter waits on them at exit and
            # `halide calibrate` sat ~7 s in the terminal after its window had closed (measured).
            # concurrent.futures has no public "terminate", hence the private process table.
            if self.isInterruptionRequested():
                for process in list(getattr(executor, "_processes", {}).values()):
                    process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)


class FrameLoader(QThread):
    """Loads one frame at full resolution (picks sample full-resolution pixels); emits
    loaded(path, image, error)."""

    loaded = Signal(object, object, object)

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path

    def run(self) -> None:
        from halide.processing import load_working_space_image

        try:
            image = load_working_space_image(self._path)
        except Exception as exc:  # noqa: BLE001 - surface any I/O error in the UI, don't crash it
            self.loaded.emit(self._path, None, str(exc))
            return
        self.loaded.emit(self._path, image, None)
