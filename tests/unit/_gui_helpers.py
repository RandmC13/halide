"""Shared by the picker tests: record every thread a MainWindow starts, and a FrameLoader stand-in
that stays mid-load until released (the state Task 12's close-time crash needed)."""

import threading
import time

from PySide6.QtCore import QThread, Signal

from halide.gui.main_window import MainWindow


def record_started_threads(monkeypatch) -> list:
    """Every QThread `MainWindow._start_thread` starts, in order."""
    started = []
    real = MainWindow._start_thread

    def recording(self, thread):
        started.append(thread)
        real(self, thread)

    monkeypatch.setattr(MainWindow, "_start_thread", recording)
    return started


class ParkedLoader(QThread):
    """A FrameLoader that is mid-load until `release` is set. Asked to stop (closeEvent's
    requestInterruption) it still takes 0.5 s to finish, so a window that did not wait for it would
    be left with a running thread when close() returns. Reset with `ParkedLoader.reset()`."""

    loaded = Signal(int, object, object, object)
    started_for: list = []
    release = threading.Event()

    @classmethod
    def reset(cls):
        cls.started_for, cls.release = [], threading.Event()

    def __init__(self, path, generation, parent=None):
        super().__init__(parent)
        ParkedLoader.started_for.append(path)

    def run(self):
        deadline = time.monotonic() + 30
        while not ParkedLoader.release.is_set() and not self.isInterruptionRequested() and time.monotonic() < deadline:
            time.sleep(0.01)
        if self.isInterruptionRequested():
            time.sleep(0.5)
