"""Runs a halide command as a child process and turns its HALIDE_PROGRESS=machine lines into
signals. Every other line (the run sheet, the plain progress) is ignored."""

from __future__ import annotations

import os
import signal
from collections import deque

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

_STDERR_TAIL = 20


class DevelopRunner(QObject):
    frameStarted = Signal(int, str)  # index, scan name
    frameDone = Signal(int, bool, str)  # index, ok, detail or error
    summaryRow = Signal(str, str)  # label, value
    finished = Signal(int, int, bool)  # developed, total, cancelled
    failed = Signal(str)  # a child that stopped without an end line: its last stderr lines

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._queue: list[list[str]] = []
        self._buffer = b""
        self._stderr: deque[str] = deque(maxlen=_STDERR_TAIL)
        self._ended = False
        self._last_end: tuple[int, int, bool] | None = None
        self.step = -1  # which command of the chain is running: 0 the first, 1 the one run_after added

    def is_running(self) -> bool:
        return self._process is not None

    def pending(self) -> bool:
        """Whether a run_after command is still to come (true inside the first one's `finished`)."""
        return bool(self._queue)

    def run(self, command: list[str]) -> None:
        self._queue = []
        self.step = -1
        self._start(command)

    def run_after(self, command: list[str]) -> None:
        """Run `command` once the current one has ended well (exit 0, not cancelled). `finished` is
        emitted for each command; `step` says which one it was."""
        self._queue.append(command)

    def _start(self, command: list[str]) -> None:
        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("HALIDE_PROGRESS", "machine")
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        self._buffer = b""
        self._stderr.clear()
        self._ended = False
        self._last_end = None
        self.step += 1
        self._process = process
        process.start(command[0], command[1:])

    def _read_stdout(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            self._line(raw.decode("utf-8", "replace").rstrip("\r").split("\t"))

    def _read_stderr(self) -> None:
        text = bytes(self._process.readAllStandardError()).decode("utf-8", "replace")
        self._stderr.extend(line for line in text.splitlines() if line.strip())

    def _line(self, f: list[str]) -> None:
        try:
            if f[0] == "start" and len(f) == 3:
                self.frameStarted.emit(int(f[1]), f[2])
            elif f[0] == "done" and len(f) == 4 and f[2] in ("ok", "failed"):
                self.frameDone.emit(int(f[1]), f[2] == "ok", f[3])
            elif f[0] == "summary" and len(f) == 3:
                self.summaryRow.emit(f[1], f[2])
            elif f[0] == "end" and len(f) == 4:
                self._ended = True
                self._last_end = (int(f[1]), int(f[2]), f[3] == "1")
        except ValueError:
            pass  # not a protocol line after all

    def _on_finished(self, code: int, status) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._buffer:
            self._line(self._buffer.decode("utf-8", "replace").split("\t"))
        self._process.deleteLater()
        self._process = None
        if not self._ended:
            self._queue = []
            tail = "\n".join(self._stderr) or f"halide stopped (exit code {code})"
            self.failed.emit(tail)
            return
        developed, total, cancelled = self._last_end
        if self._queue and not cancelled and code == 0:
            self.finished.emit(developed, total, cancelled)
            self._start(self._queue.pop(0))
            return
        self._queue = []
        self.finished.emit(developed, total, cancelled)

    def cancel(self) -> None:
        """Ctrl-C for the child: it finishes the frames in hand and ends as the CLI does."""
        self._queue = []
        if self._process is not None and self._process.processId():
            os.kill(self._process.processId(), signal.SIGINT)

    def wait(self, msecs: int = 30000) -> bool:
        """Block until the child has ended (closing the window during a run)."""
        if self._process is None:
            return True
        return self._process.waitForFinished(msecs)
