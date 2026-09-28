"""One exiftool process per halide process, reused for every output file.

A fresh exiftool costs ~0.66 s per output, and most of that is exiftool's own start-up and
tag-table loading, not the file rewrite (on a 1-pixel TIFF the same copy still takes ~0.49 s).
Kept running in its `-stay_open` mode it is ~0.29 s per file after the first, and the files it
writes are byte-for-byte identical (docs/investigations/gpu-batch-throughput.md, "Corrections").

exiftool stays optional: with none installed, `execute` returns False exactly as the old
one-shot call did. Anything that goes wrong with the kept-open process itself (it died, it hung,
it can't be started) is never a failure of the frame: the command is retried on a restarted
process once, then run one-shot exactly as before this module existed. What exiftool itself
reports as a failure of the file (`Error: ...`, nothing updated) is raised, as the one-shot's
non-zero exit was.

Stdlib only: this is imported by halide.io.tiff, on the CLI's start-up path.
"""

from __future__ import annotations

import atexit
import itertools
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

EXECUTABLE = "exiftool"

# How long one command may take before its exiftool is considered hung, killed, and the command
# retried. Real copies take well under a second; this only has to catch a process that will never
# answer, without failing a slow disk. Read when a session is created (tests shorten it).
DEFAULT_TIMEOUT = 120.0

# Appended by exiftool to every command of the session. `-charset filename=utf8`: file names go
# over the pipe as UTF-8 (on Windows exiftool would otherwise read them in the system code page);
# on Linux it doesn't change a byte of the output (checked against the real exiftool, task A1).
# `-echo3`: after each command, exiftool prints this line with ${status} replaced by the exit
# status the same command would have had as its own process (exiftool >= 12.10), so "failed" means
# exactly what the one-shot's non-zero exit meant.
_COMMON_ARGS = ["-charset", "filename=utf8", "-echo3", "{status=${status}}"]
_STATUS_LINE = re.compile(r"^\{status=(\d+)\}$")
# Fallback for an exiftool too old to fill in ${status}: its own summary of the command.
_WRITTEN_LINE = re.compile(r"^\s*([1-9]\d*) image files (updated|unchanged)$")


class ExifToolError(RuntimeError):
    """exiftool reported that the command failed (what a non-zero exit meant for the one-shot)."""


class ExifToolSessionError(ExifToolError):
    """The kept-open process itself failed - it exited, hung past the timeout, or couldn't be
    written to - so the command's outcome is unknown. Never the file's fault: retried elsewhere."""


class ExifToolSession:
    """One exiftool process reused for many commands (its -stay_open mode). Most of a fresh
    exiftool's ~0.66 s per file is its own start-up and tag-table loading, not the file
    rewrite: kept open it is ~0.29 s per file, with byte-identical output
    (docs/investigations/gpu-batch-throughput.md)."""

    def __init__(self, executable: str = EXECUTABLE, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._proc = subprocess.Popen(
            [executable, "-stay_open", "True", "-@", "-", "-common_args", *_COMMON_ARGS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # errors and results in one stream, in the order written
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_die_with_this_process(),
        )
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        # Replies are read on a helper thread, so a hung exiftool is caught by a timeout on the
        # queue - portable, unlike select() on a pipe, which Windows doesn't have.
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read, args=(self._proc.stdout,), name="exiftool-reader", daemon=True
        )
        self._reader.start()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def _read(self, stdout) -> None:
        try:
            for line in stdout:
                self._lines.put(line)
        except (OSError, ValueError):  # the pipe was closed under us: the session is ending
            pass
        self._lines.put(None)

    def run(self, args: list[str]) -> str:
        """Run one command (exiftool's arguments, without the program name); return its output.
        Raises ExifToolError if exiftool reports the command failed, or ExifToolSessionError if
        the process died or didn't answer within the timeout (it is then killed)."""
        with self._lock:
            if self._proc is None:
                raise ExifToolSessionError("exiftool session is closed")
            number = next(self._counter)
            try:
                self._proc.stdin.write("".join(f"{arg}\n" for arg in args) + f"-execute{number}\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                self._kill()
                raise ExifToolSessionError(f"exiftool stopped accepting commands ({exc})") from exc
            ready = f"{{ready{number}}}"
            deadline = time.monotonic() + self._timeout
            lines: list[str] = []
            while True:
                remaining = deadline - time.monotonic()
                try:
                    line = self._lines.get(timeout=max(remaining, 0.0))
                except queue.Empty:
                    self._kill()
                    raise ExifToolSessionError(f"exiftool didn't answer within {self._timeout:g} s") from None
                if line is None:
                    self._kill()
                    raise ExifToolSessionError("exiftool exited in the middle of a command")
                line = line.rstrip("\r\n")
                if line == ready:
                    break
                lines.append(line)
        return _check_reply(lines)

    def close(self) -> None:
        """Ask exiftool to exit (killing it if it won't). Safe to call more than once."""
        with self._lock:
            if self._proc is None:
                return
            if self._proc.poll() is None:
                try:
                    self._proc.stdin.write("-stay_open\nFalse\n")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
                    self._proc.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            self._kill()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        # The reader sees end of output once exiftool is gone; let it finish before closing the
        # pipe it is reading (closing under a blocked read waits on the same lock anyway).
        self._reader.join(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


def _check_reply(lines: list[str]) -> str:
    status = _STATUS_LINE.match(lines[-1]) if lines else None
    if status is not None:
        lines = lines[:-1]
        failed = int(status.group(1)) != 0
    else:
        failed = any(line.startswith("Error") for line in lines) or not any(
            _WRITTEN_LINE.match(line) for line in lines
        )
    output = "\n".join(lines)
    if failed:
        raise ExifToolError(output or "exiftool failed without a message")
    return output


def _die_with_this_process() -> dict:
    """Popen arguments that make exiftool exit when the process that started it dies.

    exiftool's -stay_open never exits on end of input - it polls the pipe every 10 ms forever - so
    a process killed before its atexit hook runs (the GUI terminates its worker pool; the OOM
    killer) would leave an exiftool running with no one to talk to. On Linux the kernel's
    parent-death signal ends it. Strictly the signal follows the *thread* that started it: if that
    thread ends first, exiftool is stopped early, found dead on the next command and restarted.
    Other platforms don't get a session at all (_KEPT_OPEN_SUPPORTED)."""
    if not sys.platform.startswith("linux"):
        return {}
    try:
        import ctypes
        import signal

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return {}
    parent = os.getpid()

    def preexec() -> None:
        prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:  # the parent died before the signal was armed
            os._exit(1)

    # `preexec_fn` is documented as unsafe in a process with threads: only the forking thread
    # survives the fork, so a lock another thread held at that instant stays held in the child, and
    # the child deadlocks if it needs it before exec. That process *can* have threads here: a
    # session starts lazily, after the worker's first frame has developed; in per-worker GPU mode
    # CUDA's own threads already exist; a restart happens mid-batch, when an earlier session's
    # reader thread may still be alive. And the ctypes call below is not allocation-free (argument
    # conversion runs Python code). The risk is accepted because the child does nothing but that one
    # prctl call and a getppid before exec, which is what CPython's documentation asks of a
    # preexec_fn ("keep it trivial"): the forking thread holds the GIL through the fork, and CPython
    # and glibc reset their own allocator locks in the child, so nothing it runs waits on a lock a
    # vanished thread could hold. There is no other way to set PR_SET_PDEATHSIG from subprocess.
    return {"preexec_fn": preexec}


# This process's session. A forked child inherits these globals, but not the right to use the
# session in them: writing to it would interleave with the parent's commands on the same pipe,
# and closing it would close the parent's. So everything is keyed by the PID that made it.
_lock = threading.Lock()

# The kept-open session is used only where the OS guarantees exiftool dies with the process that
# started it: Linux, via the parent-death signal (_die_with_this_process). exiftool's -stay_open
# never exits on its own when its input closes, so elsewhere a worker that is terminated (the GUI
# closing its pool) or killed (out of memory) would leave its exiftool running forever. There,
# every copy stays the one-shot call, exactly as before this module.
_KEPT_OPEN_SUPPORTED = sys.platform.startswith("linux")

_session: ExifToolSession | None = None
_session_pid: int | None = None
_oneshot_pid: int | None = None  # a restarted session failed in this process: one-shot from now on
_atexit_pid: int | None = None


def session() -> ExifToolSession | None:
    """This process's shared session, started on first use. None if exiftool isn't installed,
    can't be started, has already failed after a restart in this process, or this platform can't
    guarantee it would die with us (_KEPT_OPEN_SUPPORTED)."""
    global _session, _session_pid, _atexit_pid
    if not _KEPT_OPEN_SUPPORTED:
        return None
    with _lock:
        pid = os.getpid()
        if _session_pid != pid:
            _session, _session_pid = None, pid  # inherited over a fork: not ours to use or close
        if _oneshot_pid == pid:
            return None
        if _session is None:
            _session = _start()
            if _session is not None and _atexit_pid != pid:
                # Per PID, not once: multiprocessing's forkserver clears atexit in each worker.
                atexit.register(_close_at_exit)
                _atexit_pid = pid
        return _session


def _start() -> ExifToolSession | None:
    global _oneshot_pid
    if shutil.which(EXECUTABLE) is None:
        return None
    try:
        return ExifToolSession(EXECUTABLE, DEFAULT_TIMEOUT)
    except OSError:
        _oneshot_pid = os.getpid()
        return None


def _discard(broken: ExifToolSession, give_up: bool) -> None:
    global _session, _oneshot_pid
    broken.close()
    with _lock:
        if _session is broken:
            _session = None
        if give_up:
            _oneshot_pid = os.getpid()


def _close_at_exit() -> None:
    if _session is not None and _session_pid == os.getpid():
        _session.close()


def _reset() -> None:
    """Close this process's session and forget every failure (tests)."""
    global _session, _session_pid, _oneshot_pid
    if _session is not None and _session_pid == os.getpid():
        _session.close()
    _session = _session_pid = _oneshot_pid = None


def _fits_argument_file(args: list[str]) -> bool:
    """Whether every argument survives being one line of exiftool's argument file unchanged. It
    can't hold a line break; it strips leading white space and trailing CR/LF; a line starting
    with '#' is a comment. (Trailing spaces are refused too, to be safe.) Those go one-shot."""
    for arg in args:
        if not arg or "\n" in arg or "\r" in arg or arg[0].isspace() or arg[-1].isspace() or arg[0] == "#":
            return False
        try:
            arg.encode("utf-8")
        except UnicodeEncodeError:  # e.g. an undecodable file name held as surrogates
            return False
    return True


def _one_shot(args: list[str]) -> bool:
    """The original call, unchanged: a fresh exiftool for this one command."""
    try:
        subprocess.run([EXECUTABLE, *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def _remove_leftover(written: str | Path | None) -> None:
    # exiftool writes `<file>_exiftool_tmp` and renames it over the file. One killed mid-write
    # leaves it behind, and the next exiftool then refuses the file ("Temporary file already
    # exists"). Only this process's exiftool writes this output, and it has been killed.
    if written is not None:
        Path(f"{written}_exiftool_tmp").unlink(missing_ok=True)


def execute(args: list[str], written: str | Path | None = None) -> bool:
    """Run one exiftool command (arguments without the program name) on this process's session.
    Returns False if exiftool isn't installed. `written` is the file the command rewrites, so a
    temporary file left by a killed exiftool can be cleared before the command is retried."""
    if not _fits_argument_file(args):
        return _one_shot(args)
    current = session()
    if current is None:
        return _one_shot(args)
    try:
        current.run(args)
        return True
    except ExifToolSessionError:
        _remove_leftover(written)
        _discard(current, give_up=False)
    restarted = session()
    if restarted is None:
        return _one_shot(args)
    try:
        restarted.run(args)
        return True
    except ExifToolSessionError:
        # Twice in a row: don't restart for every remaining frame of this process.
        _remove_leftover(written)
        _discard(restarted, give_up=True)
    return _one_shot(args)
