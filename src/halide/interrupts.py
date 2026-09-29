"""Who stops a batch, and how (F12, review 2.4-9): the parent process, always.

Ctrl-C and closing the terminal (SIGHUP) reach the terminal's whole foreground process group — the
batch parent, its worker processes, the pool's forkserver and the GPU service alike. Left to their
defaults, each child died on its own: a forkserver interrupted while preloading colour/scipy printed
pages of tracebacks, and a terminal closed mid-batch killed everything that could have cleaned up
the shared frames in /dev/shm. So the children ignore those two signals and the parent turns them
(and SIGTERM, which a system shutdown or `kill` sends) into one orderly cancel: stop the pool, stop
the GPU service, sweep the shared frames.

SIGTERM stays at its default in the children: it is how the parent (and the picker's windows)
stop a worker that won't finish (`Process.terminate`).

Standard library only: imported by the batch orchestrator and the GPU service, neither of which may
pull anything heavy into the CLI's start-up.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Iterator

# The signals a terminal sends its whole foreground process group.
_TERMINAL_SIGNALS = tuple(getattr(signal, name) for name in ("SIGINT", "SIGHUP") if hasattr(signal, name))
# The signals the batch parent turns into a cancel, besides Ctrl-C's own KeyboardInterrupt.
_CANCEL_SIGNALS = tuple(getattr(signal, name) for name in ("SIGHUP", "SIGTERM") if hasattr(signal, name))


def _in_main_thread() -> bool:
    # signal.signal only works there; elsewhere (the picker's background threads) these are no-ops.
    return threading.current_thread() is threading.main_thread()


def ignore_terminal_signals() -> None:
    """For a child process (a pool worker's initializer, the GPU service): ignore Ctrl-C and
    SIGHUP — the parent decides when this process stops."""
    for signum in _TERMINAL_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)


@contextmanager
def _handlers(signals: tuple, handler) -> Iterator[None]:
    if not _in_main_thread():
        yield
        return
    previous = {signum: signal.signal(signum, handler) for signum in signals}
    try:
        yield
    finally:
        for signum, old in previous.items():
            if old is not None:  # None: the handler was installed from C, and can't be put back
                signal.signal(signum, old)


def children_ignore_terminal_signals():
    """A context manager for the moment a child process is launched (the forkserver, the GPU
    service): an ignored signal stays ignored across fork and exec, and Python keeps it ignored at
    start-up, so the child ignores Ctrl-C and SIGHUP from its very first instruction — before it
    could run any code of its own to do so (the forkserver's preload, 2.4-9). Keep it short: a
    Ctrl-C landing inside it is lost."""
    return _handlers(_TERMINAL_SIGNALS, signal.SIG_IGN)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def cancel_on_hangup_and_term():
    """A context manager for the batch parent: SIGHUP (the terminal closed) and SIGTERM raise
    KeyboardInterrupt, so they take exactly the same orderly cancel path as Ctrl-C. The previous
    handlers come back on the way out; nested use is fine."""
    return _handlers(_CANCEL_SIGNALS, _raise_interrupt)
