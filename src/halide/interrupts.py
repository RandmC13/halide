"""Who stops a batch, and how: the parent process, always.

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

import multiprocessing
import os
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


def exit_with_parent() -> None:
    """For a pool worker: exit the moment the process that owns the pool dies.

    concurrent.futures workers block forever on their call queue and never ask whether their
    owner is alive; they also hold the forkserver's "alive" pipe, so after `kill -9` of the
    owner the forkserver, the resource tracker and every worker (~150 MB each) would stay for
    good. A daemon thread waits on the parent's sentinel (readable when it dies, forkserver
    children included) and exits at once; once the workers are gone the forkserver and tracker
    follow by themselves. `os._exit` is deliberate: nobody is left to collect a result, and a
    half-written output is only ever a hidden `.<stem>.halide-partial-<pid>` file that roll
    discovery skips. A no-op outside a child process."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()  # blocks on the sentinel until the parent is gone
        os._exit(1)

    threading.Thread(target=watch, name="halide-exit-with-parent", daemon=True).start()


def pool_worker_init() -> None:
    """The batch pool's worker initializer: ignore the terminal's signals (the parent decides
    when this process stops) and exit when the parent dies (see `exit_with_parent`)."""
    ignore_terminal_signals()
    exit_with_parent()


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
