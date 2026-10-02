"""halide.interrupts: the signal helpers behind orderly cancelling (F12, review 2.4-9)."""

import os
import selectors
import signal
import subprocess
import sys
import time

import pytest

from halide import interrupts


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")
def test_cancel_on_hangup_and_term_raises_keyboard_interrupt_and_restores():
    before = signal.getsignal(signal.SIGTERM)
    with interrupts.cancel_on_hangup_and_term():
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_handler_installed_from_c_is_left_alone_on_restore(monkeypatch):
    """signal.signal returns None for a handler installed from C: there is nothing to put back,
    and passing None back would raise TypeError on the way out."""
    calls = []

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        return None

    monkeypatch.setattr(interrupts.signal, "signal", fake_signal)
    with interrupts.cancel_on_hangup_and_term():
        pass
    assert all(handler is interrupts._raise_interrupt for _, handler in calls)


# Liveness is read from /proc, so these run where that exists rather than pass vacuously elsewhere.
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")

_POOL_SCRIPT = """
import multiprocessing, sys, time
from concurrent.futures import ProcessPoolExecutor
from halide import interrupts

def sleepy():
    time.sleep(600)

def pid():
    import os
    time.sleep(0.5)
    return os.getpid()

if __name__ == "__main__":
    init = interrupts.pool_worker_init if sys.argv[1] == "init" else None
    ctx = multiprocessing.get_context("forkserver")
    ex = ProcessPoolExecutor(max_workers=2, mp_context=ctx, initializer=init)
    pids = {f.result() for f in [ex.submit(pid) for _ in range(2)]}
    for _ in range(2):
        ex.submit(sleepy)
    print(" ".join(map(str, pids)), flush=True)
    time.sleep(600)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers kill(0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _run_and_kill_owner(mode: str, script) -> list[int]:
    import os
    import subprocess
    import sys
    import time

    script.write_text(_POOL_SCRIPT)  # a real file: the forkserver re-imports __main__ by path
    owner = subprocess.Popen([sys.executable, str(script), mode], stdout=subprocess.PIPE, text=True)
    pids: list[int] = []
    try:
        ready = selectors.DefaultSelector()
        ready.register(owner.stdout, selectors.EVENT_READ)
        assert ready.select(timeout=60), "the pool's workers never reported their pids"
        pids = [int(p) for p in owner.stdout.readline().split()]
        assert pids
        owner.kill()
        owner.wait()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.1)
        return [p for p in pids if _alive(p)]
    finally:
        owner.kill()
        owner.stdout.close()
        for p in pids:
            if _alive(p):
                os.kill(p, signal.SIGKILL)


@LINUX_ONLY
def test_pool_workers_exit_when_the_owner_is_killed(tmp_path):
    assert _run_and_kill_owner("init", tmp_path / "owner.py") == []


@LINUX_ONLY
def test_control_pool_workers_without_initializer_are_orphaned(tmp_path):
    assert _run_and_kill_owner("none", tmp_path / "owner.py") != []
