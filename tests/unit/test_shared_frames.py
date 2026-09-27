"""halide.shared_frames: full-resolution decoded buffers in shared memory (see docs/investigations/
gpu-batch-throughput.md, Part B) — a CPU worker's frame and the GPU service that will attach to it
(Task B2) must see the same bytes without a pipe copy, and a worker that dies while holding one must
never leak a /dev/shm segment (the plan's Review Focus item 5)."""

from __future__ import annotations

import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from halide.shared_frames import SharedMemoryUnavailable, attach_frame, new_frame


def _segment_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False, track=False)
    except FileNotFoundError:
        return False
    shm.close()
    return True


def test_new_frame_yields_a_writable_buffer_of_the_requested_shape_and_dtype():
    with new_frame((2, 3, 3), np.float32) as frame:
        assert frame.array.shape == (2, 3, 3)
        assert frame.array.dtype == np.float32
        frame.array[:] = 1.0
        assert np.all(frame.array == 1.0)


def test_new_frame_removes_its_segment_on_normal_exit():
    with new_frame((4, 4, 3), np.float32) as frame:
        name = frame.name
        assert _segment_exists(name)
    assert not _segment_exists(name)


def test_new_frame_removes_its_segment_on_exception():
    name = None
    with pytest.raises(RuntimeError, match="boom"):
        with new_frame((4, 4, 3), np.float32) as frame:
            name = frame.name
            assert _segment_exists(name)
            raise RuntimeError("boom")
    assert not _segment_exists(name)


def test_attach_frame_sees_the_creators_data_and_does_not_unlink():
    with new_frame((2, 2, 3), np.float32) as frame:
        frame.array[:] = 5.0
        with attach_frame(frame.name, frame.shape, frame.dtype) as attached:
            assert attached is not frame.array  # a separate mapping of the same bytes
            np.testing.assert_array_equal(attached, frame.array)
        # attach_frame's own exit must not have unlinked the segment — only the creator does that.
        assert _segment_exists(frame.name)
    assert not _segment_exists(frame.name)


def test_new_frame_raises_shared_memory_unavailable_when_creation_fails(monkeypatch):
    import halide.shared_frames as shared_frames_module

    def _boom(*args, **kwargs):
        raise OSError("no space left on /dev/shm")

    monkeypatch.setattr(shared_frames_module.shared_memory, "SharedMemory", _boom)

    with pytest.raises(SharedMemoryUnavailable) as exc_info:
        with new_frame((4, 4, 3), np.float32):
            pass
    assert isinstance(exc_info.value.__cause__, OSError)


def test_new_frame_raises_shared_memory_unavailable_when_the_segment_has_no_real_room(monkeypatch):
    """Reproduced directly against this dev sandbox's 64 MiB /dev/shm: `SharedMemory(create=True,
    size=...)` succeeds even for a size bigger than the tmpfs's free space (ftruncate on tmpfs is
    lazy), and only writing to the oversized segment later crashes the whole interpreter with an
    uncatchable SIGBUS — no Python exception at all. `new_frame` guards against this with
    `os.posix_fallocate` right after creation; this test forces that guard's failure branch via
    monkeypatch rather than actually exhausting /dev/shm (which would be a real, hard-to-clean-up
    crash if the guard were ever removed by mistake)."""
    import halide.shared_frames as shared_frames_module

    def _enospc(fd, offset, length):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shared_frames_module.os, "posix_fallocate", _enospc, raising=False)

    created_names = []
    real_shared_memory = shared_frames_module.shared_memory.SharedMemory

    def _tracking_shared_memory(*args, **kwargs):
        shm = real_shared_memory(*args, **kwargs)
        created_names.append(shm.name)
        return shm

    monkeypatch.setattr(shared_frames_module.shared_memory, "SharedMemory", _tracking_shared_memory)

    with pytest.raises(SharedMemoryUnavailable) as exc_info:
        with new_frame((4, 4, 3), np.float32):
            pass
    assert isinstance(exc_info.value.__cause__, OSError)

    # The segment created before the fallocate check failed must not be left behind either.
    assert created_names and not _segment_exists(created_names[0])


@pytest.mark.skipif(sys.platform != "linux", reason="resource_tracker's leaked-segment cleanup is exercised on Linux")
def test_killed_child_holding_a_frame_leaves_no_segment_once_its_process_tree_exits():
    """Review Focus item 5: a worker OOM-killed mid-batch while it holds a shared-memory frame must
    not leak a /dev/shm segment. Python's resource_tracker only unlinks leaked segments once *every*
    process sharing its registration pipe has exited (it fires on EOF, not the instant one process
    dies) — so this drives the kill from a throwaway subprocess (standing in for a batch's worker
    pool) and checks the segment only after that whole subprocess has exited, matching the plan's own
    framing: "check the segment is gone after the pool shuts down."
    """
    script = """
import multiprocessing
import os
import signal
import time

import numpy as np

from halide.shared_frames import new_frame

ctx = multiprocessing.get_context("fork")
ready = ctx.Event()
name_queue = ctx.Queue()


def hold_frame_forever():
    with new_frame((4, 4, 3), np.float32) as frame:
        name_queue.put(frame.name)
        ready.set()
        time.sleep(60)


proc = ctx.Process(target=hold_frame_forever)
proc.start()
assert ready.wait(timeout=10)
name = name_queue.get(timeout=10)
os.kill(proc.pid, signal.SIGKILL)
proc.join(timeout=10)
print(name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=True
    )
    name = result.stdout.strip().splitlines()[-1]

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _segment_exists(name):
        time.sleep(0.2)
    assert not _segment_exists(name), f"segment {name} leaked after its owning process tree exited"
