"""halide.shared_frames: full-resolution decoded buffers in shared memory (see docs/investigations/
gpu-batch-throughput.md, Part B) — a CPU worker's frame and the GPU service that will attach to it
(Task B2) must see the same bytes without a pipe copy, and a worker that dies while holding one must
never leak a /dev/shm segment (the plan's Review Focus item 5)."""

from __future__ import annotations

import subprocess
import sys
import time
from multiprocessing import shared_memory
from textwrap import dedent

import numpy as np
import pytest

from halide.shared_frames import SharedMemoryUnavailable, attach_frame, batch_prefix, new_frame, sweep


def _segment_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False, track=False)
    except FileNotFoundError:
        return False
    shm.close()
    return True


def _run(script: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """Run `script` as a fresh interpreter and return the completed process — used whenever a test
    needs to observe something that would be unsafe or misleading to do inside the live pytest
    process itself: a real segfault (a regression would kill the whole test run, not just fail one
    test), or a resource_tracker/interpreter-shutdown effect that depends on being the *only* thing
    using this interpreter's tracker."""
    return subprocess.run([sys.executable, "-c", dedent(script)], capture_output=True, text=True, timeout=timeout)


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


def test_attach_frame_works_before_python_3_13_via_manual_unregister():
    """`SharedMemory(..., track=False)` only exists from Python 3.13 (gh-82300) — this sandbox runs
    newer, so the gate is patched rather than the interpreter itself
    (`halide.shared_frames._UNTRACKED_ATTACH_SUPPORTED`, not `sys.version_info`). Below that
    version, `attach_frame` must still attach successfully (not raise TypeError for an unknown
    `track` keyword) and must not leave a second, spurious resource_tracker registration for a
    segment its creator already owns — the module docstring's whole reason `track=False` exists in
    the first place. Run in a subprocess so a real "leaked shared_memory objects" warning at
    interpreter exit, if the manual-unregister shim were wrong, shows up as this test's own stderr
    rather than bleeding into some unrelated test."""
    result = _run(
        """
        import halide.shared_frames as sf
        sf._UNTRACKED_ATTACH_SUPPORTED = False
        import numpy as np

        with sf.new_frame((2, 2, 3), np.float32) as frame:
            frame.array[:] = 5.0
            with sf.attach_frame(frame.name, frame.shape, frame.dtype) as attached:
                assert float(attached[0, 0, 0]) == 5.0
        """
    )
    assert result.returncode == 0, f"subprocess crashed (exit {result.returncode}): {result.stderr}"
    assert "leaked shared_memory" not in result.stderr


def test_sweep_works_before_python_3_13_via_manual_unregister(monkeypatch):
    """Same gate as attach_frame's own pre-3.13 shim (see above), exercised on sweep()'s attach."""
    import halide.shared_frames as shared_frames_module

    monkeypatch.setattr(shared_frames_module, "_UNTRACKED_ATTACH_SUPPORTED", False)
    prefix = "sweep-pre313-test-"
    shm = shared_frames_module.shared_memory.SharedMemory(create=True, size=48, name=f"{prefix}leaked")
    try:
        assert _segment_exists(shm.name)
        removed = sweep(prefix)
        assert removed == 1
        assert not _segment_exists(shm.name)
    finally:
        try:
            shm.close()
        except BufferError:
            pass


def test_shared_frame_descriptor_is_a_plain_picklable_tuple():
    import pickle

    with new_frame((2, 2, 3), np.float32) as frame:
        descriptor = frame.descriptor()
        assert descriptor == (frame.name, frame.shape, frame.dtype)
        # Round-trips through pickle without dragging the (unpicklable-at-this-size-for-our-
        # purposes) array along — the whole point of sending a descriptor instead of `frame` itself.
        assert pickle.loads(pickle.dumps(descriptor)) == descriptor


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


def test_array_kept_past_the_with_block_stays_valid_and_the_segment_name_is_gone():
    """Review finding: `np.ndarray(shape, buffer=shm.buf)` (the original implementation) copies out
    the raw pointer and holds no real buffer-protocol export, so `shm.close()` in `new_frame`'s
    `finally` succeeds even while a caller still holds `frame.array` — and a later write into that
    now-dangling pointer segfaults (reproduced directly: exit code 139 with the pre-fix
    implementation). Run in a subprocess so a regression shows up as *this test failing*, not the
    whole pytest process dying."""
    result = _run(
        """
        import numpy as np
        from halide.shared_frames import new_frame

        with new_frame((4, 4, 3), np.float32) as frame:
            name = frame.name
            kept = frame.array
            kept[:] = 1.0

        # The `with` block has exited (segment unlinked); `kept` must still be safely writable, not
        # a dangling pointer into now-invalid memory.
        kept[:] = 3.0
        assert float(kept[0, 0, 0]) == 3.0
        print(name)
        """
    )
    assert result.returncode == 0, f"subprocess crashed (exit {result.returncode}): {result.stderr}"
    name = result.stdout.strip().splitlines()[-1]
    assert not _segment_exists(name)


def test_batch_prefix_is_keyed_on_this_processs_own_pid():
    import os

    prefix = batch_prefix()
    assert prefix.startswith("halide-")
    assert str(os.getpid()) in prefix


def test_new_frame_with_a_prefix_names_the_segment_accordingly():
    with new_frame((2, 2, 3), np.float32, prefix="my-prefix-") as frame:
        assert frame.name.startswith("my-prefix-")
        assert _segment_exists(frame.name)


@pytest.mark.skipif(sys.platform != "linux", reason="/dev/shm-based sweep is Linux-only")
def test_sweep_removes_a_leaked_segment_by_prefix_and_unregisters_it():
    prefix = "sweep-leak-test-"
    # A segment "leaked" the same way a killed worker would leave one: created, then never unlinked
    # (the with-block's own cleanup is bypassed by reaching in and calling __enter__/never __exit__
    # would be awkward with a contextmanager — instead, create one directly and leave it unlinked).
    import halide.shared_frames as shared_frames_module

    shm = shared_frames_module.shared_memory.SharedMemory(create=True, size=48, name=f"{prefix}leaked")
    try:
        assert _segment_exists(shm.name)
        removed = sweep(prefix)
        assert removed == 1
        assert not _segment_exists(shm.name)
    finally:
        try:
            shm.close()
        except BufferError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="/dev/shm-based sweep is Linux-only")
def test_sweep_leaves_a_differently_prefixed_segment_alone():
    """Selectivity: sweeping one batch's prefix must never touch another batch's (or an unrelated
    live frame's) segment, even one sitting right next to it in /dev/shm."""
    import halide.shared_frames as shared_frames_module

    leaked_prefix = "sweep-selective-leak-"
    other_prefix = "sweep-selective-unrelated-"
    leaked = shared_frames_module.shared_memory.SharedMemory(create=True, size=48, name=f"{leaked_prefix}x")
    try:
        with new_frame((2, 2, 3), np.float32, prefix=other_prefix) as unrelated:
            removed = sweep(leaked_prefix)
            assert removed == 1
            assert not _segment_exists(leaked.name)
            assert _segment_exists(unrelated.name)
    finally:
        try:
            leaked.close()
        except BufferError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="forkserver + resource_tracker sharing is exercised on Linux")
def test_sweep_removes_a_sigkilled_pool_workers_segment_with_no_resource_tracker_warning(tmp_path):
    """Review Focus item 5, under the *real* batch topology: a batch's worker pool is a forkserver
    pool, and every worker shares the *parent's* resource_tracker (its registration pipe is
    inherited at pool start), not a fresh one per worker — so a segment a SIGKILLed worker leaves
    behind survives until the whole *parent* process exits, not the instant that worker dies
    (reproduced directly: without `sweep()`, `executor.shutdown()` returning does not remove the
    segment, and the interpreter prints a "leaked shared_memory objects" UserWarning at its own
    exit). `sweep()` is what lets the still-alive orchestrator clean this up immediately after
    noticing the pool lost a worker, instead of waiting for its own process to end.

    Run as a real script (not `-c`): a forkserver/spawn worker's target function must be importable
    by module+qualname in the (separately bootstrapped) forkserver process, which isn't possible for
    a function defined inside a `-c` string's throwaway `__main__`.
    """
    coord_path = tmp_path / "coord.txt"
    script = tmp_path / "kill_pool_worker.py"
    script.write_text(
        dedent(f"""
        import multiprocessing
        import os
        import signal
        import time
        from concurrent.futures import ProcessPoolExecutor

        import numpy as np

        from halide.shared_frames import batch_prefix, new_frame, sweep

        COORD_PATH = {str(coord_path)!r}


        def _hold_a_frame(prefix):
            with new_frame((4, 4, 3), np.float32, prefix=prefix) as frame:
                with open(COORD_PATH, "w") as f:
                    f.write(f"{{os.getpid()}} {{frame.name}}")
                time.sleep(60)


        if __name__ == "__main__":
            prefix = batch_prefix()
            context = multiprocessing.get_context("forkserver")
            executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
            executor.submit(_hold_a_frame, prefix)

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not os.path.exists(COORD_PATH):
                time.sleep(0.1)
            with open(COORD_PATH) as f:
                pid_str, name = f.read().split()
            os.kill(int(pid_str), signal.SIGKILL)

            executor.shutdown(wait=True, cancel_futures=True)

            removed = sweep(prefix)
            print("removed", removed)
            print("name", name)
        """)
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"exit {result.returncode}, stderr:\\n{result.stderr}"
    lines = result.stdout.strip().splitlines()
    values = dict(line.split(" ", 1) for line in lines)
    assert values["removed"] == "1"
    assert not _segment_exists(values["name"])
    stderr_lower = result.stderr.lower()
    assert "leaked" not in stderr_lower, result.stderr
    assert "resource_tracker" not in stderr_lower, result.stderr


# --- Task B3 carry-over fixes -----------------------------------------------------------------


def test_batch_prefix_is_different_for_every_batch():
    # One process can run two batches (the picker's contact sheet, rebuilt): sweeping one batch's
    # prefix must never match the other's live frames.
    assert batch_prefix() != batch_prefix()


def test_standalone_prefix_is_keyed_on_this_processs_own_pid_not_its_parents():
    import os

    with new_frame((2, 2, 3), np.float32) as frame:
        assert frame.name.startswith(f"halide-{os.getpid()}-")
        assert not frame.name.startswith(f"halide-{os.getppid()}-")


@pytest.mark.skipif(sys.platform != "linux", reason="/dev/shm-based sweep is Linux-only")
def test_a_frame_already_swept_by_name_still_closes_cleanly_with_no_tracker_noise():
    # new_frame's own unlink must tolerate the name being gone already (an early sweep), and still
    # close its mapping — with nothing left in the tracker to warn about at exit.
    result = _run(
        """
        import numpy as np
        from halide.shared_frames import new_frame, sweep

        with new_frame((4, 4, 3), np.float32, prefix="early-sweep-test-") as frame:
            frame.array[...] = 1.0
            assert sweep("early-sweep-test-") == 1
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert "leaked" not in result.stderr.lower() and "Traceback" not in result.stderr, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="/dev/shm-based sweep is Linux-only")
def test_sweeping_a_segment_this_processs_tracker_never_registered_prints_nothing():
    # E.g. an orphan from an earlier SIGKILLed run whose pid was reused: an UNREGISTER for a name
    # the tracker doesn't hold made the tracker process print a KeyError traceback.
    result = _run(
        """
        from multiprocessing import shared_memory
        from halide.shared_frames import sweep

        orphan = shared_memory.SharedMemory(create=True, size=48, name="orphan-sweep-test-x", track=False)
        orphan.close()
        print(sweep("orphan-sweep-test-"))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"
    assert "KeyError" not in result.stderr and "Traceback" not in result.stderr, result.stderr
    assert not _segment_exists("orphan-sweep-test-x")
