"""Full-resolution decoded frames living in shared memory (`/dev/shm` on Linux) instead of a
process-private heap — Part B of docs/plans/gpu-acceleration.md's "faster batch" work
(docs/investigations/gpu-batch-throughput.md): a CPU-only worker decodes a scan straight into one of
these buffers, hands its name to a separate GPU service process, and the service attaches to the
same bytes instead of the frame being pickled/copied through a pipe.

Kept free of heavy imports (only numpy) at module import time: nothing on the CLI parser's own
import path may reach this module, since that would pull numpy into `halide --help`'s startup, which
tests/unit/test_cli_startup.py forbids. This module is only ever imported from inside batch worker
functions and the GPU service (halide/gpu_service.py), the same way `halide.processing` already is.

Two roles, two functions, deliberately not symmetric:
- `new_frame` (the creating side — a CPU worker): allocates a segment sized for one frame, yields
  it, and unlinks it again once the `with` block exits, normally or via exception. It does NOT pass
  `track=False`, on purpose: Python's own `multiprocessing.resource_tracker`, which is watching this
  process by default, is exactly the mechanism that removes the segment if this process dies (e.g.
  the OOM killer) before its own `finally` can run — see Review Focus item 5 in the plan.
- `attach_frame` (the service side): opens an existing segment by name and never unlinks it — the
  creator owns that lifecycle, and the service is just borrowing the bytes for as long as the `with`
  block runs. It attaches with `track=False` (Python >= 3.13, matching this project's floor) so the
  *service's own* resource tracker doesn't also register the segment: if it did, the segment would
  have two independent trackers claiming to own its cleanup, defeating the "let the resource tracker
  clean it up when its creator dies" guarantee above.

Review Focus item 5, under the *real* batch topology (found during review, not the first pass):
a batch's worker pool is a forkserver (or spawn) pool, and every process spawned from it shares the
*parent's* resource_tracker (its registration pipe is inherited at pool start) — not a fresh one per
worker. So when a worker is SIGKILLed while it holds a frame, the segment isn't cleaned up the
instant that worker dies; it survives until *every* process sharing that pipe exits, i.e. until the
whole batch's own parent process exits — reproduced directly (see tests/unit/test_shared_frames.py).
That's too late for a long-running `halide batch`. `batch_prefix()`/`sweep()` below exist so the
*orchestrator* (the parent, still alive) can proactively find and remove a dead worker's segment as
soon as the pool notices the worker is gone, without waiting for its own process to exit — B3 wires
this into `batch/orchestrator.py`; this module only provides the primitive.
"""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Iterator

import numpy as np

_NAME_STEM = "halide"

# Linux only: this is where POSIX shared_memory objects actually live on this platform, and it's
# the one thing that lets sweep() find a leftover segment by name rather than needing a live handle
# to it. Not applicable elsewhere (see sweep's docstring).
_SHM_DIR = "/dev/shm"


class SharedMemoryUnavailable(RuntimeError):
    """Raised by `new_frame` when the platform can't back a shared-memory segment of the requested
    size — e.g. `/dev/shm` missing or too small (a small Docker container defaults to 64 MiB; Review
    Focus item 4). Callers fall back to today's non-shared path with a warning; this is never
    allowed to crash a batch."""


@dataclass(frozen=True)
class SharedFrame:
    """A frame backed by a shared-memory segment, as seen by the process that created it.

    Only `name`, `shape` and `dtype` should ever cross a process boundary (to the GPU service, via
    `descriptor()` below) — `array` is a real numpy array and pickling it would serialize the whole
    frame's pixel data over a pipe, exactly the copy this module exists to avoid. `dtype` itself is
    kept as `np.dtype(...).str` (e.g. "<f4") rather than a live `np.dtype` object so that triple
    stays trivially picklable as a plain tuple of built-in types.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    array: np.ndarray

    def descriptor(self) -> tuple[str, tuple[int, ...], str]:
        """The `(name, shape, dtype)` a caller should actually send to another process — e.g. as
        `attach_frame(*frame.descriptor())` on the receiving end (Task B2). Never send `frame`
        itself, or `frame.array`, across a process boundary (see the class docstring)."""
        return (self.name, self.shape, self.dtype)


def batch_prefix() -> str:
    """A predictable, batch-scoped name prefix for this batch's shared-memory segments — call this
    once in the orchestrator (the pool's own parent process) and pass the result down to every
    worker's call to `new_frame(..., prefix=...)`. `sweep()` then only ever removes segments this
    particular batch created, never another `halide batch` (or anything unrelated) that happens to
    be running at the same time. Keyed on *this* process's own pid — the pool owner's, not a
    worker's — plus a random token per call: one process can run two batches over its life (the
    picker's contact sheet window, rebuilt), and sweeping one must never unlink the other's live
    frames; nor can a leftover from an earlier, killed run whose pid happens to be reused ever
    match."""
    return f"{_NAME_STEM}-{os.getpid()}-{uuid.uuid4().hex[:8]}-"


def _standalone_prefix() -> str:
    """`new_frame`'s default prefix when no orchestrator-provided one is given — direct/non-batch
    use, or a unit test. Keyed on this process's own pid, the one that creates (and unlinks) the
    segment: its parent may be anything — a shell — and names nothing useful."""
    return f"{_NAME_STEM}-{os.getpid()}-"


def _nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    size = dtype.itemsize
    for dim in shape:
        size *= dim
    return size


def _close_tolerating_live_exports(shm: shared_memory.SharedMemory) -> None:
    """`shm.close()`, but survives a still-live consumer of its buffer (this process's own returned
    array, or a caller who kept it past its `with` block) instead of leaving a landmine for
    `SharedMemory.__del__` to step on later.

    `np.frombuffer(shm.buf, ...)` (used by both `new_frame` and `attach_frame`) keeps a real
    buffer-protocol export open against the underlying mmap for as long as *any* reference to the
    resulting array survives — which, in practice, is almost always true right when a `with` block
    exits: the caller's own `frame`/`array` variable is typically still in scope for the rest of its
    function, simply because Python doesn't un-bind a `with ... as x` name early. `shm.close()`
    correctly refuses to unmap memory that's still reachable and raises `BufferError` instead
    (confirmed directly: this is not rare, it is close to the *default* case).

    Left alone, that failure is only half the story: `shm`'s own `__del__` retries the very same
    `close()` call as soon as the wrapper itself is garbage-collected (typically immediately —
    nothing else references it), and `__del__` only tolerates `OSError`, not `BufferError`, so that
    retry prints a distracting-but-harmless "Exception ignored while calling deallocator" traceback
    to stderr — reproduced on nearly every test in this suite before this helper existed. This
    function finishes what `close()` itself would have done on success — releasing the file
    descriptor (safe independently of the mapping: POSIX doesn't need it open once `mmap()` has
    returned) and letting go of this wrapper's own reference to the `mmap` object — without touching
    the mapping itself, which is still legitimately reachable through the live consumer's array.
    That mapping is released for real, silently, whenever *that* array is eventually
    garbage-collected: a `Py_buffer` export always keeps its exporting object's refcount above zero,
    so the underlying `mmap` object cannot be deallocated while an export against it survives, and by
    the time the last one goes away there is nothing left to conflict with a normal, ordinary
    deallocation. Nothing here touches the segment's *name*: the caller is expected to have already
    unlinked it (see `new_frame`), which is independent of whether this process itself can still
    close its own local mapping of it.
    """
    try:
        shm.close()
    except BufferError:
        fd = getattr(shm, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            shm._fd = -1
        shm._mmap = None


@contextmanager
def new_frame(shape: tuple[int, ...], dtype, *, prefix: str | None = None) -> Iterator[SharedFrame]:
    """Create a shared-memory segment sized for `shape`/`dtype`, yield it as a `SharedFrame`, and
    unlink it again on the way out — normal return or exception alike. Raises
    `SharedMemoryUnavailable` (wrapping the original OSError/ValueError as `__cause__`) if the
    segment can't be created at all, rather than ever getting partway into a frame's lifetime with no
    backing memory.

    `prefix`, if given (see `batch_prefix()`), becomes the start of the segment's name — the rest is
    a random per-frame suffix, so two frames from the same batch never collide. Defaults to
    `_standalone_prefix()` for direct/non-batch callers.

    Also raises `SharedMemoryUnavailable` if the segment *is* created but there isn't really room
    for it — found by testing this against this project's own dev sandbox's 64 MiB `/dev/shm`
    (Review Focus item 4's own example of a small container): on tmpfs, `SharedMemory(create=True,
    size=...)` only calls `ftruncate`, which happily reports success for a size bigger than the
    filesystem's actual free space (tmpfs allocates backing pages lazily, on first write) — so the
    constructor never raises, and the failure instead shows up as an **uncatchable SIGBUS** the
    moment something writes past the real available space (reproduced directly: `arr[:] = 1.0` on an
    oversized segment kills the interpreter, no Python exception in sight). `os.posix_fallocate`
    forces real allocation immediately and reports a shortfall as an ordinary, catchable `OSError`
    instead — so it's called right after creation, before this segment is ever handed to a caller
    that might write into it. POSIX-only (mirrors `os.posix_fallocate`'s own availability: present
    on Linux, absent on Windows and macOS) — those platforms don't get this pre-check, matching
    Windows's own shared memory, which commits its backing store at creation time already.
    """
    np_dtype = np.dtype(dtype)
    size = _nbytes(shape, np_dtype)
    name = f"{prefix or _standalone_prefix()}{uuid.uuid4().hex}"
    try:
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except (OSError, ValueError) as exc:
        raise SharedMemoryUnavailable(f"could not create a {size}-byte shared-memory segment: {exc}") from exc
    fd = getattr(shm, "_fd", -1)  # no public accessor; only used on the POSIX path (see above)
    if fd >= 0 and hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(fd, 0, size)
        except OSError as exc:
            shm.close()
            shm.unlink()
            raise SharedMemoryUnavailable(f"no room for a {size}-byte shared-memory segment: {exc}") from exc
    try:
        # np.frombuffer, not np.ndarray(..., buffer=shm.buf): the latter copies out the raw pointer
        # and drops the buffer-protocol export once construction finishes, so a caller who keeps
        # the returned array alive past this `with` block (as `attach_frame`'s whole purpose implies
        # some other process will, briefly, once B2 lands) ends up holding a *dangling* pointer once
        # `shm.close()` below actually unmaps the memory — a real, reproduced segfault
        # (tests/unit/test_shared_frames.py). `frombuffer` keeps a live export instead, which turns
        # that same mistake into a catchable `BufferError` from `close()` (below) rather than memory
        # corruption — and if nothing still references the array, there's no export left and close()
        # succeeds normally, exactly as before.
        array = np.frombuffer(shm.buf, dtype=np_dtype).reshape(shape)
        yield SharedFrame(name=shm.name, shape=tuple(shape), dtype=np_dtype.str, array=array)
    finally:
        # unlink() before close(): the name should stop being attachable regardless of whether this
        # process itself can still close its own mapping (see below) — and unlink() also unregisters
        # this segment from the resource_tracker (self._track is True here), so a normal exit never
        # leaves a "leaked shared_memory" registration behind either.
        try:
            shm.unlink()
        except FileNotFoundError:
            # Already removed by name — a sweep() that ran early (an orchestrator cleaning up after
            # a pool it believed was finished). The segment is gone either way; what's left is the
            # tracker registration unlink() would have dropped, and this process's own mapping.
            _unregister(shm._name)
        _close_tolerating_live_exports(shm)


def _unregister(tracked_name: str) -> None:
    """Drop `tracked_name` (with its leading slash, as the tracker keeps it) from this process's
    resource tracker without ever making it print a traceback.

    The tracker is a separate process that keeps a *set* of names and answers an UNREGISTER for a
    name it doesn't hold with a KeyError traceback on stderr (Python 3.14's resource_tracker main
    loop) — and there's no way to ask it what it holds. A REGISTER first makes the pair safe either
    way: a name it already holds stays one entry (a set), a name it never saw (an orphan from
    another run, a worker that died between creating and registering) is added and removed again.
    Both go down the same pipe from this process, so they arrive in order."""
    try:
        resource_tracker.register(tracked_name, "shared_memory")
        resource_tracker.unregister(tracked_name, "shared_memory")
    except Exception:  # noqa: BLE001 — best-effort bookkeeping; the segment is gone either way
        pass


@contextmanager
def attach_frame(name: str, shape: tuple[int, ...], dtype) -> Iterator[np.ndarray]:
    """Attach to an existing segment by name and yield it as an array of `shape`/`dtype` — the
    service side of `new_frame`. Never unlinks: the creator (`new_frame`) owns that. `track=False`
    keeps this process's own resource tracker from also registering the segment (see the module
    docstring)."""
    shm = shared_memory.SharedMemory(name=name, create=False, track=False)
    try:
        yield np.frombuffer(shm.buf, dtype=np.dtype(dtype)).reshape(shape)
    finally:
        _close_tolerating_live_exports(shm)


def sweep(prefix: str) -> int:
    """Unlink every leftover shared-memory segment whose name starts with `prefix` (see
    `batch_prefix()`). For an orchestrator to call **only after its worker pool has fully shut
    down** — every worker of a batch shares one prefix, so a sweep while any of them is still
    running would unlink frames they are still using (the GPU service would then fail to attach
    to them). What it catches is a worker that died holding a frame (SIGKILL, the OOM killer):
    Review Focus item 5, under the real pool topology — a forkserver/spawn worker pool shares the
    *parent's* resource_tracker, so such a segment would otherwise survive until the whole parent
    process exits, not the instant that one worker dies (reproduced directly; see
    tests/unit/test_shared_frames.py). Returns how many segments were removed.

    Also drops each removed name from `multiprocessing.resource_tracker`'s registry, not just the
    segment's bytes: `SharedMemory.unlink()` normally does this itself, but only in the process that
    created the segment — here we unlink on behalf of a worker that's already dead. Without it, the
    tracker still lists the name and prints its own "leaked shared_memory objects" `UserWarning`
    when the parent eventually exits, even though the segment is already gone. Done via
    `_unregister`, which never makes the tracker print a KeyError for a name it doesn't hold.

    Linux only: `/dev/shm` is listable, so leftover segments can be found by name from outside the
    process that created them. Other platforms return 0 without raising, deliberately not attempted:
    on Windows, a shared-memory segment's backing object is destroyed with its last open handle, so
    there's nothing a name-based sweep could still find; macOS has no equivalent listing of POSIX
    shared-memory objects (no visible `/dev/shm`), so a leftover segment there can only be cleaned up
    by the OS itself or by the resource_tracker's own at-exit pass — not proactively, by name, from
    here.
    """
    if sys.platform != "linux":
        return 0
    try:
        entries = os.listdir(_SHM_DIR)
    except OSError:
        return 0
    removed = 0
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        try:
            leftover = shared_memory.SharedMemory(name=entry, create=False, track=False)
        except FileNotFoundError:
            continue
        # This handle never called .buf (no numpy export was ever taken against it), so a plain
        # close() is always safe here — unlike new_frame/attach_frame's own cleanup.
        try:
            leftover.unlink()
        except FileNotFoundError:
            leftover.close()  # removed by someone else between the open and here: theirs to track
            continue
        _unregister(f"/{entry}")
        leftover.close()
        removed += 1
    return removed
