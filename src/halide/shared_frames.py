"""Full-resolution decoded frames living in shared memory (`/dev/shm` on Linux) instead of a
process-private heap — Part B of docs/plans/gpu-acceleration.md's "faster batch" work
(docs/investigations/gpu-batch-throughput.md): a CPU-only worker decodes a scan straight into one of
these buffers, hands its name to a separate GPU service process, and the service attaches to the
same bytes instead of the frame being pickled/copied through a pipe.

Kept free of heavy imports (only numpy) at module import time: nothing on the CLI parser's own
import path may reach this module, since that would pull numpy into `halide --help`'s startup, which
tests/unit/test_cli_startup.py forbids. This module is only ever imported from inside batch worker
functions and the (not-yet-built) GPU service, the same way `halide.processing` already is.

Two roles, two functions, deliberately not symmetric:
- `new_frame` (the creating side — a CPU worker): allocates a segment sized for one frame, yields
  it, and unlinks it again once the `with` block exits, normally or via exception. It does NOT pass
  `track=False`, on purpose: Python's own `multiprocessing.resource_tracker`, which is watching this
  process by default, is exactly the mechanism that removes the segment if this process dies (e.g.
  the OOM killer) before its own `finally` can run — see Review Focus item 5 in the plan, and
  tests/unit/test_shared_frames.py's killed-child test.
- `attach_frame` (the service side): opens an existing segment by name and never unlinks it — the
  creator owns that lifecycle, and the service is just borrowing the bytes for as long as the `with`
  block runs. It attaches with `track=False` (Python >= 3.13, matching this project's floor) so the
  *service's own* resource tracker doesn't also register the segment: if it did, the segment would
  have two independent trackers claiming to own its cleanup, defeating the "let the resource tracker
  clean it up when its creator dies" guarantee above.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Iterator

import numpy as np


class SharedMemoryUnavailable(RuntimeError):
    """Raised by `new_frame` when the platform can't back a shared-memory segment of the requested
    size — e.g. `/dev/shm` missing or too small (a small Docker container defaults to 64 MiB; Review
    Focus item 4). Callers fall back to today's non-shared path with a warning; this is never
    allowed to crash a batch."""


@dataclass(frozen=True)
class SharedFrame:
    """A frame backed by a shared-memory segment, as seen by the process that created it. `dtype` is
    kept as `np.dtype(...).str` (e.g. "<f4") rather than a live `np.dtype` object so it's trivially
    picklable/sendable as part of a plain message to the GPU service (Task B2)."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    array: np.ndarray


def _nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    size = dtype.itemsize
    for dim in shape:
        size *= dim
    return size


@contextmanager
def new_frame(shape: tuple[int, ...], dtype) -> Iterator[SharedFrame]:
    """Create a shared-memory segment sized for `shape`/`dtype`, yield it as a `SharedFrame`, and
    unlink it again on the way out — normal return or exception alike. Raises
    `SharedMemoryUnavailable` (wrapping the original OSError/ValueError as `__cause__`) if the
    segment can't be created at all, rather than ever getting partway into a frame's lifetime with no
    backing memory.

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
    Windows's own shared memory, which commits its backing store at creation time already."""
    np_dtype = np.dtype(dtype)
    size = _nbytes(shape, np_dtype)
    try:
        shm = shared_memory.SharedMemory(create=True, size=size)
    except (OSError, ValueError) as exc:
        raise SharedMemoryUnavailable(f"could not create a {size}-byte shared-memory segment: {exc}") from exc
    fd = getattr(shm, "_fd", None)  # no public accessor; only used on the POSIX path (see above)
    if fd is not None and hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(fd, 0, size)
        except OSError as exc:
            shm.close()
            shm.unlink()
            raise SharedMemoryUnavailable(f"no room for a {size}-byte shared-memory segment: {exc}") from exc
    try:
        array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
        yield SharedFrame(name=shm.name, shape=tuple(shape), dtype=np_dtype.str, array=array)
    finally:
        shm.close()
        shm.unlink()


@contextmanager
def attach_frame(name: str, shape: tuple[int, ...], dtype) -> Iterator[np.ndarray]:
    """Attach to an existing segment by name and yield it as an array of `shape`/`dtype` — the
    service side of `new_frame`. Never unlinks: the creator (`new_frame`) owns that. `track=False`
    keeps this process's own resource tracker from also registering the segment (see the module
    docstring)."""
    shm = shared_memory.SharedMemory(name=name, create=False, track=False)
    try:
        yield np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    finally:
        shm.close()
