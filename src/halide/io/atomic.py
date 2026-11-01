"""Atomic output writes: every file halide writes as its final result — a developed TIFF,
a print, an export, a contact sheet, a saved profile — goes through here first, so a kill -9, a
full disk, or any other failure partway through a write never leaves a truncated file sitting under
the real output name, indistinguishable by name from a complete one.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output(path: str | Path):
    """Yield a hidden temporary path beside `path`; on success move it over `path` in one step
    (os.replace), so a killed or failed write never leaves a truncated file under the real name.
    The temp keeps the real suffix (exiftool picks the file type from it) and starts with "."
    (roll discovery skips hidden files, so a leftover is never mistaken for a frame)."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.halide-partial-{os.getpid()}{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_PARTIAL = re.compile(r"^\..+\.halide-partial-(\d+)(?:\.\w+)?$")


def sweep_partials(folder: str | Path) -> int:
    """Remove the unfinished `.<name>.halide-partial-<pid>` files a killed run left in `folder` (not
    recursive), skipping any whose process is still alive (another run writing there) and our own.
    Returns how many were removed."""
    import psutil

    removed = 0
    for entry in Path(folder).iterdir():
        match = _PARTIAL.match(entry.name)
        if not match or not entry.is_file():
            continue
        pid = int(match.group(1))
        if pid == os.getpid() or psutil.pid_exists(pid):
            continue
        try:
            entry.unlink()
            removed += 1
        except OSError:
            pass
    return removed
