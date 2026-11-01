"""atomic_output: a killed or failed write must never leave a truncated file under the real
output name."""

import pytest

from halide.io.atomic import atomic_output


def test_success_replaces_target_and_leaves_no_temp(tmp_path):
    target = tmp_path / "out.tif"
    with atomic_output(target) as tmp:
        assert not target.exists()  # nothing under the real name until the write finishes
        tmp.write_bytes(b"new bytes")
    assert target.read_bytes() == b"new bytes"
    assert list(tmp_path.iterdir()) == [target]  # no leftover temp file


def test_exception_inside_leaves_target_untouched_and_removes_temp(tmp_path):
    target = tmp_path / "out.tif"
    target.write_bytes(b"old bytes")

    with pytest.raises(RuntimeError, match="disk full"):
        with atomic_output(target) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("disk full")

    assert target.read_bytes() == b"old bytes"  # untouched
    assert list(tmp_path.iterdir()) == [target]  # temp file removed, nothing else left behind


def test_temp_is_hidden_same_folder_same_suffix(tmp_path):
    target = tmp_path / "out.tif"
    with atomic_output(target) as tmp:
        assert tmp.parent == target.parent
        assert tmp.name.startswith(".")
        assert tmp.suffix == ".tif"  # exiftool picks the file type from the extension
        tmp.write_bytes(b"bytes")


def test_keyboardinterrupt_also_cleans_up(tmp_path):
    target = tmp_path / "out.tif"
    target.write_bytes(b"old bytes")

    with pytest.raises(KeyboardInterrupt):
        with atomic_output(target) as tmp:
            tmp.write_bytes(b"partial")
            raise KeyboardInterrupt

    assert target.read_bytes() == b"old bytes"
    assert list(tmp_path.iterdir()) == [target]


def test_sweep_partials_removes_only_dead_processes_partials(tmp_path):
    import os
    import subprocess
    import sys

    from halide.io.atomic import sweep_partials

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    dead = tmp_path / f".a.halide-partial-{child.pid}.tif"
    live = tmp_path / f".b.halide-partial-{os.getppid()}.png"
    ours = tmp_path / f".c.halide-partial-{os.getpid()}.tif"
    keep = [tmp_path / "a.tif", tmp_path / ".DS_Store", tmp_path / ".d.halide-partial-x.tif"]
    for p in (dead, live, ours, *keep):
        p.write_bytes(b"x")
    assert sweep_partials(tmp_path) == 1
    assert not dead.exists() and all(p.exists() for p in (live, ours, *keep))
