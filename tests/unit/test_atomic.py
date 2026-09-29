"""atomic_output (F04): a killed or failed write must never leave a truncated file under the real
output name. See docs/investigations/codebase-review-evidence/2.3.md 2.3-3."""

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
