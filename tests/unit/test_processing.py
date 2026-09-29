import numpy as np
import pytest

import halide.processing
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.exiftool import ExifToolError
from halide.io.icc import output_profile_bytes
from halide.io.tiff import write_tiff
from halide.processing import (
    ScanColorError,
    Stage,
    estimate_roll_density_profile,
    export_delivery_image,
    print_scan,
    process_scan,
)
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)
PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))


def _write_negative(path, seed):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


def test_estimate_roll_density_profile_skips_unreadable_files(tmp_path, capsys):
    good_paths = []
    for i in range(3):
        path = tmp_path / f"good_{i}.tiff"
        _write_negative(path, seed=i)
        good_paths.append(path)

    bad_path = tmp_path / "bad.tiff"
    write_tiff(bad_path, np.zeros((16, 16, 3), dtype=np.float32))  # no ICC profile

    # Must not raise, despite the unreadable file among otherwise-good ones.
    profile = estimate_roll_density_profile(good_paths + [bad_path])
    assert profile.source == "auto"
    assert "skipping" in capsys.readouterr().out


def test_estimate_roll_density_profile_skips_a_genuinely_corrupt_file(tmp_path, capsys):
    # A missing-ICC file (above) raises ScanColorError, a clean/expected error. A genuinely
    # corrupt file (e.g. truncated during a scanner hiccup) raises straight from `tifffile`
    # instead — a real bug found by testing with an actual non-TIFF file, not just a synthetic
    # ScanColorError case. Must be tolerated the same way.
    good_paths = []
    for i in range(3):
        path = tmp_path / f"good_{i}.tiff"
        _write_negative(path, seed=i)
        good_paths.append(path)

    corrupt_path = tmp_path / "corrupt.tiff"
    corrupt_path.write_bytes(b"not a tiff file")

    profile = estimate_roll_density_profile(good_paths + [corrupt_path])
    assert profile.source == "auto"
    assert "skipping" in capsys.readouterr().out


def test_estimate_roll_density_profile_raises_when_all_files_unreadable(tmp_path):
    bad_paths = []
    for i in range(2):
        path = tmp_path / f"bad_{i}.tiff"
        write_tiff(path, np.zeros((4, 4, 3), dtype=np.float32))
        bad_paths.append(path)

    with pytest.raises(ScanColorError, match="no readable frames"):
        estimate_roll_density_profile(bad_paths)


# --- Atomic writes (F04) ------------------------------------------------------------------------
# See docs/investigations/codebase-review-evidence/2.3.md 2.3-3 (truncated file on a crashed
# write) and 2.3-4 (an exiftool failure after a good write must not fail the whole frame).


def test_process_scan_exif_failure_is_a_warning_not_an_error(tmp_path, monkeypatch):
    scan = tmp_path / "neg.tif"
    _write_negative(scan, seed=1)
    out = tmp_path / "out.tif"

    def _broken_exif(*args, **kwargs):
        raise ExifToolError("disk full")

    monkeypatch.setattr(halide.processing, "copy_exif_metadata", _broken_exif)
    warnings = []
    resolved = process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams(), on_warning=warnings.append)

    assert resolved is not None
    assert out.exists()  # the pixels are still written and moved into place
    assert len(warnings) == 1
    assert "developed, but its camera metadata couldn't be copied" in warnings[0]
    assert "disk full" in warnings[0]
    assert sorted(tmp_path.iterdir()) == sorted([scan, out])  # no leftover temp file


def test_process_scan_write_failure_leaves_existing_output_untouched(tmp_path, monkeypatch):
    scan = tmp_path / "neg.tif"
    _write_negative(scan, seed=1)
    out = tmp_path / "out.tif"
    out.write_bytes(b"old output")

    def _broken_write_tiff(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(halide.processing, "write_tiff", _broken_write_tiff)
    with pytest.raises(RuntimeError, match="disk full"):
        process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams())

    assert out.read_bytes() == b"old output"
    assert sorted(tmp_path.iterdir()) == sorted([scan, out])  # no leftover temp file


def test_print_scan_exif_failure_is_a_warning_not_an_error(tmp_path, monkeypatch):
    flat = tmp_path / "flat.tif"
    _write_negative(flat, seed=2)
    out = tmp_path / "print.tif"

    def _broken_exif(*args, **kwargs):
        raise ExifToolError("disk full")

    monkeypatch.setattr(halide.processing, "copy_exif_metadata", _broken_exif)
    warnings = []
    resolved, _ = print_scan(flat, out, ToneCurveParams(), on_warning=warnings.append)

    assert resolved is not None
    assert out.exists()
    assert len(warnings) == 1
    assert "printed, but its camera metadata couldn't be copied" in warnings[0]
    assert sorted(tmp_path.iterdir()) == sorted([flat, out])  # no leftover temp file


def test_print_scan_write_failure_leaves_existing_output_untouched(tmp_path, monkeypatch):
    flat = tmp_path / "flat.tif"
    _write_negative(flat, seed=2)
    out = tmp_path / "print.tif"
    out.write_bytes(b"old output")

    def _broken_write_tiff(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(halide.processing, "write_tiff", _broken_write_tiff)
    with pytest.raises(RuntimeError, match="disk full"):
        print_scan(flat, out, ToneCurveParams())

    assert out.read_bytes() == b"old output"
    assert sorted(tmp_path.iterdir()) == sorted([flat, out])


def test_export_delivery_image_write_failure_leaves_existing_output_untouched(tmp_path, monkeypatch):
    processed = tmp_path / "processed.tif"
    write_tiff(processed, np.full((8, 8, 3), 0.3, dtype=np.float32), icc_profile=output_profile_bytes())
    out = tmp_path / "delivery.png"
    out.write_bytes(b"old output")

    def _broken_write(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(halide.processing, "write_delivery_image", _broken_write)
    with pytest.raises(RuntimeError, match="disk full"):
        export_delivery_image(processed, out)

    assert out.read_bytes() == b"old output"
    assert sorted(tmp_path.iterdir()) == sorted([processed, out])
