"""R13: a TIFF whose compressed strips are damaged is reported plainly by every command, never as
library text (libdeflate, tifffile)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from halide.cli.main import main
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc


@pytest.fixture
def corrupt_tiff(tmp_path):
    path = tmp_path / "bad.tif"
    write_tiff(path, np.random.default_rng(0).random((300, 400, 3), dtype=np.float32) * 0.1 + 0.05,
               icc_profile=build_icc(LINEAR_TAGS))
    with tifffile.TiffFile(path) as tif:
        offset, count = tif.pages[0].dataoffsets[1], tif.pages[0].databytecounts[1]
    data = bytearray(path.read_bytes())
    span = min(count - 8, 200)
    data[offset + 4 : offset + 4 + span] = bytes([0xA5]) * span
    path.write_bytes(bytes(data))
    return path


def _plain(text):
    assert "libdeflate" not in text.lower() and "LIBDEFLATE" not in text and "Traceback" not in text
    assert "bad.tif isn't a TIFF halide can read" in text


def test_invert_reports_a_damaged_tiff_plainly(corrupt_tiff, tmp_path):
    with pytest.raises(SystemExit) as info:
        main(["invert", str(corrupt_tiff), str(tmp_path / "o.tif"), "--rm", "1", "--bm", "1", "--rs", "1", "--bs", "1"])
    _plain(str(info.value))


def test_batch_print_export_contact_workers_report_it_plainly(corrupt_tiff, tmp_path):
    from halide.batch.orchestrator import BatchJob, _export_worker, _print_worker, _thumbnail_worker, _worker
    from halide.core.types import Stage, ToneCurveParams

    job = BatchJob(input_path=corrupt_tiff, output_path=tmp_path / "out.tif", thumbnail_path=tmp_path / "t.png")
    results = [
        _worker(job, Stage.INVERT_ONLY, None, ToneCurveParams()),
        _print_worker(job, ToneCurveParams()),
        _export_worker(BatchJob(input_path=corrupt_tiff, output_path=tmp_path / "o.png"), 95),
        _thumbnail_worker(job, 200),
    ]
    for result in results:
        assert result.error is not None
        _plain(result.error)
