"""End-to-end tests for contact sheets: `halide contact` on processed folders, and
`halide batch --contact-sheet` as a no-TIFFs preview."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from halide.cli.main import main
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]
SMALL = ["--frame-width", "120", "--quiet", "--workers", "1"]


@pytest.fixture
def roll(tmp_path):
    rng = np.random.default_rng(0)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    folder = tmp_path / "roll"
    folder.mkdir()
    for i in range(3):
        img = base * 10.0 ** (-rng.uniform(0.0, 1.0, size=(40, 60, 1)).astype(np.float32))
        write_tiff(folder / f"IMG_{i:04d}.tif", img, icc_profile=build_icc(LINEAR_TAGS))
    return folder


def _record_tempdirs(monkeypatch):
    made = []
    real = tempfile.mkdtemp

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", recording)
    return made


def test_batch_preview_writes_only_the_sheet(roll, tmp_path, monkeypatch):
    made = _record_tempdirs(monkeypatch)
    sheet = tmp_path / "preview.jpg"
    before = set(tmp_path.rglob("*"))
    assert main(["batch", str(roll), "--contact-sheet", str(sheet), *MANUAL, *SMALL]) == 0
    assert set(tmp_path.rglob("*")) - before == {sheet}  # no TIFFs (or anything else) left behind
    ours = [d for d in made if os.path.basename(d).startswith("halide-contact-")]  # not multiprocessing's own
    assert ours and all(not os.path.exists(d) for d in ours)  # thumbnail folder deleted
    with Image.open(sheet) as image:
        assert image.width > 3 * 120


def test_batch_can_keep_outputs_and_make_a_sheet(roll, tmp_path):
    out, sheet = tmp_path / "out", tmp_path / "sheet.png"
    assert main(["batch", str(roll), str(out), "--contact-sheet", str(sheet), *MANUAL, *SMALL]) == 0
    assert len(list(out.glob("*.tif"))) == 3 and sheet.exists()


def test_batch_needs_an_output_or_a_sheet(roll):
    with pytest.raises(SystemExit, match="--contact-sheet"):
        main(["batch", str(roll), *MANUAL])


def test_contact_from_processed_tiffs_and_from_exports(roll, tmp_path):
    out = tmp_path / "out"
    assert main(["batch", str(roll), str(out), *MANUAL, "--quiet", "--workers", "1"]) == 0
    assert main(["contact", str(out), str(out / "sheet.jpg"), *SMALL]) == 0
    assert (out / "sheet.jpg").exists()
    # Run again with the sheet now inside the folder: it must not proof itself.
    assert main(["contact", str(out), str(tmp_path / "again.jpg"), *SMALL]) == 0
    with Image.open(out / "sheet.jpg") as first, Image.open(tmp_path / "again.jpg") as second:
        assert first.size == second.size

    exported = tmp_path / "exported"
    assert main(["export", str(out), str(exported), "--quiet", "--workers", "1"]) == 0
    assert main(["contact", str(exported), str(tmp_path / "exports.png"), *SMALL]) == 0
    assert (tmp_path / "exports.png").exists()


def test_contact_rejects_a_bad_sheet_format(roll, tmp_path):
    with pytest.raises(SystemExit, match=".jpg, .jpeg or .png"):
        main(["contact", str(roll), str(tmp_path / "sheet.tif")])
