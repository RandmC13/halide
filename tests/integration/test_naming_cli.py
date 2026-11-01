"""`--name` templates through batch and the folder forms of print and export."""

from __future__ import annotations

import os

import numpy as np
import pytest

from halide.cli.main import main
from halide.core.types import DensityProfile
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


def _negative(path, seed):
    rng = np.random.default_rng(seed)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    density = rng.uniform(0.0, 1.0, size=(24, 24, 1)).astype(np.float32)
    write_tiff(path, base * 10.0 ** (-density), icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    for i in range(3):
        _negative(d / f"frame_{i:02d}.tif", i)
    return d


def _batch(roll, out, *extra, calibration=MANUAL):
    return main(["batch", str(roll), str(out), *calibration, "--workers", "1", "--no-log", *extra])


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


def test_name_template_names_the_outputs(roll, tmp_path):
    assert _batch(roll, tmp_path / "out", "--name", "{roll}_{n:02}") == 0
    assert _names(tmp_path / "out") == ["roll_01.tif", "roll_02.tif", "roll_03.tif"]


def test_a_template_that_maps_two_frames_to_one_name_is_refused(roll, tmp_path):
    with pytest.raises(SystemExit, match=r"frame_00.tif and frame_01.tif would both be written.*add \{n\} or \{stem\} to --name"):
        _batch(roll, tmp_path / "out", "--name", "{roll}")
    assert not list((tmp_path / "out").glob("*"))


def test_name_and_suffix_together_are_an_argument_error(roll, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _batch(roll, tmp_path / "out", "--name", "{n}", "--suffix", "x")
    assert exc.value.code == 2


def test_a_bad_template_lists_the_fields_and_writes_nothing(roll, tmp_path):
    with pytest.raises(SystemExit, match=r"unknown field \{date\}.*\{stem\}"):
        _batch(roll, tmp_path / "out", "--name", "{date}")
    assert not (tmp_path / "out").exists()


def test_resuming_keeps_the_frame_numbers(roll, tmp_path):
    out = tmp_path / "out"
    assert _batch(roll, out, "--name", "{roll}_{n:02}") == 0
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    (out / "roll_02.tif").unlink()
    os.utime(out / "roll_01.tif", ns=(1, 1))
    assert _batch(roll, out, "--name", "{roll}_{n:02}", "--skip-existing") == 0
    assert _names(out) == ["roll_01.tif", "roll_02.tif", "roll_03.tif"]
    assert (out / "roll_01.tif").stat().st_mtime_ns == 1
    assert (out / "roll_03.tif").stat().st_mtime_ns == before["roll_03.tif"]


def test_dry_run_lists_the_renames_and_writes_nothing(roll, tmp_path, capsys):
    assert _batch(roll, tmp_path / "out", "--name", "{roll}_{n:02}", "--dry-run") == 0
    text = capsys.readouterr().out
    assert "frame_00.tif → roll_01.tif" in text and "frame_02.tif → roll_03.tif" in text
    assert not (tmp_path / "out").exists()


def test_stock_comes_from_the_profile(roll, tmp_path, monkeypatch):
    from halide.calibration.profile_store import save_named_profile

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    save_named_profile(DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78),
                                      film_stock="Portra 400"), "p")
    assert _batch(roll, tmp_path / "out", "--name", "{stock}_{n}", "--profile", "p", calibration=[]) == 0
    assert _names(tmp_path / "out") == ["Portra 400_1.tif", "Portra 400_2.tif", "Portra 400_3.tif"]


def test_folder_print_and_export_take_name(roll, tmp_path):
    assert _batch(roll, tmp_path / "flat", "--output", "flat") == 0
    assert main(["print", str(tmp_path / "flat"), str(tmp_path / "printed"), "--workers", "1", "--no-log",
                 "--name", "{n:02}"]) == 0
    assert _names(tmp_path / "printed") == ["01.tif", "02.tif", "03.tif"]
    assert main(["export", str(tmp_path / "printed"), str(tmp_path / "png"), "--workers", "1", "--no-log",
                 "--name", "{n:02}"]) == 0
    assert _names(tmp_path / "png") == ["01.png", "02.png", "03.png"]


def test_name_on_a_contact_sheet_preview_is_refused(roll, tmp_path):
    # a preview keeps no outputs, so --name would do nothing
    with pytest.raises(SystemExit, match="--name names the developed frames"):
        main(["batch", str(roll), "--contact-sheet", str(tmp_path / "s.jpg"), *MANUAL, "--name", "{n}"])
    assert not (tmp_path / "s.jpg").exists()
