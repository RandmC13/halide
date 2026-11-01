"""`batch --recursive` and `--into`: a folder of rolls in one run."""

from __future__ import annotations

import re

import numpy as np
import pytest

from halide.batch import orchestrator
from halide.cli.commands import batch_cmd
from halide.cli.main import main
from halide.io.roll import RUN_LOG_NAME
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\033\[[0-9;]*m")
MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


def _negative(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    density = rng.uniform(0.0, 1.0, size=(24, 24, 1)).astype(np.float32)
    write_tiff(path, base * 10.0 ** (-density), icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def holiday(tmp_path):
    """The user's layout: raw files beside a pre-processed folder of linear TIFFs, per roll."""
    top = tmp_path / "Holiday"
    for r, roll in enumerate(("Roll1", "Roll2")):
        (top / roll).mkdir(parents=True)
        (top / roll / "IMG_0001.CR2").write_bytes(b"raw")
        for i in range(2):
            _negative(top / roll / "pre-processed" / f"IMG_{i + 1:04d}.tif", r * 10 + i)
    return top


def _batch(*args, calibration=MANUAL):
    return main(["batch", *map(str, args), *calibration, "--workers", "1"])


def _tifs(folder):
    return sorted(p.name for p in folder.glob("*.tif"))


def _screen(capsys):
    return _ANSI.sub("", capsys.readouterr().out).replace("\xa0", " ")


def test_each_roll_develops_into_its_own_folder(holiday, capsys):
    assert _batch(holiday, "--recursive", "--into", "../inversion") == 0
    for roll in ("Roll1", "Roll2"):
        out = holiday / roll / "inversion"
        assert _tifs(out) == ["IMG_0001.tif", "IMG_0002.tif"]
        assert (out / RUN_LOG_NAME).exists()
    screen = _screen(capsys)
    assert re.search(r"^  Roll +Roll1 · 2 frames", screen, re.M)
    assert "2 rolls · 4 frames developed in" in screen
    assert not re.search(r"^  Skipped +$", screen, re.M)  # nothing left out, so no empty row


def test_running_again_skips_the_output_folders(holiday, capsys):
    assert _batch(holiday, "--recursive", "--into", "../inversion") == 0
    before = {p: p.stat().st_mtime_ns for p in holiday.rglob("inversion/*.tif")}
    capsys.readouterr()
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--skip-existing") == 0
    screen = _screen(capsys)
    assert "2 folders of halide output" in screen and "nothing new to develop" in screen
    assert {p: p.stat().st_mtime_ns for p in holiday.rglob("inversion/*.tif")} == before


def test_running_again_without_a_policy_refuses_once(holiday):
    assert _batch(holiday, "--recursive", "--into", "../inversion") == 0
    stamp = {p: p.stat().st_mtime_ns for p in holiday.rglob("inversion/*.tif")}
    # A new scan in Roll1 puts its inversion folder back among the destinations, not the skipped.
    _negative(holiday / "Roll1" / "pre-processed" / "IMG_0003.tif", 5)
    with pytest.raises(SystemExit, match=r"4 of 5 outputs already exist.*--overwrite.*--skip-existing"):
        _batch(holiday, "--recursive", "--into", "../inversion")
    assert {p: p.stat().st_mtime_ns for p in holiday.rglob("inversion/*.tif")} == stamp
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--skip-existing") == 0
    assert _tifs(holiday / "Roll1" / "inversion") == ["IMG_0001.tif", "IMG_0002.tif", "IMG_0003.tif"]


def test_existing_outputs_are_refused_once_for_all_rolls(holiday, tmp_path):
    out = tmp_path / "Out"
    assert _batch(holiday, out, "--recursive") == 0
    with pytest.raises(SystemExit, match=r"4 of 4 outputs already exist in 2 folders.*--overwrite.*--skip-existing") as exc:
        _batch(holiday, out, "--recursive")
    assert str(exc.value).count("already exist") == 1


def test_dry_run_writes_nothing_and_lists_every_roll(holiday, capsys):
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--dry-run") == 0
    assert not list(holiday.rglob("inversion"))
    screen = _screen(capsys)
    assert screen.count("Dry run — nothing developed.") == 3  # one per roll, then the whole run
    assert "Roll1 · 2 frames" in screen and "Roll2 · 2 frames" in screen
    assert "Dry run — nothing developed. 2 rolls · 4 frames would be developed." in screen


def test_a_positional_output_folder_gets_a_subfolder_per_roll(holiday, tmp_path):
    out = tmp_path / "Out"
    assert _batch(holiday, out, "--recursive") == 0
    assert _tifs(out / "Roll1") == ["IMG_0001.tif", "IMG_0002.tif"]
    assert _tifs(out / "Roll2") == ["IMG_0001.tif", "IMG_0002.tif"]


def test_contact_sheet_is_a_file_name_written_into_each_roll(holiday, tmp_path):
    with pytest.raises(SystemExit, match="file name only"):
        _batch(holiday, "--recursive", "--into", "../inversion", "--contact-sheet", "sheets/s.jpg")
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--contact-sheet", "s.jpg", "--quiet") == 0
    assert (holiday / "Roll1" / "inversion" / "s.jpg").exists()
    assert (holiday / "Roll2" / "inversion" / "s.jpg").exists()


def test_a_recursive_preview_is_refused(holiday):
    with pytest.raises(SystemExit, match="needs --into or an output folder"):
        _batch(holiday, "--recursive", "--contact-sheet", "s.jpg")
    with pytest.raises(SystemExit, match="--into FOLDER"):
        _batch(holiday, "--recursive")


def test_into_rules(holiday, tmp_path):
    with pytest.raises(SystemExit, match="absolute path.*2 rolls"):
        _batch(holiday, "--recursive", "--into", tmp_path / "abs")
    with pytest.raises(SystemExit, match="give one of them"):
        _batch(holiday, tmp_path / "Out", "--recursive", "--into", "../inversion")
    assert not (tmp_path / "abs").exists()


def test_into_without_recursive_is_resolved_against_the_input(holiday):
    assert _batch(holiday / "Roll1" / "pre-processed", "--into", "../inversion") == 0
    assert _tifs(holiday / "Roll1" / "inversion") == ["IMG_0001.tif", "IMG_0002.tif"]


def test_saving_a_profile_from_a_per_roll_estimate_is_refused(holiday):
    with pytest.raises(SystemExit, match="each roll gets its own estimate"):
        _batch(holiday, "--recursive", "--into", "../inversion", "--auto-density-roll",
               "--save-profile-as", "X", calibration=[])


def test_saving_a_manual_profile_saves_it_once(holiday, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--save-profile-as", "hol") == 0
    assert (tmp_path / "cfg" / "halide" / "profiles" / "hol.json").exists()


def test_name_template_uses_the_roll_name(holiday):
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--name", "{roll}_{n:02}") == 0
    assert _tifs(holiday / "Roll1" / "inversion") == ["Roll1_01.tif", "Roll1_02.tif"]
    assert _tifs(holiday / "Roll2" / "inversion") == ["Roll2_01.tif", "Roll2_02.tif"]


def test_name_is_allowed_with_into_alone(holiday):
    assert _batch(holiday / "Roll1" / "pre-processed", "--into", "../inversion", "--name", "x{n}") == 0
    assert _tifs(holiday / "Roll1" / "inversion") == ["x1.tif", "x2.tif"]


def test_a_failing_roll_does_not_stop_the_next(holiday, capsys):
    for p in (holiday / "Roll1" / "pre-processed").glob("*.tif"):
        p.write_bytes(b"not a tiff")
    assert _batch(holiday, "--recursive", "--into", "../inversion") == 1
    assert _tifs(holiday / "Roll2" / "inversion") == ["IMG_0001.tif", "IMG_0002.tif"]
    screen = _screen(capsys)
    assert "2 of 4 frames developed (2 failed: Roll1)" in screen


def test_cancelling_roll_one_starts_no_other_roll(holiday, capsys, monkeypatch):
    calls = []

    def cancelled(jobs, *args, **kwargs):
        calls.append(jobs)
        return []  # what a Ctrl-C returns when nothing finished

    monkeypatch.setattr(batch_cmd, "run_batch", cancelled)
    assert _batch(holiday, "--recursive", "--into", "../inversion") == 130
    assert len(calls) == 1
    assert not list((holiday / "Roll2").glob("inversion/*"))
    assert "Cancelled - Roll1 cancelled, Roll2 not started" in _screen(capsys)


def test_the_calibration_question_is_asked_once(holiday, monkeypatch):
    asked = []
    monkeypatch.setattr("halide.cli._calibration_args.is_interactive", lambda: True)
    monkeypatch.setattr("halide.cli._calibration_args.console.menu",
                        lambda prompt, options: asked.append(prompt) or "auto")
    monkeypatch.setattr("halide.cli._output_policy.is_interactive", lambda: False)
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--quiet", calibration=[]) == 0
    assert len(asked) == 1 and "these rolls" in asked[0]
    assert _tifs(holiday / "Roll2" / "inversion")


def test_a_log_file_collects_every_roll(holiday, tmp_path):
    log = tmp_path / "all.txt"
    assert _batch(holiday, "--recursive", "--into", "../inversion", "--log", log) == 0
    assert log.read_text(encoding="utf-8").count("━━ finished") == 2
    assert not (holiday / "Roll1" / "inversion" / RUN_LOG_NAME).exists()
