"""The end-of-run summary and halide-log.txt through the real commands."""

from __future__ import annotations

import re

import numpy as np
import pytest

from halide.cli.main import main
from halide.io.roll import RUN_LOG_NAME
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\033\[[0-9;]*m")
MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


def _negative(path, seed):
    rng = np.random.default_rng(seed)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    density = rng.uniform(0.0, 1.0, size=(24, 24, 1)).astype(np.float32)
    write_tiff(path, base * 10.0 ** (-density), icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll(tmp_path):
    d = tmp_path / "roll one"
    d.mkdir()
    for i in range(3):
        _negative(d / f"frame_{i:02d}.tif", i)
    return d


def _batch(roll, out, *extra):
    return main(["batch", str(roll), str(out), *MANUAL, "--workers", "1", *extra])


def test_batch_writes_a_log_into_the_output_folder(roll, tmp_path, capsys):
    out = tmp_path / "out"
    assert _batch(roll, out) == 0
    text = (out / RUN_LOG_NAME).read_text(encoding="utf-8")
    assert "halide batch" in text and "roll one'" in text  # the path is quoted so it pastes back
    assert text.count("frame_0") == 3 and "━━ finished" in text
    screen = _ANSI.sub("", capsys.readouterr().out)
    # Long paths wrap in an 80-column test terminal, so check the rows, not the whole path.
    assert re.search(r"^  Prints ", screen, re.M) and re.search(r"^  Log ", screen, re.M)


def test_log_elsewhere_and_no_log(roll, tmp_path):
    notes = tmp_path / "notes.txt"
    assert _batch(roll, tmp_path / "a", "--log", str(notes)) == 0
    assert notes.exists() and not (tmp_path / "a" / RUN_LOG_NAME).exists()
    assert _batch(roll, tmp_path / "b", "--no-log") == 0
    assert not (tmp_path / "b" / RUN_LOG_NAME).exists()


def test_log_and_no_log_together_is_an_error(roll, tmp_path):
    with pytest.raises(SystemExit):
        _batch(roll, tmp_path / "out", "--log", "x.txt", "--no-log")


def test_an_unwritable_log_is_a_warning_not_a_failure(roll, tmp_path, capsys):
    assert _batch(roll, tmp_path / "out", "--log", str(tmp_path / "no-such-folder" / "log.txt")) == 0
    screen = _ANSI.sub("", capsys.readouterr().out)
    assert re.search(r"^  Log\s+⚠ not written", screen, re.M)


def test_a_log_pointing_at_a_folder_is_a_warning(roll, tmp_path, capsys):
    assert _batch(roll, tmp_path / "out", "--log", str(tmp_path)) == 0
    assert "not written" in _ANSI.sub("", capsys.readouterr().out)


def test_quiet_keeps_warnings_but_not_prints_or_log_rows(roll, tmp_path, capsys):
    assert _batch(roll, tmp_path / "out", "--quiet") == 0
    screen = _ANSI.sub("", capsys.readouterr().out)
    assert "Prints" not in screen and "Log " not in screen
    assert (tmp_path / "out" / RUN_LOG_NAME).exists()  # the log ignores --quiet


def test_skip_existing_logs_kept_frames_and_appends(roll, tmp_path):
    out = tmp_path / "out"
    assert _batch(roll, out) == 0
    (out / "frame_01.tif").unlink()
    assert _batch(roll, out, "--skip-existing") == 0
    text = (out / RUN_LOG_NAME).read_text(encoding="utf-8")
    assert text.count("━━ finished") == 2
    assert len([line for line in text.splitlines() if " kept " in line]) == 2


def test_every_frame_failing_still_logs_and_exits_1(tmp_path):
    roll = tmp_path / "bad"
    roll.mkdir()
    for i in range(2):
        write_tiff(roll / f"f{i}.tif", np.full((8, 8, 3), 0.1, dtype=np.float32))  # no ICC profile
    out = tmp_path / "out"
    assert _batch(roll, out) == 1
    text = (out / RUN_LOG_NAME).read_text(encoding="utf-8")
    assert len([line for line in text.splitlines() if " failed " in line]) == 2
    assert "Prints" not in text


def test_preview_without_output_folder_writes_no_log_unless_asked(roll, tmp_path):
    assert main(["batch", str(roll), "--contact-sheet", str(tmp_path / "s.jpg"), *MANUAL, "--workers", "1"]) == 0
    assert list(tmp_path.glob("**/" + RUN_LOG_NAME)) == []
    log = tmp_path / "preview-log.txt"
    assert main(["batch", str(roll), "--contact-sheet", str(tmp_path / "t.jpg"), *MANUAL, "--workers", "1",
                 "--log", str(log)]) == 0
    assert log.exists()


def test_an_unexpected_error_mid_run_still_closes_the_block(roll, tmp_path, monkeypatch):
    import halide.cli.commands.batch_cmd as batch_cmd

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(batch_cmd, "run_batch", boom)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _batch(roll, out)
    lines = (out / RUN_LOG_NAME).read_text(encoding="utf-8").splitlines()
    i = lines.index("✗ Stopped by an error: disk on fire")
    assert lines[i + 1].startswith("━━ finished")


def test_print_and_export_folders_log_and_single_files_do_not(roll, tmp_path):
    flat = tmp_path / "flat"
    assert _batch(roll, flat, "--output", "flat", "--no-log") == 0
    prints = tmp_path / "prints"
    assert main(["print", str(flat), str(prints), "--workers", "1"]) == 0
    assert "printed" in (prints / RUN_LOG_NAME).read_text(encoding="utf-8")
    pngs = tmp_path / "pngs"
    assert main(["export", str(prints), str(pngs), "--workers", "1"]) == 0
    assert "exported" in (pngs / RUN_LOG_NAME).read_text(encoding="utf-8")
    single = tmp_path / "single"
    single.mkdir()
    assert main(["print", str(flat / "frame_00.tif"), str(single / "a.tif")]) == 0
    assert not (single / RUN_LOG_NAME).exists()


def test_print_on_a_batch_output_folder_does_not_report_the_log(roll, tmp_path, capsys):
    flat = tmp_path / "flat"
    assert _batch(roll, flat, "--output", "flat") == 0
    capsys.readouterr()
    assert main(["print", str(flat), str(tmp_path / "prints"), "--workers", "1"]) == 0
    assert "Skipped" not in _ANSI.sub("", capsys.readouterr().out)


def test_a_cancelled_batch_still_closes_its_log(roll, tmp_path, monkeypatch):
    import halide.cli.commands.batch_cmd as batch_cmd
    from halide.batch.orchestrator import BatchResult

    def cancelled(jobs, *args, on_result=None, **kwargs):
        result = BatchResult(job=jobs[0], error=None)
        on_result(result)
        return [result]

    monkeypatch.setattr(batch_cmd, "run_batch", cancelled)
    out = tmp_path / "out"
    assert _batch(roll, out) == 130
    text = (out / RUN_LOG_NAME).read_text(encoding="utf-8")
    assert "Cancelled" in text and "━━ finished" in text
