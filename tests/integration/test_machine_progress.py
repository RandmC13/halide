"""HALIDE_PROGRESS=machine: the tab-separated lines the window reads from a halide child."""

from __future__ import annotations

import os
import subprocess
import sys

from halide.cli.main import main
from tests.integration.test_run_log_cli import MANUAL, _negative

KINDS = ("start", "done", "summary", "end")


def _machine_lines(text):
    return [line.split("\t") for line in text.splitlines() if line.split("\t")[0] in KINDS]


def _roll(tmp_path, n=2):
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(n):
        _negative(roll / f"frame_{i:02d}.tif", i)
    return roll


def test_a_real_batch_child_writes_every_line_kind_in_order(tmp_path):
    roll = _roll(tmp_path)
    env = {**os.environ, "HALIDE_PROGRESS": "machine", "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run(
        [sys.executable, "-m", "halide.cli.main", "batch", str(roll), str(tmp_path / "out"), *MANUAL,
         "--workers", "1"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
    lines = _machine_lines(out)
    kinds = [line[0] for line in lines]
    assert sorted(kinds[:4]) == ["done", "done", "start", "start"] and set(kinds[4:-1]) == {"summary"}
    starts = {int(line[1]): line[2] for line in lines if line[0] == "start"}
    assert starts == {0: "frame_00.tif", 1: "frame_01.tif"}
    dones = [line for line in lines if line[0] == "done"]
    assert sorted(int(d[1]) for d in dones) == [0, 1] and all(d[2] == "ok" and len(d) == 4 for d in dones)
    summaries = {line[1]: line[2] for line in lines if line[0] == "summary"}
    assert "Prints" in summaries and summaries["Log"].endswith("halide-log.txt")
    assert kinds.index("summary") > max(i for i, k in enumerate(kinds) if k == "done")
    assert lines[-1] == ["end", "2", "2", "0"]
    assert all(len(line) == {"start": 3, "done": 4, "summary": 3, "end": 4}[line[0]] for line in lines)


def _fake_batch(monkeypatch, results_for):
    import halide.cli.commands.batch_cmd as batch_cmd
    from halide.batch.orchestrator import BatchResult

    def fake(jobs, *args, on_result=None, on_start=None, **kwargs):
        done = []
        for job, error in results_for(jobs):
            on_start(job)
            result = BatchResult(job=job, error=error)
            on_result(result)
            done.append(result)
        return done

    monkeypatch.setattr(batch_cmd, "run_batch", fake)


def test_a_failed_frame_and_a_tab_in_its_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HALIDE_PROGRESS", "machine")
    _fake_batch(monkeypatch, lambda jobs: [(jobs[0], None), (jobs[1], "bad\tscan\nreally")])
    assert main(["batch", str(_roll(tmp_path)), str(tmp_path / "out"), *MANUAL, "--workers", "1"]) == 1
    lines = _machine_lines(capsys.readouterr().out)
    assert ["done", "1", "failed", "bad scan really"] in lines
    assert lines[-1] == ["end", "1", "2", "0"]


def test_a_cancelled_run_ends_with_its_flag_set(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HALIDE_PROGRESS", "machine")
    _fake_batch(monkeypatch, lambda jobs: [(jobs[0], None)])
    assert main(["batch", str(_roll(tmp_path, 3)), str(tmp_path / "out"), *MANUAL, "--workers", "1"]) == 130
    assert _machine_lines(capsys.readouterr().out)[-1] == ["end", "1", "3", "1"]


def test_without_the_variable_no_machine_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HALIDE_PROGRESS", raising=False)
    assert main(["batch", str(_roll(tmp_path)), str(tmp_path / "out"), *MANUAL, "--workers", "1"]) == 0
    assert _machine_lines(capsys.readouterr().out) == []


def test_export_speaks_it_too(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    assert main(["batch", str(_roll(tmp_path)), str(out), *MANUAL, "--workers", "1"]) == 0
    monkeypatch.setenv("HALIDE_PROGRESS", "machine")
    capsys.readouterr()
    assert main(["export", str(out), str(tmp_path / "share"), "--workers", "1"]) == 0
    lines = _machine_lines(capsys.readouterr().out)
    assert [line[0] for line in lines].count("done") == 2 and lines[-1] == ["end", "2", "2", "0"]
