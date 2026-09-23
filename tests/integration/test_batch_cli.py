"""End-to-end tests for `halide batch` and the underlying orchestrator, including the specific
robustness property called out in the build plan: one frame's failure must not corrupt or block
the rest of the batch."""

import numpy as np
import pytest

from halide.batch.orchestrator import discover_jobs, run_batch
from halide.cli.main import main
from halide.core.types import ToneCurveParams
from halide.io.tiff import read_tiff, write_tiff
from halide.processing import Stage
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def _write_negative(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll_dir(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(4):
        _write_negative(in_dir / f"frame_{i:02d}.tiff", seed=i)
    return in_dir


def test_discover_jobs_applies_suffix(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    jobs = discover_jobs(roll_dir, out_dir, suffix="_positive")
    assert len(jobs) == 4
    assert all(job.output_path.stem.endswith("_positive") for job in jobs)


def test_run_batch_processes_all_jobs(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams())
    assert len(results) == 4
    assert all(r.error is None for r in results)
    for job in jobs:
        assert job.output_path.exists()


def test_one_bad_frame_does_not_corrupt_the_rest(roll_dir, tmp_path):
    # Sabotage one input file: no embedded ICC profile at all -> guaranteed ScanColorError.
    bad_path = roll_dir / "frame_02.tiff"
    write_tiff(bad_path, np.zeros((16, 16, 3), dtype=np.float32))  # overwrite, no icc_profile

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams())

    assert len(results) == 4
    failed = [r for r in results if r.error is not None]
    succeeded = [r for r in results if r.error is None]
    assert len(failed) == 1
    assert failed[0].job.input_path.name == "frame_02.tiff"
    assert len(succeeded) == 3
    for r in succeeded:
        assert r.job.output_path.exists()
        result = read_tiff(r.job.output_path)
        assert np.all(np.isfinite(result.image))
    assert not failed[0].job.output_path.exists()


def test_run_batch_keyboard_interrupt_returns_partial_results_instead_of_raising(roll_dir, tmp_path, monkeypatch):
    # Regression test for the raw-traceback-on-Ctrl+C bug: _run_pool must catch KeyboardInterrupt
    # around its wait() loop and return whatever completed so far, not propagate it and discard
    # every already-finished result.
    import halide.batch.orchestrator as orchestrator

    def _raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "wait", _raise_interrupt)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams(), max_workers=2)
    assert results == []


def test_batch_cli_end_to_end(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main(
        ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"]
    )
    assert exit_code == 0
    outputs = sorted(out_dir.glob("*.tiff"))
    assert len(outputs) == 4
    for path in outputs:
        result = read_tiff(path)
        assert np.all(np.isfinite(result.image))
        assert result.icc_profile is not None


def test_batch_cli_reports_failure_exit_code(roll_dir, tmp_path):
    write_tiff(roll_dir / "frame_02.tiff", np.zeros((16, 16, 3), dtype=np.float32))
    out_dir = tmp_path / "out"
    exit_code = main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"])
    assert exit_code == 1


def test_batch_cli_missing_input_directory_errors(tmp_path):
    # Regression test: a nonexistent input directory used to propagate a raw FileNotFoundError
    # traceback straight out of discover_jobs' iterdir() call.
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="input directory not found"):
        main(["batch", str(tmp_path / "does-not-exist"), str(out_dir)])


def test_batch_cli_empty_directory_errors(tmp_path):
    in_dir = tmp_path / "empty_in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    exit_code = main(["batch", str(in_dir), str(out_dir)])
    assert exit_code == 1


def test_auto_density_roll_conflicts_with_other_sources(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="cannot be combined"):
        main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--rm", "2.0", "--quiet"])


def test_batch_cli_prints_its_settings_as_one_run_sheet_before_developing(roll_dir, tmp_path, capsys):
    import re

    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--contrast", "0.8", "--workers", "1"]
    assert main(args) == 0
    out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out)
    sheet = out[: out.index("Developing")].splitlines()

    assert sheet[0] == sheet[-1] and set(sheet[0]) == {"▫", " "}  # framed by sprocket rules
    rows: dict[str, str] = {}
    for line in sheet[1:-1]:  # a continuation line (blank label) belongs to the row above it
        label = line[2:17].strip() or label
        rows[label] = f"{rows[label]} {line[17:].strip()}" if label in rows else line[17:]
    assert list(rows) == ["Roll", "Scans", "Calibration", "Output", "Workers"]
    nbsp = "\u00a0"  # RunSheet.SEP's non-breaking space
    assert rows["Roll"].replace(" ", "") == f"in{nbsp}·4frames→{out_dir}"  # the long tmp path wraps
    assert rows["Calibration"] == f"auto{nbsp}· one profile for the whole roll, from 4 frames"
    assert rows["Output"] == f"print{nbsp}· grade 0.80{nbsp}· exposure fitted per frame"
    assert rows["Workers"] == "1 (--workers)"


def test_batch_cli_run_sheet_names_the_roll_when_run_on_the_current_directory(roll_dir, tmp_path, monkeypatch, capsys):
    # Path(".").name is "", which printed a blank roll name for `halide batch . ...`.
    monkeypatch.chdir(roll_dir)
    assert main(["batch", ".", str(tmp_path / "out"), "--auto-density-roll", "--workers", "1"]) == 0
    import re

    out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out)
    roll_line = next(line for line in out.splitlines() if line.startswith("  Roll"))
    assert roll_line.split("Roll", 1)[1].split()[0].startswith("in")
