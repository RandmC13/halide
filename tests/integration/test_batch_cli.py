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
