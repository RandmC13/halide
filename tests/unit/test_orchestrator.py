"""Unit tests for halide.batch.orchestrator's own resilience logic, using a fake executor so a
process pool crash can be simulated deterministically rather than relying on an actual OOM."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from halide.batch.orchestrator import (
    BatchJob,
    BatchResult,
    _FALLBACK_PER_WORKER_BYTES,
    default_export_worker_count,
    default_worker_count,
    estimate_export_worker_memory_bytes,
    estimate_worker_memory_bytes,
    export_memory_budget_warning,
    memory_budget_warning,
    run_batch,
    run_export_batch,
)
from halide.core.types import ToneCurveParams
from halide.io.tiff import write_tiff
from halide.processing import Stage


class _FakeExecutor:
    """Stands in for ProcessPoolExecutor, returning pre-made futures instead of really spawning
    worker processes — lets a broken-pool crash be simulated deterministically."""

    def __init__(self, futures_by_job, *args, **kwargs):
        self._futures_by_job = futures_by_job

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, job, *rest):
        return self._futures_by_job[job]


def _jobs(tmp_path, n):
    return [
        BatchJob(input_path=tmp_path / f"in_{i}.tiff", output_path=tmp_path / f"out_{i}.tiff")
        for i in range(n)
    ]


def test_run_batch_records_a_broken_pool_as_a_failed_result_per_job_not_a_raised_exception(tmp_path):
    # Real bug found via testing on full-resolution scans: enough parallel workers on a
    # memory-constrained machine gets one OOM-killed, which breaks the whole pool — every other
    # pending future then raises BrokenProcessPool too. Without explicit handling, that propagates
    # out of run_batch entirely, discarding every already-completed result. This must not happen —
    # completed work survives, and the rest is reported as ordinary per-job failures.
    jobs = _jobs(tmp_path, 3)

    ok_future = Future()
    ok_future.set_result(BatchResult(job=jobs[0], error=None))
    crashed_future_1 = Future()
    crashed_future_1.set_exception(BrokenProcessPool("simulated crash"))
    crashed_future_2 = Future()
    crashed_future_2.set_exception(BrokenProcessPool("simulated crash"))

    futures_by_job = {jobs[0]: ok_future, jobs[1]: crashed_future_1, jobs[2]: crashed_future_2}

    reported = []
    with patch(
        "halide.batch.orchestrator.ProcessPoolExecutor",
        lambda *a, **k: _FakeExecutor(futures_by_job, *a, **k),
    ):
        results = run_batch(jobs, Stage.FULL, None, ToneCurveParams(), on_result=reported.append)

    assert len(results) == 3  # nothing was silently dropped
    by_job = {r.job: r for r in results}
    assert by_job[jobs[0]].error is None
    assert "crashed" in by_job[jobs[1]].error
    assert "--workers" in by_job[jobs[1]].error  # actionable, not a bare exception message
    assert "crashed" in by_job[jobs[2]].error
    assert len(reported) == 3  # progress callback still fires for the crashed jobs too


def _job_with_image(tmp_path, name, shape):
    path = tmp_path / name
    write_tiff(path, np.zeros(shape, dtype=np.float32))
    return BatchJob(input_path=path, output_path=tmp_path / f"out_{name}")


def test_estimate_worker_memory_bytes_scales_with_decoded_image_size(tmp_path):
    # Regression test for the real crash this sizing exists to prevent: a fixed worker-count
    # default ignored that full-resolution scans are memory-heavy (~4GB peak RSS measured on a
    # real 3276x4849 scan) — the estimate must actually grow with the image's pixel count, not
    # just be a flat constant.
    small = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    large = [_job_with_image(tmp_path, "large.tiff", (2000, 3000, 3))]
    assert estimate_worker_memory_bytes(large) > estimate_worker_memory_bytes(small)


def test_estimate_worker_memory_bytes_uses_the_largest_job_in_the_batch(tmp_path):
    small = _job_with_image(tmp_path, "small.tiff", (64, 64, 3))
    large = _job_with_image(tmp_path, "large.tiff", (2000, 3000, 3))
    assert estimate_worker_memory_bytes([small, large]) == estimate_worker_memory_bytes([large])


def test_estimate_worker_memory_bytes_falls_back_when_header_unreadable(tmp_path):
    # A job whose input file doesn't exist (or is unreadable) must not crash worker-count sizing —
    # it should fall back to a conservative fixed estimate instead.
    missing = BatchJob(input_path=tmp_path / "does_not_exist.tiff", output_path=tmp_path / "out.tiff")
    assert estimate_worker_memory_bytes([missing]) == _FALLBACK_PER_WORKER_BYTES


def test_default_worker_count_is_capped_by_available_memory(tmp_path):
    # Real bug this fixes: a memory-heavy full-res image with several CPU cores available used to
    # default to a CPU-sized worker count regardless of how much RAM was actually free, which is
    # exactly what let a real batch run OOM-kill a worker even without --workers being misused.
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]  # ~137MB decoded -> several GB estimate
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        assert default_worker_count(jobs) == 1


def test_default_worker_count_is_capped_by_cpu_when_memory_is_plentiful(tmp_path):
    jobs = [_job_with_image(tmp_path, f"small_{i}.tiff", (64, 64, 3)) for i in range(10)]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)), patch(
        "os.cpu_count", return_value=4
    ):
        assert default_worker_count(jobs) == 4


def test_default_worker_count_never_exceeds_the_number_of_jobs(tmp_path):
    jobs = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)), patch(
        "os.cpu_count", return_value=8
    ):
        assert default_worker_count(jobs) == 1


def test_default_worker_count_falls_back_to_cpu_heuristic_if_psutil_is_unavailable(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", side_effect=RuntimeError("no such API on this platform")), patch(
        "os.cpu_count", return_value=4
    ):
        assert default_worker_count(jobs) == 4


def test_memory_budget_warning_fires_when_requested_workers_exceed_the_safe_estimate(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        warning = memory_budget_warning(jobs, requested_workers=8)
    assert warning is not None
    assert "--workers 8" in warning


def test_memory_budget_warning_is_none_when_requested_workers_look_safe(tmp_path):
    jobs = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)):
        assert memory_budget_warning(jobs, requested_workers=4) is None


def test_run_batch_rejects_a_non_positive_explicit_worker_count(tmp_path):
    jobs = _jobs(tmp_path, 1)
    try:
        run_batch(jobs, Stage.FULL, None, ToneCurveParams(), max_workers=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- export's own worker-pool sizing/execution: same machinery, its own calibrated constants ---


def test_estimate_export_worker_memory_bytes_scales_with_decoded_image_size(tmp_path):
    small = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    large = [_job_with_image(tmp_path, "large.tiff", (2000, 3000, 3))]
    assert estimate_export_worker_memory_bytes(large) > estimate_export_worker_memory_bytes(small)


def test_estimate_export_worker_memory_bytes_uses_its_own_calibrated_constants(tmp_path):
    # export does far less per-file work than the full inversion pipeline (no white/density
    # balance, invert, or tone-render stages) — its calibrated estimate for the same file must be
    # smaller than the pipeline's own estimate, not the same value reused unchanged.
    jobs = [_job_with_image(tmp_path, "large.tiff", (2000, 3000, 3))]
    assert estimate_export_worker_memory_bytes(jobs) < estimate_worker_memory_bytes(jobs)


def test_default_export_worker_count_is_capped_by_available_memory(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        assert default_export_worker_count(jobs) == 1


def test_default_export_worker_count_is_capped_by_cpu_when_memory_is_plentiful(tmp_path):
    jobs = [_job_with_image(tmp_path, f"small_{i}.tiff", (64, 64, 3)) for i in range(10)]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)), patch(
        "os.cpu_count", return_value=4
    ):
        assert default_export_worker_count(jobs) == 4


def test_export_memory_budget_warning_fires_when_requested_workers_exceed_the_safe_estimate(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        warning = export_memory_budget_warning(jobs, requested_workers=8)
    assert warning is not None
    assert "--workers 8" in warning


def test_export_memory_budget_warning_is_none_when_requested_workers_look_safe(tmp_path):
    jobs = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)):
        assert export_memory_budget_warning(jobs, requested_workers=4) is None


def test_run_export_batch_records_a_broken_pool_as_a_failed_result_per_job(tmp_path):
    jobs = _jobs(tmp_path, 3)

    ok_future = Future()
    ok_future.set_result(BatchResult(job=jobs[0], error=None))
    crashed_future_1 = Future()
    crashed_future_1.set_exception(BrokenProcessPool("simulated crash"))
    crashed_future_2 = Future()
    crashed_future_2.set_exception(BrokenProcessPool("simulated crash"))

    futures_by_job = {jobs[0]: ok_future, jobs[1]: crashed_future_1, jobs[2]: crashed_future_2}

    reported = []
    with patch(
        "halide.batch.orchestrator.ProcessPoolExecutor",
        lambda *a, **k: _FakeExecutor(futures_by_job, *a, **k),
    ):
        results = run_export_batch(jobs, quality=95, on_result=reported.append)

    assert len(results) == 3
    by_job = {r.job: r for r in results}
    assert by_job[jobs[0]].error is None
    assert "crashed" in by_job[jobs[1]].error
    assert "crashed" in by_job[jobs[2]].error
    assert len(reported) == 3


def test_run_export_batch_rejects_a_non_positive_explicit_worker_count(tmp_path):
    jobs = _jobs(tmp_path, 1)
    try:
        run_export_batch(jobs, max_workers=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
