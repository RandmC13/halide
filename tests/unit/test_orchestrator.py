"""Unit tests for halide.batch.orchestrator's own resilience logic, using a fake executor so a
process pool crash can be simulated deterministically rather than relying on an actual OOM."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch

from halide.batch.orchestrator import BatchJob, BatchResult, run_batch
from halide.core.types import ToneCurveParams
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
