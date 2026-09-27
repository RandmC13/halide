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
    _cpu_cap,
    _pool_context,
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
        "halide.batch.orchestrator._cpu_cap", return_value=4
    ):
        assert default_worker_count(jobs) == 4


def test_default_worker_count_never_exceeds_the_number_of_jobs(tmp_path):
    jobs = [_job_with_image(tmp_path, "small.tiff", (64, 64, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)), patch(
        "halide.batch.orchestrator._cpu_cap", return_value=8
    ):
        assert default_worker_count(jobs) == 1


def test_default_worker_count_falls_back_to_cpu_heuristic_if_psutil_is_unavailable(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", side_effect=RuntimeError("no such API on this platform")), patch(
        "halide.batch.orchestrator._cpu_cap", return_value=4
    ):
        assert default_worker_count(jobs) == 4


def test_cpu_cap_is_the_physical_core_count():
    # One worker per physical core: a hyperthread sibling adds little speed to a numpy-bound
    # worker but costs a whole extra frame of memory. No longer a fixed cap of 6.
    with patch("psutil.cpu_count", side_effect=lambda logical=True: 16 if logical else 8):
        assert _cpu_cap() == 8


def test_cpu_cap_falls_back_to_logical_cores_when_physical_is_unknown():
    with patch("psutil.cpu_count", return_value=None), patch("os.cpu_count", return_value=12):
        assert _cpu_cap() == 12


def test_cpu_cap_is_at_least_one():
    with patch("psutil.cpu_count", return_value=None), patch("os.cpu_count", return_value=None):
        assert _cpu_cap() == 1


def test_pool_context_preloads_halide_into_the_forkserver_where_available():
    with patch("multiprocessing.get_all_start_methods", return_value=["fork", "spawn", "forkserver"]):
        with patch("multiprocessing.context.ForkServerContext.set_forkserver_preload") as preload:
            context = _pool_context()
    assert context.get_start_method() == "forkserver"
    modules = preload.call_args.args[0]
    assert "halide.processing" in modules and "__main__" in modules  # keeps Python's own default


def test_pool_context_leaves_platforms_without_forkserver_on_their_default():
    with patch("multiprocessing.get_all_start_methods", return_value=["spawn"]):
        assert _pool_context() is None


def test_run_pool_hands_the_preload_context_to_the_executor(tmp_path):
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    future: Future = Future()
    future.set_result(BatchResult(job=job, error=None))
    seen = {}

    def fake_executor(*args, **kwargs):
        seen.update(kwargs)
        return _FakeExecutor({job: future})

    sentinel = object()
    with patch("halide.batch.orchestrator.ProcessPoolExecutor", side_effect=fake_executor), patch(
        "halide.batch.orchestrator._pool_context", return_value=sentinel
    ):
        run_batch([job], Stage.FULL, None, ToneCurveParams(), max_workers=1)
    assert seen["mp_context"] is sentinel


def test_memory_budget_warning_fires_when_requested_workers_exceed_the_safe_estimate(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        warning = memory_budget_warning(jobs, requested_workers=8)
    assert warning is not None
    assert warning.startswith("--workers 8")  # no "Warning:" of its own — console.warning() adds that


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
    # export's calibrated estimate for a file must actually use export's own (baseline, multiplier)
    # pair, not silently fall back to the pipeline's — checked by requiring the two diverge, not by
    # a fixed ordering: after the full pipeline's own memory-optimization fix (buffer reuse across
    # core/*.py, no needless float64 upcast in the ICC matrix multiply — see CLAUDE.md), the pipeline
    # is now the *cheaper* per-pixel estimate of the two for a large image, since none of that fix
    # touches export's own path through colour.RGB_to_RGB, which computes internally in float64
    # regardless of input dtype.
    jobs = [_job_with_image(tmp_path, "large.tiff", (2000, 3000, 3))]
    assert estimate_export_worker_memory_bytes(jobs) != estimate_worker_memory_bytes(jobs)


def test_default_export_worker_count_is_capped_by_available_memory(tmp_path):
    jobs = [_job_with_image(tmp_path, "large.tiff", (3000, 4000, 3))]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1 * 1024**3)):
        assert default_export_worker_count(jobs) == 1


def test_default_export_worker_count_is_capped_by_cpu_when_memory_is_plentiful(tmp_path):
    jobs = [_job_with_image(tmp_path, f"small_{i}.tiff", (64, 64, 3)) for i in range(10)]
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)), patch(
        "halide.batch.orchestrator._cpu_cap", return_value=4
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


# --- GPU workers (docs/plans/gpu-acceleration.md §3.5, Task 7) ---
#
# There is no GPU here, and a fake device can't cross into a forkserver worker, so the worker
# functions are called in-process (with tests/unit/_fake_device.py's fake GPU) and the pool
# plumbing — what each worker is handed, how many there are, what the forkserver preloads — is
# tested separately.

import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402

import halide.batch.orchestrator as orchestrator  # noqa: E402
import halide.device  # noqa: E402
from halide.batch.orchestrator import (  # noqa: E402
    _CUDA_CONTEXT_BYTES,
    _export_worker,
    _print_worker,
    _worker,
    estimate_worker_device_bytes,
    run_print_batch,
)
from halide.device import ComputeDevice, DeviceUnavailableError  # noqa: E402
from tests.unit.test_device_pipeline import (  # noqa: E402,F401 — fixtures
    PROFILE,
    OutOfMemoryError,
    _fail_on_device_call,
    _write_positive,
    _write_scan,
    fake_gpu,
    uploads,
)

_MIB = 2**20
_FRAME = 181 * _MIB  # a real full-resolution scan, decoded


def _cupy_is_loaded() -> bool:
    return "cupy" in sys.modules


def test_forkserver_preload_never_includes_cupy():
    # A CUDA context made in the forkserver would be inherited, unusably, by every forked worker.
    assert not any(name == "cupy" or name.startswith("cupy.") for name in orchestrator._FORKSERVER_PRELOAD)


def test_a_forkserver_worker_has_not_imported_cupy_before_its_first_job():
    from concurrent.futures import ProcessPoolExecutor

    context = _pool_context()
    if context is None:
        pytest.skip("no forkserver on this platform")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        assert executor.submit(_cupy_is_loaded).result(timeout=120) is False


def test_estimate_worker_device_bytes_is_context_plus_frames(tmp_path):
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=None)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        estimate = estimate_worker_device_bytes([job])
    assert estimate == orchestrator._DEVICE_CONTEXT_BYTES + orchestrator._DEVICE_FRAME_MULTIPLIER * _FRAME
    # Never less than the frame itself plus a CUDA context: the frame stays resident on the device.
    assert estimate > _FRAME + _CUDA_CONTEXT_BYTES


def test_default_worker_count_on_a_gpu_is_capped_by_free_device_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(16)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=4 * 2**30, memory_total=8 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 2**30)
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=16):
        per_worker = estimate_worker_device_bytes(jobs)
        assert default_worker_count(jobs, device=gpu) == max(1, (4 * 2**30) // per_worker)
        assert default_worker_count(jobs, device=gpu) < 16
        # The CPU (or no device at all) is unchanged: RAM and cores only.
        cpu_count = default_worker_count(jobs)
        assert default_worker_count(jobs, device=ComputeDevice(kind="cpu")) == cpu_count == 16


def test_default_worker_count_on_a_gpu_is_at_least_one(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=10 * _MIB, memory_total=8 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        assert default_worker_count(jobs, device=gpu) == 1


def test_default_export_worker_count_on_a_gpu_is_capped_by_free_device_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(16)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=4 * 2**30, memory_total=8 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 2**30)
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=16):
        assert default_export_worker_count(jobs, device=gpu) == max(1, (4 * 2**30) // estimate_worker_device_bytes(jobs))


class _RecordingExecutor(_FakeExecutor):
    def __init__(self, futures_by_job, submitted):
        super().__init__(futures_by_job)
        self._submitted = submitted

    def submit(self, fn, job, *rest):
        self._submitted.append((fn, rest))
        return super().submit(fn, job, *rest)


def _run_recording(runner, job, *args, **kwargs):
    future: Future = Future()
    future.set_result(BatchResult(job=job, error=None))
    submitted = []
    with patch("halide.batch.orchestrator.ProcessPoolExecutor",
               lambda *a, **k: _RecordingExecutor({job: future}, submitted)):
        runner([job], *args, **kwargs)
    return submitted


_GPU = ComputeDevice(kind="gpu", name="Fake", memory_free=6 * 2**30, memory_total=8 * 2**30)


@pytest.mark.parametrize(
    "runner, args",
    [
        (run_batch, (Stage.FULL, None, ToneCurveParams())),
        (run_export_batch, ()),
        (run_print_batch, (ToneCurveParams(),)),
    ],
)
def test_runners_hand_workers_the_device_kind_and_their_share_of_device_memory(tmp_path, runner, args):
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    (fn, rest), = _run_recording(runner, job, *args, max_workers=2, device=_GPU)
    # The resolved *kind*, not the ComputeDevice: each worker resolves the device itself, so CUDA is
    # never initialised in the forkserver. Its pool limit: an equal share of free VRAM, less the
    # worker's own CUDA context.
    assert rest[-2:] == ("gpu", (6 * 2**30) // 2 - _CUDA_CONTEXT_BYTES)

    (fn, rest), = _run_recording(runner, job, *args, max_workers=2, device=ComputeDevice(kind="cpu"))
    assert rest[-2:] == ("cpu", None)
    (fn, rest), = _run_recording(runner, job, *args, max_workers=2)
    assert rest[-2:] == ("cpu", None)


def test_run_batch_default_worker_count_uses_the_device(tmp_path):
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    seen = {}

    def fake_default(jobs, **kwargs):
        seen.update(kwargs)
        return 1

    with patch("halide.batch.orchestrator.default_worker_count", side_effect=fake_default):
        _run_recording(run_batch, job, Stage.FULL, None, ToneCurveParams(), device=_GPU)
    assert seen["device"] is _GPU


@pytest.fixture
def worker_process(monkeypatch):
    """A fresh worker process's state: no device resolved yet."""
    monkeypatch.setattr(orchestrator, "_WORKER_DEVICES", {})


def _fake_cupy_pool(limits):
    cupy = types.ModuleType("cupy")
    pool = SimpleNamespace(set_limit=lambda size=None: limits.append(size), free_all_blocks=lambda: None)
    cupy.get_default_memory_pool = lambda: pool

    class ndarray:  # core/_xp.py asks whether an array is one; none are
        pass

    cupy.ndarray = ndarray
    return cupy


@pytest.fixture
def worker_gpu(monkeypatch, fake_gpu, worker_process):
    """The worker's own resolve_device finds (fake) GPU; records every pool limit it sets."""
    calls = []
    monkeypatch.setattr(halide.device, "resolve_device", lambda requested=None: calls.append(requested) or fake_gpu)
    limits = []
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_pool(limits))
    return SimpleNamespace(device=fake_gpu, resolve_calls=calls, limits=limits)


def test_worker_develops_on_the_gpu_it_resolves_itself(tmp_path, worker_gpu, uploads):
    scan = _write_scan(tmp_path / "neg.tif")
    job = BatchJob(input_path=scan, output_path=tmp_path / "gpu.tif")
    result = _worker(job, Stage.FULL, PROFILE, ToneCurveParams(), 64, "gpu", 512 * _MIB)
    assert result.error is None and result.warning is None
    assert uploads  # the frame really went to the device
    assert worker_gpu.resolve_calls == ["gpu"]
    assert worker_gpu.limits == [512 * _MIB]  # this worker's share of the card

    # Once per worker process, not per frame: the second frame reuses the device and its pool.
    job2 = BatchJob(input_path=scan, output_path=tmp_path / "gpu2.tif")
    assert _worker(job2, Stage.FULL, PROFILE, ToneCurveParams(), 64, "gpu", 512 * _MIB).error is None
    assert worker_gpu.resolve_calls == ["gpu"] and worker_gpu.limits == [512 * _MIB]


def test_worker_on_cpu_never_touches_the_device(tmp_path, worker_gpu, uploads):
    scan = _write_scan(tmp_path / "neg.tif")
    result = _worker(BatchJob(input_path=scan, output_path=tmp_path / "cpu.tif"),
                     Stage.FULL, PROFILE, ToneCurveParams(), 64, "cpu", None)
    assert result.error is None and result.warning is None
    assert uploads == [] and worker_gpu.resolve_calls == []


def test_worker_frame_that_fell_back_to_cpu_reports_it_in_its_result(tmp_path, worker_gpu, monkeypatch):
    from halide.processing import process_scan

    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    _fail_on_device_call(monkeypatch, "negative_to_positive", OutOfMemoryError("out of memory"), on_call=1)
    job = BatchJob(input_path=scan, output_path=tmp_path / "gpu.tif")
    result = _worker(job, Stage.FULL, PROFILE, ToneCurveParams(), 64, "gpu", 512 * _MIB)
    assert result.error is None
    assert result.warning is not None and "out of GPU memory" in result.warning and "CPU" in result.warning
    # The CLI names the frame itself; the worker doesn't repeat the path.
    assert not result.warning.startswith(str(scan))
    np.testing.assert_array_equal(read_tiff_pixels(tmp_path / "gpu.tif"), read_tiff_pixels(tmp_path / "cpu.tif"))


def read_tiff_pixels(path):
    from halide.io.tiff import read_tiff

    return read_tiff(path).image


def test_worker_whose_gpu_is_unusable_develops_on_the_cpu_and_says_so(tmp_path, monkeypatch, fake_gpu,
                                                                       worker_process, uploads):
    # E.g. the card has no room left for this worker's CUDA context.
    def unusable(requested=None):
        try:
            raise OutOfMemoryError("out of memory allocating context")
        except OutOfMemoryError as exc:
            raise DeviceUnavailableError("the GPU isn't usable") from exc

    calls = []
    monkeypatch.setattr(halide.device, "resolve_device", lambda requested=None: calls.append(1) or unusable())
    scan = _write_scan(tmp_path / "neg.tif")
    for name in ("a.tif", "b.tif"):
        result = _worker(BatchJob(input_path=scan, output_path=tmp_path / name),
                         Stage.FULL, PROFILE, ToneCurveParams(), 64, "gpu", 512 * _MIB)
        assert result.error is None  # never fails the frame
        assert "out of GPU memory" in result.warning and "CPU" in result.warning
        assert (tmp_path / name).exists()
    assert uploads == []
    assert calls == [1]  # not re-probed for every frame


def test_export_worker_passes_the_device_and_reports_fallback(tmp_path, worker_gpu, uploads, monkeypatch):
    positive = _write_positive(tmp_path / "pos.tif")
    result = _export_worker(BatchJob(input_path=positive, output_path=tmp_path / "out.png"), 95, "gpu", None)
    assert result.error is None and result.warning is None and uploads
    assert worker_gpu.limits == []  # no share given: no limit set

    _fail_on_device_call(monkeypatch, "to_srgb_8bit", RuntimeError("cudaErrorLaunchFailure"), on_call=1)
    result = _export_worker(BatchJob(input_path=positive, output_path=tmp_path / "out2.png"), 95, "gpu", None)
    assert result.error is None and "cudaErrorLaunchFailure" in result.warning


def test_print_worker_passes_the_device_and_reports_fallback(tmp_path, worker_gpu, uploads, monkeypatch):
    positive = _write_positive(tmp_path / "flat.tif")
    result = _print_worker(BatchJob(input_path=positive, output_path=tmp_path / "print.tif"), ToneCurveParams(), "gpu", None)
    assert result.error is None and uploads
    _fail_on_device_call(monkeypatch, "apply_tone", RuntimeError("cudaErrorLaunchFailure"), on_call=1)
    result = _print_worker(BatchJob(input_path=positive, output_path=tmp_path / "print2.tif"), ToneCurveParams(), "gpu", None)
    assert result.error is None and "cudaErrorLaunchFailure" in result.warning
