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


def test_device_budget_warning_fires_when_requested_workers_exceed_free_gpu_memory(tmp_path):
    from halide.batch.orchestrator import device_budget_warning

    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=int(6.8 * 2**30), memory_total=8 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        safe = max(1, gpu.memory_free // estimate_worker_device_bytes(jobs))
        warning = device_budget_warning(jobs, safe + 1, gpu)
        assert warning is not None
        assert warning.startswith(f"--workers {safe + 1}")  # console.warning() adds "Warning:"
        assert "GPU memory" in warning and "CPU" in warning
        assert device_budget_warning(jobs, safe, gpu) is None


def test_device_budget_warning_is_none_without_a_gpu_or_its_free_memory(tmp_path):
    from halide.batch.orchestrator import device_budget_warning

    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        assert device_budget_warning(jobs, 64, None) is None
        assert device_budget_warning(jobs, 64, ComputeDevice(kind="cpu")) is None
        assert device_budget_warning(jobs, 64, ComputeDevice(kind="gpu", name="Fake")) is None


def test_device_worker_cap_is_what_bounds_the_default_on_a_small_card(tmp_path):
    from halide.batch.orchestrator import device_worker_cap

    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(16)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=4 * 2**30, memory_total=8 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 2**30)
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=16):
        assert device_worker_cap(jobs, gpu) == default_worker_count(jobs, device=gpu)
        assert device_worker_cap(jobs, ComputeDevice(kind="cpu")) is None


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


# The user's RTX 3070 benchmark (docs/plans/gpu-acceleration.md §7): a GPU worker's host memory is
# ~1.2 GiB against a CPU worker's ~0.4 GiB (CUDA/CuPy's own host-side libraries), so the RAM cap must
# count that for GPU pools — or a machine with a big card and little RAM starts too many workers.
_USER_RAM = int(7.6 * 2**30)
_USER_VRAM = int(6.58 * 2**30)


def test_a_gpu_worker_is_estimated_to_need_more_host_memory_than_a_cpu_worker(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=_USER_VRAM)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        cpu_estimate = estimate_worker_memory_bytes(jobs)
        gpu_estimate = estimate_worker_memory_bytes(jobs, device=gpu)
    assert gpu_estimate == cpu_estimate + orchestrator._GPU_HOST_OVERHEAD_BYTES
    # Measured: largest GPU worker 1235 MiB on a 181 MiB frame — the estimate must cover it.
    assert gpu_estimate >= 1235 * _MIB


def test_default_worker_count_on_a_gpu_is_also_capped_by_host_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(37)]
    lots_of_vram = ComputeDevice(kind="gpu", name="Fake", memory_free=48 * 2**30)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=4 * 2**30)
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=16):
        per_worker = estimate_worker_memory_bytes(jobs, device=lots_of_vram)
        assert default_worker_count(jobs, device=lots_of_vram) == (4 * 2**30) // per_worker
        assert default_worker_count(jobs, device=lots_of_vram) < default_worker_count(jobs)


def test_on_the_users_machine_the_gpu_default_is_the_benchmarked_four(tmp_path):
    # 8 physical cores, 7.6 GiB RAM and 6.58 GiB VRAM free: the benchmark ran 4 GPU workers with no
    # CPU fallbacks and ~1.2 GiB host memory each; 5 would leave too little RAM.
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(37)]
    gpu = ComputeDevice(kind="gpu", name="NVIDIA GeForce RTX 3070", memory_free=_USER_VRAM)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=_USER_RAM)
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=8):
        assert default_worker_count(jobs, device=gpu) == 4


def test_measured_device_memory_fits_the_device_estimate(tmp_path):
    # Benchmark, one 181 MiB frame: CuPy pool peak 878 MiB (--auto-density; 821 with a profile) plus
    # ~168 MiB CUDA context/library overhead = 1046 MiB. The estimate must cover that.
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=None)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME):
        assert estimate_worker_device_bytes([job]) >= (878 + 168) * _MIB


def test_memory_budget_warning_counts_gpu_host_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    gpu = ComputeDevice(kind="gpu", name="Fake", memory_free=_USER_VRAM)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(available=_USER_RAM)
    ):
        assert memory_budget_warning(jobs, requested_workers=6) is None  # fine for CPU workers
        assert memory_budget_warning(jobs, requested_workers=6, device=gpu) is not None


# --- The shared GPU service (docs/plans/gpu-batch-throughput.md Task B3) ---
#
# No GPU here: real-pool tests run a "gpu"-kind service whose own process installs the strict fake
# device (tests/unit/_fake_device.py's install_as_gpu) — so every frame really crosses into the
# service through shared memory, the service really runs the device code, and the result must still
# be bit-identical to the CPU (the fake computes with numpy). Frames are small: /dev/shm is 64 MiB in
# this sandbox.

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402

import psutil  # noqa: E402

from halide.batch.orchestrator import (  # noqa: E402
    BatchCompute,
    GpuService,
    batch_compute,
    estimate_service_host_bytes,
    estimate_service_worker_memory_bytes,
    service_budget_warnings,
    service_worker_count,
)
from halide.io.contact_sheet_defaults import DEFAULT_FRAME_WIDTH  # noqa: E402
from halide.io.tiff import read_tiff_description  # noqa: E402
from halide.processing import export_delivery_image, print_scan, process_scan  # noqa: E402
from tests.unit import _fake_device  # noqa: E402


def _probe_worker(job, *args):
    """orchestrator._worker, reporting what the worker process looked like afterwards: whether it
    had imported CuPy (it must never — no CUDA in a service-mode worker) and what it was handed."""
    result = _worker(job, *args)
    handed = type(args[-1]).__name__ if args else None
    return BatchResult(job=result.job, error=result.error,
                       warning=f"cupy={'cupy' in sys.modules};last_arg={handed};pid={os.getpid()};{result.warning}")


def _shm_entries() -> set:
    return set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()


@pytest.fixture
def clean_shm():
    """A test that runs a real pool must leave no shared-memory segment behind."""
    before = _shm_entries()
    yield
    assert _shm_entries() - before == set()


@pytest.fixture
def fake_gpu_service(monkeypatch):
    """batch_compute starts a real "gpu" service process whose GPU is the strict fake."""
    monkeypatch.setattr(orchestrator, "_SERVICE_INITIALIZER", _fake_device.install_as_gpu)
    starts = []
    real = orchestrator._start_service

    def counting(kind, initializer, cancel=None):
        starts.append(kind)
        return real(kind, initializer, cancel)

    monkeypatch.setattr(orchestrator, "_start_service", counting)
    return starts


def _no_service(monkeypatch):
    def refuse(kind, initializer, cancel=None):
        raise AssertionError("a service was started")

    monkeypatch.setattr(orchestrator, "_start_service", refuse)


def _provenance(path):
    return json.loads(read_tiff_description(path))["halide"]


def _roll(tmp_path, n, writer=_write_scan):
    folder = tmp_path / "in"
    folder.mkdir()
    return [writer(folder / f"f{i}.tif", seed=i + 1) for i in range(n)]


@pytest.mark.parametrize(
    "runner, args, expected",
    [
        (run_batch, (Stage.FULL, None, ToneCurveParams()),
         (Stage.FULL, None, ToneCurveParams(), DEFAULT_FRAME_WIDTH, "cpu", None)),
        (run_export_batch, (), (95, "cpu", None)),
        (run_print_batch, (ToneCurveParams(),), (ToneCurveParams(), "cpu", None)),
    ],
)
@pytest.mark.parametrize("device", [None, ComputeDevice(kind="cpu")])
def test_the_cpu_path_hands_workers_exactly_what_it_always_did(tmp_path, monkeypatch, runner, args, expected, device):
    _no_service(monkeypatch)
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    (fn, rest), = _run_recording(runner, job, *args, max_workers=2, device=device)
    assert rest == expected  # no service argument, nothing new


def test_the_cpu_path_is_plain_cpu_and_starts_nothing(monkeypatch):
    _no_service(monkeypatch)
    for device in (None, ComputeDevice(kind="cpu")):
        with batch_compute([], device) as compute:
            assert compute.mode == "cpu" and compute.service is None and compute.shm_prefix is None
            assert compute.worker_args(4) == ("cpu", None)


def test_a_gpu_batch_shares_one_service_and_no_worker_touches_cuda(tmp_path, monkeypatch, fake_gpu_service, clean_shm):
    scans = _roll(tmp_path, 4)
    out = tmp_path / "out"
    out.mkdir()
    jobs = [BatchJob(input_path=s, output_path=out / s.name) for s in scans]
    monkeypatch.setattr(orchestrator, "_worker", _probe_worker)
    results = run_batch(jobs, Stage.FULL, PROFILE, ToneCurveParams(), max_workers=2, device=_GPU)

    assert fake_gpu_service == ["gpu"]  # one service for the whole batch
    assert [r.error for r in results] == [None] * 4
    for r in results:
        cupy, handed, pid, warning = r.warning.split(";", 3)
        assert cupy == "cupy=False"  # the worker never imported CuPy, let alone made a CUDA context
        assert handed == "last_arg=GpuService"  # the service's address, not ("gpu", pool share)
        assert warning == "None"  # no frame fell back to the CPU
    for scan in scans:
        # Developed by the service (its device, the fake GPU, is recorded) and bit-identical to the
        # CPU, since the fake computes with numpy.
        assert _provenance(out / scan.name)["device"] == "gpu"
        process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
        np.testing.assert_array_equal(read_tiff_pixels(out / scan.name), read_tiff_pixels(tmp_path / "cpu.tif"))


def test_service_mode_export_and_print_match_the_in_process_gpu_path(tmp_path, fake_gpu_service, clean_shm, fake_gpu):
    # Compared with the in-process GPU path on the same (fake) device: the service runs the same
    # device code. (A device export may differ from the CPU's by one 8-bit code value — the device
    # branch's own sRGB encoding, docs/plans/gpu-acceleration.md D2 — so the CPU isn't the reference.)
    positives = _roll(tmp_path, 2, writer=_write_positive)
    exports = [BatchJob(input_path=p, output_path=tmp_path / f"{p.stem}.png") for p in positives]
    assert [r.error for r in run_export_batch(exports, max_workers=2, device=_GPU)] == [None, None]
    prints = [BatchJob(input_path=p, output_path=tmp_path / f"{p.stem}-print.tif") for p in positives]
    results = run_print_batch(prints, ToneCurveParams(), max_workers=2, device=_GPU)
    assert [(r.error, r.warning) for r in results] == [(None, None), (None, None)]
    assert fake_gpu_service == ["gpu", "gpu"]
    from tests.unit.test_device_pipeline import _png_pixels

    for p in positives:
        export_delivery_image(p, tmp_path / "in-process.png", device=fake_gpu)
        np.testing.assert_array_equal(_png_pixels(tmp_path / f"{p.stem}.png"), _png_pixels(tmp_path / "in-process.png"))
        print_scan(p, tmp_path / "in-process-print.tif", ToneCurveParams(), device=fake_gpu)
        np.testing.assert_array_equal(read_tiff_pixels(tmp_path / f"{p.stem}-print.tif"),
                                      read_tiff_pixels(tmp_path / "in-process-print.tif"))
        assert _provenance(tmp_path / f"{p.stem}-print.tif")["device"] == "gpu"


def test_when_the_service_dies_mid_batch_the_rest_develop_on_the_cpu(tmp_path, fake_gpu_service, clean_shm):
    scans = _roll(tmp_path, 4)
    out = tmp_path / "out"
    out.mkdir()
    jobs = [BatchJob(input_path=s, output_path=out / s.name) for s in scans]
    with batch_compute(jobs, _GPU) as compute:
        assert compute.mode == "service"
        pid = compute.service.address.pid

        def kill_after_the_first(result):
            if result.job == jobs[0]:
                os.kill(pid, signal.SIGKILL)

        # One worker, so the next frame is only handed out after the service is already dead.
        results = run_batch(jobs, Stage.FULL, PROFILE, ToneCurveParams(), max_workers=1, device=_GPU,
                            compute=compute, on_result=kill_after_the_first)
    assert [r.error for r in results] == [None] * 4  # the batch completed; no frame lost
    assert results[0].warning is None
    for r in results[1:]:
        assert r.warning and "GPU service" in r.warning and "on the CPU instead" in r.warning, [x.warning for x in results]
    for scan, expected_device in zip(scans, ["gpu", "cpu", "cpu", "cpu"]):
        assert _provenance(out / scan.name)["device"] == expected_device
        process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
        np.testing.assert_array_equal(read_tiff_pixels(out / scan.name), read_tiff_pixels(tmp_path / "cpu.tif"))


def test_the_service_is_off_below_python_3_13_with_the_reason(monkeypatch):
    """Finding 1, final whole-branch review: SharedMemory(..., track=False) — what
    shared_frames.attach_frame uses to attach to a service-mode worker's frame — is a Python 3.13
    addition; this project's own floor is 3.11 (pyproject.toml). Patch the gate helper itself
    (batch.orchestrator._service_python_supported), not sys.version_info, so this runs the same way
    regardless of the interpreter actually running the suite."""
    _no_service(monkeypatch)
    monkeypatch.setattr(orchestrator, "_service_python_supported", lambda: False)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "per_worker" and compute.service is None
        assert compute.fallback_reason == "the shared GPU service needs Python 3.13 or newer"
        assert compute.worker_args(2) == ("gpu", (6 * 2**30) // 2 - _CUDA_CONTEXT_BYTES)  # as before the service


def test_the_service_starts_on_a_supported_python_even_when_patched_true(monkeypatch, fake_gpu_service, clean_shm):
    monkeypatch.setattr(orchestrator, "_service_python_supported", lambda: True)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "service"
    assert fake_gpu_service == ["gpu"]


def test_a_service_that_wont_start_means_per_worker_gpu_mode_with_the_reason(tmp_path, monkeypatch):
    from tests.unit.test_gpu_service import _fail_at_startup

    monkeypatch.setattr(orchestrator, "_SERVICE_INITIALIZER", _fail_at_startup)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "per_worker" and compute.service is None
        assert "cudaErrorInsufficientDriver" in compute.fallback_reason
        assert compute.worker_args(2) == ("gpu", (6 * 2**30) // 2 - _CUDA_CONTEXT_BYTES)  # as before the service


@pytest.mark.parametrize("value", ["0", "off", " OFF ", "false", "no"])
def test_halide_gpu_service_off_means_per_worker_gpu_mode_and_starts_no_service(monkeypatch, value):
    _no_service(monkeypatch)
    monkeypatch.setenv("HALIDE_GPU_SERVICE", value)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "per_worker" and compute.service is None and compute.shm_prefix is None
        assert compute.fallback_reason == f"turned off by HALIDE_GPU_SERVICE={value.strip()}"
        assert compute.worker_args(2) == ("gpu", (6 * 2**30) // 2 - _CUDA_CONTEXT_BYTES)  # as before the service


@pytest.mark.parametrize("value", [None, "", "1", "on", "yes"])
def test_the_gpu_service_is_on_unless_halide_gpu_service_turns_it_off(monkeypatch, fake_gpu_service, clean_shm, value):
    if value is None:
        monkeypatch.delenv("HALIDE_GPU_SERVICE", raising=False)
    else:
        monkeypatch.setenv("HALIDE_GPU_SERVICE", value)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "service" and compute.fallback_reason is None
    assert fake_gpu_service == ["gpu"]


def test_halide_gpu_service_off_leaves_a_cpu_batch_alone(monkeypatch):
    _no_service(monkeypatch)
    monkeypatch.setenv("HALIDE_GPU_SERVICE", "0")
    with batch_compute([], ComputeDevice(kind="cpu")) as compute:
        assert compute.mode == "cpu" and compute.fallback_reason is None


@pytest.mark.parametrize(
    "runner, args",
    [
        (run_batch, (Stage.FULL, None, ToneCurveParams())),
        (run_export_batch, ()),
        (run_print_batch, (ToneCurveParams(),)),
    ],
)
def test_halide_gpu_service_off_hands_every_runners_workers_their_own_gpu(tmp_path, monkeypatch, fake_gpu_service,
                                                                         runner, args):
    # A service *could* start here (the fake-GPU initializer is set); the switch alone keeps it off.
    monkeypatch.setenv("HALIDE_GPU_SERVICE", "off")
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    (fn, rest), = _run_recording(runner, job, *args, max_workers=2, device=_GPU)
    assert rest[-2:] == ("gpu", (6 * 2**30) // 2 - _CUDA_CONTEXT_BYTES)  # per-worker mode's arguments
    assert fake_gpu_service == []


def test_too_little_shared_memory_means_per_worker_gpu_mode_with_the_reason(tmp_path, monkeypatch):
    _no_service(monkeypatch)
    monkeypatch.setattr(orchestrator, "_shared_memory_free", lambda: 64 * _MIB)  # a small container
    monkeypatch.setattr(orchestrator, "_shared_frame_bytes", lambda jobs, workload: _FRAME)
    with batch_compute([], _GPU) as compute:
        assert compute.mode == "per_worker"
        assert "shared memory" in compute.fallback_reason and "64 MiB" in compute.fallback_reason
        assert compute.worker_args(2)[0] == "gpu"


def test_shared_memory_caps_service_mode_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "_shared_memory_free", lambda: 1024 * _MIB)
    monkeypatch.setattr(orchestrator, "_shared_frame_bytes", lambda jobs, workload: _FRAME)
    assert orchestrator.shared_memory_worker_cap([]) == int(1024 * _MIB * 0.8) // _FRAME == 4
    monkeypatch.setattr(orchestrator, "_shared_memory_free", lambda: None)  # Windows: no separate limit
    assert orchestrator.shared_memory_worker_cap([]) is None


def test_shared_frame_bytes_is_the_float32_decode_plus_an_exports_8bit_output(tmp_path):
    path = tmp_path / "u16.tif"
    import tifffile

    tifffile.imwrite(path, np.zeros((10, 20, 3), dtype=np.uint16), photometric="rgb")
    job = BatchJob(input_path=path, output_path=None)
    assert orchestrator._shared_frame_bytes([job], "develop") == 10 * 20 * 3 * 4  # decoded as float32
    assert orchestrator._shared_frame_bytes([job], "export") == 10 * 20 * 3 * 5


_SERVICE = BatchCompute(device=_GPU, service=GpuService(address="unused", shm_prefix="p-"))


def test_service_mode_worker_count_has_no_vram_cap(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(16)]
    tiny_card = BatchCompute(device=ComputeDevice(kind="gpu", name="Fake", memory_free=10 * _MIB),
                             service=_SERVICE.service)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "halide.batch.orchestrator._shared_frame_bytes", return_value=_FRAME
    ), patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 2**30)), patch(
        "halide.batch.orchestrator._cpu_cap", return_value=16
    ):
        assert service_worker_count(jobs, tiny_card) == 16  # cores and jobs only: the card holds one frame


def test_service_mode_worker_count_is_ram_after_the_service_and_shared_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(37)]
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "halide.batch.orchestrator._shared_frame_bytes", return_value=_FRAME
    ), patch("halide.batch.orchestrator._cpu_cap", return_value=16):
        per_worker = estimate_service_worker_memory_bytes(jobs)
        # A CPU worker's estimate, plus its shared frame — no CUDA/CuPy libraries in the worker.
        assert per_worker == estimate_worker_memory_bytes(jobs) + _FRAME
        service = estimate_service_host_bytes(jobs)
        assert service == orchestrator._GPU_SERVICE_HOST_BYTES + _FRAME
        available = service + 3 * per_worker + per_worker // 2
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=available)):
            assert service_worker_count(jobs, _SERVICE) == 3
            capped = BatchCompute(device=_GPU, service=_SERVICE.service, shm_cap=2)
            assert service_worker_count(jobs, capped) == 2


def test_on_the_users_machine_service_mode_runs_more_workers_than_per_worker_gpu(tmp_path):
    # 8 physical cores, 7.6 GiB RAM: per-worker GPU mode fits 4 (~1.2 GiB each); CPU-only workers
    # sharing one service fit more — the point of Part B. (Constants measured and kept in B4.)
    jobs = [BatchJob(input_path=tmp_path / f"{i}.tif", output_path=None) for i in range(37)]
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "halide.batch.orchestrator._shared_frame_bytes", return_value=_FRAME
    ), patch("psutil.virtual_memory", return_value=SimpleNamespace(available=_USER_RAM)), patch(
        "halide.batch.orchestrator._cpu_cap", return_value=8
    ):
        gpu = ComputeDevice(kind="gpu", name="NVIDIA GeForce RTX 3070", memory_free=_USER_VRAM)
        assert default_worker_count(jobs, device=gpu) == 4
        assert service_worker_count(jobs, BatchCompute(device=gpu, service=_SERVICE.service)) > 4


def test_service_budget_warnings_name_ram_and_shared_memory(tmp_path):
    jobs = [BatchJob(input_path=tmp_path / "a.tif", output_path=None)]
    capped = BatchCompute(device=_GPU, service=_SERVICE.service, shm_cap=2)
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "halide.batch.orchestrator._shared_frame_bytes", return_value=_FRAME
    ), patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 2**30)):
        assert service_budget_warnings(jobs, 2, capped) == []
        (warning,) = service_budget_warnings(jobs, 3, capped)
        assert warning.startswith("--workers 3") and "shared memory" in warning and "CPU" in warning
    with patch("halide.batch.orchestrator._decoded_pixel_bytes", return_value=_FRAME), patch(
        "halide.batch.orchestrator._shared_frame_bytes", return_value=_FRAME
    ), patch("psutil.virtual_memory", return_value=SimpleNamespace(available=2 * 2**30)):
        (warning,) = service_budget_warnings(jobs, 2, _SERVICE)
        assert "memory" in warning


def test_a_worker_with_no_room_in_shared_memory_develops_that_frame_on_the_cpu(tmp_path, monkeypatch):
    import halide.shared_frames
    from halide.shared_frames import SharedMemoryUnavailable

    def full(*args, **kwargs):
        raise SharedMemoryUnavailable("no room for a 23532-byte shared-memory segment")

    monkeypatch.setattr(halide.shared_frames, "new_frame", full)
    monkeypatch.setattr(orchestrator, "_WORKER_CLIENTS", {})
    scan = _write_scan(tmp_path / "neg.tif")
    # The service is never reached (its client only connects on a first request), so any address does.
    result = _worker(BatchJob(input_path=scan, output_path=tmp_path / "out.tif"), Stage.FULL, PROFILE,
                     ToneCurveParams(), 64, "cpu", None, _SERVICE.service)
    assert result.error is None
    assert "no room in shared memory" in result.warning and "on the CPU instead" in result.warning
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    np.testing.assert_array_equal(read_tiff_pixels(tmp_path / "out.tif"), read_tiff_pixels(tmp_path / "cpu.tif"))
    assert _provenance(tmp_path / "out.tif")["device"] == "cpu"


class _OrderedExecutor(_FakeExecutor):
    def __init__(self, futures_by_job, events):
        super().__init__(futures_by_job)
        self._events = events

    def __exit__(self, *exc_info):
        self._events.append("pool shut down")
        return False

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_frames_are_swept_only_after_the_pool_has_shut_down(tmp_path, monkeypatch, interrupt):
    import halide.shared_frames

    events = []
    monkeypatch.setattr(halide.shared_frames, "sweep", lambda prefix: events.append(f"sweep {prefix}"))
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    future: Future = Future()
    future.set_result(BatchResult(job=job, error=None))

    def on_result(result):
        events.append("result")
        if interrupt:
            raise KeyboardInterrupt

    with patch("halide.batch.orchestrator.ProcessPoolExecutor", lambda *a, **k: _OrderedExecutor({job: future}, events)):
        orchestrator._run_pool([job], _worker, (), 1, on_result=on_result, shm_prefix="halide-1-abc-")
    assert events == ["result", "pool shut down", "sweep halide-1-abc-"]


def test_no_sweep_without_shared_frames(tmp_path, monkeypatch):
    import halide.shared_frames

    monkeypatch.setattr(halide.shared_frames, "sweep", lambda prefix: pytest.fail("swept"))
    job = BatchJob(input_path=tmp_path / "a.tif", output_path=tmp_path / "b.tif")
    future: Future = Future()
    future.set_result(BatchResult(job=job, error=None))
    with patch("halide.batch.orchestrator.ProcessPoolExecutor", lambda *a, **k: _FakeExecutor({job: future})):
        orchestrator._run_pool([job], _worker, (), 1)


def test_the_service_stops_and_its_frames_are_swept_when_the_batch_ends(tmp_path, fake_gpu_service, clean_shm):
    with batch_compute([], _GPU) as compute:
        pid = compute.service.address.pid
        assert psutil.pid_exists(pid)
        # A worker that died holding a frame would leave one like this behind.
        from multiprocessing import shared_memory

        leftover = shared_memory.SharedMemory(create=True, size=64, name=f"{compute.shm_prefix}dead")
        leftover.close()
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert f"{compute.shm_prefix}dead" not in _shm_entries()


def test_sweep_runs_after_the_service_has_actually_stopped(monkeypatch, fake_gpu_service, clean_shm):
    """Finding 2, final whole-branch review: batch_compute's sweep callback must unwind *after* the
    service's own context manager has stopped it (ExitStack is LIFO: the callback registered first
    unwinds last) — not before, which is what the code did despite its own comment claiming
    otherwise. Verified directly here, not just by inspection: record whether the service process is
    already gone at the moment sweep actually runs."""
    import halide.shared_frames

    service_pid = []
    seen_service_gone_at_sweep = []
    real_sweep = halide.shared_frames.sweep

    def recording_sweep(prefix):
        pid = service_pid[0]
        gone = not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        seen_service_gone_at_sweep.append(gone)
        return real_sweep(prefix)

    monkeypatch.setattr(halide.shared_frames, "sweep", recording_sweep)
    with batch_compute([], _GPU) as compute:
        service_pid.append(compute.service.address.pid)
    assert seen_service_gone_at_sweep == [True]


def test_decoded_pixel_bytes_reads_what_read_tiff_decodes(tmp_path):
    # Two pages stacked as one series: read_tiff decodes the whole series (read_tiff_shape), not
    # just the first page.
    import tifffile

    from halide.io.tiff import read_tiff_shape

    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros((2, 8, 8, 3), dtype=np.float32), photometric="rgb")
    shape = read_tiff_shape(path)
    assert orchestrator._decoded_pixel_bytes(path) == int(np.prod(shape)) * 4


# --- Exit paths (F12, review 2.4-2 and 2.4-9): stale frames, SIGHUP/SIGTERM, early Ctrl-C --------

import subprocess  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402

from multiprocessing import shared_memory  # noqa: E402
from pathlib import Path  # noqa: E402


def _stale_name(pid: int) -> str:
    from halide.shared_frames import _base36

    return f"hl{_base36(pid)}00f00d-{time.time_ns():x}"[:30]


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 13), reason="/dev/shm sweep, Python 3.13+")
def test_a_gpu_batch_first_sweeps_frames_a_killed_batch_left_behind(fake_gpu_service, clean_shm):
    """A whole-group SIGKILL (or the OOM killer taking the parent) leaves its frames in /dev/shm —
    RAM — with nothing left alive to sweep them. The next GPU batch removes them before it starts."""
    stale = shared_memory.SharedMemory(create=True, size=4096, name=_stale_name(_dead_pid()),
                                       track=False)
    try:
        with batch_compute([], _GPU) as compute:
            assert compute.mode == "service"
            assert stale.name not in _shm_entries()
    finally:
        stale.close()


# The real CLI in a child process, as a terminal runs it. SERVICE_MODE: its GPU is a stand-in (a
# "cpu" service process behind a "gpu" device), so a batch goes through the shared GPU service and
# /dev/shm exactly as on a real card.
_CLI = textwrap.dedent(
    """\
    import signal, sys
    # A background process started by a non-interactive shell (or pytest) may inherit Ctrl-C ignored;
    # a terminal's foreground job has Python's own handler.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if sys.argv[1] == "service":
        import halide.cli._device_args as device_args
        import halide.device as device
        from halide.batch import orchestrator

        def fake(requested=None, isolated=False):
            wanted = device.requested_device(requested)
            return device.ComputeDevice(kind="cpu") if wanted == "cpu" else device.ComputeDevice(kind="gpu", name="Fake")

        device.resolve_device = device_args.resolve_device = fake
        orchestrator._SERVICE_KIND = "cpu"
    from halide.cli.main import run_cli
    sys.exit(run_cli(sys.argv[2:]))
    """
)


def _start_cli(tmp_path, mode, frames, shape):
    roll = tmp_path / "in"
    roll.mkdir()
    for i in range(frames):
        _write_scan(roll / f"f{i:02d}.tif", shape=shape, seed=i + 1)
    env = dict(os.environ, HALIDE_NO_COMPLETION="1", HALIDE_DEVICE="gpu" if mode == "service" else "cpu")
    process = subprocess.Popen(
        [sys.executable, "-c", _CLI, mode, "batch", str(roll), str(tmp_path / "out"),
         "--rm", "0.9", "--bm", "1.1", "--rs", "1", "--bs", "1", "--workers", "2", "--quiet"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, env=env,
    )
    return process


def _group_gone(pgid, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(p.info["pid"] != pgid and _pgid(p.info["pid"]) == pgid for p in psutil.process_iter(["pid"])):
            return True
        time.sleep(0.05)
    return False


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs POSIX process groups")
@pytest.mark.parametrize("delay", [0.3, 0.6, 0.9])
def test_ctrl_c_early_in_a_batch_prints_cancelled_not_tracebacks(tmp_path, delay):
    """2.4-9: Ctrl-C in the first second of a batch (a terminal sends it to the whole process group)
    used to interrupt the forkserver's preload of colour/scipy and print pages of tracebacks."""
    process = _start_cli(tmp_path, "cpu", frames=6, shape=(300, 450, 3))
    time.sleep(delay)
    os.killpg(process.pid, signal.SIGINT)
    out, err = process.communicate(timeout=60)
    assert "Traceback" not in err and "Traceback" not in out, err
    assert "Cancelled" in out + err
    assert process.returncode == 130
    assert _group_gone(process.pid)


@pytest.mark.skipif(sys.platform != "linux" or sys.version_info < (3, 13), reason="service mode: /dev/shm, Python 3.13+")
@pytest.mark.parametrize("signum, whole_group", [(signal.SIGHUP, True), (signal.SIGTERM, False), (signal.SIGINT, True)],
                         ids=["terminal closed", "SIGTERM", "Ctrl-C"])
def test_a_signalled_gpu_batch_cancels_in_order_and_leaves_nothing_behind(tmp_path, signum, whole_group):
    """F12 (review 2.4-2): closing the terminal (SIGHUP to the whole group) mid-batch used to kill
    every process that could clean up, leaving the shared frames in /dev/shm — RAM — until reboot.
    SIGHUP and SIGTERM now take Ctrl-C's orderly path: the frames in progress finish, the service
    stops, the shared frames are swept."""
    process = _start_cli(tmp_path, "service", frames=6, shape=(600, 900, 3))
    from halide.shared_frames import _base36
    ours = f"hl{_base36(process.pid)}"
    deadline = time.monotonic() + 60
    while not any(name.startswith(ours) for name in _shm_entries()):  # a frame is in flight
        assert process.poll() is None and time.monotonic() < deadline, process.communicate()
        time.sleep(0.01)
    if whole_group:
        os.killpg(process.pid, signum)
    else:
        os.kill(process.pid, signum)
    out, err = process.communicate(timeout=60)
    assert "Traceback" not in err and "Traceback" not in out, err
    assert "Cancelled" in out + err
    assert process.returncode == 130
    assert _group_gone(process.pid)
    assert not any(name.startswith(ours) for name in _shm_entries())
    outputs = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert not any("partial" in name for name in outputs)  # frames in progress finished, whole
    counted = int(re.search(r"Cancelled - (\d+) of 6", out + err).group(1))
    assert len(outputs) == counted  # and every frame written is counted


def test_after_a_cancel_the_wait_for_frames_in_progress_is_announced_once():
    """Ruling R9: while the frames in progress finish after Ctrl-C, the caller is told why the batch
    hasn't stopped yet — and only when there is something to wait for."""
    job = BatchJob(input_path=Path("a.tif"), output_path=Path("b.tif"))
    future: Future = Future()
    future.set_result(BatchResult(job=job, error=None))
    notices, recorded = [], []
    orchestrator._finish_in_flight({future: job}, recorded.append, [], notices.append)
    assert notices == [orchestrator.FINISHING_NOTICE]
    assert orchestrator.FINISHING_NOTICE == "Finishing the frames in progress — Ctrl-C again to stop now"
    assert [r.job for r in recorded] == [job]
    notices.clear()
    orchestrator._finish_in_flight({}, recorded.append, [], notices.append)
    assert notices == []
