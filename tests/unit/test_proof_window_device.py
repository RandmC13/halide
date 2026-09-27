"""The picker's contact-sheet window hands its resolved device to batch's workers (Task 7).

gui/proof_window.py needs Qt, which can't load in every environment the suite runs in (this
sandbox has no libglib), and the part under test — ProofRenderer.run's pool plumbing — uses none of
Qt's behaviour. So the module is loaded under a private name against a stub of the few PySide6
names it imports, leaving the real `halide.gui.proof_window` (and PySide6, where it exists)
untouched. The executor is a fake; what's checked is what each worker is handed and how many there
are, as tests/unit/test_orchestrator.py does for run_batch.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import halide.gui
from halide.batch.orchestrator import _CUDA_CONTEXT_BYTES, BatchResult, _worker
from halide.core.types import DensityProfile, ToneCurveParams
from halide.device import ComputeDevice


class _QThread:
    def __init__(self, parent=None):
        pass

    def isInterruptionRequested(self):
        return False


def _qt_stub(name):
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: MagicMock(name=f"{name}.{attr}")
    return module


@pytest.fixture
def proof_window(monkeypatch):
    core = _qt_stub("PySide6.QtCore")
    core.QThread = _QThread
    core.Signal = lambda *types_: MagicMock()
    for name, module in {
        "PySide6": _qt_stub("PySide6"),
        "PySide6.QtCore": core,
        "PySide6.QtGui": _qt_stub("PySide6.QtGui"),
        "PySide6.QtWidgets": _qt_stub("PySide6.QtWidgets"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(halide.gui.__file__).parent / "proof_window.py"
    spec = importlib.util.spec_from_file_location("_proof_window_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Executor:
    def __init__(self, submitted, **kwargs):
        self.kwargs = kwargs
        self._submitted = submitted

    def submit(self, fn, job, *rest):
        self._submitted.append((fn, rest))
        future = Future()
        future.set_result(BatchResult(job=job, error="not really developed"))
        return future

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize(
    "device, expected",
    [
        (ComputeDevice(kind="gpu", name="Fake", memory_free=6 * 2**30, memory_total=8 * 2**30),
         ("gpu", 6 * 2**30 // 2 - _CUDA_CONTEXT_BYTES)),
        (ComputeDevice(kind="cpu"), ("cpu", None)),
        (None, ("cpu", None)),
    ],
)
def test_proof_renderer_develops_on_the_pickers_device(proof_window, monkeypatch, tmp_path, device, expected):
    submitted, counted = [], []

    def default_worker_count(jobs, **kwargs):
        counted.append(kwargs.get("device"))
        return 2

    monkeypatch.setattr(proof_window, "default_worker_count", default_worker_count)
    monkeypatch.setattr(proof_window, "ProcessPoolExecutor", lambda **kw: _Executor(submitted, **kw))
    profile = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))
    renderer = proof_window.ProofRenderer(
        [tmp_path / "a.tif", tmp_path / "b.tif"], [1.0, 1.0], profile, ToneCurveParams(), device
    )
    renderer.run()
    assert counted == [device]  # the default worker count knows a GPU run (VRAM cap)
    assert [fn for fn, _ in submitted] == [_worker, _worker]
    assert all(rest[-2:] == expected for _, rest in submitted)


# --- The GPU service is started and stopped with the window's pool (Task B3) ---
#
# A real pool and a real service process: a "gpu" service whose GPU is the strict fake, or a "cpu"
# one whose develop never finishes (to close the window while a frame is in the service's hands).

import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import psutil  # noqa: E402

import halide.batch.orchestrator as orchestrator  # noqa: E402
from tests.unit import _fake_device  # noqa: E402
from tests.unit.test_device_pipeline import PROFILE, _write_scan  # noqa: E402
from tests.unit.test_gpu_service import _slow_develop_in_child  # noqa: E402

_GPU = ComputeDevice(kind="gpu", name="Fake", memory_free=6 * 2**30, memory_total=8 * 2**30)


def _shm() -> set:
    return set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()


@pytest.fixture
def services(monkeypatch):
    """Every service the window starts: its address (pid) as it starts."""
    started = []
    real = orchestrator._start_service

    def recording(kind, initializer):
        from contextlib import contextmanager

        @contextmanager
        def run():
            with real(kind, initializer) as address:
                started.append(address)
                yield address

        return run()

    monkeypatch.setattr(orchestrator, "_start_service", recording)
    return started


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


class _RecordingPool(orchestrator.ProcessPoolExecutor):
    """The real pool, remembering its worker processes (it forgets them on shutdown)."""

    seen: list = []

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        _RecordingPool.seen.extend(p for p in self._processes.values() if p not in _RecordingPool.seen)
        return future


def test_the_window_runs_its_frames_through_one_service_and_stops_it(proof_window, monkeypatch, tmp_path, services):
    monkeypatch.setattr(orchestrator, "_SERVICE_INITIALIZER", _fake_device.install_as_gpu)
    before = _shm()
    scans = [_write_scan(tmp_path / f"f{i}.tif", seed=i + 1) for i in range(3)]
    renderer = proof_window.ProofRenderer(scans, [1.0] * 3, PROFILE, ToneCurveParams(), _GPU)
    emitted = []
    renderer.frameDone = MagicMock(emit=lambda *args: emitted.append(args))
    renderer.run()
    assert sorted(index for index, *_ in emitted) == [0, 1, 2]
    assert all(error is None and record["device"] == "gpu" for _, _, record, error in emitted)
    (address,) = services
    assert _gone(address.pid)  # stopped with the pool
    assert _shm() - before == set()


def test_closing_the_window_mid_render_stops_the_workers_and_the_service(proof_window, monkeypatch, tmp_path,
                                                                        services):
    # The service holds a frame and never answers: the window is closed while it's in its hands.
    arrived = tmp_path / "arrived"
    monkeypatch.setattr(orchestrator, "_SERVICE_KIND", "cpu")
    monkeypatch.setattr(orchestrator, "_SERVICE_INITIALIZER", partial(_slow_develop_in_child, str(arrived)))
    monkeypatch.setattr(proof_window, "ProcessPoolExecutor", _RecordingPool)
    _RecordingPool.seen = []
    before = _shm()
    scans = [_write_scan(tmp_path / f"f{i}.tif", seed=i + 1) for i in range(2)]
    renderer = proof_window.ProofRenderer(scans, [1.0] * 2, PROFILE, ToneCurveParams(), _GPU)
    renderer.frameDone = MagicMock()
    closed = threading.Event()
    renderer.isInterruptionRequested = closed.is_set
    thread = threading.Thread(target=renderer.run)
    thread.start()
    deadline = time.monotonic() + 60
    while not arrived.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert arrived.exists()
    assert _shm() - before  # a worker's frame is in shared memory, in the service's hands
    started = time.monotonic()
    closed.set()  # closeEvent's requestInterruption
    thread.join(30)
    assert not thread.is_alive()
    assert time.monotonic() - started < 15  # stopped, not waited out (the request would take 60 s)
    renderer.frameDone.emit.assert_not_called()
    (address,) = services
    assert _gone(address.pid)
    assert _RecordingPool.seen and all(not p.is_alive() for p in _RecordingPool.seen)
    assert _shm() - before == set()  # the killed workers' frames were swept
