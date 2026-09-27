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
