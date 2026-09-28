"""Which compute device halide runs on (`halide/device.py`). CuPy is an optional dependency (see
CLAUDE.md/docs/plans/gpu-acceleration.md), so `resolve_device` must behave correctly whether or not
it's installed, and whether or not the machine it's installed on actually has a working GPU driver
— the three real cases seen in practice: no CuPy at all, CuPy installed with a working card, and
CuPy installed but the driver is missing/too old (this sandbox: CuPy is installed, but the first
CUDA call raises `cudaErrorInsufficientDriver`).

Every scenario here monkeypatches `sys.modules["cupy"]` with a fake rather than relying on the
sandbox's real CuPy/driver state, so these tests pass on any machine (with or without a GPU).
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

import halide.device as device_module
from halide.device import (
    DEFAULT_DEVICE,
    DEVICE_ENV,
    ComputeDevice,
    DeviceUnavailableError,
    requested_device,
    resolve_device,
)


def _fake_cupy(*, device_count=1, device_count_error=None, device_name="NVIDIA GeForce RTX 3070",
                mem_free=4 * 2**30, mem_total=8 * 2**30):
    """A minimal fake of the bits of CuPy `resolve_device` touches: `cuda.runtime.getDeviceCount`,
    a smoke kernel (`cupy.arange(...).sum()`), `getDeviceProperties`, and `cuda.Device(0).mem_info`.
    """
    cupy = types.ModuleType("cupy")

    def getDeviceCount():
        if device_count_error is not None:
            raise device_count_error
        return device_count

    def getDeviceProperties(index):
        return {"name": device_name.encode()}

    class _Device:
        def __init__(self, index):
            self.index = index

        @property
        def mem_info(self):
            return (mem_free, mem_total)

    runtime = types.SimpleNamespace(getDeviceCount=getDeviceCount, getDeviceProperties=getDeviceProperties)
    cupy.cuda = types.SimpleNamespace(runtime=runtime, Device=_Device)

    def arange(n, dtype=None):
        import numpy as np

        return np.arange(n, dtype=dtype)

    cupy.arange = arange
    cupy.float32 = __import__("numpy").float32
    return cupy


def test_cpu_request_never_imports_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import would raise
    assert resolve_device("cpu").kind == "cpu"


def test_auto_without_cupy_is_cpu_with_no_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert resolve_device("auto") == ComputeDevice(kind="cpu")


def test_auto_with_broken_driver_falls_back_and_says_why(monkeypatch):
    error = RuntimeError(
        "cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version"
    )
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count_error=error))
    device = resolve_device("auto")
    assert device.kind == "cpu" and "driver" in device.fallback_reason


def test_auto_with_working_gpu_returns_gpu_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    device = resolve_device("auto")
    assert device == ComputeDevice(
        kind="gpu", name="NVIDIA GeForce RTX 3070", memory_free=4 * 2**30, memory_total=8 * 2**30
    )


def test_gpu_request_with_working_gpu_returns_gpu_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    assert resolve_device("gpu").kind == "gpu"


def test_gpu_request_without_cupy_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(DeviceUnavailableError, match=r"halide gpu --install"):
        resolve_device("gpu")


def test_gpu_request_with_broken_driver_raises_with_reason(monkeypatch):
    error = RuntimeError(
        "cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version"
    )
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(device_count_error=error))
    with pytest.raises(DeviceUnavailableError, match="driver"):
        resolve_device("gpu")


def test_env_var_is_used_when_no_flag(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "cpu")
    assert resolve_device(None).kind == "cpu"


def test_no_flag_no_env_defaults_to_auto(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert resolve_device(None) == ComputeDevice(kind="cpu")


def test_default_device_constant_is_auto():
    assert DEFAULT_DEVICE == "auto"


def test_default_device_constant_actually_governs_resolve_device(monkeypatch):
    """Not just documentation: resolve_device must read this constant at call time, not have its
    own separately hardcoded "auto" fallback that this constant merely happens to match."""
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    # A GPU "auto" would happily pick, so switching DEFAULT_DEVICE to "cpu" is the only thing that
    # can make resolve_device(None) come back CPU here — proving the constant is actually consulted,
    # not just documented to match resolve_device's own separate literal.
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(device_module, "DEFAULT_DEVICE", "cpu")
    assert resolve_device(None) == ComputeDevice(kind="cpu")


def test_bad_env_value_is_an_error(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "tpu")
    with pytest.raises(ValueError, match="HALIDE_DEVICE"):
        resolve_device(None)


def test_bad_requested_value_is_an_error(monkeypatch):
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_empty_env_value_counts_as_unset(monkeypatch, value):
    # `HALIDE_DEVICE= halide invert ...` (or an exported-but-empty variable) must not be an error.
    monkeypatch.setenv(DEVICE_ENV, value)
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert requested_device(None) == DEFAULT_DEVICE
    assert resolve_device(None) == ComputeDevice(kind="cpu")


@pytest.mark.parametrize("value", ["CPU", " cpu ", "Cpu\n"])
def test_env_value_ignores_case_and_surrounding_space(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "cupy", None)  # "cpu" must never touch CuPy
    monkeypatch.setenv(DEVICE_ENV, value)
    assert requested_device(None) == "cpu"
    assert resolve_device(None).kind == "cpu"


def test_requested_value_ignores_case_and_surrounding_space(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert requested_device(" GPU ") == "gpu"
    assert resolve_device(" CPU").kind == "cpu"
    with pytest.raises(DeviceUnavailableError):
        resolve_device("Gpu")


def test_requested_device_rejects_a_bad_value_naming_the_variable(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "tpu")
    with pytest.raises(ValueError, match="HALIDE_DEVICE"):
        requested_device(None)


def test_importing_device_does_not_import_cupy():
    code = "import sys, halide.device; print('cupy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_importing_processing_does_not_import_cupy():
    code = "import sys, halide.processing; print('cupy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


class _FakeCupyArray:
    """Stands in for a real `cupy.ndarray` just enough to test `to_device`/`to_host`'s plumbing:
    `.get(out=...)` is CuPy's own device-to-host transfer method."""

    def __init__(self, data):
        self._data = data

    def get(self, out=None):
        import numpy as np

        if out is not None:
            np.copyto(out, self._data)
            return out
        return self._data.copy()


def _fake_cupy_for_transfer():
    cupy = types.ModuleType("cupy")
    cupy.asarray = lambda a: _FakeCupyArray(a)
    return cupy


def test_to_device_calls_cupy_asarray(monkeypatch):
    import numpy as np

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_for_transfer())
    from halide.device import to_device

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    on_device = to_device(a)
    assert isinstance(on_device, _FakeCupyArray)
    assert np.array_equal(on_device._data, a)


def test_to_host_calls_get_and_supports_out(monkeypatch):
    import numpy as np

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_for_transfer())
    from halide.device import to_device, to_host

    a = np.array([1.0, 2.0], dtype=np.float32)
    on_device = to_device(a)

    assert np.array_equal(to_host(on_device), a)

    out = np.empty_like(a)
    result = to_host(on_device, out=out)
    assert result is out
    assert np.array_equal(out, a)


def _fake_probe_child(monkeypatch, stdout):
    """Stand in for the child process of an isolated probe; the in-process probe must never run."""
    monkeypatch.setattr(device_module, "_probe_gpu", lambda: pytest.fail("the parent probed the GPU itself"))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=stdout))


def test_an_isolated_probe_learns_the_card_without_touching_cupy_here(monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    _fake_probe_child(monkeypatch, '{"name": "RTX", "memory_free": 5, "memory_total": 8}\n')
    device = device_module.resolve_device("auto", isolated=True)
    assert (device.kind, device.name, device.memory_free, device.memory_total) == ("gpu", "RTX", 5, 8)
    assert "cupy" not in sys.modules  # no CuPy, so no CUDA context, in this process


def test_an_isolated_probe_reports_failures_like_the_in_process_one(monkeypatch):
    _fake_probe_child(monkeypatch, '{"error": "cudaErrorInsufficientDriver", "import_error": false}\n')
    auto = device_module.resolve_device("auto", isolated=True)
    assert auto.kind == "cpu" and auto.fallback_reason == "cudaErrorInsufficientDriver"
    with pytest.raises(DeviceUnavailableError, match="cudaErrorInsufficientDriver"):
        device_module.resolve_device("gpu", isolated=True)


def test_a_cupy_that_is_installed_but_cannot_be_imported_gives_its_error_not_none(monkeypatch):
    monkeypatch.setattr(device_module, "gpu_support_installed", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", None)  # `import cupy` raises ImportError
    reason = device_module.resolve_device("auto").fallback_reason
    assert reason and reason != "None" and "cupy" in reason
    with pytest.raises(DeviceUnavailableError, match="installed but couldn't be loaded"):
        device_module.resolve_device("gpu")


def test_no_cupy_at_all_is_still_silent(monkeypatch):
    monkeypatch.setattr(device_module, "gpu_support_installed", lambda: False)
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert device_module.resolve_device("auto").fallback_reason is None
