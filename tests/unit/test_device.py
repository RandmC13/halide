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
from halide.device import DEFAULT_DEVICE, DEVICE_ENV, ComputeDevice, DeviceUnavailableError, resolve_device


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
