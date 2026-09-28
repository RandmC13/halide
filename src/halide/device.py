"""Which compute device halide runs on: the CPU (always available) or an NVIDIA GPU via CuPy (an
optional install — see CLAUDE.md's "GPU support is optional" priority and
docs/plans/gpu-acceleration.md). This module is the single place that decides; nothing else should
import cupy directly or guess whether it works.

Deliberately does not import cupy at module level, or even for a `"cpu"` request: cupy is a ~1 GB
optional dependency, and `halide --help`/every CPU-only run must stay fast and dependency-free (see
CLAUDE.md's "Starting the CLI imports no numpy, tifffile, Pillow or colour-science" entry, and
`tests/unit/test_cli_startup.py`, which this module's callers must keep passing). `resolve_device`
imports cupy lazily, only when asked for anything other than `"cpu"`.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from dataclasses import dataclass

DEVICE_ENV = "HALIDE_DEVICE"

# R3 (docs/plans/gpu-acceleration.md §6 D3): the user's one default, used by every command, once
# neither --device nor $HALIDE_DEVICE says otherwise. Defined here (not just in cli/_device_args.py,
# which re-exports it for CLI callers) because this is the single place that actually acts on it.
DEFAULT_DEVICE = "auto"

_VALID_REQUESTS = ("auto", "cpu", "gpu")


@dataclass(frozen=True)
class ComputeDevice:
    """The device a run actually used, or will use. `fallback_reason` is set only when `"auto"`
    wanted a GPU and couldn't get one on a machine where CuPy is installed (a driver mismatch, a
    dead card, ...) — it's None both when the device really is a GPU and when CuPy simply isn't
    installed at all (the ordinary, not-a-problem case), so callers can tell "nothing to report"
    from "here's why we didn't use your GPU"."""

    kind: str  # "cpu" | "gpu"
    name: str | None = None  # e.g. "NVIDIA GeForce RTX 3070"
    memory_free: int | None = None
    memory_total: int | None = None
    fallback_reason: str | None = None


class DeviceUnavailableError(Exception):
    """Raised when the user explicitly asked for `--device gpu` and it isn't usable. The message
    is written for a film photographer, not a programmer: what's wrong and the exact command that
    fixes it, not a CUDA error code."""


def requested_device(requested: str | None = None) -> str:
    """What was asked for — `"auto" | "cpu" | "gpu"` — without resolving it (no CuPy, no CUDA):
    `requested` if given, else `$HALIDE_DEVICE`, else DEFAULT_DEVICE. Case and surrounding space
    are ignored, and an empty (or all-space) `$HALIDE_DEVICE` counts as unset, so
    `HALIDE_DEVICE= halide ...` isn't an error. Anything else raises ValueError."""
    if requested is None:
        requested = os.environ.get(DEVICE_ENV, "").strip() or DEFAULT_DEVICE
    normalised = requested.strip().lower()
    if normalised not in _VALID_REQUESTS:
        raise ValueError(
            f"{DEVICE_ENV}/--device is {requested!r}, but it must be one of {', '.join(_VALID_REQUESTS)}"
        )
    return normalised


def resolve_device(requested: str | None = None) -> ComputeDevice:
    """Decide which device to run on. `requested` is `"auto" | "cpu" | "gpu"`; if None, falls back
    to the `HALIDE_DEVICE` environment variable, then to `"auto"` (see requested_device).

    `"cpu"` never touches CuPy — it can't fail and returns immediately. `"auto"` and `"gpu"` both
    probe for a real, working GPU the same way (import CuPy, count devices, run a tiny kernel);
    they differ only in what happens when that probe fails: `"auto"` quietly falls back to CPU
    (recording why only if CuPy was installed at all — the normal case, no CuPy, gets no reason to
    show), `"gpu"` raises so the user's explicit request isn't silently downgraded.
    """
    requested = requested_device(requested)
    if requested == "cpu":
        return ComputeDevice(kind="cpu")

    try:
        return _probe_gpu()
    except Exception as exc:  # broad on purpose — see _probe_gpu's docstring
        if requested == "gpu":
            raise DeviceUnavailableError(_install_hint(exc)) from exc
        # "auto": a GPU is a bonus, not a requirement. Only say why it's missing when CuPy is
        # actually installed — a plain ImportError just means the normal, GPU-less case.
        reason = None if isinstance(exc, ImportError) else str(exc)
        return ComputeDevice(kind="cpu", fallback_reason=reason)


def _probe_gpu() -> ComputeDevice:
    """Try, end to end, to actually use a GPU: import CuPy, ask it for a device, and run a real
    kernel rather than trusting `getDeviceCount` alone — a stale/mismatched driver can report a
    device that then fails on first use. Any failure anywhere in here (CuPy's own exception types
    live in `cupy_backends`, and a broken install can raise plain `RuntimeError`/`OSError` instead)
    is treated the same way by the caller, so this catches broadly and lets it propagate."""
    # Imported here, not at module level — see the module docstring.
    import cupy

    cupy.cuda.runtime.getDeviceCount()
    # A real kernel, not just a device count: a broken driver often reports a device that then
    # fails on first actual use (this sandbox's cudaErrorInsufficientDriver, for instance).
    (cupy.arange(4, dtype=cupy.float32) ** 1.5).sum().item()
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    memory_free, memory_total = cupy.cuda.Device(0).mem_info
    return ComputeDevice(kind="gpu", name=name, memory_free=memory_free, memory_total=memory_total)


def _install_hint(exc: Exception) -> str:
    if isinstance(exc, ImportError):
        return (
            "GPU support isn't installed. Run 'halide gpu --install' to add it, or use "
            "--device cpu/auto."
        )
    return f"the GPU isn't usable ({exc}). Run 'halide gpu --install' to check/fix it, or use --device cpu/auto."


def to_device(a):
    """Upload a numpy array to the current GPU. Only meaningful once `resolve_device` has returned
    a `"gpu"` device — callers on a `"cpu"` device never call this."""
    import cupy  # see module docstring

    return cupy.asarray(a)


def to_host(a, out=None):
    """Download a CuPy array back to numpy, optionally into a caller-owned buffer (`out`) so a
    banded pipeline (see `banding.py`) can reuse one host array instead of allocating per band."""
    return a.get(out=out)


@dataclass(frozen=True)
class NvidiaDriver:
    """What halide can learn about an NVIDIA driver *without* CuPy installed at all — just enough
    to tell a user with a card that GPU support exists, and which CuPy build fits it. `cuda_version`
    is exactly `cuDriverGetVersion`'s own integer form (e.g. `13040` = CUDA 13.4: `major*1000 +
    minor*10`)."""

    cuda_version: int
    device_name: str | None = None  # the first device's name, or None if that lookup itself failed


def detect_nvidia_driver() -> NvidiaDriver | None:
    """Look for an NVIDIA driver directly via `ctypes` — no CuPy, no subprocess, and no parsing
    `nvidia-smi`'s output (its header format changes between driver versions; the user's own reads
    "CUDA UMD Version", not "CUDA Version"). This runs in the main process only (see `halide gpu`
    and the CLI's GPU hint), and only bothers when CuPy itself isn't already importable — most
    people running this have no NVIDIA card at all, so "not found" must be the fast, silent, normal
    result, and any surprise here (a stub library, a driver returning nonsense) must come back as
    that same "not found" rather than a traceback.

    Loads `libcuda.so.1` (Linux) / `nvcuda.dll` (Windows) and calls `cuDriverGetVersion`, then
    `cuInit`/`cuDeviceGet`/`cuDeviceGetName` for the first device's name — each guarded by its own
    CUresult check (0 = success), so a driver that answers the version call but not the name call
    still gets its version reported."""
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("nvcuda.dll")
        elif sys.platform.startswith("linux"):
            lib = ctypes.CDLL("libcuda.so.1")
        else:
            return None

        version = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
        cuda_version = version.value

        device_name: str | None = None
        device = ctypes.c_int(0)
        if lib.cuInit(0) == 0 and lib.cuDeviceGet(ctypes.byref(device), 0) == 0:
            name_buf = ctypes.create_string_buffer(256)
            if lib.cuDeviceGetName(name_buf, 256, device) == 0:
                device_name = name_buf.value.decode("utf-8", errors="replace")
        return NvidiaDriver(cuda_version=cuda_version, device_name=device_name)
    except Exception:  # noqa: BLE001 -- see docstring: this is a "does a card exist" probe, never fatal
        return None


def cupy_package_for(driver: NvidiaDriver) -> str | None:
    """Which pip extra (`pyproject.toml`'s `cuda12`/`cuda13`) fits this driver's CUDA version, or
    None if the driver is too old for either CuPy build currently packaged (needs CUDA 12+)."""
    if driver.cuda_version >= 13000:
        return "cupy-cuda13x[ctk]"
    if driver.cuda_version >= 12000:
        return "cupy-cuda12x[ctk]"
    return None


def gpu_support_installed() -> bool:
    """Whether CuPy is importable, without actually importing it: `import cupy` alone costs ~0.2s
    even before touching a device (see the module docstring's speed priority), so this is the check
    used on every command's run-sheet Compute row and the once-per-machine invert hint, not just
    `halide gpu`."""
    return importlib.util.find_spec("cupy") is not None


def release_memory() -> None:
    """Hand CuPy's cached device memory back to the driver. CuPy keeps freed blocks in a pool for
    reuse, which is right for a batch worker developing frame after frame (Task 7 keeps the pool
    warm), but a one-frame run — or a frame that just ran out of GPU memory and is about to be
    redone on the CPU — has no use for the cache. A no-op when CuPy was never imported (nothing to
    release, and importing it here would cost exactly what the module docstring avoids); never
    raises, since it runs on the way out of an error path."""
    import sys

    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    try:
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 — best effort; a broken driver is already being reported
        pass
