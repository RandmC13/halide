"""Shared `--device` CLI argument, resolution, and run-sheet row — used by every command that
develops a frame (see docs/plans/gpu-acceleration.md Tasks 5-7). `contact` takes the flag too, but
its thumbnails of already-developed frames stay on the CPU, so it resolves only an explicit `gpu`
(to fail fast, like every command) and never probes for `auto`.

Deliberately tiny and importable at parser-build time: `halide.device` itself imports no heavy
dependency (not even cupy) at module level, so pulling in `resolve_device`/`ComputeDevice` here
doesn't cost `halide --help` anything (see tests/unit/test_cli_startup.py).
"""

from __future__ import annotations

import argparse

from halide.device import DEFAULT_DEVICE, ComputeDevice, DeviceUnavailableError, requested_device, resolve_device

# Re-exported from halide.device (the single place that actually acts on it — resolve_device falls
# back to this same value when neither --device nor $HALIDE_DEVICE is given) so CLI code, and a
# later "halide gpu" command/hint, can import the one true default from here without reaching past
# the CLI package. R3 (docs/plans/gpu-acceleration.md §6 D3): the user chose "auto" as the one
# default used by every command, once GPU support exists at all.
__all__ = [
    "DEFAULT_DEVICE",
    "add_device_argument",
    "device_fallback_warning",
    "device_row",
    "gpu_hint",
    "requested_device_arg",
    "resolve_device_arg",
]


def _device_value(value: str) -> str:
    """`--device GPU` / `--device " cpu"` mean what they say, as $HALIDE_DEVICE does."""
    return value.strip().lower()


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        type=_device_value,
        default=None,
        help="where to do the arithmetic: auto (GPU if one is usable; default), cpu, or gpu (fail "
        "if unusable). Also HALIDE_DEVICE.",
    )


def resolve_device_arg(args: argparse.Namespace, *, isolated: bool = False) -> ComputeDevice:
    """Resolve --device/$HALIDE_DEVICE into the device this run actually uses. A bad value (a
    mistyped $HALIDE_DEVICE — --device itself is constrained by argparse's `choices`) or an
    explicit --device gpu that isn't usable becomes the CLI's usual clean error exit rather than a
    traceback or a stack of CUDA internals.

    `isolated`: for a batch's parent process, which only needs to know what card there is — the
    GPU service or the workers do the computing — the probe runs in a child process, so no CUDA
    context exists in the parent (see halide.device.resolve_device)."""
    try:
        return resolve_device(getattr(args, "device", None), **({"isolated": True} if isolated else {}))
    except (ValueError, DeviceUnavailableError) as exc:
        raise SystemExit(str(exc)) from exc


def requested_device_arg(args: argparse.Namespace) -> str:
    """What --device/$HALIDE_DEVICE asks for (`"auto" | "cpu" | "gpu"`), without resolving it —
    for a command that only needs to know whether a GPU was explicitly requested (`contact`). A
    bad $HALIDE_DEVICE is the same clean error exit as resolve_device_arg's."""
    try:
        return requested_device(getattr(args, "device", None))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def device_fallback_warning(device: ComputeDevice) -> str | None:
    """Why an `auto` choice ended up on the CPU after trying for a GPU — for a caller to show
    alongside the Compute row (a `sheet.warn`, or a plain warning line for a command with no run
    sheet). None when there's nothing to say: a real GPU, or no CuPy installed at all (the normal,
    not-a-problem case — see `ComputeDevice.fallback_reason`'s own docstring)."""
    if device.fallback_reason:
        return f"GPU not usable, using CPU instead: {device.fallback_reason}"
    return None


def device_row(device: ComputeDevice) -> str:
    """The run sheet's Compute row text (label added by the caller — `sheet.row("Compute", ...)`,
    or a plain print for a command with no run sheet): `"CPU"` or `"GPU — <name>"`. Deliberately
    just the base text, not the fallback reason or the GPU-available hint (a caller shows those
    separately — `device_fallback_warning`/`gpu_hint` below — since `invert`'s once-per-machine
    hint needs to show the hint text without repeating it on every run's summary line)."""
    if device.kind == "gpu":
        return f"GPU — {device.name}"
    return "CPU"


def gpu_hint(device: ComputeDevice) -> str | None:
    """Only when it's worth mentioning: CuPy isn't installed, but the machine actually has an
    NVIDIA card (Task 6b, docs/plans/gpu-acceleration.md §3.7). None when there's nothing to add —
    a real GPU is already in use, or there's no card at all (nobody without an NVIDIA card should
    be pitched a ~1 GB download). Shared, word for word, by two surfaces that show it differently:
    `cli/_run_sheet.py::compute_row` appends it to the Compute row on every batch/print/export run,
    while `cli/commands/invert_cmd.py` prints it as its own line once per machine (invert has no
    run sheet, and would otherwise repeat it on every single-frame run)."""
    from halide.device import detect_nvidia_driver, gpu_support_installed

    if device.kind == "gpu" or gpu_support_installed():
        return None
    driver = detect_nvidia_driver()
    if driver is None:
        return None
    name = driver.device_name or "An NVIDIA GPU"
    return f"{name} found; add GPU support with: halide gpu --install"
