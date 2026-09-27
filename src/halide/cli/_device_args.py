"""Shared `--device` CLI argument, resolution, and run-sheet row — used by every command that
develops a frame (or will once later tasks wire batch/contact/calibrate workers up to it; see
CLAUDE.md/docs/plans/gpu-acceleration.md Task 5-7).

Deliberately tiny and importable at parser-build time: `halide.device` itself imports no heavy
dependency (not even cupy) at module level, so pulling in `resolve_device`/`ComputeDevice` here
doesn't cost `halide --help` anything (see tests/unit/test_cli_startup.py).
"""

from __future__ import annotations

import argparse

from halide.device import DEFAULT_DEVICE, ComputeDevice, DeviceUnavailableError, resolve_device

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
    "resolve_device_arg",
]


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help="where to do the arithmetic: auto (GPU if one is usable; default), cpu, or gpu (fail "
        "if unusable). Also HALIDE_DEVICE.",
    )


def resolve_device_arg(args: argparse.Namespace) -> ComputeDevice:
    """Resolve --device/$HALIDE_DEVICE into the device this run actually uses. A bad value (a
    mistyped $HALIDE_DEVICE — --device itself is constrained by argparse's `choices`) or an
    explicit --device gpu that isn't usable becomes the CLI's usual clean error exit rather than a
    traceback or a stack of CUDA internals."""
    try:
        return resolve_device(getattr(args, "device", None))
    except (ValueError, DeviceUnavailableError) as exc:
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
    just the base text, not the fallback reason (a caller shows that separately, e.g. as a
    sheet.warn) — kept separate so a later task (6b) can grow this into `"CPU — NVIDIA GeForce
    RTX 3070 found; add GPU support with: halide gpu --install"` without touching every caller."""
    if device.kind == "gpu":
        return f"GPU — {device.name}"
    return "CPU"
