"""Which array library an array belongs to — numpy, or CuPy for a frame on the GPU — so core/'s
functions are written once and run on either (see docs/plans/gpu-acceleration.md). Pure: never
imports CuPy; a CuPy array can only exist if the caller already imported it."""

from __future__ import annotations

import sys

import numpy as np

_REGISTERED: dict[type, object] = {}


def register_namespace(array_type: type, namespace) -> None:
    """Tests only: route arrays of `array_type` to `namespace` (tests/unit/_fake_device.py's strict
    CPU stand-in for a GPU library)."""
    _REGISTERED[array_type] = namespace


def array_namespace(a):
    """The module whose functions operate on `a`: `cupy` for a CuPy array, else numpy."""
    for array_type, namespace in _REGISTERED.items():
        if isinstance(a, array_type):
            return namespace
    # Looked up, never imported: importing CuPy costs start-up time and needs a GPU driver.
    cupy = sys.modules.get("cupy")
    if cupy is not None and isinstance(a, cupy.ndarray):
        return cupy
    return np
