"""Minimal reader for 1D .cube LUTs (the format both reference repos ship their curves in).

Only 1D LUTs are supported — that's all this project needs (the print-paper response curve, and
the reference LUTs used in golden tests). A 3D LUT reader is not needed anywhere in this pipeline,
since color-space conversion is handled by explicit matrices (see io/icc.py), not baked LUTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from halide.core._xp import array_namespace


@dataclass(frozen=True)
class Cube1D:
    values: np.ndarray  # shape (N,) — every reference curve used here is channel-identical
    domain_min: float
    domain_max: float

    def _table(self, xp, dtype) -> np.ndarray:
        """`values` in `xp`'s memory and `dtype`, made once per (namespace, dtype) rather than per
        call — on a GPU that is one upload per curve, not one per band of rows. `self.values`
        itself stays a host numpy array (fit_print reads it). Cached in the instance's __dict__,
        not a field, so equality/repr are unchanged; set with object.__setattr__ because the
        dataclass is frozen."""
        cache = self.__dict__.get("_tables")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_tables", cache)
        key = (id(xp), np.dtype(dtype))
        if key not in cache:
            host = self.values if self.values.dtype == dtype else self.values.astype(dtype)
            cache[key] = xp.asarray(host)  # numpy: the same array back, no copy
        return cache[key]

    def lookup(self, x: np.ndarray) -> np.ndarray:
        """Linearly interpolate this LUT at x (any shape). Inputs outside
        [domain_min, domain_max] are clamped to the LUT's endpoints — the same floor/ceiling a
        physical curve (e.g. photographic paper) has.

        Follows `x`'s own dtype throughout (used to hardcode float64/int64 regardless of caller
        dtype) — on a full-size image array, forcing float64 here silently doubled the memory of
        this function's several image-shaped temporaries even when the rest of the pipeline had
        carefully stayed in float32. `x`'s own precision already bounds the result's accuracy, so
        this changes nothing about correctness, only which dtype the arithmetic runs in.
        """
        xp = array_namespace(x)
        x = xp.asarray(x)
        size = self.values.shape[0]
        values = self._table(xp, x.dtype)

        t = x - self.domain_min  # fresh array — safe to keep mutating in place from here
        t *= 1.0 / (self.domain_max - self.domain_min)
        xp.clip(t, 0.0, 1.0, out=t)
        t *= size - 1
        floor_t = xp.floor(t)
        lo = floor_t.astype(xp.int32)
        t -= floor_t  # t now holds the interpolation fraction
        hi = lo + 1
        xp.minimum(hi, size - 1, out=hi)

        # lo + frac * (hi - lo), algebraically identical to lo*(1-frac) + hi*frac but reuses the
        # two gathered buffers in place instead of allocating separate product/sum temporaries.
        gathered_lo = values[lo]
        gathered_hi = values[hi]
        gathered_hi -= gathered_lo
        gathered_hi *= t
        gathered_lo += gathered_hi
        return gathered_lo


def load_1d_cube(path: str | Path) -> Cube1D:
    """Load a 1D .cube LUT. If the file stores 3 identical columns (as every curve currently
    vendored in this project does), only the first column is kept."""
    domain_min, domain_max = 0.0, 1.0
    rows: list[list[float]] = []

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("LUT_1D_SIZE"):
            continue  # size is inferred from the row count, not needed separately
        if line.startswith("LUT_3D_SIZE"):
            raise ValueError(f"{path} is a 3D LUT; load_1d_cube only supports 1D LUTs")
        if line.startswith("DOMAIN_MIN"):
            domain_min = float(line.split()[1])
            continue
        if line.startswith("DOMAIN_MAX"):
            domain_max = float(line.split()[1])
            continue
        if line.startswith("TITLE"):
            continue
        rows.append([float(v) for v in line.split()])

    table = np.asarray(rows, dtype=np.float64)
    values = table[:, 0]
    if not np.allclose(table, values[:, np.newaxis]):
        raise ValueError(f"{path}: expected identical R/G/B columns, got channel-varying data")
    return Cube1D(values=values, domain_min=domain_min, domain_max=domain_max)
