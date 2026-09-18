"""Minimal reader for 1D .cube LUTs (the format both reference repos ship their curves in).

Only 1D LUTs are supported — that's all this project needs (the print-paper response curve, and
the reference LUTs used in golden tests). A 3D LUT reader is not needed anywhere in this pipeline,
since color-space conversion is handled by explicit matrices (see io/icc.py), not baked LUTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Cube1D:
    values: np.ndarray  # shape (N,) — every reference curve used here is channel-identical
    domain_min: float
    domain_max: float

    def lookup(self, x: np.ndarray) -> np.ndarray:
        """Linearly interpolate this LUT at x (any shape). Inputs outside
        [domain_min, domain_max] are clamped to the LUT's endpoints — the same floor/ceiling a
        physical curve (e.g. photographic paper) has."""
        size = self.values.shape[0]
        t = (np.asarray(x, dtype=np.float64) - self.domain_min) / (self.domain_max - self.domain_min)
        t = np.clip(t, 0.0, 1.0)
        idx = t * (size - 1)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, size - 1)
        frac = idx - lo
        return self.values[lo] * (1 - frac) + self.values[hi] * frac


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
