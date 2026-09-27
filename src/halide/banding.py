"""Apply a per-pixel function to an image a band of rows at a time, writing into a buffer the caller
owns — the memory-saving half of every full-resolution code path (see processing.py).

Why this exists: every core/ stage is a pure whole-array function, and several of them allocate
full-frame temporaries internally (colour-science's float64 conversions, Cube1D.lookup's
interpolation buffers). On a real 181 MiB scan that put a single frame's peak at 1.3-2.6 GiB —
7-14 frames' worth of scratch memory — which is what limited how many frames a batch could
develop at once. Mapped over ~4 MiB bands, the same functions' temporaries are band-sized instead.

Why it can't change the output: every function mapped this way is per-pixel (each output pixel
depends only on the same input pixel), so computing it on a slice of rows gives exactly the values
the whole-array call gives. Whole-frame statistics — the print fit's percentiles, auto calibration
— are never banded; callers run them on the full buffer between banded passes. Verified bit-for-bit
against the whole-array pipeline on real scans at band sizes from 1 to 1000 rows, and pinned by
tests/unit/test_banding.py.

Lives outside core/ because it writes into a buffer — core/ stays pure functions only.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

# ~64 rows of a real full-resolution scan. A byte budget rather than a row count, so portrait and
# landscape frames (or small test images) get the same working-set size. Larger bands measurably
# cost memory (1000 rows: ~713 MiB peak vs ~372 MiB at 64) for no speed gain.
_BAND_BYTES = 4 * 1024**2

# The band budget for a frame resident on a GPU (processing.py's device path). Larger than the
# host's: each band is a kernel launch per stage, and a GPU needs big launches to be busy, while its
# memory only has to hold the frame plus a few bands' temporaries (docs/plans/gpu-acceleration.md,
# §3.3: ~181 MiB frame + ~6 x 64 MiB). Banding stays bit-identical at any size (module docstring).
DEVICE_BAND_BYTES = 64 * 2**20


def band_slices(n_rows: int, row_bytes: int, band_bytes: int | None = None) -> Iterator[slice]:
    """Row slices covering [0, n_rows) exactly once, each about `band_bytes` (default _BAND_BYTES,
    read at call time) and at least one row."""
    if band_bytes is None:
        band_bytes = _BAND_BYTES
    rows_per_band = max(1, band_bytes // max(1, row_bytes))
    for start in range(0, n_rows, rows_per_band):
        yield slice(start, min(start + rows_per_band, n_rows))


def map_in_bands(
    src, fn: Callable, out=None, band_bytes: int | None = None
):
    """`out[band] = fn(src[band])` for every band of rows; returns `out`. `out=None` writes back
    into `src` itself — only for a buffer the caller owns. `fn` must be per-pixel (see module
    docstring) and return an array of the band's shape; assignment casts it to `out`'s dtype.

    `src`/`out` may be numpy arrays or device (CuPy) arrays — only slicing, `.shape` and `.dtype`
    are used. `band_bytes` defaults to _BAND_BYTES; the device path passes DEVICE_BAND_BYTES."""
    if out is None:
        out = src
    # From shape and dtype rather than `.nbytes`, which not every array type offers.
    row_bytes = int(np.prod(src.shape[1:], dtype=np.int64)) * src.dtype.itemsize
    for band in band_slices(src.shape[0], row_bytes, band_bytes):
        out[band] = fn(src[band])
    return out
