"""Regression tests against .cube LUTs vendored from abpy/color-neg-resources (MIT licensed, see
tests/golden/luts/LICENSE-abpy.txt) — these encode the reference implementation's exact expected
input->output mappings, independent of anything in this codebase.

Note on the vendored 0.01 constant: `inverse_01.cube` bakes the reference's ad hoc display-range
scalar into the LUT itself (positive = 0.01 / negative). This project's core.invert.invert() is
deliberately the bare reciprocal with no such constant (see core/invert.py's docstring for why) —
so the comparison below reintroduces the 0.01 factor only in the test, to confirm the underlying
reciprocal relationship matches exactly while keeping that magic number out of shipped code.
"""

from pathlib import Path

import numpy as np
import pytest

from halide.core.invert import invert
from halide.io.lut import load_1d_cube

LUTS_DIR = Path(__file__).parent / "luts"


def _domain_samples(cube, skip: int = 5):
    """Sample indices/x-values from a loaded cube, skipping the first few entries: LUTs generated
    near x=0 clamp to their own floor value, which need not match this project's MIN_TRANSMITTANCE
    floor exactly, so we compare where both are safely away from that clamp."""
    size = cube.values.shape[0]
    idx = np.arange(skip, size)
    x = cube.domain_min + idx / (size - 1) * (cube.domain_max - cube.domain_min)
    return idx, x


def test_invert_matches_reference_inverse_lut():
    cube = load_1d_cube(LUTS_DIR / "inverse_01.cube")
    idx, x = _domain_samples(cube)
    expected = cube.values[idx]
    actual = invert(x) * 0.01
    assert actual == pytest.approx(expected, rel=1e-3)


def test_invert_implies_reference_density_lut():
    """log10(invert(x)) == log10(1/x), the exact density formula `density.cube` encodes."""
    cube = load_1d_cube(LUTS_DIR / "density.cube")
    idx, x = _domain_samples(cube)
    expected = cube.values[idx]
    actual = np.log10(invert(x))
    assert actual == pytest.approx(expected, rel=1e-3)
