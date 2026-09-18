import numpy as np
import pytest

from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.invert import invert


def test_invert_is_reciprocal():
    x = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    assert invert(x) == pytest.approx(1.0 / x)


def test_invert_is_unbounded_above_one():
    # A dense (low-transmittance) pixel must produce a large positive value, not something
    # clipped into [0, 1] — that clipping is exactly the bug this rewrite removes.
    x = np.array([0.001])
    assert invert(x)[0] > 100


def test_invert_clamps_zero_and_negative_input():
    x = np.array([0.0, -1.0])
    result = invert(x)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0 / MIN_TRANSMITTANCE)
