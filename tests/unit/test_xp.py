"""core/ runs on numpy or a GPU array library with one implementation (core/_xp.py). There is no GPU
here, so the device path runs on tests/unit/_fake_device.py: numpy underneath, but strict the way
CuPy is — any stage that reaches for numpy directly raises, and the result must be bit-identical to
the CPU path."""

import numpy as np
import pytest

from halide.core._xp import array_namespace, register_namespace
from halide.core.pipeline import negative_to_positive
from halide.core.tone_render import (
    ResolvedTone,
    _DEFAULT_CURVE_PATH,
    _load_curve,
    apply_tone,
    estimate_linear_scale,
    negative_density_range,
)
from halide.core.types import DensityProfile
from tests.unit._fake_device import FakeDeviceArray, fake_xp, to_device, to_host

PROFILE = DensityProfile(white_balance=(1.3, 1.0, 0.7), density_scale=(0.9, 1.0, 1.15))


@pytest.fixture(autouse=True)
def _fake():
    register_namespace(FakeDeviceArray, fake_xp)


def _negative(shape=(33, 17, 3)):
    return np.random.default_rng(2).uniform(1e-3, 0.9, shape).astype(np.float32)


def test_numpy_input_uses_numpy():
    assert array_namespace(np.zeros(3)) is np


def test_registered_type_uses_its_namespace():
    assert array_namespace(to_device(np.zeros(3))) is fake_xp


def test_negative_to_positive_on_device_equals_cpu():
    neg = _negative()
    assert np.array_equal(to_host(negative_to_positive(to_device(neg), PROFILE)), negative_to_positive(neg, PROFILE))


def test_print_fit_statistics_on_device_equal_cpu():
    pos = negative_to_positive(_negative(), PROFILE)
    assert negative_density_range(to_device(pos)) == negative_density_range(pos)
    assert estimate_linear_scale(to_device(pos)) == estimate_linear_scale(pos)


@pytest.mark.parametrize(
    "resolved", [ResolvedTone("paper", exposure=0.4, contrast=0.8), ResolvedTone("linear", linear_scale=0.05)]
)
def test_apply_tone_on_device_equals_cpu(resolved):
    pos = negative_to_positive(_negative(), PROFILE)
    assert np.array_equal(to_host(apply_tone(to_device(pos), resolved)), apply_tone(pos, resolved))


def test_curve_lookup_on_device_equals_cpu_and_keeps_host_table():
    curve = _load_curve(str(_DEFAULT_CURVE_PATH))
    x = np.random.default_rng(3).uniform(-3, 3, (40, 9)).astype(np.float32)
    for _ in range(2):  # the second call reuses the uploaded table
        assert np.array_equal(to_host(curve.lookup(to_device(x))), curve.lookup(x))
    assert type(curve.values) is np.ndarray  # the curve itself stays a host array for fit_print


# The fake is only a useful witness if it really refuses what CuPy refuses.
def test_fake_device_refuses_numpy_leaks():
    a = to_device(np.ones(3, np.float32))
    with pytest.raises(TypeError):
        np.asarray(a)
    with pytest.raises(TypeError):
        np.maximum(a, 1.0)
    with pytest.raises(TypeError):
        np.percentile(a, 50)
    with pytest.raises(TypeError):
        a * np.ones(3, np.float32)
    with pytest.raises(TypeError):
        np.ones(3, np.float32) * a
    with pytest.raises(TypeError):
        fake_xp.maximum(a, np.ones(3, np.float32))
    # A profile tuple must be uploaded with xp.asarray, as CuPy requires — not multiplied in raw.
    with pytest.raises(TypeError):
        a * (1.0, 2.0, 3.0)
    with pytest.raises(TypeError):
        (1.0, 2.0, 3.0) * a
    with pytest.raises(TypeError):
        a *= [1.0, 2.0, 3.0]
    with pytest.raises(TypeError):
        fake_xp.maximum(a, [1, 2, 3])


def test_fake_device_allows_scalars_and_returns_device_reductions():
    a = to_device(np.arange(4, dtype=np.float32))
    b = a * np.float32(2) + 1.0
    b *= 3
    assert isinstance(b, FakeDeviceArray)
    median = fake_xp.percentile(b, 50)
    assert isinstance(median, FakeDeviceArray) and median.ndim == 0
    assert fake_xp.percentile(b, [10, 90]).shape == (2,)  # CuPy takes a plain sequence for q
    assert isinstance(fake_xp.asarray((1.0, 2.0, 3.0), dtype=np.float32) * a[:3], FakeDeviceArray)
    assert float(median) == float(np.percentile(np.arange(4, dtype=np.float32) * 2 * 3 + 3, 50))


def test_core_never_imports_cupy():
    # CuPy is optional and slow to import; _xp only recognises it once a caller has imported it.
    import subprocess
    import sys

    code = "import sys, halide.core.pipeline, halide.io.lut; sys.exit('cupy' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
