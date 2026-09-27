"""A CPU stand-in for a GPU array library, strict in the ways CuPy is, so the GPU plumbing can be
tested on a machine with no GPU.

Computes with numpy underneath, so results must be bit-identical to the CPU path — any difference
is a plumbing bug (a stage that skipped `xp`, or a dtype that changed on the way), not device
arithmetic.

What it emulates, because each is a real way CuPy code breaks:
  - no implicit conversion to numpy: `np.asarray(a)`, `np.array(a)` and any numpy module function
    (`np.maximum(a, …)`, `np.percentile(a, …)`) raise — a stage must call `xp.<name>`;
  - no mixing with a plain numpy array: an operator or `fake_xp` function given one raises — it has
    to be uploaded with `xp.asarray` first. Python and numpy *scalars* mix freely, as with CuPy;
  - reductions return a 0-d device array, not a host scalar (CuPy's `percentile`, `median`, …),
    which `float()`/`bool()` bring back to the host;
  - operators, in-place operators, indexing, `.astype`, `.reshape`, `.shape`/`.ndim`/`.dtype` work
    on the array itself, since CuPy arrays support them natively.

What it deliberately doesn't: asynchronous execution, device memory limits, CuPy's own dtype
promotion quirks or its subset of numpy's functions — `fake_xp` exposes only the names core/
needs, and adding one here is not evidence CuPy has it.

It is a wrapper around an ndarray, not an ndarray subclass: numpy converts a subclass to a plain
array silently (`np.asarray`), which would hide exactly the leaks this exists to catch.
"""

from __future__ import annotations

import operator
import types

import numpy as np


class FakeDeviceArray:
    # Makes numpy defer every ufunc and binary operator involving this type to its own methods
    # (or raise), instead of converting it: `ndarray * fake` lands in __rmul__, which rejects it.
    __array_ufunc__ = None

    def __init__(self, data: np.ndarray):
        self._data = data

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit conversion of a device array to numpy — use to_host / xp")

    shape = property(lambda self: self._data.shape)
    ndim = property(lambda self: self._data.ndim)
    dtype = property(lambda self: self._data.dtype)
    size = property(lambda self: self._data.size)

    def astype(self, dtype, copy=True):
        return _wrap(self._data.astype(dtype, copy=copy))

    def reshape(self, *shape):
        return _wrap(self._data.reshape(*shape))

    def copy(self):
        return _wrap(self._data.copy())

    def __getitem__(self, key):
        return _wrap(self._data[_unwrap(key)])

    def __setitem__(self, key, value):
        self._data[_unwrap(key)] = _unwrap(value)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return (_wrap(row) for row in self._data)

    # Host round-trips a 0-d result makes explicitly, as with CuPy (each synchronises there).
    def __float__(self):
        return float(self._data)

    def __int__(self):
        return int(self._data)

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return f"FakeDeviceArray({self._data!r})"


def _binary(op):
    def forward(self, other):
        return _wrap(op(self._data, _unwrap(other)))

    def reflected(self, other):
        return _wrap(op(_unwrap(other), self._data))

    return forward, reflected


def _inplace(op):
    def method(self, other):
        # The ufunc writes into this array's own buffer — same aliasing as numpy's `a *= b`.
        op(self._data, _unwrap(other), out=self._data)
        return self

    return method


for _name, _op in {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "truediv": np.true_divide,
    "floordiv": np.floor_divide, "pow": np.power, "matmul": np.matmul,
    "lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
    "eq": np.equal, "ne": np.not_equal, "and": np.bitwise_and, "or": np.bitwise_or,
}.items():
    _fwd, _ref = _binary(_op)
    setattr(FakeDeviceArray, f"__{_name}__", _fwd)
    if _name not in ("lt", "le", "gt", "ge", "eq", "ne"):
        setattr(FakeDeviceArray, f"__r{_name}__", _ref)
    if _name not in ("lt", "le", "gt", "ge", "eq", "ne", "matmul"):
        setattr(FakeDeviceArray, f"__i{_name}__", _inplace(_op))
FakeDeviceArray.__neg__ = lambda self: _wrap(operator.neg(self._data))
FakeDeviceArray.__hash__ = None  # defines __eq__ elementwise, like numpy/CuPy arrays


def _wrap(x):
    if isinstance(x, np.ndarray):
        return FakeDeviceArray(x)
    if isinstance(x, np.generic):  # a reduction's result stays on the device, 0-d
        return FakeDeviceArray(np.asarray(x))
    if isinstance(x, tuple):
        return tuple(_wrap(v) for v in x)
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _unwrap(x):
    if isinstance(x, FakeDeviceArray):
        return x._data
    if isinstance(x, np.ndarray):
        raise TypeError("plain numpy array mixed with a device array — upload it with xp.asarray")
    if isinstance(x, tuple):
        return tuple(_unwrap(v) for v in x)
    if isinstance(x, list):
        return [_unwrap(v) for v in x]
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _namespaced(fn):
    def call(*args, **kwargs):
        return _wrap(fn(*_unwrap(args), **_unwrap(kwargs)))

    call.__name__ = fn.__name__
    return call


def _upload(fn):
    # `xp.asarray`/`xp.array` are the one door from the host: they accept plain numpy arrays.
    def call(a, *args, **kwargs):
        host = a._data if isinstance(a, FakeDeviceArray) else a
        return FakeDeviceArray(fn(host, *args, **kwargs))

    call.__name__ = fn.__name__
    return call


fake_xp = types.SimpleNamespace(
    **{
        name: _namespaced(getattr(np, name))
        for name in ("maximum", "minimum", "power", "divide", "log10", "clip", "floor", "percentile")
    },
    asarray=_upload(np.asarray),
    array=_upload(np.array),
    float32=np.float32,
    float64=np.float64,
    int32=np.int32,
    __name__="fake_xp",
)


def to_device(a: np.ndarray) -> FakeDeviceArray:
    return FakeDeviceArray(np.array(a, copy=True))


def to_host(a: FakeDeviceArray) -> np.ndarray:
    if not isinstance(a, FakeDeviceArray):
        raise TypeError(f"expected a device array back, got {type(a).__name__}")
    return np.array(a._data, copy=True)
