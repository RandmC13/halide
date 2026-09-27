"""Suite-wide fixtures.

`DEFAULT_DEVICE` ("auto", `halide.device.DEFAULT_DEVICE`) is live on every command as of the
gpu-acceleration work (`cli/_device_args.py`), so any CLI test that doesn't pass `--device` would
otherwise resolve "auto" against *this machine's* real state. That's fine in this sandbox (CPU
only), but on a developer's own machine with a working NVIDIA card — e.g. the user's RTX 3070,
which runs this same suite — "auto" would actually take the GPU path. The GPU path is allowed to
differ from the CPU path within the D2 tolerance (docs/plans/gpu-acceleration.md §4: float32
pixels within 1e-5 relative), so an exact-pixel/exact-provenance test that assumes the CPU path
could fail there for a reason that has nothing to do with whatever's actually being tested — and a
developer's own stray `$HALIDE_DEVICE` in their shell would have the same effect. Forcing
`HALIDE_DEVICE=cpu` for every test makes the suite's device choice part of the suite's own
environment, not the machine's or the shell's.

Tests that are actually about device *selection* (tests/unit/test_device.py,
tests/unit/test_device_cli.py) override this within the test as needed —
monkeypatch.setenv/delenv inside a test wins over this fixture, since both use the same per-test
`monkeypatch` instance. Tests that build/inject a `ComputeDevice` directly
(tests/unit/test_device_pipeline.py's fake-GPU fixture, tests/gpu/test_gpu_parity.py's `_DEVICE`)
never consult `$HALIDE_DEVICE` at all, so they're unaffected either way; test_gpu_parity.py in
particular resolves its device from the literal `"auto"` at import time (before any fixture runs),
so it still targets a real GPU on a machine that has one.
"""

import pytest


@pytest.fixture(autouse=True)
def _default_device_is_cpu_in_tests(monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "cpu")
