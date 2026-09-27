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
    # Likewise a developer's own HALIDE_GPU_SERVICE=0 (the troubleshooting switch back to per-worker
    # GPU mode) mustn't change which mode the service tests get; tests that want it set it.
    monkeypatch.delenv("HALIDE_GPU_SERVICE", raising=False)


@pytest.fixture(autouse=True)
def _no_real_gpu_service_in_tests(request, monkeypatch):
    """The same reasoning for batches: on a GPU `ComputeDevice` (tests build fake ones freely),
    `batch.orchestrator.batch_compute` starts the shared GPU service, which resolves the machine's
    *real* GPU in its own process — on the user's RTX 3070 a test meant for plumbing would develop
    on the card. So a real "gpu" service is refused here, as a service that couldn't start (the
    batch then runs in per-worker GPU mode, which tests drive with fakes as before). Tests about the
    service itself ask for a "cpu" one, or one whose child installs the fake GPU (an initializer),
    and those still start.

    Tests marked `gpu` (tests/gpu, run with `pytest -m gpu`) are exempt: they are the real-GPU
    checks, and the service on the real card is exactly what they compare. tests/gpu skips entirely
    on a machine without a usable GPU."""
    if request.node.get_closest_marker("gpu") is not None:
        return
    import halide.batch.orchestrator as orchestrator
    from halide.gpu_service import ServiceUnavailable

    real = orchestrator._start_service

    def start(kind, initializer):
        if kind == "gpu" and initializer is None:
            raise ServiceUnavailable("the GPU service isn't started in tests")
        return real(kind, initializer)

    monkeypatch.setattr(orchestrator, "_start_service", start)
