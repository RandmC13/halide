"""The GPU benchmark script (docs/plans/gpu-acceleration-bench.py) is run by the user on their own
card, not here — there's no GPU in the dev sandbox. These tests exercise its helpers (with a fake
CuPy where the GPU would be) so a mistake in them shows up before the user spends a long run on it."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from halide.device import ComputeDevice
from halide.io.tiff import write_tiff

_BENCH = Path(__file__).resolve().parents[2] / "docs" / "plans" / "gpu-acceleration-bench.py"
_MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("_gpu_bench_under_test", _BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_worker_gpu_rows_skip_eight_workers_but_the_service_and_the_cpu_keep_them(bench):
    assert bench._worker_counts("per-worker", None) == [4]
    assert bench._worker_counts("service", None) == [4, 8]
    assert bench._worker_counts("-", None) == [1, 2, 4, 8]  # the CPU
    assert bench._worker_counts("service", [8]) == [8]  # an explicit list is used as given


def test_the_gpu_is_timed_in_both_modes_and_per_worker_mode_is_halides_own_switch(bench):
    assert [mode for mode, _ in bench._MODES["gpu"]] == ["service", "per-worker"]
    env = dict(bench._MODES["gpu"])
    assert env["per-worker"] == {"HALIDE_GPU_SERVICE": "0"}
    assert env["service"] == {"HALIDE_GPU_SERVICE": "1"}  # explicit, so a stray =0 in the shell can't leak in
    from halide.batch.orchestrator import SERVICE_ENV

    assert SERVICE_ENV == "HALIDE_GPU_SERVICE"


def test_host_rss_poller_reports_the_gpu_service_on_its_own(bench):
    """A real service process (the "cpu" kind: the same spawned process, numpy instead of CUDA) is
    found among the parent's children, and its memory is reported separately from the workers'."""
    from halide.gpu_service import running_service

    import psutil

    with running_service("cpu") as address:
        assert bench._is_gpu_service(psutil.Process(address.pid).cmdline())
        poller = bench._HostRssPoller(os.getpid())
        poller.sample()
    assert poller.service_pids == {address.pid}
    assert poller.service_rss > 0
    assert poller.peak_total > poller.service_rss
    assert not bench._is_gpu_service([sys.executable, "-c", "from multiprocessing.forkserver import main"])


def test_where_workers_are_spawned_too_nothing_is_attributed_to_the_service(bench):
    """Windows: no forkserver, so the pool's workers are spawned like the service and the cmdline
    can't tell them apart — the poller must not guess (the service columns then read "unknown")."""
    from halide.gpu_service import running_service

    with running_service("cpu") as address:
        poller = bench._HostRssPoller(os.getpid(), identify_service=False)
        poller.sample()
    assert poller.service_pids == set() and poller.service_rss == 0
    assert poller.peak_process > 0


def test_the_service_is_identifiable_exactly_where_the_pool_uses_forkserver(bench):
    from halide.batch.orchestrator import _pool_context

    assert bench._SERVICE_IDENTIFIABLE == (_pool_context() is not None)


def test_host_rss_poller_sees_a_process_and_its_children(bench):
    child = "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); time.sleep(3); p.wait()"
    proc = subprocess.Popen([sys.executable, "-c", child])
    try:
        poller = bench._HostRssPoller(proc.pid, interval=0.05)
        import time

        time.sleep(1.0)  # both processes up
        poller.sample()
    finally:
        proc.wait()
    assert poller.peak_total > poller.peak_process > 0


def test_run_reports_host_ram_and_output(bench):
    result = bench._run([sys.executable, "-c", "print('hello on the CPU instead')"], "cpu")
    assert result["ok"] and "hello" in result["output"]
    assert result["fallbacks"] == 1
    assert "host_peak_total_mib" in result and "host_peak_process_mib" in result
    assert result["vram_peaks_mib"] == []
    assert result["service_seen"] is False and result["service_rss_mib"] is None
    assert result["service_not_used"] is None


def test_run_passes_the_mode_environment_and_spots_a_batch_that_didnt_use_the_service(bench):
    script = "import os; print('⚠ Warning: GPU service not used: turned off by HALIDE_GPU_SERVICE=' + os.environ['HALIDE_GPU_SERVICE'])"
    result = bench._run([sys.executable, "-c", script], "cpu", {"HALIDE_GPU_SERVICE": "0"})
    assert result["ok"]
    assert result["service_not_used"] == "⚠ Warning: GPU service not used: turned off by HALIDE_GPU_SERVICE=0"


def test_auto_workers_asks_a_fresh_process(bench, tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.tif").write_bytes(b"not a real tiff")  # header unreadable: fallback estimate
    auto = bench._auto_workers(tmp_path, "cpu")
    assert auto["kind"] == "cpu" and auto["gpu_cap"] is None
    assert 1 <= auto["workers"] <= 3


class _FakePool:
    def __init__(self):
        self.held = 0

    def free_all_blocks(self):
        self.held = 0

    def total_bytes(self):
        return self.held


def test_memory_on_device_measures_the_given_calibration_and_auto(bench, tmp_path, monkeypatch):
    import halide.device
    import halide.processing

    pool = _FakePool()
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.get_default_memory_pool = lambda: pool
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    gpu = ComputeDevice(kind="gpu", name="Fake GPU", memory_free=6 * 2**30, memory_total=8 * 2**30)
    monkeypatch.setattr(halide.device, "resolve_device", lambda requested: gpu)
    profiles = []

    def fake_process_scan(scan, out, stage, density_profile, tone, *, device, on_warning):
        assert device is gpu
        profiles.append(density_profile)
        pool.held = (100 if density_profile is not None else 150) * 2**20
        Path(out).write_bytes(b"")

    monkeypatch.setattr(halide.processing, "process_scan", fake_process_scan)
    scan = tmp_path / "scan.tif"
    write_tiff(scan, np.full((8, 8, 3), 0.1, dtype=np.float32))

    lines = bench._memory_on_device(scan, _MANUAL, tmp_path)

    assert profiles[0] is not None and profiles[1] is None  # the given profile, then per-frame auto
    text = "\n".join(lines)
    assert "given calibration (its peak): 100 MiB" in text
    assert "--auto-density (its peak): 150 MiB" in text
    assert not (tmp_path / "m.tif").exists()


def test_the_bench_runs_end_to_end_on_the_cpu_batch_rows_only(tmp_path):
    """`--devices cpu --only batch` on a tiny roll through the real `halide batch`: a table row per
    worker count, and nothing left in the scratch folder."""
    roll = tmp_path / "roll"
    roll.mkdir()
    from tests.unit.test_icc import LINEAR_TAGS, build_icc

    for i in range(2):
        image = np.random.default_rng(i).uniform(0.01, 0.3, size=(24, 36, 3)).astype(np.float32)
        write_tiff(roll / f"f{i}.tif", image, icc_profile=build_icc(LINEAR_TAGS))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    proc = subprocess.run(
        [sys.executable, str(_BENCH), "--roll", str(roll), "--devices", "cpu", "--only", "batch", "--workers", "1",
         "--scratch", str(scratch), *_MANUAL],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [line for line in proc.stdout.splitlines() if line.startswith("| batch")]
    assert [row.split("|")[4].strip() for row in rows] == ["1", "auto"]
    assert all("| yes |" in row for row in rows)
    assert not any(line.startswith("| invert") for line in proc.stdout.splitlines())  # --only batch
    assert list(scratch.iterdir()) == []
