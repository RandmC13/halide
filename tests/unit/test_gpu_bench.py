"""The GPU benchmark script (docs/plans/gpu-acceleration-bench.py) is run by the user on their own
card, not here — there's no GPU in the dev sandbox. These tests exercise its helpers (with a fake
CuPy where the GPU would be) so a mistake in them shows up before the user spends a long run on it."""

from __future__ import annotations

import importlib.util
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


def test_gpu_rows_skip_eight_workers_by_default_but_the_cpu_keeps_them(bench):
    assert bench._worker_counts("gpu", None) == [1, 2, 4]
    assert bench._worker_counts("cpu", None) == [1, 2, 4, 8]
    assert bench._worker_counts("gpu", [8]) == [8]  # an explicit list is used as given


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
