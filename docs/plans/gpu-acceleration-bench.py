"""GPU benchmark for docs/plans/gpu-acceleration.md, Task 7 — run on the machine with the NVIDIA card.

Unlike the probe beside it, this times halide itself: the real `halide invert` and `halide batch`
commands, at `--device cpu` and `--device gpu`, and `batch` at several `--workers` counts plus the
automatic choice. It answers what the plans leave to measurement:

  1. Is the GPU faster for a whole roll, and with how many workers (a GPU worker's time is mostly
     decode + write, so fewer may be enough) — which sets the default device and worker count.
  2. How much GPU memory one frame really needs, to fit the constants in
     src/halide/batch/orchestrator.py (`_CUDA_CONTEXT_BYTES`, `_DEVICE_FRAME_MULTIPLIER`): one
     frame is developed in this process on the GPU and CuPy's memory pool is read afterwards (it
     keeps every block it allocated, so what it holds is the frame's peak) — once with the given
     calibration and once with per-frame auto calibration (`--auto-density`, the least-margin
     case); `nvidia-smi` adds the CUDA context on top, and is also polled during every GPU batch
     for each process's peak.
  3. How much host RAM a batch uses while it runs (all of halide's processes, sampled with psutil:
     RSS and, where the platform has it, PSS — shared-memory frames are mapped by both a worker and
     the GPU service, so RSS counts them twice and PSS doesn't), and whether the machine swapped.
  4. (docs/plans/gpu-batch-throughput.md, Task B4) The two ways a GPU batch can use the card, in
     the same run: the shared GPU service (the default: one process holds the only CUDA context,
     CPU-only workers hand it frames through shared memory) at `--workers` auto, 4 and 8; and the
     previous per-worker mode (`HALIDE_GPU_SERVICE=0`: every worker its own CUDA context) at auto
     and 4. For service rows the service process is reported on its own — its peak host RSS/PSS
     and VRAM — since it is the one process holding CUDA, to refit `_GPU_SERVICE_HOST_BYTES`; the
     "largest worker" column then excludes it. A service row whose batch didn't actually use the
     service (the run sheet's "GPU service not used: ..." — e.g. /dev/shm too small) is flagged.

First run on the user's RTX 3070 on 2026-09-27; its results and what they set are in
gpu-acceleration.md §7 (items 1-3) and gpu-batch-throughput.md, Task B4 (item 4).
Rerun it on new hardware to refit.

Every batch row also shows how many workers `--workers` auto would pick in that mode. Per-worker
GPU mode skips 8 workers: about 4 per-worker GPU workers fit on an 8 GiB card, so at 8 most frames
would likely fall back to the CPU and the row would time that instead. Before any GPU timing, one
untimed GPU `invert` warms up CuPy (its first-ever run compiles kernels).

    .venv/bin/python docs/plans/gpu-acceleration-bench.py --scan IMG_0158.tif --roll Roll16-Testing \\
        --profile Roll16-KodakGold200 > gpu-bench.txt

`--only batch` skips the single-frame `invert` rows and the in-process memory measurement (a
shorter rerun); `--devices gpu` skips the CPU rows. Any other halide calibration flags
(`--rm 1.9 --bm 1.4 ...`) are passed through in place of `--profile`. Outputs go to a scratch
folder next to the roll (a real roll is ~5 GB of TIFFs — not /tmp, which is often RAM) and are
deleted after every run; redirect the report to a file on disk as above, not in /tmp either.
`--devices cpu` runs without a GPU (only for checking the script). Takes a while: every run is a
full roll.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_MIB = 2**20
_FALLBACK = "on the CPU instead"  # the text of every GPU -> CPU fallback warning (processing.py)


def _halide(*args: str) -> list[str]:
    return [sys.executable, "-m", "halide.cli.main", *args]


class _VramPoller:
    """Peak GPU memory per process while a command runs, from `nvidia-smi` (None if unavailable)."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peaks: dict[str, int] = {}
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
            except Exception:  # noqa: BLE001 — a missed sample, not a failed benchmark
                out = ""
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[1].isdigit():
                    self.peaks[parts[0]] = max(self.peaks.get(parts[0], 0), int(parts[1]))
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.available:
            self._thread.join()
        return False


def _is_gpu_service(cmdline: list[str]) -> bool:
    """halide's GPU service is the only process halide starts with the spawn method (the forkserver
    and its workers, and multiprocessing's resource tracker, run other entry points)."""
    return any("spawn_main" in part for part in cmdline)


class _HostRssPoller:
    """Peak host RAM of a process and all its descendants (halide's parent, its forkserver, every
    worker and the GPU service), sampled with psutil: the largest total seen at once (RSS, and PSS
    where the platform has it), the largest single process other than the GPU service, the GPU
    service's own peak (RSS and PSS), and the most swap in use. Zero/None if psutil can't read them."""

    def __init__(self, pid: int, interval: float = 0.2) -> None:
        self.pid = pid
        self.interval = interval
        self.peak_total = 0
        self.peak_total_pss: int | None = None
        self.peak_process = 0
        self.service_pids: set[int] = set()
        self.service_rss = 0
        self.service_pss: int | None = None
        self.swap_start: int | None = None
        self.swap_peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        try:
            import psutil

            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
            swap = psutil.swap_memory().used
            self.swap_start = swap if self.swap_start is None else self.swap_start
            self.swap_peak = max(self.swap_peak or 0, swap)
        except Exception:  # noqa: BLE001 — the process has already exited, or no psutil
            return
        total = total_pss = 0
        have_pss = True
        for process in processes:
            try:
                try:
                    info = process.memory_full_info()  # PSS/USS: reads smaps_rollup on Linux
                except Exception:  # noqa: BLE001 — not on this platform, or not permitted
                    info = process.memory_info()
                rss, pss = info.rss, getattr(info, "pss", None)
                if process.pid not in self.service_pids and _is_gpu_service(process.cmdline()):
                    self.service_pids.add(process.pid)
            except Exception:  # noqa: BLE001 — exited between listing and reading
                continue
            total += rss
            if pss is None:
                have_pss = False
            else:
                total_pss += pss
            if process.pid in self.service_pids:
                self.service_rss = max(self.service_rss, rss)
                if pss is not None:
                    self.service_pss = max(self.service_pss or 0, pss)
            else:
                self.peak_process = max(self.peak_process, rss)
        self.peak_total = max(self.peak_total, total)
        if have_pss and total:
            self.peak_total_pss = max(self.peak_total_pss or 0, total_pss)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


_SERVICE_NOT_USED = "GPU service not used"  # the run sheet's warning when a GPU batch falls back (cli/_run_sheet.py)


def _mib(n: int | None) -> int | None:
    return None if n is None else n // _MIB


def _run(cmd: list[str], device: str, env: dict[str, str] | None = None) -> dict:
    """Run one halide command; its peak host RAM (all its processes), and on the GPU the peak GPU
    memory of every process that used it (the halide parent's own CUDA context — it resolves the
    device — plus each worker's in per-worker mode, or the GPU service's in service mode)."""
    with _VramPoller() if device == "gpu" else contextlib.nullcontext() as poller:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})})
        with _HostRssPoller(proc.pid) as host:
            output, _ = proc.communicate()
        elapsed = time.perf_counter() - start
    peaks = poller.peaks if poller is not None else {}
    service_pids = {str(pid) for pid in host.service_pids}
    return {
        "seconds": elapsed,
        "ok": proc.returncode == 0,
        "output": output,
        "fallbacks": output.count(_FALLBACK),
        "service_not_used": next((line.strip() for line in output.splitlines() if _SERVICE_NOT_USED in line), None),
        "vram_peaks_mib": sorted((v for pid, v in peaks.items() if pid not in service_pids), reverse=True),
        "service_vram_mib": max((v for pid, v in peaks.items() if pid in service_pids), default=None),
        "service_seen": bool(host.service_pids),
        "host_peak_total_mib": host.peak_total // _MIB,
        "host_peak_total_pss_mib": _mib(host.peak_total_pss),
        "host_peak_process_mib": host.peak_process // _MIB,
        "service_rss_mib": _mib(host.service_rss) if host.service_pids else None,
        "service_pss_mib": _mib(host.service_pss),
        "swap_mib": None if host.swap_start is None else (host.swap_start // _MIB, host.swap_peak // _MIB),
    }


_AUTO_WORKERS = """
import json, sys
from pathlib import Path
from halide.batch.orchestrator import (BatchCompute, GpuService, default_worker_count, device_worker_cap,
                                       discover_jobs, service_worker_count, shared_memory_worker_cap)
from halide.device import resolve_device
device = resolve_device(sys.argv[2])
jobs = discover_jobs(Path(sys.argv[1]), Path("."))
result = {"kind": device.kind, "workers": default_worker_count(jobs, device=device),
          "gpu_cap": device_worker_cap(jobs, device), "memory_free": device.memory_free}
if device.kind == "gpu":
    # What `batch` would pick with the service running (service_worker_count), without starting it.
    shm_cap = shared_memory_worker_cap(jobs)
    compute = BatchCompute(device=device, service=GpuService(address=None, shm_prefix=""), shm_cap=shm_cap)
    result.update(service_workers=service_worker_count(jobs, compute), shm_cap=shm_cap)
print(json.dumps(result))
"""


def _auto_workers(roll: Path, device: str) -> dict:
    """What `halide batch --workers` auto would pick on this device (per-worker mode's count in
    "workers", service mode's in "service_workers"), asked in a fresh process (so this one never
    makes a CUDA context of its own that would hold GPU memory during the runs)."""
    import json

    proc = subprocess.run([sys.executable, "-c", _AUTO_WORKERS, str(roll), device],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


# The ways a batch reaches the device: (label, device, environment). "service" and "per-worker" are
# the two GPU modes (docs/plans/gpu-batch-throughput.md Part B); HALIDE_GPU_SERVICE=0 is halide's own
# switch back to per-worker mode (batch/orchestrator.py).
_MODES = {
    "cpu": [("-", {})],
    "gpu": [("service", {"HALIDE_GPU_SERVICE": "1"}), ("per-worker", {"HALIDE_GPU_SERVICE": "0"})],
}


def _worker_counts(mode: str, requested: list[int] | None) -> list[int]:
    """The explicit worker counts to time in this mode (the automatic choice is always added)."""
    if requested is not None:
        return requested
    return {"service": [4, 8], "per-worker": [4]}.get(mode, [1, 2, 4, 8])


def _frames(roll: Path) -> list[Path]:
    return sorted(f for f in roll.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff"))


def _memory_on_device(scan: Path, calibration: list[str], scratch: Path) -> list[str]:
    """Develop one frame in this process on the GPU, twice — with the given calibration, then with
    per-frame auto calibration (`--auto-density`, the least-margin case for the worker-count
    constants) — and report what the device needed for each."""
    import cupy

    from halide.cli._calibration_args import resolve_density_profile, resolve_tone_params
    from halide.cli.main import build_parser
    from halide.core.types import Stage
    from halide.device import resolve_device
    from halide.io.tiff import read_tiff
    from halide.processing import process_scan

    device = resolve_device("gpu")
    frame_bytes = read_tiff(scan).image.nbytes
    args = build_parser().parse_args(["invert", str(scan), str(scratch / "m.tif"), *calibration])
    profile, saved_tone = resolve_density_profile(args)
    tone = resolve_tone_params(args, saved_tone=saved_tone)
    pool = cupy.get_default_memory_pool()
    lines = [
        f"GPU: {device.name}, free before {device.memory_free / 2**30:.2f} / {device.memory_total / 2**30:.2f} GiB",
        f"frame (decoded): {frame_bytes / _MIB:.0f} MiB",
    ]
    largest_pool = 0
    for label, density_profile in (("given calibration", profile), ("--auto-density", None)):
        pool.free_all_blocks()
        warnings: list[str] = []
        process_scan(scan, scratch / "m.tif", Stage.FULL, density_profile, tone, device=device,
                     on_warning=warnings.append)
        pool_bytes = pool.total_bytes()
        largest_pool = max(largest_pool, pool_bytes)
        lines.append(f"CuPy pool after one frame, {label} (its peak): {pool_bytes / _MIB:.0f} MiB = "
                     f"{pool_bytes / frame_bytes:.2f} x frame")
        if warnings:
            lines.append(f"!! the frame ({label}) fell back to the CPU: {warnings}")
    with _VramPoller() as poller:
        time.sleep(1.0)
    used = poller.peaks.get(str(os.getpid()))
    if used is not None:
        context = used * _MIB - largest_pool
        lines.append(f"nvidia-smi, this process: {used} MiB -> CUDA context + CuPy/library overhead ~{context / _MIB:.0f} MiB")
    else:
        lines.append("nvidia-smi per-process memory unavailable (context size not measured)")
    os.remove(scratch / "m.tif")
    return lines


def _auto_pick(auto: dict, mode: str) -> str:
    if "error" in auto:
        return "?"
    return str(auto.get("service_workers" if mode == "service" else "workers", "?"))


def _or_dash(value) -> str:
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scan", type=Path, help="one full-resolution scan (invert, memory; not needed with --only batch)")
    parser.add_argument("--roll", required=True, type=Path, help="a roll folder (batch)")
    parser.add_argument("--profile", help="saved profile to calibrate with (or pass --rm/--bm/... instead)")
    parser.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    parser.add_argument("--only", choices=["batch"],
                        help="batch: only the batch rows (no invert rows, no in-process GPU memory measurement)")
    parser.add_argument("--workers", nargs="+", type=int, default=None,
                        help="worker counts to time in every mode (default: 1 2 4 8 on the CPU; 4 8 for the GPU "
                             "service; 4 for per-worker GPU)")
    parser.add_argument("--scratch", type=Path, help="where outputs go meanwhile (default: beside the roll)")
    args, calibration = parser.parse_known_args()
    if args.profile:
        calibration = ["--profile", args.profile, *calibration]
    if not calibration:
        parser.error("give --profile NAME or halide's manual calibration flags")
    if args.only != "batch" and args.scan is None:
        parser.error("give --scan (or --only batch)")

    frames = _frames(args.roll)
    scratch = Path(tempfile.mkdtemp(prefix="halide-bench-", dir=args.scratch or args.roll.resolve().parent))
    print(f"python {platform.python_version()}  {platform.platform()}  cpus {os.cpu_count()}")
    try:
        import psutil

        print(f"physical cores {psutil.cpu_count(logical=False)}  RAM available "
              f"{psutil.virtual_memory().available / 2**30:.1f} GiB  swap in use "
              f"{psutil.swap_memory().used / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001
        pass
    if os.path.isdir("/dev/shm"):
        shm = shutil.disk_usage("/dev/shm")
        print(f"/dev/shm free {shm.free / 2**30:.1f} of {shm.total / 2**30:.1f} GiB (the GPU service's frames live there)")
    print(f"scan {args.scan or '-'}; roll {args.roll} ({len(frames)} frames); calibration {' '.join(calibration)}")
    print(f"scratch {scratch}\n")

    rows = []
    autos = {}
    try:
        for device in args.devices:
            autos[device] = _auto_workers(args.roll, device)
            if args.only != "batch":
                out = scratch / "invert.tif"
                if device == "gpu":  # untimed: CuPy compiles its kernels on the first-ever run
                    _run(_halide("invert", str(args.scan), str(out), "--device", device, *calibration), device)
                    out.unlink(missing_ok=True)
                result = _run(_halide("invert", str(args.scan), str(out), "--device", device, *calibration), device)
                out.unlink(missing_ok=True)
                rows.append(("invert", device, "-", "-", result, 1))
            for mode, env in _MODES[device]:
                for workers in [*_worker_counts(mode, args.workers), None]:
                    out_dir = scratch / "batch"
                    cmd = _halide("batch", str(args.roll), str(out_dir), "--device", device, "--quiet", *calibration)
                    if workers is not None:
                        cmd += ["--workers", str(workers)]
                    result = _run(cmd, device, env)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    rows.append(("batch", device, mode, "auto" if workers is None else str(workers), result, len(frames)))
                    print(f"  done: batch {device} {mode} workers {workers or 'auto'}: {result['seconds']:.1f} s",
                          file=sys.stderr)

        for device, auto in autos.items():
            if "error" in auto:
                print(f"--workers auto on {device}: couldn't ask ({auto['error']})")
                continue
            cap = f", GPU memory fits {auto['gpu_cap']}" if auto.get("gpu_cap") is not None else ""
            free = f", {auto['memory_free'] / 2**30:.2f} GiB GPU memory free" if auto.get("memory_free") else ""
            label = "per-worker GPU mode" if auto["kind"] == "gpu" else device
            print(f"--workers auto on {device} ({label}): {auto['workers']} (resolved to {auto['kind']}{cap}{free})")
            if "service_workers" in auto:
                shm = f", /dev/shm fits {auto['shm_cap']}" if auto.get("shm_cap") is not None else ""
                print(f"--workers auto on {device} (GPU service): {auto['service_workers']}{shm}")
        print("\nMode: on the GPU, \"service\" is one GPU process shared by all workers (the default);")
        print("\"per-worker\" is HALIDE_GPU_SERVICE=0, every worker its own CUDA context (before Part B).")
        print("GPU procs: every process that used the card other than the GPU service; the smallest is usually")
        print("the halide parent's own CUDA context (it resolves the device), the rest are per-worker mode's")
        print("workers. Host RAM: all of halide's processes together at their peak (RSS, and PSS, which counts")
        print("a shared-memory frame once rather than in both the worker and the service), and the largest")
        print("single process other than the GPU service. Service: the GPU service process's own peak host")
        print("RSS / PSS and VRAM. Swap: in use at the start / at most during the run.\n")
        print("| command | device | mode | workers | auto picks | wall s | s/frame | ok | CPU fallbacks "
              "| GPU procs | peak VRAM/proc MiB | host RAM peak MiB (RSS total / PSS total / largest) "
              "| service RSS / PSS MiB | service VRAM MiB | swap MiB |")
        print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---|")
        for command, device, mode, workers, r, n in rows:
            peaks = r["vram_peaks_mib"]
            auto = _auto_pick(autos.get(device, {}), mode) if command == "batch" else "-"
            ok = "yes" if r["ok"] else "NO"
            if mode == "service" and r["service_not_used"]:
                ok += " (service NOT used)"
            swap = f"{r['swap_mib'][0]} / {r['swap_mib'][1]}" if r["swap_mib"] else "-"
            service = (f"{_or_dash(r['service_rss_mib'])} / {_or_dash(r['service_pss_mib'])}"
                       if r["service_seen"] else "-")
            print(
                f"| {command} | {device} | {mode} | {workers} | {auto} | {r['seconds']:.1f} | "
                f"{r['seconds'] / n:.2f} | {ok} | {r['fallbacks']} | {len(peaks) if device == 'gpu' else '-'} | "
                f"{', '.join(map(str, peaks)) if peaks else '-'} | "
                f"{r['host_peak_total_mib']} / {_or_dash(r['host_peak_total_pss_mib'])} / {r['host_peak_process_mib']} | "
                f"{service} | {_or_dash(r['service_vram_mib'])} | {swap} |"
            )
        for command, device, mode, workers, r, _ in rows:
            if r["service_not_used"] and mode == "service":
                print(f"\n{command} {device} {mode} workers {workers}: {r['service_not_used']}")
        failed = [r for *_, r, _ in rows if not r["ok"]]
        for r in failed[:3]:
            print("\nA run failed; its output ends:\n" + "\n".join(r["output"].splitlines()[-15:]))

        if "gpu" in args.devices and args.only != "batch":
            print("\nOne frame on the GPU, in this process:")
            try:
                for line in _memory_on_device(args.scan, calibration, scratch):
                    print(f"  {line}")
            except Exception as exc:  # noqa: BLE001
                print(f"  couldn't measure: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
