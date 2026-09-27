"""GPU benchmark for docs/plans/gpu-acceleration.md, Task 7 — run on the machine with the NVIDIA card.

Unlike the probe beside it, this times halide itself: the real `halide invert` and `halide batch`
commands, at `--device cpu` and `--device gpu`, and `batch` at several `--workers` counts plus the
automatic choice. It answers three things the plan leaves to measurement:

  1. Is the GPU faster for a whole roll, and with how many workers (a GPU worker's time is mostly
     decode + write, so fewer may be enough) — which sets the default device and worker count.
  2. How much GPU memory one worker really needs, to refit the PROVISIONAL constants in
     src/halide/batch/orchestrator.py (`_CUDA_CONTEXT_BYTES`, `_DEVICE_CONTEXT_BYTES`,
     `_DEVICE_FRAME_MULTIPLIER`): one frame is developed in this process on the GPU and CuPy's
     memory pool is read afterwards (it keeps every block it allocated, so what it holds is the
     frame's peak) — once with the given calibration and once with per-frame auto calibration
     (`--auto-density`, the least-margin case); `nvidia-smi` adds the CUDA context on top, and is
     also polled during every GPU batch for each worker's peak.
  3. How much host RAM the workers use while they run (all of halide's processes, sampled with
     psutil) — a GPU worker's host footprint isn't modelled by the worker-count default yet.

Every batch row also shows how many workers `--workers` auto would pick on that device. The GPU
rows skip 8 workers by default: the provisional estimate fits about 4 GPU workers on an 8 GiB
card, so at 8 most frames would likely fall back to the CPU and the row would time that instead. Before any GPU
timing, one untimed GPU `invert` warms up CuPy (its first-ever run compiles kernels).

    .venv/bin/python docs/plans/gpu-acceleration-bench.py --scan IMG_0158.tif --roll Roll16-Testing \\
        --profile Roll16-KodakGold200 > gpu-bench.txt

Any other halide calibration flags (`--rm 1.9 --bm 1.4 ...`) are passed through in place of
`--profile`. Outputs go to a scratch folder next to the roll (a real roll is ~5 GB of TIFFs — not
/tmp, which is often RAM) and are deleted after every run. `--devices cpu` runs without a GPU (only
for checking the script). Takes a while: every run is a full roll.
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


class _HostRssPoller:
    """Peak host RAM of a process and all its descendants (halide's parent, its forkserver and
    every worker), sampled with psutil: the largest total seen at once, and the largest single
    process. Zero if psutil can't read them."""

    def __init__(self, pid: int, interval: float = 0.2) -> None:
        self.pid = pid
        self.interval = interval
        self.peak_total = 0
        self.peak_process = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        try:
            import psutil

            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
        except Exception:  # noqa: BLE001 — the process has already exited, or no psutil
            return
        sizes = []
        for process in processes:
            try:
                sizes.append(process.memory_info().rss)
            except Exception:  # noqa: BLE001 — exited between listing and reading
                pass
        if sizes:
            self.peak_total = max(self.peak_total, sum(sizes))
            self.peak_process = max(self.peak_process, max(sizes))

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


def _run(cmd: list[str], device: str) -> dict:
    """Run one halide command; its peak host RAM (all its processes), and on the GPU the peak GPU
    memory of every process that used it (the halide parent's own CUDA context — it resolves the
    device — plus each worker's)."""
    with _VramPoller() if device == "gpu" else contextlib.nullcontext() as poller:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                stdin=subprocess.DEVNULL)
        with _HostRssPoller(proc.pid) as host:
            output, _ = proc.communicate()
        elapsed = time.perf_counter() - start
    peaks = poller.peaks if poller is not None else {}
    return {
        "seconds": elapsed,
        "ok": proc.returncode == 0,
        "output": output,
        "fallbacks": output.count(_FALLBACK),
        "vram_peaks_mib": sorted(peaks.values(), reverse=True),
        "host_peak_total_mib": host.peak_total // _MIB,
        "host_peak_process_mib": host.peak_process // _MIB,
    }


_AUTO_WORKERS = """
import json, sys
from pathlib import Path
from halide.batch.orchestrator import default_worker_count, device_worker_cap, discover_jobs
from halide.device import resolve_device
device = resolve_device(sys.argv[2])
jobs = discover_jobs(Path(sys.argv[1]), Path("."))
print(json.dumps({"kind": device.kind, "workers": default_worker_count(jobs, device=device),
                  "gpu_cap": device_worker_cap(jobs, device), "memory_free": device.memory_free}))
"""


def _auto_workers(roll: Path, device: str) -> dict:
    """What `halide batch --workers` auto would pick on this device, asked in a fresh process (so
    this one never makes a CUDA context of its own that would hold GPU memory during the runs)."""
    import json

    proc = subprocess.run([sys.executable, "-c", _AUTO_WORKERS, str(roll), device],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _worker_counts(device: str, requested: list[int] | None) -> list[int]:
    """The explicit worker counts to time on this device (the automatic choice is always added)."""
    if requested is not None:
        return requested
    return [1, 2, 4] if device == "gpu" else [1, 2, 4, 8]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scan", required=True, type=Path, help="one full-resolution scan (invert, memory)")
    parser.add_argument("--roll", required=True, type=Path, help="a roll folder (batch)")
    parser.add_argument("--profile", help="saved profile to calibrate with (or pass --rm/--bm/... instead)")
    parser.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    parser.add_argument("--workers", nargs="+", type=int, default=None,
                        help="worker counts to time (default: 1 2 4 8 on the CPU, 1 2 4 on the GPU)")
    parser.add_argument("--scratch", type=Path, help="where outputs go meanwhile (default: beside the roll)")
    args, calibration = parser.parse_known_args()
    if args.profile:
        calibration = ["--profile", args.profile, *calibration]
    if not calibration:
        parser.error("give --profile NAME or halide's manual calibration flags")

    frames = _frames(args.roll)
    scratch = Path(tempfile.mkdtemp(prefix="halide-bench-", dir=args.scratch or args.roll.resolve().parent))
    print(f"python {platform.python_version()}  {platform.platform()}  cpus {os.cpu_count()}")
    try:
        import psutil

        print(f"physical cores {psutil.cpu_count(logical=False)}  RAM available "
              f"{psutil.virtual_memory().available / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001
        pass
    print(f"scan {args.scan}; roll {args.roll} ({len(frames)} frames); calibration {' '.join(calibration)}")
    print(f"scratch {scratch}\n")

    rows = []
    autos = {}
    try:
        for device in args.devices:
            autos[device] = _auto_workers(args.roll, device)
            out = scratch / "invert.tif"
            if device == "gpu":  # untimed: CuPy compiles its kernels on the first-ever run
                _run(_halide("invert", str(args.scan), str(out), "--device", device, *calibration), device)
                out.unlink(missing_ok=True)
            result = _run(_halide("invert", str(args.scan), str(out), "--device", device, *calibration), device)
            out.unlink(missing_ok=True)
            rows.append(("invert", device, "-", result, 1))
            for workers in [*_worker_counts(device, args.workers), None]:
                out_dir = scratch / "batch"
                cmd = _halide("batch", str(args.roll), str(out_dir), "--device", device, "--quiet", *calibration)
                if workers is not None:
                    cmd += ["--workers", str(workers)]
                result = _run(cmd, device)
                shutil.rmtree(out_dir, ignore_errors=True)
                rows.append(("batch", device, "auto" if workers is None else str(workers), result, len(frames)))
                print(f"  done: batch {device} workers {workers or 'auto'}: {result['seconds']:.1f} s", file=sys.stderr)

        for device, auto in autos.items():
            if "error" in auto:
                print(f"--workers auto on {device}: couldn't ask ({auto['error']})")
            else:
                cap = f", GPU memory fits {auto['gpu_cap']}" if auto.get("gpu_cap") is not None else ""
                free = f", {auto['memory_free'] / 2**30:.2f} GiB GPU memory free" if auto.get("memory_free") else ""
                print(f"--workers auto on {device}: {auto['workers']} (resolved to {auto['kind']}{cap}{free})")
        print("\nGPU procs: every process that used the card; the smallest is usually the halide parent's own")
        print("CUDA context (it resolves the device), the rest are workers. Host RAM: all of halide's")
        print("processes together at their peak, and the largest single one.\n")
        print("| command | device | workers | auto picks | wall s | s/frame | ok | CPU fallbacks | GPU procs "
              "| peak VRAM/proc MiB | host RAM peak MiB (total / largest) |")
        print("|---|---|---:|---:|---:|---:|---|---:|---:|---|---|")
        for command, device, workers, r, n in rows:
            peaks = r["vram_peaks_mib"]
            auto = autos.get(device, {}).get("workers", "?") if command == "batch" else "-"
            print(
                f"| {command} | {device} | {workers} | {auto} | {r['seconds']:.1f} | {r['seconds'] / n:.2f} | "
                f"{'yes' if r['ok'] else 'NO'} | {r['fallbacks']} | {len(peaks) if device == 'gpu' else '-'} | "
                f"{', '.join(map(str, peaks)) if peaks else '-'} | "
                f"{r['host_peak_total_mib']} / {r['host_peak_process_mib']} |"
            )
        failed = [r for *_, r, _ in rows if not r["ok"]]
        for r in failed[:3]:
            print("\nA run failed; its output ends:\n" + "\n".join(r["output"].splitlines()[-15:]))

        if "gpu" in args.devices:
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
