"""GPU feasibility probe for docs/plans/gpu-acceleration.md — run on the machine with the NVIDIA card.

It does NOT use halide's GPU code (there isn't any yet). It reads one real scan with halide's own
CPU loader, then times the same kinds of array operations halide's pipeline does — the ICC matrix,
white/density balance, invert, the print fit's percentile, the paper-curve lookup, the auto
calibration's sort — on the GPU (CuPy) and on the CPU (numpy), and reports how closely the two agree.
The answers decide whether GPU is worth turning on by default, and how many batch workers can share
the card.

    .venv/bin/pip install "cupy-cuda12x[ctk]"        # or cupy-cuda13x[ctk] — see the plan, §6
    .venv/bin/python docs/plans/gpu-acceleration-probe.py IMG_0158.tif > gpu-probe.txt

Paste gpu-probe.txt back. `--xp numpy` runs the "GPU" column on numpy (only for checking the probe).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np


def _sync(xp) -> None:
    if xp.__name__ == "cupy":
        xp.cuda.Stream.null.synchronize()


def _timed(xp, fn, repeats: int = 3) -> tuple[float, object]:
    """Best of `repeats` wall-clock seconds (after synchronising the GPU), and the last result."""
    best, result = float("inf"), None
    for _ in range(repeats):
        _sync(xp)
        start = time.perf_counter()
        result = fn()
        _sync(xp)
        best = min(best, time.perf_counter() - start)
    return best, result


def _stages(xp, img, matrix, wb, scale, lut, domain):
    """The pipeline's per-pixel and whole-frame operations, written once for either namespace.
    Mirrors core/ closely enough to time it; it is not halide's implementation."""
    lo, hi = domain
    lum_weights = xp.asarray([0.2722287168, 0.6740817658, 0.0536895174], dtype=img.dtype)
    m = xp.asarray(matrix, dtype=img.dtype)
    w = xp.asarray(wb, dtype=img.dtype)
    s = xp.asarray(scale, dtype=img.dtype)
    table = xp.asarray(lut, dtype=img.dtype)

    def icc():
        return img @ m

    def negative_to_positive(x):
        y = xp.maximum(x * w, 1e-7)
        xp.power(y, s, out=y)
        y = xp.maximum(y, 1e-7)
        return xp.divide(1.0, y, out=y)

    def print_fit(pos):
        lum = pos @ lum_weights
        xp.maximum(lum, 1e-7, out=lum)
        xp.log10(lum, out=lum)
        return xp.percentile(lum, [0.1, 99.5], overwrite_input=True)

    def paper_curve(pos):
        d = xp.log10(xp.maximum(pos, 1e-7))
        d += 0.3
        t = (d - lo) * (1.0 / (hi - lo))
        xp.clip(t, 0.0, 1.0, out=t)
        t *= table.shape[0] - 1
        f = xp.floor(t)
        i = f.astype(xp.int32)
        t -= f
        j = xp.minimum(i + 1, table.shape[0] - 1)
        a, b = table[i], table[j]
        b -= a
        b *= t
        a += b
        return a

    def auto_sort(pos):
        return xp.argsort(pos.reshape(-1, 3).mean(axis=1))

    return icc, negative_to_positive, print_fit, paper_curve, auto_sort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", help="a real full-resolution linear scan (TIFF with ICC profile)")
    parser.add_argument("--xp", choices=("cupy", "numpy"), default="cupy")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]}  {platform.platform()}  cpus {os.cpu_count()}")
    try:
        import psutil

        print(f"physical cores {psutil.cpu_count(logical=False)}  "
              f"RAM available {psutil.virtual_memory().available / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001
        pass

    t0 = time.perf_counter()
    if args.xp == "cupy":
        import cupy as xp
    else:
        xp = np
    print(f"import {args.xp}: {time.perf_counter() - t0:.2f} s")

    if args.xp == "cupy":
        t0 = time.perf_counter()
        n = xp.cuda.runtime.getDeviceCount()
        props = xp.cuda.runtime.getDeviceProperties(0)
        free, total = xp.cuda.Device(0).mem_info
        print(f"CUDA context: {time.perf_counter() - t0:.2f} s   devices {n}")
        print(f"GPU 0: {props['name'].decode()}  compute {props['major']}.{props['minor']}  "
              f"VRAM free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
        print(f"CUDA driver {xp.cuda.runtime.driverGetVersion()}  runtime {xp.cuda.runtime.runtimeGetVersion()}  "
              f"cupy {xp.__version__}")
        t0 = time.perf_counter()
        (xp.arange(10, dtype=xp.float32) ** 1.5).sum().item()
        print(f"first kernel (compile or kernel-cache load): {time.perf_counter() - t0:.2f} s")

    from halide.core.tone_render import _DEFAULT_CURVE_PATH, _load_curve
    from halide.processing import load_working_space_image

    t0 = time.perf_counter()
    host = load_working_space_image(args.scan)
    print(f"\nscan {args.scan}: {host.shape} {host.dtype} {host.nbytes / 2**20:.0f} MiB "
          f"(CPU read + ICC {time.perf_counter() - t0:.2f} s)")
    curve = _load_curve(str(_DEFAULT_CURVE_PATH))
    params = dict(matrix=np.array([[0.9, 0.05, 0.05], [0.04, 0.92, 0.04], [0.02, 0.03, 0.95]]),
                  wb=(1.2, 1.0, 0.8), scale=(0.95, 1.0, 1.1), lut=curve.values,
                  domain=(curve.domain_min, curve.domain_max))

    upload, dev = _timed(xp, lambda: xp.asarray(host))
    print(f"upload to device: {upload * 1000:.0f} ms")

    rows = []
    cpu_fns = _stages(np, host, **params)
    dev_fns = _stages(xp, dev, **params)
    names = ("icc matrix", "negative_to_positive", "print fit percentile", "paper curve", "auto-calib argsort")
    cpu_pos = cpu_fns[1](host)
    dev_pos = dev_fns[1](dev)
    for name, cfn, dfn in zip(names, cpu_fns, dev_fns):
        if name == "icc matrix":
            ct, cr = _timed(np, cfn, 1)
            dt, dr = _timed(xp, dfn)
        elif name == "negative_to_positive":
            ct, cr = _timed(np, lambda: cfn(host), 1)
            dt, dr = _timed(xp, lambda: dfn(dev))
        else:
            ct, cr = _timed(np, lambda: cfn(cpu_pos.copy() if name == "print fit percentile" else cpu_pos), 1)
            dt, dr = _timed(xp, lambda: dfn(dev_pos.copy() if name == "print fit percentile" else dev_pos))
        dr = dr.get() if hasattr(dr, "get") else np.asarray(dr)
        if name == "auto-calib argsort":
            agree = f"orders equal {np.mean(cr == dr) * 100:.3f}%"
        else:
            diff = np.abs(dr.astype(np.float64) - cr) / np.maximum(np.abs(cr), 1e-6)
            agree = f"max rel diff {diff.max():.1e}"
        rows.append((name, ct, dt, agree))

    download, _ = _timed(xp, lambda: xp.asnumpy(dev_pos) if xp is not np else dev_pos.copy())
    print(f"download from device: {download * 1000:.0f} ms\n")
    print(f"{'stage':24s} {'CPU s':>7s} {args.xp + ' s':>8s}  agreement")
    for name, ct, dt, agree in rows:
        print(f"{name:24s} {ct:7.3f} {dt:8.4f}  {agree}")

    if args.xp == "cupy":
        pool = xp.get_default_memory_pool()
        print(f"\ndevice memory held by this process's pool: {pool.total_bytes() / 2**20:.0f} MiB")
        free, total = xp.cuda.Device(0).mem_info
        print(f"VRAM free after run: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
