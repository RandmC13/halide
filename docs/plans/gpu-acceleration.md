# GPU Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: **approved 2026-09-27, being built on branch `gpu-acceleration`** (subagent-driven). The
user's decisions are in §6 and their machine's numbers in §7. Written against `dff3cb2`.

**Goal:** Run halide's per-frame number crunching on an NVIDIA GPU when one is available — on by
default if (and only if) it measurably speeds up the user's real runs — with `--device cpu` /
`HALIDE_DEVICE=cpu` to turn it off.

**Architecture:** CuPy (a numpy look-alike for CUDA) as an optional extra. `core/` stays one set of
pure functions written against an "array namespace" (numpy or CuPy, whichever the input array
belongs to), so there is still exactly one definition of the pipeline's math. `processing.py` decides
where a frame lives: it uploads the decoded scan once, runs every stage on the card, and downloads
once before writing. File decode/encode, exiftool and the GUI stay on the CPU.

**Tech Stack:** Python 3.14, numpy 2.5, CuPy 14 (`cupy-cuda12x` or `cupy-cuda13x`, which have
cp314 wheels — checked), colour-science (only to *compute* 3x3 matrices once), pytest.

**Spec:** the user's request (2026-09-27): "thorough investigation … implementation plan for adding
GPU processing … if it increases performance I'd like it to be on by default but able to be
disabled." This document is both the investigation and the plan.

---

## 1. Where a frame's time goes today (measured, dev sandbox, 16 cores, no GPU)

`IMG_0158.tif`, 3266 x 4836, float32 (181 MiB decoded), each stage timed in isolation:

| stage | seconds | on GPU? |
|---|---:|---|
| `read_tiff` (deflate + predictor decode, tifffile/imagecodecs) | 0.42 | no — CPU decompression |
| ICC → ACEScg (`load_working_space_image`, colour-science, banded) | **1.4–2.3** | yes |
| `auto_density_balance` (only with `--auto-density`) | **4.9** | yes (Task 8) |
| `negative_to_positive` (white balance, density balance, invert) | 0.51 | yes |
| print fit (`resolve_tone`: luminance, log, two percentiles) | 0.22 | yes |
| paper curve (`apply_tone`: log, affine, `Cube1D.lookup`) | 0.53 | yes |
| `write_tiff` (131 MiB out) | 0.34 | no — CPU compression/disk |
| export: ACEScg → sRGB 8-bit (colour-science) | 1.70 | yes |
| export: PNG save (Pillow zlib) | 4.9 | no (JPEG save: 0.04) |

Whole CLI runs: `halide invert` with a fixed profile **4.3 s** wall; with `--auto-density` **8.9 s**.

Two findings shape the whole plan:

*Follow-up, after the user asked for D1 to be proven identical:* no faster form is bit-identical.
colour's `vecmul` broadcasts `np.matmul` over per-pixel 3-vectors, and numpy's path for that uses
fused multiply-adds (FMA) in a CPU-dependent way. A per-channel form and a one-call BLAS form
(`xyz @ M.T`) each round the last float64 bit differently in 11–18 % of values. After the cast to
float32: 0 of 188 M values differ on the four real scans, and 1 of 60 M random values differs by
one float32 step. The user then set the bar to "looks identical" (§6 D1). The BLAS form is the
fastest (0.28 s vs 1.37 s per frame), so Task 1 uses it.

1. **Half of a fixed-profile frame is colour-science overhead, not arithmetic.** The ICC step is
   mathematically one 3x3 matrix multiply. Done as a single float32 matrix on the CPU it takes
   **0.07 s instead of 1.36 s**, and differs from today's output by at most **5.9e-6 relative
   (8.9e-8 absolute)** — measured on the whole frame above. Any GPU path has to do exactly this
   (colour-science can't run on a GPU), so GPU output will differ from today's by that much anyway.
   `CLAUDE.md` records fusing the matrices as "deliberately not done, because it would change
   output ~1e-7". That was the right call while bit-identity was free; see decision **D1**.
2. **What a GPU can remove is bounded.** Of the 4.3 s fixed-profile run, ~3.5 s is GPU-able
   arithmetic (ICC + develop + fit + curve); ~0.8 s is decode/encode/startup that stays. Estimate
   for a mid-range card: the 3.5 s becomes ~0.1–0.2 s (≈20 memory passes over 181 MiB at
   300+ GB/s, plus ~30–60 ms of PCIe upload/download), but each new process pays CuPy import
   (0.17 s, measured) + CUDA context creation (0.2–1 s, unmeasured here) + first-use kernel
   compilation (seconds on the very first run ever; cached in `~/.cupy/kernel_cache` after). So:
   - single `invert`, fixed profile: 4.3 s → **~1.5–2 s** (estimate);
   - single `invert --auto-density`: 8.9 s → **~2 s** (estimate, after Task 8);
   - `batch`: today 8 CPU workers already reach ~1.06 s/frame and scaling flattens past ~4 workers
     (disk writes, memory bandwidth — see `CLAUDE.md`). GPU workers each do less CPU work, so the
     same disk becomes the limit sooner. **Expect a smaller batch gain than a single-frame gain**,
     possibly small. Only the user's machine can tell (§7).

## 2. What moves to the GPU, and what doesn't

On the GPU (every per-pixel stage plus the whole-frame statistics, so a frame crosses PCIe exactly
twice — once up after decode, once down before encode):

- ICC → ACEScg (`io/icc.py`, as one precomputed matrix), `scan_gain` multiply
- `negative_to_positive` (`core/density.py`, `core/invert.py`)
- print fit / linear scale (`core/tone_render.py`: `negative_density_range`, `estimate_linear_scale`)
- paper curve (`core/tone_render.py::apply_tone`, `io/lut.py::Cube1D.lookup`)
- `--output density-only`'s balance stages
- export's ACEScg → sRGB → 8-bit (`io/raster.py::to_srgb_8bit`) — Task 6
- per-frame auto calibration (`calibration/auto.py`) — Task 8, last, only if the probe says it pays

Staying on the CPU, deliberately:

- TIFF decode/encode, PNG/JPEG encode, exiftool — compression and disk, not array math.
- `thumbnail_from_linear` (contact sheets) — runs on the host copy that is written anyway.
- The GUI process (`gui/render.py` previews, sampling). Previews are small, and creating a CUDA
  context inside the Qt process for a few megapixels would cost more than it saves. The picker's
  **contact sheet window** develops through batch's own `_worker`, so it gets the GPU for free.
- `--auto-density-roll`'s pre-pass (`estimate_roll_density_profile`): 8x-downsampled frames, tiny.
- Profile solving (`fit_density_balance` etc.): a handful of points.

## 3. Design

### 3.1 CuPy, as an optional extra

- **Why CuPy:** halide's core is plain numpy ufunc code (`np.maximum`, `np.power(out=)`,
  `np.percentile`, fancy indexing); CuPy implements that same API, including `out=` and
  `percentile(overwrite_input=)` (checked in CuPy 14.2's signature). PyTorch/JAX would mean
  rewriting every stage in a different idiom and a multi-GB dependency; Numba CUDA would mean
  hand-written kernels for every stage. None buys accuracy or speed CuPy doesn't.
- **Optional:** `halide gpu --install` or `pip install -e ".[cuda13]"` (see §3.7). The line below
  about `.[gpu]` was written before the CUDA version mattered. `halide` must keep working, identically, with CuPy
  absent. CuPy 14 wheels for cp314 exist for `cupy-cuda12x` and `cupy-cuda13x`; the `[ctk]` extra
  pulls NVIDIA's runtime libraries (cudart, NVRTC, cuBLAS) as pip wheels so no system CUDA Toolkit
  is needed — only the NVIDIA driver. Which of the two depends on the driver (§7).
- Verified here without a GPU: `import cupy` works (0.17 s) and the first CUDA call raises
  `cupy_backends.cuda.api.runtime.CUDARuntimeError` — detection must catch that, not just
  `ImportError`.

### 3.2 One pipeline, two array namespaces

`core/` functions look up the namespace of their input (`xp = array_namespace(img)`) and call
`xp.maximum`, `xp.power`, … instead of `np.…`. On numpy input `xp is numpy`, so **the CPU path
executes exactly the same calls it does today and stays bit-identical** (the existing
`test_banding.py`, golden tests and every other test keep passing unchanged). On a CuPy array the
same function runs on the card. `core/` stays pure: no device selection, no transfers, no CuPy
import — `core/_xp.py` only recognises a CuPy array if `cupy` is already in `sys.modules`.

Not relied on: NumPy's `__array_function__`/`__array_ufunc__` dispatch. CuPy supports some of it,
but it refuses numpy-array operands (`cupy_arr * np.asarray(profile.white_balance)` raises), so
implicit dispatch would fail in exactly the places a profile's tuples become arrays. Explicit `xp`
is clearer and catches mistakes at test time (§3.6).

### 3.3 Where the frame lives (`processing.py`)

```
read_tiff (CPU) ── upload ──> ICC matrix ─ scan_gain ─ [auto calib] ─ negative_to_positive
                                (all on the device, frame resident, per-pixel work in bands)
                              ─ print fit (whole device frame) ─ paper curve ── download ──> write_tiff (CPU)
```

- The decoded host buffer is kept and reused as the download target, so host RAM stays ~1 frame.
- On the device the frame is resident (181 MiB) and the per-pixel stages run through the same
  `map_in_bands` in larger bands (default 64 MiB, a new `band_bytes` parameter) to cap VRAM
  temporaries at a few bands. Estimated device peak per worker: ~181 MiB frame + ~6 x 64 MiB
  band temporaries + ~60 MiB luminance and its sort for the percentile + CUDA context (~300 MiB)
  ≈ **1–1.2 GiB**. The probe measures the real number (§7).
- **Out-of-memory never fails a frame:** the device path works on the device copy and only
  writes into the host buffer at the very end, so on `cupy.cuda.memory.OutOfMemoryError` (or any
  CUDA error) the frame is redone on the CPU from the untouched host buffer, and the result
  carries a warning. Same for a CUDA error mid-batch.
- **ICC on the device** uses `working_space_matrix(profile)` — the source profile's
  RGB→PCS-XYZ matrix composed with colour-science's own XYZ(D50)→ACEScg Bradford matrix, computed
  once in float64 on the CPU (so colour-science stays the authority on the numbers) and applied
  as one float32 multiply. Same for export's ACEScg→sRGB (`srgb_matrix()` + the sRGB transfer
  function).
- **Provenance** gains `"device": "gpu"` / `"cpu"` in the TIFF's JSON, since the two paths are
  close but not bit-identical — a file says which one made it.

### 3.4 Choosing the device

- `halide/device.py`: `resolve_device(requested) -> ComputeDevice`, where `requested` is
  `"auto" | "cpu" | "gpu"`, from `--device` if given, else `HALIDE_DEVICE`, else `"auto"`.
  - `auto`: GPU if CuPy imports, a device exists, and a tiny kernel runs; otherwise CPU, silently
    (a machine without CuPy is the normal case, not a problem).
  - `gpu`: the same checks, but failure is an error naming why ("CuPy is not installed — pip
    install 'halide[gpu]'", "no CUDA device", "driver too old for this CuPy build: …").
  - `cpu`: never touches CuPy.
- Default = `auto` **only if the benchmark in §7 shows a gain on the user's machine**. If it shows
  a gain for single frames but not batch, the default can differ per command (decision **D3**).
- `--device` on `invert`, `batch`, `print`, `export`, `contact`, `calibrate` (its contact sheet).
  The parser must still import nothing heavy — CuPy joins `HEAVY` in `tests/unit/test_cli_startup.py`.
- The run sheet gets a **Compute** row: `GPU — NVIDIA GeForce RTX 3070, 8.0 GiB` or `CPU` (with the
  reason when `auto` fell back from a machine that has CuPy installed, e.g. driver mismatch).

### 3.5 Batch workers

- Workers resolve the device themselves (the parent passes the *resolved* choice, `"cpu"` or
  `"gpu"`, so every worker agrees with the run sheet). CUDA must never be initialised in the
  forkserver process: `_FORKSERVER_PRELOAD` gets no `cupy`, and a test asserts that importing
  `halide.processing` doesn't import CuPy. (A forked CUDA context is unusable in the child.)
- Each GPU worker holds its own CUDA context. `default_worker_count` gains a third cap:
  `free VRAM // per-worker device estimate`, alongside the CPU and RAM caps, and each worker sets
  its CuPy memory-pool limit to its share so one worker can't starve the rest.
- How many GPU workers is best is a measurement, not a formula: fewer workers than on CPU are
  probably enough, because a GPU worker's time is decode + write. Task 7 benchmarks 1/2/4/8 on the
  user's machine before a default is written down.

### 3.7 GPU support is optional, and halide says so (user's requirement)

CuPy plus NVIDIA's runtime libraries is ~1 GB, so a plain install never pulls it in. Halide finds
the card without CuPy, tells the user GPU support exists, and installs it for them on request.

- **Finding the card without CuPy** (`device.py::detect_nvidia_driver() -> NvidiaDriver | None`):
  load the driver library with `ctypes` (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows) and call
  `cuDriverGetVersion` (e.g. `13040`), plus `cuInit`/`cuDeviceGetName` for the card's name. No
  subprocess and no `nvidia-smi` parsing (its header format changes; the user's reads "CUDA UMD
  Version"). This runs only in the main process, only when CuPy isn't installed, and only where a
  hint or `halide gpu` needs it. Any failure means "no card".
- **Which package:** driver ≥ 13000 → `cupy-cuda13x[ctk]`; ≥ 12000 → `cupy-cuda12x[ctk]`; older →
  "driver too old for GPU support (needs CUDA 12 or newer); update the NVIDIA driver". pyproject
  gets matching extras, `cuda12` and `cuda13`, for anyone who prefers `pip install "halide[cuda13]"`.
- **`halide gpu`** (new command) prints what halide sees: the card, the driver's CUDA version,
  whether GPU support is installed, whether it works (smoke kernel), and what `--device auto` will
  use. **`halide gpu --install`** names the exact package and its download size (~1 GB), asks y/N,
  then runs `sys.executable -m pip install <package>` in halide's own environment and re-checks.
  If that environment has no pip (e.g. a `uv tool` install), it prints the right command for pip,
  pipx and uv instead of guessing. The command's help also prints how to uninstall.
- **Telling the user:** only when a card is found and support isn't installed.
  - The run sheet's Compute row reads `CPU — NVIDIA GeForce RTX 3070 found; add GPU support with:
    halide gpu --install`. It's a normal row, not a ⚠.
  - `halide invert` shows the same line **once per machine** (a stamp file beside the completion
    stamps, same mechanism).
  - `--device gpu` without support is an error naming `halide gpu --install`.
  - With no NVIDIA card nothing is said, so people without one aren't pitched a 1 GB download.
- README: a short "GPU acceleration (optional)" section.

### 3.6 Testing without a GPU

This sandbox has no GPU, so the plan tests in three layers:

1. **CPU bit-identity:** every existing test, unchanged. Namespace-generic `core/` must still
   produce identical bytes on numpy (`test_banding.py` already pins this bit-for-bit).
2. **A strict fake device** (`tests/unit/_fake_device.py`): an ndarray subclass `FakeDeviceArray`
   plus a `fake_xp` namespace, registered with `core/_xp.py` for tests. Like CuPy it refuses
   implicit conversion to numpy and refuses plain numpy operands, so any `np.` left in a generic
   function, any missing upload/download, or any profile tuple not converted with `xp.asarray`
   fails loudly on a CPU-only machine — which is where the plumbing bugs will actually be.
   The fake computes with numpy underneath, so its results must equal the CPU path exactly.
3. **Real GPU tests** (`@pytest.mark.gpu`, skipped unless CuPy has a device): the device path vs.
   the CPU path on synthetic images and, when present, the real scans — within the tolerance
   agreed in D2. The user runs these on their machine (`pytest -m gpu`).

## 4. Global Constraints

- CPU output: bit-identical to `dff3cb2` except for Task 1's faster conversions, which must pass
  Task 1's "looks identical" acceptance (8-bit exports pixel-identical, TIFFs < 1/65535 apart).
  Everything after Task 1 keeps the CPU path bit-identical to Task 1's output.
- CuPy is never a required dependency. `pip install -e .` and `.[dev]` never pull it in; the extras
  are `cuda12` / `cuda13`.
- GPU output matches CPU output within the D2 tolerance; the proposal is: float32 pixels within
  1e-5 relative (or 1e-7 absolute), fitted exposure/contrast within 1e-5, 8-bit exports within 1
  code value.
- `halide` with CuPy not installed behaves exactly as today; `pip install -e ".[dev]"` does not
  pull CuPy.
- Building the CLI parser imports no numpy, tifffile, Pillow, colour, scipy, PySide6 **or cupy**.
- `core/` stays pure: no I/O, no device selection, no transfers, no module-level CuPy import.
- A GPU problem never fails a frame that the CPU could have processed; it falls back with a warning.
- Whole-frame statistics (print fit, auto calibration) are never computed on bands or downsampled
  frames, on either device.

## 5. Review Focus

1. **Driver/CuPy mismatch on a machine that has CuPy installed** — `auto` must fall back to CPU and
   say why on the run sheet, not crash at the first kernel. (Task 3 test: detection with a CuPy
   whose `getDeviceCount` raises `CUDARuntimeError`.)
2. **VRAM exhausted mid-batch** (another app on the card, a larger scan) — that frame is redone on
   the CPU, the batch continues, the result shows a warning. (Task 4 test: fake device raising OOM
   inside `negative_to_positive`; output equals the CPU path.)
3. **`halide print` of a flat file** — its exposure-scale undo and fitted print must match the CPU
   path within D2; the ICC-skip for halide's own profile must still skip on the device path.
   (Task 4 test.)
4. **Forkserver + CUDA** — no CuPy import in the forkserver; workers initialise CUDA themselves.
   (Task 7 test.)
5. **Tiny and odd-shaped images** (1 x N, 2 x 2, non-contiguous crops from tifffile) through the
   device path — bands of 1 row, `percentile` on a handful of pixels. (Task 4 test at 1-row bands.)

---

## 6. Decisions (answered by the user, 2026-09-27)

- **D1 → yes, as long as the output images *look* identical** (refined by the user after the
  identity investigation below: bit-identity isn't achievable. colour-science's own float64
  result depends on the CPU's FMA use, and every faster form differed by one float32 step in
  ~1 of 60 M random values, with 0 on the real scans). Task 1's acceptance test defines "looks
  identical": 8-bit exports pixel-identical, TIFFs within one 16-bit step. The GPU's ICC step runs
  in float64 with no fused multiply-adds (a custom CuPy kernel using `__dmul_rn`/`__dadd_rn`).
  Its first step, though, is today's float32 `img @ matrix.T`, which goes through the CPU's BLAS
  library and rounds in its own way, so the GPU ICC result is held to D2, not to identity.
- **D2 → the §4 tolerances.**
- **D3 → `auto`.** GPU whenever it's usable.
- **D4 → `--device auto|cpu|gpu`.**
- **Added: GPU support must stay an optional install** (~1 GB with NVIDIA's runtime libraries).
  Halide must tell users who have an NVIDIA card that it's available, and make adding it easy.
  See §3.7 and Task 6b.

The original questions, for the record:

- **D1. Replace colour-science's per-pixel ICC conversion with one precomputed float32 matrix on
  the CPU too?** Recommended: **yes**. It is the single biggest speed-up available (1.3–2.2 s of a
  4.3 s frame, measured) with or without a GPU; it changes output by ≤ 6e-6 relative (≈ 1/170 of
  one 16-bit step at mid-grey); and the GPU path has to do exactly this anyway, so CPU and GPU then
  agree more closely. `CLAUDE.md` lists this as deliberately not done to keep output bit-identical
  — this is the new evidence to reconsider. If no, CPU stays bit-identical and only the GPU path
  uses the matrix.
- **D2. Accuracy bar for GPU vs CPU.** Bit-identical is not achievable: the GPU's `pow`/`log10`
  round differently in the last bit, and summation order differs. Proposed: the tolerances in §4.
  For scale, the float32 refactor you already accepted moved output ~2e-6 relative.
- **D3. Default.** Proposed: `auto` (GPU when usable) for every command where the §7 benchmark
  shows at least ~20 % less wall time on your machine; CPU elsewhere. Or: one default everywhere.
- **D4. Flag name.** Proposed `--device auto|cpu|gpu` + `HALIDE_DEVICE`. Alternative
  `--gpu/--no-gpu`. (`--device` leaves room for "which GPU" later; not planned.)

## 7. The user's machine (received 2026-09-27)

CachyOS, Python 3.14.7, 8 physical cores (16 threads), ~6.7 GiB RAM free. **RTX 3070, 8 GiB**
(~6.8 GiB free on the desktop), driver 615.71.09 = CUDA driver 13.4 (`13040`). `cupy-cuda13x`
14.2.0 with CUDA runtime 13.2 is installed in the repo's `.venv`. The sandbox shares that `.venv`,
so **tests here see CuPy installed but no driver**. `resolve_device("auto")` must give CPU there,
with a reason.

Probe, second (warm) run (`gpu-probe.txt`, repo root, untracked):

| | CPU s | GPU s | GPU vs CPU |
|---|---:|---:|---|
| CuPy import / CUDA context / first kernel | | 0.24 / 0.19 / 0.17 | (first run ever: 0.39 / 0.26 / 0.48) |
| upload / download 181 MiB | | 0.066 / 0.026 | |
| ICC matrix | 0.058 | 0.009 | identical |
| negative_to_positive | 2.03 | 0.005 | 1.7e-7 rel |
| print-fit percentile | 0.52 | 0.013 | 6.3e-8 rel |
| paper curve | 3.51 | 0.018 | 2.9e-6 rel |
| auto-calibration argsort | 0.96 | 0.006 | **58 % same order** |

The probe held 2.1 GiB of device memory because it didn't use bands. Halide's banded path should
need far less; Task 7 measures it.

The 58 % is expected, not a fault. Many pixels have tied luminance, and neither sort keeps ties in
a fixed order, so auto calibration's density bins can differ at their edges. Task 8 therefore
compares profiles within D2, never by sort order.

Baselines on that machine: `halide invert IMG_0158.tif --profile Roll16-KodakGold200` **4.4 s**.
`halide batch Roll16-Testing … --quiet`, 37 frames: **35.1 s** (0.95 s/frame).

What was asked (kept for the record):

1. `nvidia-smi` (card model, VRAM, driver version and the CUDA version it supports — decides
   `cupy-cuda12x` vs `cupy-cuda13x`).
2. Is halide run directly on that machine (which OS?), or inside a sandbox/container? A Docker
   sandbox needs the NVIDIA container toolkit and `--gpus` to see the card at all.
3. The probe, after installing CuPy into halide's venv:
   ```bash
   .venv/bin/pip install "cupy-cuda12x[ctk]"      # cupy-cuda13x[ctk] if nvidia-smi says CUDA 13.x
   .venv/bin/python docs/plans/gpu-acceleration-probe.py IMG_0158.tif > gpu-probe.txt
   .venv/bin/python docs/plans/gpu-acceleration-probe.py IMG_0158.tif >> gpu-probe.txt   # 2nd run: warm kernel cache
   ```
   It times each pipeline operation on the GPU and the CPU, reports upload/download time, CUDA
   start-up cost, first-kernel compile time, device memory used, and how closely the results
   agree. (Checked here with numpy standing in for CuPy; it hasn't been run against a real GPU.)
4. A baseline on the same machine, for comparing against later:
   ```bash
   time .venv/bin/halide invert IMG_0158.tif /tmp/a.tif --profile <your usual profile>
   time .venv/bin/halide batch Roll16-Testing /tmp/roll16-out --profile <same> --quiet
   ```

---

## 8. File structure

| file | change |
|---|---|
| `src/halide/core/_xp.py` | **new** — `array_namespace(a)`, `register_namespace` (tests) |
| `src/halide/core/density.py`, `invert.py`, `tone_render.py` | per-pixel/whole-frame functions namespace-generic |
| `src/halide/io/lut.py` | `Cube1D.lookup` namespace-generic, table cached per namespace |
| `src/halide/io/icc.py` | **new** `working_space_matrix(profile)`; `convert_to_working_space` uses it if D1 = yes |
| `src/halide/io/raster.py` | **new** `srgb_matrix()`, namespace-generic `to_srgb_8bit` |
| `src/halide/banding.py` | `band_bytes` parameter |
| `src/halide/device.py` | **new** — `ComputeDevice`, `resolve_device`, `to_device`, `to_host` |
| `src/halide/processing.py` | device-aware `process_scan`, `print_scan`, `export_delivery_image`; CPU fallback; provenance `device` |
| `src/halide/batch/orchestrator.py` | pass resolved device to workers; VRAM cap; pool limit |
| `src/halide/cli/_device_args.py` | **new** — `--device` flag + resolution, shared by commands |
| `src/halide/cli/commands/*.py`, `cli/_run_sheet.py` | wire `--device`; Compute row |
| `src/halide/calibration/auto.py` | namespace-generic (Task 8 only) |
| `pyproject.toml` | `gpu` extra; `gpu` pytest marker |
| `tests/unit/_fake_device.py` | **new** — strict fake device |
| `tests/unit/test_xp.py`, `test_device.py`, `test_device_pipeline.py` | **new** |
| `tests/gpu/test_gpu_parity.py` | **new**, `@pytest.mark.gpu` |
| `CLAUDE.md`, `docs/README.md` | decisions and measurements, once the user's numbers are in |

---

## 9. Tasks

### Task 1: Faster colour-profile and sRGB conversions on the CPU (D1: must *look* identical)

**Files:**
- Modify: `src/halide/io/icc.py` (`convert_to_working_space`; new `working_space_matrices`)
- Modify: `src/halide/io/raster.py` (`to_srgb_8bit`; new `srgb_matrix`)
- Test: `tests/unit/test_icc.py`, `tests/unit/test_raster.py`

**Interfaces:**
- Produces:
  - `halide.io.icc.working_space_matrices() -> tuple[np.ndarray, np.ndarray]`: `(M_CAT, M_XYZ_to_ACEScg)`,
    both 3x3 float64, cached, taken from colour-science's own objects exactly as its `XYZ_to_RGB`
    builds them:
    `matrix_chromatic_adaptation_VonKries(xyY_to_XYZ(xy_to_xyY(D50)), xyY_to_XYZ(xy_to_xyY(ACEScg.whitepoint)), transform="Bradford")`
    and `RGB_COLOURSPACES["ACEScg"].matrix_XYZ_to_RGB`.
  - `halide.io.raster.srgb_matrix() -> np.ndarray`: `colour.matrix_RGB_to_RGB(ACEScg, sRGB, "Bradford")`, cached.
  - Tasks 4 and 6 apply both on the GPU.

**What the user decided (D1, 2026-09-27).** The faster form doesn't have to be bit-identical, but
the images it produces must *look* identical. Background, all measured in the dev sandbox:

- colour's `XYZ_to_RGB` is float64 `vecmul(M_CAT, ·)` then `vecmul(M_XYZ_to_RGB, ·)`, where
  `vecmul` = `np.matmul` broadcast over per-pixel 3-vectors. That broadcast is the slow part
  (1.37 s/frame).
- `xyz64 @ M_CAT.T @ M_XYZ.T` (one BLAS call per band) gives the same maths in **0.28 s/frame**.
  In float64 it rounds the last bit differently in ~18% of values (numpy's broadcast path and
  BLAS use FMA differently, and that is CPU-dependent even for colour-science itself). After the
  existing cast back to float32: **0 of 188 M values differ on the four real scans**, and **1 of
  60 M** in a random stress test, by one float32 step (~6e-8 relative, ~1/128 of a 16-bit step).

**Acceptance (what "looks identical" means here). All must hold on every real scan present in the repo root:**
1. The ICC step vs today's colour-science code: every float32 value within **2 float32 ulps**
   (`np.testing.assert_array_max_ulp(new, old, maxulp=2)`), as a unit test on random data and on
   512-row bands of each real scan (skipped if the scans are absent).
2. End to end vs `dff3cb2` (a worktree at that commit), running the real CLI on each real scan
   with `--profile`-equivalent flags `--rm 0.9 --bm 1.1 --rs 1 --bs 1` and again with `--auto-density`:
   - `halide export` PNGs: **pixel-identical** (8-bit, what you look at);
   - `invert` TIFFs (float32 ACEScg): max |difference| **< 1/65535** (below one 16-bit step)
     everywhere. Report the actual max and the count of differing values;
   - the recorded print decision (`exposure`, `contrast`) equal to 6 decimal places.
   Report this as a table in the task report. The CLAUDE.md entry (Step 6) uses the same table.
3. If any criterion fails, stop and report. Don't loosen a criterion.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_icc.py (additions)
def _convert_with_colour(img, profile):
    """Verbatim copy of convert_to_working_space as of dff3cb2 — the D1 oracle."""
    import colour
    matrix = profile.rgb_to_pcs_xyz.astype(img.dtype, copy=False)
    pcs_xyz = img @ matrix.T
    working = colour.XYZ_to_RGB(
        pcs_xyz,
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False,
    )
    return np.asarray(working, dtype=img.dtype)


def test_working_space_matrices_reproduce_colour_science():
    import colour
    m_cat, m_xyz = working_space_matrices()
    xyz = np.random.default_rng(0).uniform(0, 1, (1000, 3))
    expected = colour.XYZ_to_RGB(xyz, colourspace=colour.RGB_COLOURSPACES["ACEScg"],
                                 illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
                                 chromatic_adaptation_transform="Bradford", apply_cctf_encoding=False)
    np.testing.assert_allclose(xyz @ m_cat.T @ m_xyz.T, expected, rtol=1e-14, atol=0)


def test_convert_to_working_space_within_two_ulps_of_colour_science():
    rng = np.random.default_rng(3)
    img = (10.0 ** rng.uniform(-5, 0.5, (1000, 1000, 3))).astype(np.float32)
    profile = parse_linear_rgb_profile(<the linear Rec.2020 profile bytes this file already builds>)
    np.testing.assert_array_max_ulp(convert_to_working_space(img, profile), _convert_with_colour(img, profile), maxulp=2)


@pytest.mark.parametrize("name", ["IMG_0151.tif", "IMG_0156.tif", "IMG_0156-nowb.tif", "IMG_0158.tif"])
def test_convert_to_working_space_on_real_scans(name):
    path = Path(__file__).resolve().parents[2] / name
    if not path.exists():
        pytest.skip(f"{name} not present (real scans are local-only)")
    scan = read_tiff(path)
    profile = parse_linear_rgb_profile(scan.icc_profile)
    band = np.ascontiguousarray(scan.image[:512])
    np.testing.assert_array_max_ulp(convert_to_working_space(band, profile), _convert_with_colour(band, profile), maxulp=2)
```

`tests/unit/test_raster.py`: `srgb_matrix` vs `colour.matrix_RGB_to_RGB` (rtol 1e-14), and the new
`to_srgb_8bit` vs a verbatim copy of today's body on 1 M random pixels spanning -0.1…1.5 (including
values right at 8-bit rounding boundaries). Require identical `uint8` output; if a handful differ
by 1, report how many out of how many and stop for review.

- [ ] **Step 2: Run to verify they fail**: `.venv/bin/python -m pytest tests/unit/test_icc.py tests/unit/test_raster.py -q` → ImportError.

- [ ] **Step 3: Implement**

```python
# src/halide/io/icc.py
@functools.cache
def working_space_matrices() -> tuple[np.ndarray, np.ndarray]:
    """colour-science's own two float64 matrices for XYZ(D50) -> ACEScg: the Bradford adaptation
    from D50 to ACEScg's white, then ACEScg's XYZ -> RGB. Built exactly as colour's XYZ_to_RGB
    builds them. Applying them as one matrix multiply per band, rather than colour's per-pixel
    broadcast, is ~5x faster; after the cast to float32 the results match colour-science to within
    one float32 step (see CLAUDE.md, "D1")."""
    import colour
    from colour.adaptation import matrix_chromatic_adaptation_VonKries
    from colour.models import xy_to_xyY, xyY_to_XYZ

    acescg = colour.RGB_COLOURSPACES["ACEScg"]
    d50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    m_cat = matrix_chromatic_adaptation_VonKries(
        xyY_to_XYZ(xy_to_xyY(d50)), xyY_to_XYZ(xy_to_xyY(acescg.whitepoint)), transform="Bradford"
    )
    return np.asarray(m_cat, dtype=np.float64), np.asarray(acescg.matrix_XYZ_to_RGB, dtype=np.float64)


def convert_to_working_space(img: np.ndarray, profile: LinearRGBProfile) -> np.ndarray:
    matrix = profile.rgb_to_pcs_xyz.astype(img.dtype, copy=False)   # unchanged (see its comment)
    pcs_xyz = (img @ matrix.T).astype(np.float64)                     # float64, as colour computed
    m_cat, m_xyz_to_acescg = working_space_matrices()
    working = pcs_xyz @ m_cat.T
    working = working @ m_xyz_to_acescg.T
    return working.astype(img.dtype)
```

The colour import moves from `convert_to_working_space` into `working_space_matrices` (still lazy,
still off the CLI start-up path). `to_srgb_8bit`: `(acescg.astype(float64) @ srgb_matrix().T)`,
then **colour's own** `colour.RGB_COLOURSPACES["sRGB"].cctf_encoding` (the function colour already
calls, identical by construction), then today's clip/round/cast. Measure first whether the matrix
or the cctf dominates `to_srgb_8bit`'s 1.7 s, and report both.

- [ ] **Step 4: Run all tests**: `.venv/bin/python -m pytest tests/ -q`. Bit-for-bit tests that
  compare the banded pipeline with the whole-array pipeline (`test_banding.py`) must still pass
  unchanged, because both sides use the new code. If a golden test compares against stored
  pre-change output, report it; don't edit it silently.
- [ ] **Step 5: End-to-end acceptance vs `dff3cb2`** (criterion 2). Include the timing of
  `load_working_space_image` and `write_delivery_image` on IMG_0158, before and after.
- [ ] **Step 6: Update `CLAUDE.md`**. The banding entry's "Deliberately *not* done" list names
  "fusing the two ICC matrices" as something that would change output. Replace it with the D1
  record: the user accepted "looks identical" (2026-09-27), the approach, the acceptance table
  from Step 5, the timings, and the finding that colour-science's own float64 result is
  CPU-dependent at the last-bit level.
- [ ] **Step 7: Commit**: `git commit -m "Colour-profile and sRGB conversions ~5x faster; visually identical output (D1)"`

### Task 2: Namespace-generic core, pinned by a strict fake device

**Files:**
- Create: `src/halide/core/_xp.py`, `tests/unit/_fake_device.py`, `tests/unit/test_xp.py`
- Modify: `src/halide/core/density.py` (`apply_white_balance`, `apply_density_balance`),
  `src/halide/core/invert.py`, `src/halide/core/tone_render.py` (`negative_density_range`,
  `estimate_linear_scale`, `apply_tone`, `linear_passthrough`), `src/halide/io/lut.py` (`Cube1D.lookup`)

**Interfaces:**
- Produces: `array_namespace(a) -> module` (numpy, cupy, or a registered test namespace);
  `register_namespace(array_type: type, namespace) -> None` (tests only).

- [ ] **Step 1: Write the fake device and failing tests**

```python
# tests/unit/_fake_device.py
"""A CPU stand-in for a GPU array library, strict in the ways CuPy is: no implicit conversion to
numpy, no mixing with plain numpy arrays. Computes with numpy underneath, so results must be
bit-identical to the CPU path — any difference is a plumbing bug, not device arithmetic."""
import types
import numpy as np


class FakeDeviceArray(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not _ACTIVE:
            raise TypeError(f"numpy ufunc {ufunc.__name__} called on a device array — use xp")
        for x in inputs + tuple(kwargs.get("out", ())):
            if isinstance(x, np.ndarray) and not isinstance(x, FakeDeviceArray):
                raise TypeError("plain numpy array mixed with a device array — upload it with xp.asarray")
        inputs = tuple(x.view(np.ndarray) if isinstance(x, FakeDeviceArray) else x for x in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(o.view(np.ndarray) for o in kwargs["out"])
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return _wrap(result)

    def __array_function__(self, func, types_, args, kwargs):
        if not _ACTIVE:
            raise TypeError(f"np.{func.__name__} called on a device array — use xp")
        return _wrap(func._implementation(*_unwrap(args), **_unwrap(kwargs)))


_ACTIVE = False  # True only while a fake_xp function is running

def _wrap(x):
    if isinstance(x, np.ndarray) and not isinstance(x, FakeDeviceArray):
        return x.view(FakeDeviceArray)
    if isinstance(x, (tuple, list)):
        return type(x)(_wrap(v) for v in x)
    return x


def _unwrap(x):
    if isinstance(x, FakeDeviceArray):
        return x.view(np.ndarray)
    if isinstance(x, (tuple, list)):
        return type(x)(_unwrap(v) for v in x)
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x

def _namespaced(fn):
    def call(*args, **kwargs):
        global _ACTIVE
        previous, _ACTIVE = _ACTIVE, True
        try:
            return _wrap(fn(*_unwrap(args), **_unwrap(kwargs)))
        finally:
            _ACTIVE = previous
    return call

fake_xp = types.SimpleNamespace(**{
    name: _namespaced(getattr(np, name))
    for name in ("asarray", "maximum", "minimum", "power", "divide", "log10", "clip", "floor",
                 "percentile", "where", "empty", "zeros_like", "argsort", "median", "concatenate")
}, float32=np.float32, int32=np.int32, __name__="fake_xp")

def to_device(a: np.ndarray) -> FakeDeviceArray:
    return np.array(a, copy=True).view(FakeDeviceArray)

def to_host(a: FakeDeviceArray) -> np.ndarray:
    return np.array(a.view(np.ndarray), copy=True)
```

(The executor adds any numpy name a generic function turns out to need to the tuple above. Operators (`*`, `@`, `+=`, indexing) on a
`FakeDeviceArray` go through `__array_ufunc__`/`__getitem__`; `@` and in-place operators must be
allowed while *not* in a namespace call too — mark them by overriding `__matmul__`, `__imul__` etc.
to set `_ACTIVE` around `super()`, since CuPy arrays support operators natively.)

```python
# tests/unit/test_xp.py
import numpy as np
import pytest
from halide.core._xp import array_namespace, register_namespace
from halide.core.pipeline import negative_to_positive
from halide.core.tone_render import ResolvedTone, apply_tone, negative_density_range, estimate_linear_scale
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


def test_negative_to_positive_on_device_equals_cpu():
    neg = _negative()
    assert np.array_equal(to_host(negative_to_positive(to_device(neg), PROFILE)), negative_to_positive(neg, PROFILE))


def test_print_fit_statistics_on_device_equal_cpu():
    pos = negative_to_positive(_negative(), PROFILE)
    assert negative_density_range(to_device(pos)) == negative_density_range(pos)
    assert estimate_linear_scale(to_device(pos)) == estimate_linear_scale(pos)


@pytest.mark.parametrize("resolved", [ResolvedTone("paper", exposure=0.4, contrast=0.8), ResolvedTone("linear", linear_scale=0.05)])
def test_apply_tone_on_device_equals_cpu(resolved):
    pos = negative_to_positive(_negative(), PROFILE)
    assert np.array_equal(to_host(apply_tone(to_device(pos), resolved)), apply_tone(pos, resolved))
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/test_xp.py -q` → `ModuleNotFoundError: halide.core._xp`, then (after Step 3's `_xp.py` alone) `TypeError: numpy ufunc maximum called on a device array`.

- [ ] **Step 3: Implement**

```python
# src/halide/core/_xp.py
"""Which array library an array belongs to — numpy, or CuPy for a frame on the GPU — so core/'s
functions are written once and run on either (see docs/plans/gpu-acceleration.md). Pure: never
imports CuPy; a CuPy array can only exist if the caller already imported it."""
from __future__ import annotations

import sys
import numpy as np

_REGISTERED: dict[type, object] = {}


def register_namespace(array_type: type, namespace) -> None:
    _REGISTERED[array_type] = namespace


def array_namespace(a):
    for array_type, namespace in _REGISTERED.items():
        if isinstance(a, array_type):
            return namespace
    cupy = sys.modules.get("cupy")
    if cupy is not None and isinstance(a, cupy.ndarray):
        return cupy
    return np
```

Then, in each listed function, `xp = array_namespace(<input>)` and replace `np.` with `xp.` for
every call on image data; profile tuples become `xp.asarray(…, dtype=img.dtype)`. Examples:

```python
# core/density.py
def apply_white_balance(img, profile):
    xp = array_namespace(img)
    return img * xp.asarray(profile.white_balance, dtype=img.dtype)

# io/lut.py — Cube1D.lookup: the table is uploaded once per (namespace, dtype), not per band
    xp = array_namespace(x)
    values = self._table(xp, x.dtype)   # functools.cache'd helper keyed on (id(xp), dtype)
    ...
    floor_t = xp.floor(t)
```

`negative_density_range`'s `@ xp.asarray(ACESCG_LUMINANCE, …)`, `xp.percentile(…, overwrite_input=True)`,
and `float(d_lo)` (works on CuPy 0-d arrays; it synchronises, which is fine — it's once per frame).
The `ndim == 3` branch's `np.array(positive_linear, copy=True)` becomes `xp.array(…, copy=True)`.
Leave `fit_print`, `paper_exposure_range`, `_curve_input_at` untouched — they only work on the
1 000-entry curve table, which stays a host numpy array.

- [ ] **Step 4: Run** — `pytest tests/unit/test_xp.py -q` passes, **and the full suite passes
  unchanged** (`pytest tests/ -q`): the CPU path is untouched bit-for-bit, `test_banding.py` proves it.

- [ ] **Step 5: Commit** — `git commit -m "core: run on numpy or a GPU array library, one implementation"`

### Task 3: Device detection and selection

**Files:**
- Create: `src/halide/device.py`, `tests/unit/test_device.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ComputeDevice:
      kind: str                     # "cpu" | "gpu"
      name: str | None = None       # "NVIDIA GeForce RTX 3070"
      memory_free: int | None = None
      memory_total: int | None = None
      fallback_reason: str | None = None   # why "auto" chose CPU on a machine with CuPy
  class DeviceUnavailableError(Exception): ...
  def resolve_device(requested: str | None = None) -> ComputeDevice   # None -> $HALIDE_DEVICE -> "auto"
  def to_device(a: np.ndarray): ...   # cupy.asarray
  def to_host(a, out: np.ndarray | None = None) -> np.ndarray   # a.get(out=out)
  DEVICE_ENV = "HALIDE_DEVICE"
  ```

- [ ] **Step 1: Write failing tests** — with `monkeypatch.setitem(sys.modules, "cupy", fake)`:

```python
def test_cpu_request_never_imports_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import would raise
    assert resolve_device("cpu").kind == "cpu"

def test_auto_without_cupy_is_cpu_with_no_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert resolve_device("auto") == ComputeDevice(kind="cpu")

def test_auto_with_broken_driver_falls_back_and_says_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _cupy_whose_getDeviceCount_raises(
        "cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version"))
    device = resolve_device("auto")
    assert device.kind == "cpu" and "driver" in device.fallback_reason

def test_gpu_request_without_cupy_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(DeviceUnavailableError, match=r"halide gpu --install"):
        resolve_device("gpu")

def test_env_var_is_used_when_no_flag(monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "cpu")
    assert resolve_device(None).kind == "cpu"

def test_bad_env_value_is_an_error(monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "tpu")
    with pytest.raises(ValueError, match="HALIDE_DEVICE"):
        resolve_device(None)

def test_importing_processing_does_not_import_cupy():
    code = "import sys, halide.processing; print('cupy' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — `resolve_device`: `"cpu"` returns at once; otherwise `import cupy`
  (ImportError → none), `cupy.cuda.runtime.getDeviceCount()`, a smoke kernel
  `(cupy.arange(4, dtype=cupy.float32) ** 1.5).sum().item()`, name from
  `getDeviceProperties(0)["name"].decode()`, memory from `cupy.cuda.Device(0).mem_info`. Catch
  `Exception` around all of it (CuPy's error classes live in `cupy_backends`, and a broken install
  can raise others); for `"gpu"` re-raise as `DeviceUnavailableError` with the reason, for `"auto"`
  return CPU with `fallback_reason` set (None when CuPy simply isn't installed).
  `pyproject.toml`: `cuda12 = ["cupy-cuda12x[ctk]>=14"]` and `cuda13 = ["cupy-cuda13x[ctk]>=14"]`
  under optional-dependencies, and `markers = ["gpu: needs CuPy and a CUDA device"]` under `[tool.pytest.ini_options]`.
- [ ] **Step 4: Run** — `pytest tests/unit/test_device.py -q` passes; full suite passes.
- [ ] **Step 5: Commit** — `git commit -m "Detect and choose the compute device (auto/cpu/gpu)"`

### Task 4: Develop a frame on the device (`process_scan`, `print_scan`)

**Files:**
- Modify: `src/halide/banding.py` (`band_bytes` parameter), `src/halide/processing.py`
- Test: `tests/unit/test_device_pipeline.py` (fake device), `tests/gpu/test_gpu_parity.py`

**Interfaces:**
- Consumes: `array_namespace` (Task 2), `ComputeDevice`, `to_device`, `to_host` (Task 3),
  `working_space_matrix` (Task 1).
- Produces: `process_scan(..., device: ComputeDevice | None = None)`,
  `print_scan(..., device=None)`; `None` = CPU (every existing caller unchanged).
  `map_in_bands(src, fn, out=None, band_bytes=_BAND_BYTES)`. `DEVICE_BAND_BYTES = 64 * 2**20`.
  Provenance gains `"device"`. `process_scan` returns `ResolvedTone | None` as before; a CPU
  fallback is reported through a new optional `on_warning: Callable[[str], None] | None` argument.

- [ ] **Step 1: Write failing tests** (fake device injected by monkeypatching
  `halide.device.to_device`/`to_host` and `register_namespace`, and a `ComputeDevice(kind="gpu")`):

```python
def test_process_scan_on_device_matches_cpu_bit_for_bit(tmp_path, fake_gpu):
    scan = write_linear_rec2020_tiff(tmp_path / "neg.tif")   # existing test helper
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu)
    assert np.array_equal(read_tiff(tmp_path / "cpu.tif").image, read_tiff(tmp_path / "gpu.tif").image)

def test_device_out_of_memory_redoes_the_frame_on_cpu(tmp_path, fake_gpu, monkeypatch):
    # make the second band of negative_to_positive raise like cupy.cuda.memory.OutOfMemoryError
    ...
    warnings = []
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu, on_warning=warnings.append)
    assert np.array_equal(read_tiff(tmp_path / "gpu.tif").image, cpu_output)
    assert "out of GPU memory" in warnings[0]

def test_provenance_records_the_device(tmp_path, fake_gpu): ...   # "device": "gpu" / "cpu"
def test_print_scan_on_device_matches_cpu(tmp_path, fake_gpu): ...  # incl. linear_scale undo
def test_one_row_bands_on_device(tmp_path, fake_gpu, monkeypatch):  # DEVICE_BAND_BYTES -> 1
def test_tiny_image_on_device(tmp_path, fake_gpu): ...               # 2 x 2 and 1 x 9 scans
```

(Bit-for-bit is the right bar here because the fake computes with numpy; the real-GPU version of
each test lives in `tests/gpu/` with the D2 tolerance.)

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** in `processing.py`:

```python
def _develop_on_device(host: np.ndarray, source_matrix: np.ndarray | None, scan_gain: float,
                       profile_or_none, stage, tone_params, device) -> tuple[ResolvedTone | None, DensityProfile]:
    """The whole of process_scan's arithmetic on the device; writes into `host` only at the end,
    so a device failure leaves `host` as decoded and the CPU path can start over from it."""
    frame = to_device(host)                                           # one upload
    if source_matrix is not None:                                     # None: already ACEScg
        map_in_bands(frame, lambda b: b @ xp.asarray(source_matrix, dtype=b.dtype), band_bytes=DEVICE_BAND_BYTES)
    if scan_gain != 1.0:
        frame *= scan_gain
    profile = profile_or_none or auto_density_balance(to_host(frame))  # Task 8 removes this round trip
    ...same stage logic as _develop_in_place, with band_bytes=DEVICE_BAND_BYTES...
    to_host(frame, out=host)                                          # one download
    return resolved, profile
```

`load_working_space_image` is split so the device path gets the *decoded, unconverted* buffer plus
the matrix (`working_space_matrix(profile)`, or `None` for halide's own output profile — keep the
existing byte-identical skip). `process_scan` calls `_develop_on_device` inside
`try/except Exception` when `device.kind == "gpu"`, and on failure re-reads nothing: it runs the
existing CPU code on `host` (still the decoded scan — convert it now) and calls `on_warning`.
Free device memory after each frame: `del frame` then
`cupy.get_default_memory_pool().free_all_blocks()` only in the single-frame CLI path (batch workers
keep the pool warm — Task 7).

- [ ] **Step 4: Run** — new tests pass; full suite passes.
- [ ] **Step 5: GPU parity tests** — `tests/gpu/test_gpu_parity.py`, `pytestmark = pytest.mark.gpu`,
  module-level `pytest.importorskip("cupy")` plus a skip if `resolve_device("auto").kind != "gpu"`:
  the same scenarios on real CuPy against the CPU path with the D2 tolerances, and — if
  `IMG_0158.tif` etc. exist in the repo root — on the real scans, asserting fitted exposure/contrast
  within 1e-5 and pixels within D2. (The user runs `pytest -m gpu`.)
- [ ] **Step 6: Commit** — `git commit -m "Develop frames on the GPU, falling back to the CPU on any device error"`

### Task 5: `--device` on the CLI, and the Compute row

**Files:**
- Create: `src/halide/cli/_device_args.py`
- Modify: `src/halide/cli/commands/{invert,batch,print,export,contact,calibrate}_cmd.py`,
  `src/halide/cli/_run_sheet.py`, `src/halide/cli/completion.py` (`_KINDS` needs no entry — choices are static)
- Test: `tests/unit/test_cli_startup.py` (add `"cupy"` to `HEAVY`), `tests/unit/test_device_cli.py`

**Interfaces:**
- Produces: `add_device_argument(parser)`; `resolve_device_arg(args) -> ComputeDevice`
  (turns `DeviceUnavailableError`/bad env into the CLI's usual error exit); `device_row(device) -> row`.

- [ ] **Step 1: Failing tests** — `HEAVY` includes `cupy`; `halide invert --device cpu …` writes
  `"device": "cpu"`; `--device gpu` on a machine without CuPy exits non-zero naming `halide gpu --install`;
  `--device` absent + `HALIDE_DEVICE=cpu` → cpu; run sheet shows `Compute  CPU` / `Compute  GPU — <name>`
  and the fallback reason when set.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — `parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=None,
  help="where to do the arithmetic: auto (GPU if one is usable; default), cpu, or gpu (fail if unusable). "
  "Also HALIDE_DEVICE.")`. Resolve once, early in `run()`, after calibration-source prompts (the order
  rule in `CLAUDE.md` for `choose_calibration_source`) and before the run sheet closes. `invert`
  prints the device on its existing summary line. The default (`auto` vs `cpu`) is one constant,
  `DEFAULT_DEVICE`, set per D3 after Task 7's numbers.
- [ ] **Step 4: Run** — new tests, `test_cli_startup.py`, `test_completion.py` pass; full suite passes.
  Regenerate completions by running `halide --help` once in a terminal and confirm `--device` completes in zsh.
- [ ] **Step 5: Commit** — `git commit -m "Add --device auto|cpu|gpu and HALIDE_DEVICE; show it on the run sheet"`

### Task 6: Export on the device

**Files:**
- Modify: `src/halide/io/raster.py` (`to_srgb_8bit` namespace-generic via `srgb_matrix` + `encode_srgb`),
  `src/halide/processing.py::export_delivery_image(…, device=None)`
- Test: `tests/unit/test_raster.py`, `tests/gpu/test_gpu_parity.py`

- [ ] **Step 1: Failing tests** — fake device: exported PNG pixels equal the CPU path's exactly;
  real GPU: within 1 code value, and ≥ 99.9 % of pixels identical (reported, so a systematic
  offset can't hide inside "≤ 1").
- [ ] **Step 2–4:** implement (`to_srgb_8bit(xp_array)` → matrix, `encode_srgb`, `xp.clip`,
  `(… * 255).round().astype(xp.uint8)`; the 8-bit result is downloaded band by band into the
  existing preallocated host `uint8` buffer), run, pass.
- [ ] **Step 5: Commit** — `git commit -m "export: sRGB conversion on the GPU"`

Note for the user, not part of this plan: PNG export's real cost is Pillow's zlib (4.9 s of 6.7 s,
measured); a GPU can't touch that. A lower PNG `compress_level` would, at the cost of bigger files.

### Task 6b: GPU support is optional, discoverable, and one command to add (§3.7)

**Files:**
- Modify: `src/halide/device.py` (add `NvidiaDriver`, `detect_nvidia_driver`, `cupy_package_for`)
- Create: `src/halide/cli/commands/gpu_cmd.py`; register it in `src/halide/cli/main.py`
- Modify: `src/halide/cli/_device_args.py` / `_run_sheet.py` (Compute row hint), `invert_cmd.py`
  (once-per-machine note), `src/halide/cli/completion.py` (the new command must complete: it is
  generated from `build_parser()`, so check `test_completion.py` still passes and `gpu` appears)
- Modify: `pyproject.toml` (extras `cuda12 = ["cupy-cuda12x[ctk]>=14"]`, `cuda13 = ["cupy-cuda13x[ctk]>=14"]`;
  remove any `gpu` extra added in Task 3), `README.md` ("GPU acceleration (optional)" section)
- Test: `tests/unit/test_gpu_cmd.py`

**Interfaces:**
- Consumes: `resolve_device`, `ComputeDevice`, `DeviceUnavailableError` (Task 3); run sheet rows (Task 5).
- Produces:
  ```python
  @dataclass(frozen=True)
  class NvidiaDriver:
      cuda_version: int          # cuDriverGetVersion, e.g. 13040
      device_name: str | None    # first device, None if cuDeviceGetName failed
  def detect_nvidia_driver() -> NvidiaDriver | None     # ctypes only; never raises
  def cupy_package_for(driver: NvidiaDriver) -> str | None
      # >= 13000 -> "cupy-cuda13x[ctk]"; >= 12000 -> "cupy-cuda12x[ctk]"; else None (driver too old)
  def gpu_support_installed() -> bool                  # importlib.util.find_spec("cupy") is not None
  ```

- [ ] **Step 1: Failing tests**
  - `cupy_package_for`: 13040 → cuda13x, 12080 → cuda12x, 11080 → None.
  - `detect_nvidia_driver` with `ctypes.CDLL` monkeypatched to raise `OSError` → None; with a fake
    library whose `cuDriverGetVersion` writes 13040 and `cuDeviceGetName` writes
    `b"NVIDIA GeForce RTX 3070"` → both fields. It must never raise, even if the fake returns error codes.
  - `halide gpu` (status), for four machines: (a) no NVIDIA driver → "No NVIDIA GPU found" and exit 0;
    (b) driver, no CuPy → names the card and says "GPU support is not installed — add it with:
    halide gpu --install (about 1 GB download)"; (c) CuPy installed and working → "in use by default
    (--device auto)"; (d) CuPy installed, device fails (the sandbox's real case: `CUDARuntimeError`
    insufficient driver) → shows the reason.
  - `halide gpu --install`, with `subprocess.run` monkeypatched: answers "n" → nothing runs;
    answers "y" → runs `[sys.executable, "-m", "pip", "install", "cupy-cuda13x[ctk]"]` exactly;
    with no pip in the environment (`importlib.util.find_spec("pip")` → None) → runs nothing and
    prints the pip, pipx (`pipx inject halide 'cupy-cuda13x[ctk]'`) and uv
    (`uv tool install --with 'cupy-cuda13x[ctk]' …`) commands. Non-interactive stdin without `--yes`
    → refuses with a message naming `--yes`. With a driver too old → explains, runs nothing, exit 1.
  - Run sheet Compute row: driver found + CuPy absent → `CPU — NVIDIA GeForce RTX 3070 found; add GPU
    support with: halide gpu --install`. No driver → just `CPU`.
  - `invert` prints the hint once: a stamp file under the same state directory
    `cli/completion.py` uses (reuse its helper for the path). Second run → no hint.
    `HALIDE_NO_GPU_HINT=1` → never.
  - `--device gpu` without CuPy: the error names `halide gpu --install`.
  - `test_cli_startup.py` still passes (`gpu_cmd` imports nothing heavy at parser build; ctypes is fine).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** `detect_nvidia_driver`: `ctypes.CDLL("libcuda.so.1")` on Linux,
  `ctypes.WinDLL("nvcuda.dll")` on Windows, other platforms → None. Call `cuDriverGetVersion(byref(c_int))`,
  then `cuInit(0)`, `cuDeviceGet(byref(dev), 0)`, `cuDeviceGetName(buf, 256, dev)`. Check each CUresult
  (0 = success). Wrap everything in `try/except Exception`. The install prompt text:
  ```
  GPU support for halide: NVIDIA GeForce RTX 3070 (driver supports CUDA 13.4)
  This installs cupy-cuda13x[ctk] into halide's Python environment (<sys.prefix>) — about 1 GB,
  mostly NVIDIA's CUDA libraries. Remove it later with:
      <sys.executable> -m pip uninstall cupy-cuda13x
  Install now? [y/N]
  ```
  After install, re-run the status check in a **fresh subprocess** (the current process may
  have cached the failed import) and print the result.
- [ ] **Step 4: Run** the new tests, `test_completion.py`, `test_cli_startup.py`, and the full suite.
  In the sandbox (CuPy installed, no driver), run `halide gpu` for real and paste its output in the report.
- [ ] **Step 5: README section** (short: what it does, "needs an NVIDIA card", `halide gpu`,
  `halide gpu --install`, `--device cpu` / `HALIDE_DEVICE=cpu` to turn it off, the ~1 GB size).
- [ ] **Step 6: Commit**: `git commit -m "halide gpu: find the card, offer and install optional GPU support"`

### Task 7: Batch workers on the GPU, and the benchmark that sets the default

**Files:**
- Modify: `src/halide/batch/orchestrator.py` (workers take `device_kind: str`; `default_worker_count`
  gains a VRAM cap; per-worker CuPy pool limit), `src/halide/cli/commands/batch_cmd.py`
- Create: `scripts/bench_device.py` (times `invert` and `batch` at `--device cpu` vs `gpu` and
  `--workers 1 2 4 8`, prints a table; used once to fill in `CLAUDE.md`)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `ComputeDevice` (Task 3), `process_scan(device=…)` (Task 4).
- Produces: `estimate_worker_device_bytes(jobs) -> int`
  (`_DEVICE_CONTEXT_BYTES + _DEVICE_FRAME_MULTIPLIER * decoded_pixel_bytes`, constants from the
  §7 probe, rounded up); `default_worker_count(jobs, *, device: ComputeDevice | None = None, …)`.

- [ ] **Step 1: Failing tests**
  - `_FORKSERVER_PRELOAD` contains no `cupy`, and a forkserver pool worker reports
    `"cupy" not in sys.modules` before its first job.
  - `default_worker_count` with `ComputeDevice(kind="gpu", memory_free=4 * 2**30)` and a 181 MiB
    job is capped by `4 GiB // estimate_worker_device_bytes`; with `kind="cpu"` unchanged.
  - A worker whose frame hit the CPU fallback returns a `BatchResult` with that warning.
- [ ] **Step 2–4:** implement; the worker resolves `device_kind == "gpu"` into a `ComputeDevice`
  on first use (module-level cache), sets
  `cupy.get_default_memory_pool().set_limit(size=share)`, and passes it to `process_scan`.
- [ ] **Step 5: The user runs `scripts/bench_device.py` on their machine** (Roll16-Testing and the
  single scans). Record the table in `CLAUDE.md` ("Decisions and why"), set `DEFAULT_DEVICE` per D3,
  and tune the GPU worker default (it may well be lower than the CPU one).
- [ ] **Step 6: Commit** — `git commit -m "batch: GPU workers with a VRAM-aware default count"`

### Task 8 (only if the probe shows `--auto-density` is worth it): auto calibration on the device

**Files:** `src/halide/calibration/auto.py` (`_density_local_saturation`, `_saturation`,
`_neutral_candidate_mask`, `_neutral_candidates`, `_shadow_and_highlight_from_candidates`
namespace-generic; `solve_density_balance` stays CPU — it receives 3-element host arrays),
`tests/unit/test_auto_calibration.py`, `tests/gpu/test_gpu_parity.py`.

- [ ] **Step 1: Failing tests** — fake device: `auto_density_balance(to_device(img)) == auto_density_balance(img)`
  exactly (same `DensityProfile`), including the existing oracle test of the old two-step form.
  Real GPU: white balance and density scale within 1e-5 relative on the real scans. Note the
  argsort is not stable-equal across devices (ties in luminance may order differently), which can
  move a bin boundary by a pixel — the parity test must use the D2 tolerance, not equality.
- [ ] **Step 2–4:** implement with `xp = array_namespace(pixels)`; `np.array_split` of the sort
  order becomes explicit slice bounds (the existing code already computes them); the final
  shadow/highlight percentiles return `to_host(...)` 3-vectors. Remove Task 4's `to_host` round trip.
- [ ] **Step 5: Commit** — `git commit -m "Auto calibration on the GPU"`

### Task 9: Record it

- [ ] `CLAUDE.md`: a "Decisions and why" entry — CuPy and why; namespace-generic `core/`; frame
  resident on the device, one upload/one download; CPU fallback; provenance `device`; the D1/D2/D3
  outcomes; the user's measured numbers (probe + benchmark); the GPU worker default and how it was
  chosen; forkserver/CUDA rule. Update the Architecture tree (`device.py`, `core/_xp.py`) and
  Commands (`--device`, `halide gpu --install` / `pip install -e ".[cuda13]"`, `pytest -m gpu`).
- [ ] `docs/README.md`: move this plan's line to "Implemented".
- [ ] Commit.

---

## 10. Out of scope

- Multiple GPUs, AMD/Apple GPUs (CuPy's ROCm build could be a later `--device` value; the
  namespace design doesn't preclude it), GPU TIFF/PNG codecs (nvCOMP), GPU in the picker's live
  preview, `float16` anywhere.
