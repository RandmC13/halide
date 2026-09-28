# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`halide` inverts digital camera/scanner captures of developed color negative film into positive
images. The user is a film photographer who develops, scans, and edits their own film, and the
single non-negotiable priority is **colorimetric faithfulness to what the film actually recorded**
— not a "nice-looking" or "punchy" result. Anything that trades accuracy for a subjectively
pleasing look, or that requires per-image manual eyeballing to get right, is against the point of
this project.

The math implements the method from Aaron Buchler's blog post "Scanning Color Negative Film"
(local copy: `filminversion-ab.pdf`, gitignored/untracked — ask the user if you need it) and its
reference implementations at `github.com/abpy/color-neg-resources` and
`github.com/alchemy-color/color-negative-inversion`. Core idea: white balance (linear per-channel
multiply) + density balance (per-channel power function equalizing dye-layer contrast) + invert
(reciprocal) + a print-emulating tone curve, all in a linear wide-gamut working color space.

This is a from-scratch rewrite of a single 785-line script (preserved at `legacy/halide_v1.py` for
reference/comparison — do not extend it, it is superseded). The rewrite exists because the old
script clipped inconsistently, had no real color management, calibrated density balance from
statistically unsound guesses, and had zero test coverage. All of that has been fixed; see
"Decisions and why" below before assuming any of it needs revisiting.

## Commands

Environment setup (Python 3.11+; a `.venv` already exists in this repo — use it):
```bash
.venv/bin/pip install -e ".[dev]"
```

Run the full test suite:
```bash
.venv/bin/python -m pytest tests/ -q
```
Run one file or one test:
```bash
.venv/bin/python -m pytest tests/unit/test_density.py
.venv/bin/python -m pytest tests/unit/test_density.py::test_solve_density_balance_matches_reference_script
```
There is no linter/formatter configured in this project — don't assume one and don't add one
without being asked.

CLI (installed as the `halide` console script, or run via `.venv/bin/python -m halide.cli.main`):
```bash
halide invert <negative.tif> <positive.tif> [--profile NAME | --rm/--bm/--rs/--bs | --auto-density | --pick]
                                               [--output print|flat] [--exposure E] [--contrast C]
halide batch <in_dir> <out_dir> [--auto-density-roll] [--save-profile-as NAME] [--output print|flat]
halide print <flat.tif|dir> <print.tif|dir>    # print stage only, for a flat positive edited elsewhere
halide check <roll_dir>                        # were the scans made consistently? (headers only)
halide contact <processed_dir> <sheet.jpg>     # high-res contact sheet of TIFF or exported PNG/JPEG output
halide batch <in_dir> --contact-sheet s.jpg    # preview: sheet only, no full-size TIFFs kept (add out_dir to keep them)
  (invert/batch also take --match-scan-exposure [--scan-reference FRAME])
halide export <positive.tif> <delivery.png>   # ACEScg TIFF -> delivery-ready sRGB PNG/JPEG
halide profile list|show|edit|rename|delete  # edit: film stock/process/scanner/notes
halide calibrate [roll_dir | scans… | --profile NAME]  # Qt picker: neutral points on any frames of a
                                                # roll, fitted together; --profile reopens a saved one
halide gpu [--install]                         # optional NVIDIA GPU support: what halide sees, and
                                                # add it (`pip install -e ".[cuda13]"` also works)
```
`invert`/`batch`/`print`/`export`/`contact`/`calibrate` all take `--device auto|cpu|gpu`
(also `$HALIDE_DEVICE`; default `auto` — GPU whenever one is usable) — see "Decisions and why",
GPU acceleration. On a GPU, `batch`/`print`/`export` and the contact sheet window share one GPU
service process between their workers; `HALIDE_GPU_SERVICE=0` (or `off`) makes each worker use the
GPU itself instead (per-worker mode, the troubleshooting switch). Its tests that need a real card
are marked `@pytest.mark.gpu` and skip without one; run them with `pytest -m gpu`.
Tab completion (zsh, bash, fish) sets itself up on the first run from a terminal (see "Decisions
and why").
With no calibration source given in a terminal, `invert`/`batch` offer the saved profiles (newest
first), picking (invert) or the automatic estimates, before starting.
Input TIFFs must be linear (not display/gamma-encoded) with an embedded ICC profile — the tool
validates this itself and rejects anything else with a specific error (see `io/icc.py`). The
user's normal workflow produces these via RawTherapee/darktable, exporting linear Rec.2020.

## Architecture

```
src/halide/
  core/        # PURE functions only: no file I/O, no print, no globals, no argparse.
               # density.py (white/density balance), invert.py, tone_render.py, pipeline.py
               # (run_pipeline = the single composed entry point), types.py (DensityProfile,
               # ToneCurveParams — both frozen dataclasses), _xp.py (array_namespace — numpy or
               # CuPy, whichever the input array belongs to, so every function above runs on
               # either; see "Decisions and why", GPU acceleration).
  io/          # tiff.py (read/write + dtype normalization), icc.py (validate + color-manage
               # the embedded ICC profile), raster.py (ACEScg -> sRGB delivery export),
               # lut.py (.cube reader, used by tone_render.py at runtime AND by golden tests),
               # exiftool.py (one kept-open exiftool per process, Linux only, behind tiff.py's
               # copy_exif_metadata — see "Decisions and why", worker pool).
  calibration/ # auto.py (statistical fallback calibration), profile_store.py (named,
               # reusable DensityProfile JSON files under ~/.config/halide/profiles/), anchors.py
               # (the picker's neutral-point model: scan-gain normalisation, agreement, gates).
  batch/       # orchestrator.py (ProcessPoolExecutor over processing.py, one bad frame doesn't
               # abort the batch), progress.py (terminal rendering only, no math).
  cli/         # main.py + commands/*.py (argparse) including gpu_cmd.py (`halide gpu`/`--install`),
               # _calibration_args.py (shared flag definitions/resolution used by both invert and
               # batch), _device_args.py (the shared `--device auto|cpu|gpu` flag/resolution/Compute
               # row, used by every command).
  gui/         # PySide6/Qt app. Pure, tested, no Qt: sampling.py (picking math), roll.py (the
               # session model), render.py (negative/positive display). Widgets: main_window.py
               # (the picker), filmstrip.py, step_wedge.py, point_list.py, drawers.py,
               # proof_window.py (contact sheet window), loaders.py (background loading), theme.py.
  processing.py # The glue layer: read -> validate ICC -> convert to working space -> calibrate
               # -> run_pipeline -> write. Both the single-file CLI command and the batch worker
               # call this same function rather than duplicating the chain.
  banding.py   # map_in_bands: runs per-pixel core/ functions over ~4 MiB bands of rows (64 MiB on
               # the device — DEVICE_BAND_BYTES) into a buffer the caller owns — how every
               # full-resolution path stays near 1 frame of memory. Bit-identical to the
               # whole-array call (see "Decisions and why").
  device.py    # ComputeDevice, resolve_device (auto/cpu/gpu, $HALIDE_DEVICE), to_device/to_host,
               # detect_nvidia_driver (card detection without CuPy) — optional GPU acceleration,
               # see "Decisions and why".
  gpu_service.py # the one GPU process a GPU batch's CPU-only workers share: running_service,
               # ServiceClient (develop/print_/export), ServiceUnavailable — see "Decisions and
               # why", GPU service.
  shared_frames.py # decoded frames in shared memory (/dev/shm) handed between a worker and the
               # GPU service: new_frame, attach_frame, batch_prefix/sweep.
```

The internal working color space is **ACEScg**, chosen (not just used) — the blog is explicit that
the inversion math's result depends on which working space it runs in, so the working-space choice
is not an implementation detail. `core/` never touches color management directly; `io/icc.py`
converts into ACEScg on read, output is always written back tagged with a real ACEScg ICC profile
(vendored from Elle Stone's `elles_icc_profiles`, cross-validated against `colour-science`'s own
ACEScg definition in `tests/unit/test_icc.py` — not hand-authored).

Calibration is three-tier, all producing the same `DensityProfile`: ColorChecker (not yet built),
manual picking of neutral points - any number, on any frames of a roll, fitted by least squares
(`halide calibrate`, `gui/main_window.py`; or `--rm/--bm/--rs/--bs` on the CLI) - and
statistical auto-detection (`calibration/auto.py`, `--auto-density`/`--auto-density-roll`).
Profiles are meant to be solved once per film-stock/process/scanner combination and reused
(`--save-profile-as`, `halide profile`), not re-solved per image.

## Write-ups

Plans, specs and investigations live in `docs/` (`plans/`, `specs/`, `investigations/` - see
`docs/README.md`), not the repo root, which the user asked to keep uncluttered. This file stays at
the root. Put new write-ups in the matching folder and add a line to `docs/README.md`.

## Decisions and why (don't re-litigate these without new evidence)

- **TIFF-in, not RAW-in.** The user's darktable/RawTherapee export already handles
  demosaic/crop/dust-removal/lens-correction correctly; reimplementing that would duplicate work
  for no accuracy gain. `io/icc.py` owns validating that the color-management handoff is correct.
- **Hand-rolled ICC parser in `io/icc.py`, not Pillow.** Pillow's `ImageCms` only exposes
  profile-level metadata (name, color space, rendering intent) — no way to check "is this a linear
  matrix-shaper profile," which is exactly what's needed here. The matrix/TRC subset of the ICC
  spec this project reads is small and precisely documented; no PyPI package does this narrow job.
  Pillow is still a dependency, but only for `io/raster.py`'s PNG/JPEG writing — an unrelated job.
- **The default tone-render curve is a vendored real paper response curve** (`src/halide/assets/tone_curves`,
  from `abpy/color-neg-resources`, MIT), not an invented analytic curve — deliberately, to keep the
  "faithful over flashy" priority. `ToneCurveParams.exposure` (enlarger exposure) and `contrast`
  (paper grade) both default to `None` = **fitted per image** by `core/tone_render.py::fit_print`,
  and both exist because of real bugs found on actual scans — see `ToneCurveParams`' docstring for
  the history. Don't "simplify" them back to fixed constants.
- **The print fit matches paper grade to the negative, before the curve — never a post-curve
  stretch.** Found via real use: the previous defaults (fixed `contrast=0.5`, plus a shadow-only
  `estimate_exposure`, now removed) covered only about half the paper on all three real scans —
  blacks at sRGB ~45, whites ~220-233, against the paper's own ~11/255 — so the user was tempted to
  stretch levels in darktable, which is a second, non-physical tone curve on top of the paper
  (a linear-light black point reshapes the paper's toe; unlinked levels would also overwrite the
  density balance). `fit_print` instead measures this frame's robust luminance density range
  (0.1/99.5th percentiles) and solves exactly for the grade that fills the paper's ISO 6846 range
  (computed from the curve file itself: 0.04 above paper white to 90% of D-max) and the exposure
  that puts the highlights on the paper's highlight point. Grade is capped at 1.0 (the real paper)
  so a genuinely flat scene prints soft rather than being normalised. The two scalars are applied
  identically to every channel, so the fit can't create a cast — but a harder grade (~0.8-0.9 on the
  real scans vs 0.5) makes an existing calibration residual ~1.7x more visible. Known, accepted
  trade-off of highlight anchoring: highlight-heavy frames print with dark midtones (IMG_0151's
  shaded crowd, IMG_0158's rider: median sRGB ~80 -> ~40). The anchor is 99.5 (see next entry)
  — don't re-tune it from one image. Every output records its decision (fitted or
  pinned) as JSON in the TIFF ImageDescription (`processing.py::provenance_json`), and `invert`
  prints it.
- **Per-frame grade (not a per-roll grade) was re-confirmed on a full real roll, and the roll's
  "odd contrast" turned out to be bad crops, not the fit.** On Roll 16 (37 frames), three frames
  had a ~3-stop bright tail from an opaque edge strip (film holder) left in the crop, which dropped
  their fitted grade to ~0.46; re-cropped, the tails were 0.14-0.29 stops and the fit behaved. A
  roll grade (median of per-frame fits) was tried and rejected: 21/37 frames sit at the 1.0 cap,
  so the "roll grade" is just the paper's own grade, and it only differs from per-frame on the
  genuinely long-range negatives, which a printer *would* print softer. The highlight anchor
  moved from the 99.9th to the 99.5th percentile by the user's choice from greyscale proof sheets
  of the whole roll at 99.9/99.5/99.0: at 99.9, specular-heavy frames (chrome, IMG_0158: top 1%
  spans ~1 stop) printed dark; 99.0 pushed high-key frames too close to paper white. Don't change
  it without the user's say-so.
- **Scan consistency matters for colour, not just brightness, and is checked from file headers
  (`halide check`, and automatically at the start of `batch`).** Density balance is a per-channel
  power function, so a frame digitized brighter by k comes out scaled by k**density_scale — a
  colour shift. Measured on Roll 16 with one fixed profile: frames digitized at 1/50 and 1/60 vs a
  1/25 calibration frame printed ~0.13 and ~0.22 density bluer once matched (i.e. that much
  warmer uncorrected) — CC13-CC22, clearly visible. `--match-scan-exposure` corrects it exactly
  from EXIF (one global multiply on linear sensor data commutes with every colour matrix; verified
  end to end in tests/unit/test_scan_consistency.py), against the profile's recorded scan settings
  (profiles carry a "scan" sidecar, written whenever one is saved; `--scan-reference FRAME` for older ones; batch
  falls back to the roll's most common setting with a warning). Deliberately **not** corrected:
  per-frame raw white balance (a per-channel multiply in the camera's own colour space, before the
  raw converter's camera matrix — not invertible from the export without that matrix, so it's
  reported, never approximated) and active tone/colour modules in darktable's embedded history
  (the export isn't linear). Found on the same roll: the camera was metering each frame (1/25-1/60)
  and "as shot" white balance was the camera's auto WB (14 distinct values, R ±5%, B ±7%); one
  frame had shadows & highlights active. Not yet validated against a real two-exposure scan of one
  frame (see docs/plans/tone-output.md follow-ups).
- **Contact sheets are a feature, not just a test aid** (`io/contact_sheet.py`, `halide contact`,
  `halide batch --contact-sheet`) — asked for after proof sheets kept proving the most useful way to
  compare settings across a real roll. Decisions: (1) the batch *preview* develops every frame at
  full resolution exactly as a real run (including the per-frame print fit — fitting on a
  downsampled frame would give slightly different grades), but keeps only a thumbnail per frame in a
  temporary folder that's always deleted — not full TIFFs in a temp folder, which for a real roll is
  ~5 GB (more than this sandbox had free). (2) Thumbnails are block-averaged in *linear* light
  before sRGB encoding, so fine detail doesn't darken. (3) Each frame is captioned with its recorded
  printing decision (grade, exposure, scan gain — from the provenance JSON) and the sheet header
  with the run's settings, so sheets from different settings are self-describing. (4) Sheets are
  marked (JPEG comment / PNG text) and skipped by `halide contact`: found via testing, a sheet
  written into the folder it proofs became an extra "frame" on the next sheet. (5) Captions use a
  plain "x", not "×" — Pillow's built-in font has no multiplication sign (it drew an empty box).
  (6) The look is a real contact print, after the user's own printed Portra 400 sheet (asked for
  explicitly): black wherever the film is (rebate and frame lines print black; no sprocket holes
  show), frames butted in neat strips of six, orange edge print - frame number and film stock above
  ("INVERTED BY HALIDE" when unknown), number + "A" half-frame number + edge-code bars below - then a
  dim line with the file name and printing decision. The stock is the profile's `film_stock`,
  recorded in each output's provenance and printed only when every frame agrees (a mixed folder
  gets the generic edge print, not a wrong stock). DejaVu Sans Bold for the edge print, Pillow's
  built-in font as a fallback. One renderer (`render_sheet`, geometry in `SheetLayout`) for `halide
  contact`, `batch --contact-sheet` and the picker's contact sheet window.
- **`halide print` exists for the flat -> darktable -> print round trip, and range-setting belongs
  to it, not to the editor.** The contract: edits between `--output flat` and `print` stay linear
  and scene-referred (crop, spot removal, lens, denoise, global exposure; no filmic/sigmoid, curves,
  levels, local contrast). The fitted print is invariant to a global multiply, so an exposure change
  in darktable (or losing halide's metadata) doesn't change the print — verified on the real scans:
  flat -> print matches direct `invert` to ~4e-7, and a +0.6 EV, metadata-stripped copy to within
  1 8-bit sRGB step. A *pinned* `--exposure` is not scale-invariant: `print` reproduces it exactly
  only when the flat file's provenance (its exposure scale) survived, otherwise warns and fits.
  `load_working_space_image` skips ICC conversion when the embedded profile is byte-identical to
  halide's own ACEScg output profile: the conversion isn't a true identity (the profile's
  s15Fixed16 matrix round-trips ACEScg only to ~1e-4 per channel), which broke exact round trips.
  `export`'s console verb became "Exporting" (it was "Printing" before a real print step existed).
- **`mode="linear"` (`--output flat`) output is scaled, not a bare
  unbounded passthrough** — one global multiply is the *only* adjustment, since per the reference
  blog exposure and white balance are the only operations that keep a flat positive faithful. It
  looks flat because it is the film's own recorded contrast (negative gamma ~0.6), and its black is
  the film base, not zero — both are data, not headroom to trim. Undoing film gamma would need a
  measured gamma (ColorChecker tier), so it is deliberately not guessed. (`estimate_linear_scale`
  in `core/tone_render.py`). Found via real use: a real scan's raw `invert()` reciprocal is
  typically in the tens, so an unscaled passthrough put ~100% of pixels above 1.0 — solid white in
  darktable or any standard viewer. Scaled via a robust highlight percentile (not the true max,
  which on both real test scans was an out-of-range artifact from the `MIN_TRANSMITTANCE` floor
  clamp, ~10,000,000) to a target with headroom, so linear output stays genuinely flat/uncurved but
  is actually usable as a starting point for external grading, per the user's explicit priority:
  "avoid clipping ... don't want to lose data" over a perfectly-exposed flat output.
- **`batch.orchestrator.run_batch` treats a crashed worker process as a per-job failure, not a
  fatal error.** Found via real testing: full-resolution scans are memory-heavy (a single frame's
  pipeline run can peak around 4GB RSS — `core/pipeline.py`'s chain of elementwise numpy ops
  allocates a fresh float64 array at nearly every stage, not a leak), so enough parallel `--workers`
  on a memory-constrained machine can get a worker OOM-killed. Without handling, that breaks the
  whole `ProcessPoolExecutor` and every other pending future raises `BrokenProcessPool` too —
  discarding every already-completed result and surfacing a bare traceback. Each crashed future is
  now recorded as its own `BatchResult` with an actionable message instead. `estimate_roll_density_profile`
  also had to `.copy()` its downsampled frames: a strided slice is a view that pins the whole
  full-resolution parent (~180 MiB per real scan), and a real 37-frame `--auto-density-roll` got the
  main process OOM-killed (exit 137) before any worker started. It
  similarly needed to catch broad `Exception`, not just `ScanColorError` — a genuinely corrupt (not
  just unsupported-ICC) file raises straight from `tifffile` and was aborting the whole roll estimate.
- **`batch.orchestrator.default_worker_count` sizes the worker pool from available RAM, not just
  CPU count, and this is what `run_batch`/`halide batch` actually use whenever `--workers` isn't
  given explicitly.** Found via real testing: even with the `BrokenProcessPool` handling above
  making an OOM-killed worker survivable (no longer a hard crash), the *old* default
  (`min(os.cpu_count(), 6)`) still picked a worker count that reliably triggered that OOM path on a
  real memory-constrained machine, including at `--workers 2` — full-resolution 32-bit scans are
  several hundred MB on disk and multiple GB decoded, and `core/pipeline.py`'s chain of elementwise
  numpy ops holds many float64-sized copies of that pixel grid alive at once rather than freeing
  intermediates eagerly. CPU thread count was never the actual limiting factor on such a machine.
  `estimate_worker_memory_bytes` predicts a single worker's peak RSS as `baseline + K *
  decoded_pixel_bytes` (reading each job's TIFF *header* only — page shape/dtype — never decoding
  pixel data just to size the pool), where `baseline ≈ 150 MiB` and `K = 24` were fit from real
  peak-RSS measurements (`/proc/<pid>/status`'s `VmHWM`) across a real 3276×4849 full-res scan
  (~182 MiB decoded → ~4.25 GiB peak) and a small 500×500 synthetic image (~2.9 MiB decoded → ~175
  MiB peak) — both constants rounded up from that fit for safety margin, not exact physics.
  `default_worker_count` then takes `min(cpu_cap, available_memory // per_worker_estimate,
  len(jobs))`, using `psutil.virtual_memory().available` (new dependency — the only reliable
  cross-platform way to ask this; falls back to the old CPU-only heuristic if `psutil` can't answer,
  e.g. an unsupported platform). An explicit `--workers N` is still fully respected as an override
  (this is a *default*, not a hard cap) — but `memory_budget_warning` prints a heads-up when `N`
  looks likely to exceed available memory, so a user overriding the default at least sees why it
  might crash instead of just hitting the crash. Verified end-to-end against real full-res scans in
  a genuinely memory-constrained environment: auto-selection correctly dropped to 1 worker and
  completed successfully where the old default's implicit worker count (and an explicit
  `--workers 6` override) both reproduced the real OOM/crash.
  - **What looked like a separate bug but wasn't**: a user reported `--workers 2` appearing to
    "spawn more than 2 processes" before crashing. Investigated by running a real batch under `ps`
    and inspecting the process tree: with `--workers N`, exactly `N` real worker processes run the
    actual pipeline, but Python's `multiprocessing` machinery itself adds a `resource_tracker`
    process plus (on platforms/Python versions defaulting to the `forkserver` start method) a
    persistent forkserver control process — both lightweight, neither runs `_worker`. These are
    easy to mistake for extra heavy workers in a process list/task manager, but they aren't — no
    code path was found that spawns more than the requested/selected worker count. Noted here so
    this isn't re-investigated as a phantom bug: the actual reason `--workers 2` still crashed was
    that 2 real full-resolution workers already exceeded the machine's available RAM, which the
    memory-aware default above now accounts for.
  - **The ~4x RSS multiplier itself (K, above) was a real, fixable inefficiency, not just something
    to size the worker pool around** — found via a follow-up memory-usage investigation after a user
    with 7 GB free and 16 cores still got auto-selected down to 1 worker on ~120 MB scans. Root
    causes, both confirmed by profiling real scans' peak RSS (`/proc/<pid>/status`'s `VmHWM`) before
    and after: (1) `io/icc.py::convert_to_working_space` multiplied the float32 image by the ICC
    matrix without casting the matrix down first — numpy silently upcast the *entire image* to
    float64 for the rest of the pipeline from that point on, a straight 2x that then compounded
    through every later stage; the matrix's own source precision (ICC s15Fixed16 tags, ~1.5e-5) is
    already coarser than float32, so casting it to the image's dtype before the matmul loses nothing
    real. (2) `core/density.py`, `core/invert.py`, `core/tone_render.py`, and `io/lut.py::Cube1D.
    lookup` each allocated a fresh full-size array at nearly every line of a multi-step elementwise
    chain instead of reusing a buffer via `out=`/in-place ops — `Cube1D.lookup` additionally forced
    its inputs to float64/int64 unconditionally regardless of the caller's dtype, undoing any
    upstream dtype fix at the single most temporary-heavy stage. Fixed by casting the ICC matrix to
    the image's dtype and downcasting `colour.XYZ_to_RGB`'s (which always computes in float64
    internally, confirmed empirically — not itself changed) result back before it re-enters this
    project's pipeline; by rewriting the elementwise stages to reuse private (never-the-caller's-
    own) temporaries in place; and by making `Cube1D.lookup` follow its input's own dtype instead of
    hardcoding float64/int64. None of this touches any core function's actual math — verified
    against all three real test scans: peak RSS dropped from ~4.3 GiB to ~1.9 GiB (measured, not
    estimated) per scan, and output pixels match the pre-fix output to float32 rounding noise only
    (max relative diff ~2e-6, i.e. reordering-of-floating-point-ops noise, not a real difference) —
    not bit-identical, but bit-identical was never the bar (the pre-fix pipeline already computed
    partly in float64 and wrote float32 output, so its own output already wasn't the "true" float64
    result either). `K`/`baseline` refit from the same real scans post-fix: `K = 11`, `baseline ≈
    150 MiB` (still generously rounded up from a measured fit of `K ≈ 9.7`, `baseline ≈ 108 MiB`).
    On the reporting user's own numbers (7 GB free, ~120 MB scans), this raises the auto-selected
    worker count from 1 to 3 — the actual point of this fix, not memory usage for its own sake.
  - **Every full-resolution path now works band by band in one owned buffer (`halide/banding.py`),
    and this one IS bit-identical — keep it that way.** After the pass above a frame still peaked at
    1.3-2.6 GiB (7-14x the 182 MiB frame): every core/ stage returned a new whole frame, colour-
    science converted the whole frame in float64 (+900 MiB in the ICC step alone; export's
    `RGB_to_RGB` was the single heaviest step at 2.6 GiB), `Cube1D.lookup` held ~6 frame-sized
    temporaries, `process_scan` kept the original negative alive through develop, `print_scan`
    decoded its input twice, tifffile's default 256 MiB read buffer held a whole ~120 MiB compressed
    scan next to the decoded frame, and auto calibration built ~500 MiB of full-frame side arrays.
    Fix: `map_in_bands` runs the *unchanged* core/ functions over ~4 MiB bands of rows (~64 rows of
    a real scan) of a buffer the caller owns — `load_working_space_image` converts in place,
    `processing.py::_develop_in_place` runs `core.pipeline.negative_to_positive` banded, the print
    fit on the full buffer exactly where `develop()` runs it, then the paper curve banded; export
    converts into one preallocated 8-bit array; `read_tiff` uses a 16 MiB `buffersize`;
    `calibration/auto.py::_density_local_saturation` scores each density bin directly. Why it can't
    move a pixel: every banded function is per-pixel, and whole-frame statistics (print fit, auto
    calibration) are never banded. Measured on the four real scans through the real CLI (peak RSS,
    MiB): invert 1870 -> 377, flat 1327 -> 500, `--auto-density` 1880 -> 521, `--density-only` 1327
    -> 324, `print` 1990 -> 365, `export` 2605 -> 402, `--auto-density-roll`'s pre-pass 1333 -> 333;
    also faster (invert ~6.5 -> ~4.7 s per frame: bands stay in CPU cache). Verified bit-for-bit, not
    to a tolerance: every artifact of invert/print/export/contact/batch on the four scans (TIFF
    pixels, provenance, EXIF, export PNGs, contact sheets) matched the pre-change code 38/38, and
    `tests/unit/test_banding.py` pins each banded path to the whole-array pipeline at 1- and 7-row
    bands plus a tracemalloc guard (9.0x the frame before, 1.4x after, on the same test). Bands
    bigger than ~4 MiB measurably cost memory (1000 rows: 713 MiB) for no speed gain. Deliberately
    *not* done, because each would change output: replacing colour-science, measuring the fit or
    auto calibration on a downsampled frame, float16 buffers. Known, pre-existing and untouched:
    export PNGs / contact-sheet JPEGs aren't byte-reproducible between runs even on unchanged code —
    Pillow stamps its synthesized sRGB ICC profile with the creation time (pixels are identical).
  - **D1 (2026-09-27): fusing the two ICC matrices (and the ACEScg->sRGB matrix) *was* done after
    all, once the user explicitly accepted "looks identical" rather than bit-identical for this one
    step** — GPU acceleration (`docs/plans/gpu-acceleration.md`) needs the same two matrix multiplies
    to run on a GPU later, and colour-science's own per-pixel broadcast (`vecmul` = `np.matmul`
    broadcast over every pixel) doesn't map onto a GPU matmul either, so this was going to have to
    change regardless. `io/icc.py::working_space_matrices()` and `io/raster.py::srgb_matrix()`
    extract colour-science's own float64 matrices (Bradford D50->ACEScg CAT + ACEScg's XYZ->RGB;
    ACEScg->sRGB, also Bradford) exactly as `colour.XYZ_to_RGB`/`colour.RGB_to_RGB` build them
    internally, then apply them as one `@` per band instead of colour's per-pixel `vecmul`. Why it
    isn't bit-identical: confirmed directly (not assumed) that BLAS and numpy's broadcast `vecmul`
    round the last float64 bit differently even for the identical matrices and identical inputs —
    `vecmul(m_cat, xyz)` vs `xyz @ m_cat.T` differ by up to 1 float64 ULP on the same seed, before
    the second matrix is even applied (both paths use FMA, just differently). This project's dev
    sandbox measured bit-identical (`maxulp=0`) float32 output on all four real scans (190,351,044
    values total) and a 60M-value synthetic stress test spanning 1e-5 to 10^0.5 — against an earlier
    60M-value stress test in this same sandbox (different random sample) that found 1 of 60M values
    off by 1 ULP when this tolerance was accepted. Both runs are the same hardware; the two results
    disagree because they sampled different values, and which of them happen to land on the rare
    last-bit divergence isn't established — this isn't evidence of CPU-dependence, just of sampling
    variance in an ad-hoc stress test. Accepted tolerance, per unit test (`test_icc.py`/`test_raster.py`): 2
    float32 ULPs on random data and on 512-row bands of each real scan. Measured cost (IMG_0158,
    3276x4849, dev sandbox, warm process): `convert_to_working_space` alone 1.86 s -> 0.67 s
    (~2.8x); `load_working_space_image` (read + convert, banded) 1.70 s -> 0.59 s. `to_srgb_8bit`'s
    matrix step alone 0.93 s -> 0.29 s (~3.2x), but colour's own `cctf_encoding` call — applied
    unchanged, so bit-identical — is the larger share of that function's cost (~2.4 s of the
    function's ~2.4-3.1 s total), so the function's own total only drops ~3.1 s -> ~2.4 s; fusing
    the matrix doesn't touch the cctf cost. End-to-end acceptance vs `dff3cb2` (pre-change), run
    through the real CLI (`--rm 0.9 --bm 1.1 --rs 1 --bs 1` and `--auto-density`) on all four real
    scans — 8/8 combinations passed every criterion, and every TIFF/PNG pair came out genuinely
    bit-identical (not merely under the accepted bound):

    | Scan | Calibration | invert max\|diff\| | values >= 1/65535 | export PNG | exposure match | contrast match |
    |---|---|---|---|---|---|---|
    | IMG_0151 | manual | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0151 | auto-density | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0156 | manual | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0156 | auto-density | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0156-nowb | manual | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0156-nowb | auto-density | 0.0 | 0 / 47,655,972 | pixel-identical | exact | exact |
    | IMG_0158 | manual | 0.0 | 0 / 47,383,128 | pixel-identical | exact | exact |
    | IMG_0158 | auto-density | 0.0 | 0 / 47,383,128 | pixel-identical | exact | exact |
  - **Worker pool after that pass**: `K = 3` (baseline 150 MiB) for inversion and `K = 2` (100 MiB)
    for export/contact, refit from real worker processes' peaks (worst: `--auto-density` 509 MiB RSS
    on a 182 MiB scan) — see the constants' comment in `batch/orchestrator.py`. exiftool has no
    term but uses part of the margin: it streams (67 MiB peak on a 130 MiB output), `process_scan`
    frees the frame before running it, and worst worker + exiftool is 576 MiB of the 695 MiB
    estimate. Workers come from a forkserver with `halide.processing` preloaded (they share numpy/
    colour-science pages copy-on-write: 4 idle workers 294 -> 73 MiB proportional memory). The old
    fixed `min(cpu_count, 6)` cap is now one worker per *physical* core (`_cpu_cap`, the user's
    choice) — a hyperthread sibling adds little to a numpy-bound worker but costs a frame of memory.
    Net effect: 4 GiB free now runs 5 workers (was 1), 7 GiB runs 9 (was 3). Measured on 16 real
    frames in the dev sandbox (5.5 GiB free, 16 cores), each version at its own default: old code 2
    workers, 46.4 s, 3.83 GiB total PSS; new code 8 workers, 16.9 s, 2.49 GiB — outputs 16/16
    bit-identical. Scaling flattens past ~4 workers there (1: 66 s, 2: 35 s, 4: 22 s, 6: 20 s, 8:
    17 s) — probably disk writes (~2 GB of TIFFs per run) and memory bandwidth, not re-tuned from
    one sandbox; re-check on the user's own machine before lowering the physical-core cap.
  - **exiftool is kept running per process on Linux** (`-stay_open`, `halide.io.exiftool`; plan
    `docs/plans/gpu-batch-throughput.md` Part A). A fresh exiftool per output cost ~0.66 s, mostly
    its own start-up and tag-table loading, not the file rewrite; kept open it is ~0.23 s per file
    after the first (dev sandbox, exiftool 13.36), and outputs are byte-identical — 15/15 real
    outputs (4 scans + 3 Roll 16 frames, ICC dropped and kept, a non-ASCII file name).
    - **Linux only, deliberately**: a `-stay_open` exiftool never exits when its input closes (it
      polls the pipe every 10 ms forever), so a worker the GUI terminates or the OOM killer takes
      would leave it running for good. Only Linux guarantees it dies with its process (the
      kernel's parent-death signal); elsewhere every copy stays the old one-shot call. Sessions
      are keyed by PID (a forked child never uses its parent's) and closed at exit.
    - **"Failed" is exiftool's own exit status** for that command, echoed after it
      (`-echo3 {status=${status}}`, exiftool >= 12.10) — the same condition the one-shot's
      non-zero exit was. Reading its summary instead ("N image files updated", no `Error:`) isn't
      the same: an "unchanged" file exits 0, and a missing source's message has no `Error:`
      prefix. That reading is only the fallback for an older exiftool. A failed file raises
      `ExifToolError` (the batch reports it as that frame's failure).
    - A session that dies or hangs (timeout, then killed; its `_exiftool_tmp` file removed) is
      restarted once; if that fails too, the process goes one-shot for the rest of its life, so a
      broken exiftool never hangs the batch. Arguments exiftool's argument file can't carry
      verbatim (a line break, leading/trailing white space, a leading `#`) go one-shot.
      `-charset filename=utf8` (needed on Windows) was verified byte-neutral on Linux.
    - The parent-death signal is set in a `preexec_fn`, which Python warns is unsafe with threads
      (a session starts lazily, mid-batch, possibly beside CUDA's threads). Accepted: the child
      only calls `prctl` and `getppid` before exec, and there's no other way to set it.
    - **On the user's machine it paid off only where one frame runs at a time** (GPU, 1 worker:
      1.44 -> 1.14 s/frame); multi-worker batches didn't move (CPU 8 workers 0.89 -> 0.89, GPU 2
      workers 0.92 -> 0.92). Those were limited by RAM (GPU) and by cores/memory bandwidth (CPU),
      not by exiftool — see the GPU service entry.
- **GPU acceleration (optional, CuPy) — `docs/plans/gpu-acceleration.md`; verified on the user's
  RTX 3070 (2026-09-27): all 46 `pytest -m gpu` tests pass, and the GPU was ~20% faster end to end
  — see the last sub-bullet. A GPU batch is now ~2x the CPU through one shared GPU service (next
  entry); the per-worker design described below is its fallback.**
  The user asked for a "thorough investigation ... if it increases performance I'd like it to be on
  by default but able to be disabled." Answered with `--device auto|cpu|gpu` (`auto` = default,
  GPU whenever it's usable).
  - **Why CuPy, and why optional.** halide's core is plain numpy ufunc code (`np.maximum`,
    `np.power(out=)`, `np.percentile`, fancy indexing) and CuPy implements the same API, including
    `out=` and `percentile(overwrite_input=)` — PyTorch/JAX would mean rewriting every stage in a
    different idiom plus a multi-GB dependency, Numba CUDA would mean hand-written kernels per
    stage, and neither buys accuracy or speed CuPy doesn't. CuPy plus NVIDIA's CUDA runtime
    libraries is ~1 GB, so `pip install -e .`/`.[dev]` never pull it in — it's the `cuda12`/`cuda13`
    extras (`cupy-cuda12x[ctk]` / `cupy-cuda13x[ctk]`; `[ctk]` bundles cudart/NVRTC/cuBLAS as pip
    wheels so no system CUDA Toolkit is needed, only the driver). `halide gpu` reports what halide
    sees (card, driver's CUDA version, whether support is installed, whether it actually works,
    what `--device auto` will use); `halide gpu --install` names the exact package and its ~1 GB
    size, asks y/N, runs `pip install` in halide's own environment, then re-checks in a **fresh
    subprocess** (the current process may have cached the failed import). No pip in that
    environment → prints the matching pip/pipx/uv commands instead of guessing.
  - **Finding the card without CuPy** (`device.py::detect_nvidia_driver()`): pure `ctypes`
    (`libcuda.so.1` / `nvcuda.dll`), `cuDriverGetVersion`/`cuInit`/`cuDeviceGetName` — no subprocess,
    no `nvidia-smi` parsing (its header format changes; the user's own machine reports "CUDA UMD
    Version", not a plain version number). Any failure means "no card", never an exception. The hint
    ("`<card>` found; add GPU support with: `halide gpu --install`") is shown only when a card is
    found and support isn't installed — nobody without an NVIDIA card gets pitched a 1 GB download.
    `invert` shows it once per machine via a stamp file (the same state directory tab-completion's
    stamps live in) — **except** on a no-card machine, which never writes that stamp and so
    re-probes on every single `invert`, forever, so a card added later still gets the hint (a
    review initially flagged the report's claim that the probe "runs once per machine" as false for
    exactly this common case; the ruling was to measure the cost before deciding whether to fix it).
    Measured in this sandbox (no `libcuda.so.1` at all — the fastest possible failure path): a
    fresh-process call is 0.11 ms median / 0.22 ms max, about 45x under a 5 ms budget the ruling set
    — negligible next to a multi-second `invert`, so the re-probe-forever behavior was kept as
    correct rather than "fixed" into a wrong stamp. (Not measured: the probe's cost on a machine
    that *does* have a driver, where the ctypes calls do more work.)
  - **`--device auto|cpu|gpu` + `HALIDE_DEVICE`** (`cli/_device_args.py`, `halide.device.
    resolve_device`), the same flag on `invert`/`batch`/`print`/`export`/`contact`/`calibrate`.
    `DEFAULT_DEVICE = "auto"` lives in `halide.device` itself, not copied into the CLI module (an
    early version defined its own copy of the same string that `resolve_device` never actually
    read, so changing it would silently have done nothing — fixed to one source of truth). The user
    chose `auto` (GPU whenever usable) before any benchmark could run on real hardware, since that
    benchmark needs a card this dev sandbox doesn't have — flag name `--device` was chosen over
    `--gpu`/`--no-gpu` to leave room for a later device value (an AMD/ROCm build is out of scope for
    now, but the namespace design doesn't preclude it). `auto` on a machine with no CuPy at all
    falls back to CPU silently (the ordinary case, not a problem); with CuPy installed but a probe
    failure (e.g. a driver mismatch) it falls back with a `fallback_reason` shown on the run sheet's
    Compute row and the CLI's own warning line. An explicit `--device gpu` that isn't usable is a
    hard error naming `halide gpu --install`, before any file I/O.
  - **Namespace-generic `core/`** (`core/_xp.py::array_namespace(a)`): one implementation of every
    core function, not a GPU-specific copy — it returns numpy for a numpy array, cupy for a cupy
    array (checked only via `sys.modules.get("cupy")`, so `core/` itself never imports cupy), or a
    namespace registered for tests. On numpy input this is `xp is numpy`, so **the CPU path calls
    exactly the functions it always has and stays bit-identical** — verified against every existing
    pin (`test_banding.py` etc.), 106 sha256'd before/after output records across the touched
    functions, and real-scan `invert` (IMG_0156, IMG_0158, `--auto-density`, print and flat) diffed
    pixel-identical against the pre-change code.
  - **A strict fake device** (`tests/unit/_fake_device.py`) stands in for CuPy in every test outside
    `tests/gpu/`. It refuses implicit conversion to numpy (`np.asarray`/`np.percentile` on it
    raise) and refuses mixing with a plain numpy array or a Python tuple/list (matching CuPy's own
    refusal — a profile tuple never uploaded with `xp.asarray` would otherwise silently work against
    the CPU-backed fake but explode on real CuPy). It computes with numpy underneath, so its output
    must be bit-identical to the CPU path — anything it catches is a plumbing bug (a missed `xp.`, a
    dtype that changed hands), never device arithmetic. It's a wrapper that blocks `__array_ufunc__`/
    `__array__`, not the originally-sketched `ndarray` subclass — numpy converts a subclass silently,
    which would have defeated the whole point.
  - **A frame is resident on the device for the whole develop/print/export pipeline**: one upload,
    per-pixel stages banded at `DEVICE_BAND_BYTES` (64 MiB, vs. 4 MiB on the CPU — bigger transfers
    suit VRAM bandwidth and PCIe better), one download at the end straight into the same host buffer
    the scan was decoded into (host RAM stays ~1 frame, as on the CPU path). Any device exception —
    OOM or otherwise — is caught, the CuPy pool freed, and the frame is redeveloped on the CPU from
    that still-untouched host buffer; if the exception surfaced during the *download* itself (CUDA
    runs asynchronously, so an earlier error can appear there, with the host buffer possibly
    half-written), the scan is re-read from disk before the CPU retry. Every fallback prints a
    warning naming what happened — "out of GPU memory" for an OOM/`MemoryError`, else "the GPU
    failed (...)" — worded per the action that fell back ("developed this frame" vs. export's
    "exported this file"; a review finding, since the shared message helper originally hardcoded
    develop's wording onto export's own fallback too).
  - **Provenance gains `"device": "cpu"`/`"gpu"`** in the TIFF's JSON (whichever path actually ran —
    a fallback still records `"cpu"`), since the two paths are held to a tolerance, not bit-identity,
    so a file should say which one made it.
  - **GPU vs. CPU accuracy bar (D2): float32 pixels within 1e-5 relative (or 1e-7 absolute), fitted
    exposure/contrast within 1e-5, 8-bit exports within 1 code value (and ≥ 99.9% of pixels
    identical, reported, so a systematic offset can't hide inside "≤ 1").** Bit-identity isn't
    achievable on a GPU even in principle — `pow`/`log10` round the last hardware bit differently,
    and summation/BLAS order differs from the CPU's — which is the same FMA-ordering effect
    documented for the CPU-only **D1** decision above (colour-profile/sRGB matrix fusion): D2 is
    that same phenomenon one step further from bit-identity, not a new one.
  - **Kept CPU-only, deliberately:** profile solving and the `--auto-density-roll` pre-pass (both
    tiny/already downsampled); and the *thumbnail* step of contact sheets — all of `halide
    contact`, and the last step of `batch --contact-sheet` (whose develop step runs on the GPU like
    any batch; only its thumbnails of the already-developed frames are CPU): all that's left there
    is decode and a block average, and uploading a 181 MiB frame just for that would cost about
    what it saves. `halide contact` therefore never resolves `auto` (it would import CuPy and make
    a CUDA context, ~0.6 s and ~300 MB of GPU memory, and could warn "GPU not usable" for a
    command that never uses one): it resolves only an explicit `gpu` (flag or `HALIDE_DEVICE`), so
    that still fails fast, and its Compute line reads plain `CPU` (plus " · developed frames need
    no GPU" after an explicit `gpu`). `$HALIDE_DEVICE`/`--device` ignore case and surrounding
    space, and an empty `$HALIDE_DEVICE` counts as unset (`halide.device.requested_device`).
  - **Auto calibration runs on the device** (`calibration/auto.py`'s `_density_local_saturation`,
    `_saturation`, `_neutral_candidate_mask`, `_shadow_and_highlight_from_candidates`, all made
    namespace-generic) because the probe showed it was worth it (below) — `solve_density_balance`
    itself stays CPU, since it only ever receives 3-element host arrays. Its one CuPy-specific
    wrinkle: `argsort` doesn't keep tied (equal-luminance) values in the same order a CPU sort does,
    so a density bin's edge pixel can differ between devices — the probe measured only 58% of sort
    orders agreeing on a real scan. The probe compared the sort only; it never solved a profile, so
    whether the solved profile stays within tolerance is exactly what the `tests/gpu` parity tests
    will check on real hardware (see the last sub-bullet for what to look at if they fail). Its parity
    tests therefore compare the **solved `DensityProfile` within D2**, never the sort order or bin
    membership.
  - **Batch workers and CUDA/forkserver — per-worker GPU mode, now the fallback** (the shared GPU
    service, next entry, is the default; this is what runs when it can't be used). Workers never
    import cupy inside the forkserver (`_FORKSERVER_PRELOAD` is unchanged — a forked CUDA context
    is unusable in the child); only a `device_kind` string (`"cpu"`/`"gpu"`) and a VRAM pool-limit
    number cross the process boundary, and each worker resolves its own `ComputeDevice`/CUDA
    context and CuPy memory-pool limit (`cupy.get_default_memory_pool().set_limit(size=share)`) on
    its first GPU job, keeping both warm across frames after that. `default_worker_count` gains a
    third cap for a GPU device: `memory_free // estimate_worker_device_bytes(jobs)`
    (`device_worker_cap`), alongside the existing CPU-core and RAM caps — an unusable GPU inside
    one worker falls that worker back to the CPU with a warning on every frame it develops, rather
    than crashing the batch. The run sheet's Workers row says "auto-selected to fit free GPU
    memory" when that cap is what set the count, and an explicit `--workers N` above it gets a
    warning (`device_budget_warning`, beside the RAM one): each worker's pool share shrinks as N
    grows, so too many GPU workers mostly means frames redone on the CPU. A GPU worker's *host*
    RAM is ~3x a CPU worker's (below), so for GPU pools the RAM estimate adds
    `_GPU_HOST_OVERHEAD_BYTES`.
    - **Constants fitted on the user's RTX 3070** (`docs/plans/gpu-acceleration-bench.py`, plan
      §7), replacing provisional estimates (768 MiB + 4 x frame = 1492 MiB per worker):
      one 181 MiB frame peaked at 821 MiB of CuPy pool with a given profile and 878 MiB with
      `--auto-density` (4.54 / 4.86 x frame — resident frame, band temporaries and whole-frame
      statistics together), plus ~168 MiB of CUDA context/library overhead (nvidia-smi 1046 MiB).
      So `_CUDA_CONTEXT_BYTES = 256 MiB` and `_DEVICE_FRAME_MULTIPLIER = 5` (1161 MiB for a real
      scan, 11% over measured). **Host memory** was the surprise: a GPU worker holds ~1.2 GiB of RAM
      (largest 1212-1235 MiB) against ~0.4 GiB for a CPU worker on the same frames — CUDA's and
      CuPy's host-side libraries, ~840 MiB — so `_GPU_HOST_OVERHEAD_BYTES = 1 GiB` is added to the
      RAM estimate for GPU pools (and to `memory_budget_warning`), or a machine with a big card and
      little RAM would be given more workers than its RAM holds. On the user's machine (7.6 GiB
      RAM, 6.58 GiB VRAM free) RAM is what limits per-worker mode to **4 workers** (1/2/4 workers:
      53/34/25 s for 37 frames), with no CPU fallbacks; the card alone would allow 5. That RAM
      limit is what the shared service removed.
  - **Test temp files are deleted per test, not kept** (`pyproject.toml`:
    `tmp_path_retention_policy = "failed"`, count 1; `tests/gpu/test_gpu_parity.py` also deletes its
    files after a *failed* test). Found via real use: pytest's default keeps every test's `tmp_path`
    from the last 3 runs, in `/tmp` — RAM on the user's CachyOS — and the real-scan GPU tests write
    ~130 MiB TIFFs each, so one `pytest -m gpu` run left 5.2 GB there and nearly ran the machine out
    of memory. Simulated (6 tests x 2 x 130 MiB, one failing): peak 1561 -> 258 MiB, left after the
    run 1561 -> 1 MiB. `"none"` looks like the obvious setting but only cleans up at session end, so
    it wouldn't cap the peak.
  - **`tests/conftest.py` forces `HALIDE_DEVICE=cpu` for the whole suite (autouse fixture).** With
    `DEFAULT_DEVICE = "auto"` live on every command, any CLI test that didn't pass `--device` would
    otherwise resolve `auto` against whatever machine actually runs the suite — harmless here (no
    usable GPU) but on the user's own RTX 3070 (which runs this same suite) it would silently take
    the GPU path, which is only held to D2, not bit-identity, so an exact-pixel/exact-provenance
    test could fail for a reason unrelated to what it's actually testing. Tests about device
    *selection itself* (`test_device.py`, `test_device_cli.py`) override this per-test via the same
    `monkeypatch` instance; tests that inject a `ComputeDevice` directly, or
    `tests/gpu/test_gpu_parity.py` (which resolves `"auto"` at **import** time, before any fixture
    runs, so it still targets a real GPU when one exists), are unaffected either way.
  - **The user's probe** (§7 of the plan — RTX 3070, 8 GiB, driver
    615.71.09 = CUDA 13.4, 6.7 GiB free, warm run): CuPy import 0.24 s / CUDA context 0.19 s / first
    kernel 0.17 s (roughly 2x on a cold run); upload of a 181 MiB frame 66 ms, download 26 ms;
    per-stage CPU→GPU: ICC matrix 0.058 s → 0.009 s, `negative_to_positive` 2.03 s → 0.005 s
    (1.7e-7 rel), print-fit percentile 0.52 s → 0.013 s (6.3e-8 rel), paper curve 3.51 s → 0.018 s
    (2.9e-6 rel), auto-calibration argsort 0.96 s → 0.006 s (58% of sort orders agree, expected —
    see above). Baselines on that machine: `halide invert` 4.4 s; `halide batch` (37 frames)
    35.1 s (0.95 s/frame). These came from a short standalone measurement script, not from halide
    itself — the per-stage speed-ups looked dramatic, but the end-to-end gain (next) is far smaller,
    because what's left of a frame is decode, TIFF write and process start-up.
  - **Verified on the user's RTX 3070 (2026-09-27).** `pytest -m gpu`: 46 passed — every device
    path against the CPU path within D2 on synthetic images and the four real scans, including both
    `--auto-density` parity checks (the tie-order caveat above didn't bite) and export. The
    benchmark (`docs/plans/gpu-acceleration-bench.py`, full table in plan §7), each device at its
    own default: one `invert` 3.4 -> 2.7 s (~21% faster), a 37-frame `batch` 33.3 -> 26.9 s
    (0.90 -> 0.73 s/frame, ~19%), no CPU fallbacks. By the user's rule ("on by default if it
    increases performance") `auto` stays the default everywhere (D3). Before this sandbox saw a real
    card, every GPU path had only run against the strict fake device; `halide gpu --install` has
    still only run against fakes (the user installed CuPy with pip themselves).
- **A GPU batch uses one GPU service process shared by CPU-only workers** (`gpu_service.py`,
  `shared_frames.py`; plan `docs/plans/gpu-batch-throughput.md` Part B, evidence
  `docs/investigations/gpu-batch-throughput.md`). `batch`, `print`, `export` and the picker's
  contact sheet window all use it on a GPU. Verified on the user's RTX 3070 (2026-09-28): 63/63
  `pytest -m gpu`, including 16 service-vs-per-worker tests that require **bit-identical** output
  (print, flat, `--auto-density`, print stage, export; synthetic and real scans), and a 37-frame
  batch at 0.45 s/frame against 0.69 for per-worker mode in the same run (1.53x).
  - **Why not more frames in VRAM at once.** The user's idea was to load many frames onto the card.
    But GPU work is only ~0.14 s of a 1.44 s one-worker frame (upload 0.066 s, all the arithmetic
    ~0.045 s, download 0.026 s); the rest is CPU file work (decode ~0.26 s, compress + write
    ~0.26 s, exiftool ~0.65 s before Part A). One 16-megapixel frame already fills the card (47M
    values vs 5,888 cores; each pass is memory-bandwidth bound, ~0.85 ms), so two frames at once
    take twice as long, and one card can keep up with ~7 frames/s. In darkroom terms: a second
    enlarger doesn't help when the queue is at the wash and the dryer.
  - **Why not threads in one process.** tifffile decodes/encodes strip by strip (3,266 per scan)
    holding Python's global lock: measured in memory, decode was 0.33 s/frame on 1 thread and
    0.47-0.58 s on 2-8. The file work needs separate processes, so sharing one GPU means one GPU
    process serving CPU-only worker processes, with frames in shared memory.
  - **Why it was needed (B0, the user's machine, after Part A, per-worker mode).** 1/2/4 workers:
    44.2/29.9/29.9 s for 37 frames (1.19/1.61/3.24 s per frame per worker). At 4 workers the
    machine swapped out ~167 MB/s, 29% of CPU time went to the kernel, the CPU was still 34% idle
    and the GPU busy ~30%: RAM-bound, because every GPU worker carried ~0.8 GiB of CUDA/CuPy host
    libraries. The disk wasn't the limit (the NVMe writes 1.3 GB/s with `fsync`; a batch needs
    ~0.45 GB/s). The plan's probe script and its go criterion (an 8-process ceiling of >= 2.8
    frames/s) were never run; the go was decided on this RAM finding instead.
  - **How a frame goes.** The worker reads the header, creates a segment (`new_frame`), decodes
    straight into it (`read_tiff(out=)`), and sends only (name, shape, dtype) plus a picklable
    request over `multiprocessing.connection` (Unix socket / named pipe, random authkey). The
    service — started with **spawn**, so CUDA only ever exists in it — attaches, uploads, runs the
    same `run_device_job`/`run_device_export` as the in-process GPU path, and downloads into the
    same segment; the worker then compresses, writes and tags. One compute thread runs frames one at
    a time; each connection has its own reader thread. Same device code on the same card is why
    service output is bit-identical to per-worker output.
  - **`/dev/shm` is RAM** (tmpfs; on the user's CachyOS, like `/tmp`). The frames are the workers'
    working buffers, not extra copies, but they count against RAM and against `/dev/shm`'s own size
    (64 MiB in a default Docker container). Details that each fixed a real failure:
    - `os.posix_fallocate` right after creating a segment: tmpfs allocates lazily, so an oversized
      segment "succeeds" and then kills the process with SIGBUS on first write (reproduced; not
      catchable). Now it's `SharedMemoryUnavailable`, and that frame is developed on the CPU with
      a warning.
    - `np.frombuffer(shm.buf)`, not `np.ndarray(buffer=)`: the latter holds no buffer export, so
      an array kept past close became a dangling pointer (reproduced segfault).
    - Segments are named with a per-batch prefix (`batch_prefix`: the pool owner's pid plus a
      random token). A forkserver pool shares the parent's resource tracker, so a worker killed
      while holding a frame (OOM killer) leaves its segment until the *parent* exits; `sweep(prefix)`
      unlinks those. It runs only after the pool and the service have both stopped (every worker
      shares the prefix, so an earlier sweep would unlink live frames).
    - The service attaches with `SharedMemory(track=False)` (Python >= 3.13), so its tracker doesn't
      claim the worker's segment. Below 3.13 `attach_frame` refuses and `sweep` does nothing: an
      earlier hand-unregister shim dropped the creator's registration from the shared tracker and
      made it print KeyError tracebacks.
  - **Failures.** An error on the service's device (out of memory, a driver error) comes back as a
    reply carrying a `DeviceFailure`; the worker raises `DeviceJobFailed` and redoes the frame on
    the CPU (`fall_back_to_cpu`, the same warning as the in-process path). A dead, unreachable or
    stuck service raises `ServiceUnavailable`: at once for a dead one (the connection breaks), after
    `REQUEST_TIMEOUT` (300 s) for a stuck one. The connection handshake is bounded by the same
    timeout (a frozen service still completes a Unix-socket connect and then never answers; before
    that fix, it hung). Startup is bounded (120 s) and cancellable — closing the contact sheet
    window mid-startup is quiet, not reported as a fallback. The service is never restarted: a
    client that lost it stays dead, and each later frame in that worker is developed on the CPU
    with a warning; the batch completes. Once a request was sent, a failure counts as
    `host_touched` — the service may have half-written the frame, or (stuck) may still write it —
    so the CPU starts from the scan re-read into a fresh private buffer, and an export's CPU
    fallback (`export_fallback`) converts into a fresh 8-bit buffer, never the shared output. A
    stuck request's mapping stays alive in the service until it stops (RAM not counted anywhere;
    only a stuck service gets there).
  - **Worker count** (`service_worker_count`): min(physical cores, RAM cap, `/dev/shm` cap, jobs).
    RAM cap = (available - the service's own memory, `_GPU_SERVICE_HOST_BYTES` 1 GiB + one frame)
    / (a CPU worker's estimate + its shared frame). No VRAM cap: the card only ever holds the
    service's one context and one frame. `/dev/shm` cap = 80% of its free space / one frame (an
    export's 8-bit output too); an explicit `--workers` above either cap gets a warning, and a
    worker that finds no room develops that frame on the CPU. Measured (B4): service ~1178 MiB RSS
    / ~933 MiB PSS (estimate 1205 MiB), 982 MiB VRAM; largest worker ~525 MiB RSS (estimate ~876
    MiB). **Not refit, deliberately**: if the service dies mid-batch, every worker develops on the
    CPU while still holding its shared frame — about what the estimate covers — and 6 workers
    (what it picks on 7.6 GiB) vs 8 measured 0.45 vs 0.44 s/frame, so the margin costs little.
  - **Falls back to per-worker GPU mode** (previous entry), with the reason on the run sheet's
    Compute row (the contact sheet window prints it), when: Python < 3.13; `/dev/shm` is missing
    or can't hold one frame; the service doesn't start (including its startup timeout); or
    `HALIDE_GPU_SERVICE=0`/`off`/`false`/`no` — the troubleshooting switch, also how the bench and
    the parity tests reach per-worker mode. It never fails the batch.
  - **Measured (B4, user's RTX 3070, 37 frames of Roll 16, 7.6 GiB RAM available, one run):**
    service 4 / 8 / auto (6) workers 18.2 / 16.5 / 16.8 s = 0.49 / 0.44 / 0.45 s/frame; per-worker
    4 / auto (4) 27.1 / 25.4 s = 0.73 / 0.69 s/frame; no CPU fallbacks. Swap stayed flat on the
    service rows (4787 -> 4819 MiB) and rose on the per-worker rows (to 5313 MiB). For context:
    per-worker GPU batch was 0.73 s/frame before this plan and CPU batch 0.90 s/frame. The plan's
    hoped-for ~0.36 s/frame (2.8 frames/s) wasn't reached; 1.53x cleared the user's "~1.5x or
    discuss" bar, so it was kept. Full table: the plan's Task B4.
- **`--auto-density-roll` selects neutral candidates per frame, then pools candidates — never pools
  raw pixels across frames first.** The per-channel median used to judge "how neutral is this
  pixel" (`calibration/auto.py::_saturation`) is only a valid proxy for the film's own systematic
  imbalance when computed from one frame's own pixels; computed from pixels pooled across frames
  with different scene content, it blends in each frame's own scene-color average too, which isn't
  shared across a roll the way the film base is. This was a real bug (found via testing on two real
  same-roll scans), now fixed — but did not fully resolve a residual color cast, see the limitation
  noted below.
- **`calibration/auto.py::_saturation` judges neutrality against a density-local reference
  (`_density_local_saturation`, formerly `_density_reference`), not one frame-wide median.** A single global median is only a valid stand-
  in for the film's systematic per-channel imbalance at the ONE density level it happens to sit at —
  it is not a fixed ratio across the whole tonal range, because the three dye layers have different
  characteristic-curve shapes. This was a real, structural bug, not just an occasional gray-world
  failure, confirmed with real numbers on `IMG_0151.tif`: a genuinely neutral black traffic light
  and a random head of (non-neutral) dark hair had near-identical raw RGB and both passed the old
  global-median filter, while a genuinely neutral, properly-exposed white sign scored far outside
  the threshold and was excluded. Mechanism: color negative film's characteristic curve has a
  compressed "toe" at the underexposed end — near-black content of *any* real hue converges toward
  nearly the same raw color there, because little image-forming density is left to differentiate
  it, so deep-shadow content passes a ratio test almost regardless of whether it's truly neutral,
  while properly-exposed content (which preserves real hue differences) is comparatively
  under-selected even when it genuinely is neutral. The density-local reference fixes this half of the
  problem by comparing each pixel only against others at a similar density (verified: the old
  bimodal-fraction test, `neutral_fraction=0.25` failing to recover a known profile, is now fixed
  even down to `neutral_fraction=0.01`; a direct synthetic reproduction of the traffic-light/hair/
  sign numbers is a permanent regression test).
- **What that fix does *not* solve, confirmed via the same investigation — don't re-attempt without
  new evidence**:
  - `_shadow_and_highlight_from_candidates` still extracts the 99.9th/0.1th percentile of whatever
    candidates survive — which assumes a photo's best real neutral references sit at its tonal
    extremes. On `IMG_0151.tif`, the user's own confirmed-good manual picks sat at the 56th and 82nd
    percentile of the frame's own luminance range — nowhere near the true extremes. No amount of
    fixing the *classification* step changes what the *extraction* step reaches for.
  - Even the density-local reference can fail on a real (not synthetic) photo: a pixel's own density
    bin can itself be dominated by *other*, non-neutral real content (e.g. building facades at a
    similar exposure to a genuinely-neutral sign) — the local gray-world assumption isn't guaranteed
    to hold locally any more than it's guaranteed to hold globally. Confirmed: after this fix, the
    `IMG_0151.tif` sign still isn't classified as a candidate, even though the overlay's overall
    coverage is visibly more sensible (correctly includes plausible neutral building material, not
    just toe-compressed hair/dark objects).
  - A "consensus" idea was tried and rejected: instead of trusting any reference ratio, search over
    candidate (shadow, highlight) pairs and score each by how much of the whole frame becomes
    neutral after applying it. This backfires — the same toe-convergence effect makes a large,
    homogeneous mass of underexposed pixels trivially "agree" with each other under almost any
    correction that treats them consistently, so the search kept preferring a wrong-but-
    self-consistent profile (drawn from two barely-separated shadow-region points) that scored
    *higher* than the real, manually-verified ground truth (0.31 vs. 0.12 on a "fraction of frame
    now neutral" metric). Do not resurrect this approach without a fundamentally different, more
    robust scoring signal.
  - A cheap, narrow safety net was added: `auto_density_balance`/`roll_auto_density_balance` now
    warn (not raise) when the solved shadow/highlight candidates are suspiciously close in density
    (`_check_density_separation`). This only catches a near-degenerate near-zero-span pair — it does
    NOT catch a wrong-but-well-separated pair, which is exactly the `IMG_0151.tif` failure mode. Be
    explicit about that limit; it is not a general correctness guarantee.
  - **Bottom line, now backed by concrete evidence rather than a vague caveat**: `--auto-density`'s
    shadow estimate is structurally more trustworthy than its highlight estimate (shadow-end
    convergence means almost any selected shadow candidate lands close to the true film base
    anyway). For a photo whose best neutral references aren't at the tonal extremes — which is
    common, not an edge case — manual calibration from neutral points the user vouches for
    (`halide calibrate`; its Details drawer's auto-detected-candidate overlay is a meaningfully
    better, though still imperfect, sanity check) or a ColorChecker are the reliable options, not further heuristic tuning of the auto tier.
- **Manual calibration fits any number of user-picked neutral points, across any frames of a
  roll** (`core/density.py::fit_density_balance`, `calibration/anchors.py`), not one shadow and one
  highlight point. Why: the anchor-frame investigation (`docs/investigations/anchor-frame.md`)
  found the dominant calibration error is whether the object
  clicked is really neutral, not which frame it's on - on Roll 16 the user's two-point
  Roll16-Profile1 (trike, IMG_0158) printed ~CC3 warm against a T-shirt and a cloud that agreed with
  each other, and the Alhambra's "white" wall was cream. How it works and why:
  - The fit is a least-squares straight line per channel, D_c = a_c + D_G / s_c (the same model
    as `solve_density_balance`, which it reproduces to float precision for two points).
  - Each point is normalised to the roll's reference scan exposure first (the multiply
    `--match-scan-exposure` makes; the reference is saved as the profile's `scan` sidecar).
    Roll 16 was digitized at 1/25-1/60, and without it mixed-exposure picks fit a wrong profile.
  - Each point's agreement is judged against the fit through the *other* points (leave-one-out),
    shown as a colour-printing filter value and direction, "CC 8 R" (Kodak CC = density x 100).
    Against a fit that includes it, a bad point hides its own error (a point truly CC 3.9 off read
    CC 2.6 while good points took the blame).
  - No reading while the others span < 0.1 D (`MIN_DENSITY_SEPARATION`): an extrapolated line
    accused the trike of "CC 8 M" when the other two points were 0.06 D apart.
  - Bands: <= CC 5 calm, 5-10 amber, > 10 red. The "was that object really neutral?" hint fires
    from amber, not red: on Roll 16 the known-suspect objects read amber (cream wall CC 7.9 R,
    sunlit cloud edge CC 9.6 Y) while trusted whites stayed within CC 4.8. The user confirmed CC 5
    felt right in use (it mostly fired on shadowed neutrals, plausibly tinted by what casts the
    shadow).
  - The hint names one point - the one whose removal leaves the others most consistent, not simply
    the furthest off: with an outlier among them, a good point at the end of the density range read
    worse than the outlier. One bad point makes the others read a few CC off in the opposite
    direction; fix the worst first.
  - Film base is never added automatically (user's choice: every neutral is one they vouch for;
    crops have no rebate, so "the clearest pixels" isn't guaranteed to be base).
  - Profiles record their picks and roll folder (`anchors`/`roll` sidecars) so `halide calibrate
    --profile NAME` reopens them to add to; points keep their stored RGB (they count even if the
    roll moved) and re-attach to a moved roll by file name.
  - Those paths are stored **absolute** (`anchors.absolute`: `os.path.abspath`, symlinks kept).
    They used to be stored as typed, so `halide calibrate Roll16` saved `"Roll16"` and the profile
    only reopened from that same folder. An older relative path is used only if it exists from the
    current folder; otherwise it stays **as recorded** (`anchors.recorded_path`) - shown as
    "recorded only as 'pre-processed'", never absolutised. Found via real use: the first version
    read it against the current folder, so opening from the repo showed every old profile's roll
    as `/home/…/halide/pre-processed`, "missing" - and saving would have baked that guess in. Which
    folder it was relative to isn't recorded anywhere; Find roll… is the way to fix one. When the roll folder has gone,
    the picker says so in a "Roll not found" dialog with "Find roll…" (opens in the nearest
    surviving parent folder; points re-attach by file name) or "Continue without"; picked frames
    missing from a roll that is still there get a "Frames not found" warning. Both chosen by the
    user. `halide profile show` prints the roll and flags a missing roll or frames.
- **The picker's design was chosen by the user, decision by decision** (see the plan
  `docs/plans/multipoint-picker.md`) - ask the same way before changing it. Landscape,
  fixed-size window (~55% x 70% of the screen, 900x700 floor; at 62% height opening a drawer
  squeezed everything), never scrolling except the point list:
  - A sprocket-edged **filmstrip** of the roll across the top: previews load in the batch forkserver
    pool (`gui/loaders.py`) and "develop" in; thumbnails follow the view switch; point counts under
    frames.
  - **Negative | Positive** switch above the image instead of a Preview popup. The positive is the
    frame's own auto-density estimate (labelled "rough auto estimate - not your final output")
    until two separated points exist, then the live fit through the real print stage. A greyscale
    positive was rejected by the user ("every point will look neutral"). Picks always sample the
    raw full-resolution negative.
  - The caption under the image states which way brightness is reversed on the negative ("real
    whites look DARK here, real shadows look LIGHT") - a user hunting for a dark-looking spot on a
    raw negative clicks a real highlight. Don't soften it.
  - Right panel: step-wedge coverage bar (roll's own density range, a tick per point - the variety
    nudge; near-duplicate picks get a note, never refused); fixed-height scrolling point list
    (number, frame, density, agreement, remove); Clear frame / Clear all under it (in the header
    they clipped its title); "Build contact sheet…"; accordion drawers **Extra information** (film
    stock/process/scanner/notes - renamed from "Roll details" as too close to "Details"), **Print**
    (the old Fine-tune: exposure/grade sliders following the frame's fit until moved, flat preview,
    "Back to fitted"), **Details** (expands into spare height - it was a cramped scroll box); one red
    Save button (Develop in `--pick`).
  - Clicking a marker selects it (Delete removes); clicking a row jumps to its frame and flashes the
    marker once the frame has loaded.
  - **Contact sheet window** (`gui/proof_window.py`): a draft at once from the previews, then every
    frame at full resolution through batch's own `_worker` (identical to `batch --contact-sheet`),
    swapped in as they finish; wheel zoom, drag pan, double-click fit, hover names the frame; marks
    itself out of date when points/print change ("Rebuild contact sheet"); "Save contact sheet…"
    once complete. Frames go through a temp folder that's always deleted; the sheet is only in
    memory unless saved. Roll 16: draft ~1 s, full quality ~28 s.
  - Closing any window stops its worker processes rather than waiting (concurrent.futures has no
    public terminate, so the pool's private process table): `halide calibrate` otherwise sat ~7 s in
    the terminal after its window closed.
  - Closing also waits (no timeout) for **every** loader thread, including ones replaced by a newer
    load (`MainWindow._threads`): Qt aborts the whole process ("QThread: Destroyed while thread is
    still running", core dump) if a window is destroyed with any running. Found via real use -
    only the preview loader was stopped, with a 3 s timeout, never the full-resolution
    `FrameLoader`; reproduced on Roll 16 by closing within ~0.5 s of loading. A mid-decode frame
    load can't be interrupted, so the window hides first and the process exits when it finishes
    (up to ~2.7 s after closing during the first load).
  - The auto estimate's "candidates unusually close in density" warning is silenced in the picker
    only (`gui/roll.py::quiet_auto_estimate`) - it's CLI advice, and printed on every launch.
- **`halide invert --pick` opens the same picker for one frame** (`gui/quick_pick.py::
  run_quick_pick`, `MainWindow(is_pick_session=True)`: no filmstrip, roll loading or contact sheet;
  the red button reads "Develop"). It runs a local `QEventLoop` so it can return the picked
  `(DensityProfile, ToneCurveParams | None)` - or None if the window was closed - to its caller.
  `--save-profile-as` still works with it.
- **A saved profile can optionally carry an exposure/contrast override alongside its density
  calibration** (`calibration/profile_store.py`'s `tone` sidecar, written from the picker's Print
  drawer). Deliberately a sidecar, not a field on `DensityProfile` (`core/types.py` stays
  untouched) - density calibration and tone rendering are different things, and exposure defaults
  to per-image fitting for good reasons (`ToneCurveParams`). Precedence in
  `cli/_calibration_args.py::resolve_tone_params`: CLI flag, then saved override, then default.
  Linear-output mode is **never** saved - silently changing a future run's output format from a
  profile is the wrong kind of thing for a profile to do.
- **Profiles carry free-text details** (`film_stock`, `process`, `scanner`, `notes`), editable with
  `halide profile edit` or the picker's Extra information drawer; `film_stock` is what contact
  sheets print on their edge. `update_profile`/`rename_profile` edit the raw JSON so every sidecar
  (tone, scan, anchors, roll) survives - round-tripping through `DensityProfile` silently dropped
  them, and a lost `scan` sidecar breaks `--match-scan-exposure`.
- **The GUI moved from dearpygui to PySide6/Qt** (a full rewrite, not an incremental port) after the
  first dearpygui-based redesign still felt "thrown together" — its default auto-stacked widget flow
  made a reasonably-sized window need scrolling to see everything, which the user explicitly didn't
  want, and the two-tab structure (Calibrate/Preview) read as confusing rather than purposeful.
  Qt's real floating windows, QSS theming and hand-positioned layout gave a fixed-size,
  non-scrolling, purpose-built window. `gui/sampling.py` (the pixel-math/coordinate logic) needed
  zero changes — it was already framework-independent. Note for anyone testing a fresh environment:
  this sandbox needed several system libraries installed (`libegl1`, `libxcb-cursor0`,
  `libxkbcommon-x11-0`, `libxcb-icccm4`, `libxcb-keysyms1`, `libxcb-shape0`, `libxcb-xkb1`) before
  Qt would even import — a normal desktop Linux machine almost certainly already has these.
- **Deferred, deliberately**: a skeuomorphic "enlarger controller" skin for the GUI's buttons (grey
  rounded body panels, chunky black secondary buttons, a large red circular primary button, a small
  red LED-style digit readout) modeled on a reference photo (`enlarger-controller.png`, repo root,
  gitignored) of a real Durst enlarger timer. The current theme (`gui/theme.py`) only reserves red
  for one primary button per window (agreement's "red" is a lighter, warmer status colour so it
  doesn't compete) — the full custom-painted-widget version needs its own plan
  (`docs/specs/enlarger-skin.md`).
- **With no calibration source, a terminal run asks for one up front** (`cli/_calibration_args.py::
  choose_calibration_source`): the saved profiles, newest first (so the one just made in `halide
  calibrate` is on top), picking points (invert), or the automatic estimates - recorded on `args`
  as if the flag had been passed. It must run before anything reads `args.profile`: the
  scan-exposure reference comes from the profile, and invert decides whether the calibration came
  from the frame itself by whether a profile was given - a profile chosen later would silently lose
  both. Without a terminal it's an error naming the missing step (pick points in `halide calibrate
  ROLL_DIR` and save a profile, or `--pick` on invert).
- **`--pick` is invert-only, deliberately.** `add_calibration_arguments(parser, allow_pick=...)`
  defaults to `False`; only `invert_cmd.py` passes `True`. The picker now has the roll/frame picker
  that `batch --pick` would have needed (the filmstrip), but the user left `batch --pick` out of
  scope - `halide calibrate ROLL_DIR`, save, then `batch` (which offers the new profile) covers it.
- **The batch/export progress display is a static contact sheet, not a moving strip**
  (`batch/progress.py`). The whole roll is drawn at once in sprocket-edged strips of six 3:2
  frames (`███`); only individual frames animate (pulse while developing, fade to near-black when
  done). It replaced a one-strip window that wiped across to the next slice, which the user found
  busy rather than film-like. The layout (full → compact shared sprockets → scrolling with
  "N frames above/below") is fixed at start from the terminal size so every redraw has the same
  height, and the status line is trimmed rather than allowed to wrap — both because the in-place
  redraw's cursor-up counts logical lines, and a taller-than-screen or wrapped frame leaves stale
  rows behind (verified by replaying real `halide batch` pty output into a virtual terminal).
- **Everything `batch`/`export`/`print` decides before developing is one aligned "run sheet"**
  (`console.RunSheet`, rows shared via `cli/_run_sheet.py`): Roll, Scans, Scan exposure,
  Calibration, Output, Workers — a label column, values/warnings wrapped with a hanging indent,
  framed by sprocket rules per the sizing convention. It replaced a run of unrelated sentences
  (roll warnings, scan-exposure lines, "Auto-selected N workers…") that blended together. Rows are
  held until the sheet closes so prompts come before it, not inside it; the roll estimate shows a
  spinner row. `--quiet` still prints warnings, as plain lines. An exposure spread that
  `--match-scan-exposure` is correcting is still a ⚠ (worded as "evened out"), not a plain row —
  it was briefly demoted, and the user noticed the warning was missing: the correction is exact
  only for a truly linear scan and isn't yet validated on a real two-exposure scan. Fixed on the way: the scan-exposure
  reference was announced twice, the exposure warning told you to pass `--match-scan-exposure`
  when it already was, the "profile doesn't record its scan exposure" warning appeared for manual
  `--rm/--bm` values (no profile involved), and an explicit `--workers` over the memory budget
  printed "Warning: Warning:".
- **Tab completion for zsh, bash and fish installs itself; there is deliberately no `halide
  completion` command** (`cli/completion.py`, called from `run_cli` after the command, so tests
  calling `main()` never trigger it). The user wanted no extra commands, and wanted it to work for
  anyone, not just their own zsh. Static scripts, not `argcomplete`: argcomplete re-runs halide on
  every Tab, and even a lean `halide --help` takes ~0.1 s. zsh and bash come from
  `shtab` (generated from `build_parser()`); shtab can't do fish, so `fish_script` walks the same
  parser, and the fish script carries a small word parser (`__halide_state`) to know which
  subcommand/positional the cursor is at.
  - When: on every run where stdin/stdout are ttys, for the shell halide was started from (its
    parent process via psutil, else `$SHELL`). The script is rewritten only if it changed (6 ms).
  - Hooking in, the first time only, with one printed note: zsh and bash append a marked block to
    `$ZDOTDIR/.zshrc` or `~/.bashrc` (`~/.bash_profile` on macOS); the user chose this over
    printing the line for them to paste. fish autoloads `~/.config/fish/completions/halide.fish`,
    so it needs no rc edit, but a `halide.fish` without halide's header is never overwritten. The
    zsh block uses `compdef` because it lands after `compinit`, and runs `compinit` itself if the
    zshrc never does.
  - Per-shell stamp files make that one-time: a removed block or fish file stays removed.
    `HALIDE_NO_COMPLETION=1` disables it all, and failures are swallowed.
  - Value completers come from the argument `dest` in `_KINDS` (tiff/file/dir/profile, with
    `_KIND_OVERRIDES` for exceptions), so a new file/profile argument needs an entry there.
    Profile names are listed by the shell itself from the profiles folder.
  - fish quirks, all found by testing: `__fish_complete_suffix` only sorts matching files first
    (fish ≥ 3.6), so TIFFs are filtered by `__halide_tiffs`. `-d` also labels an option's
    values, so values go on a separate line. An already-open fish caches "no completions" and
    never re-checks, so every note says "open a new terminal".
  - Verified by driving real interactive `zsh -i`, `bash -i` (with and without the bash-completion
    package) and `fish -i` in a pty. Not covered: tcsh, PowerShell/Windows, and the macOS system
    bash 3.2 (untested).
- **Starting the CLI imports no numpy, tifffile, Pillow, colour-science or cupy** — they're
  imported inside the functions that use them. Before, `halide --help`/`profile list` took ~0.75 s
  in the dev sandbox (over 1 s on the user's machine): `import colour` (it drags in scipy and its
  plotting module) was ~0.55 s of it, numpy/tifffile/Pillow ~0.15 s. Now ~0.1 s, the rest being
  Python and the stdlib. How: colour only inside `io/icc.py::working_space_matrices`,
  `io/raster.py::srgb_matrix`/`to_srgb_8bit`'s CPU branch, `processing.py::_acescg_matrix`;
  tifffile only in the header readers (`io/scan_metadata.py`, `batch/orchestrator.py`); the
  orchestrator's workers import `halide.processing` themselves; CLI commands import
  `halide.processing`/`io.contact_sheet` in `run()`-level functions; `Stage` moved to
  `core/types.py` and the sheet defaults to `io/contact_sheet_defaults.py` (both re-exported from
  their old homes), because the parser needs them. `cupy` is never imported at all unless
  `--device`/`$HALIDE_DEVICE` actually resolves to a GPU (`halide.device`, GPU acceleration above)
  — even `halide gpu`'s own status check only uses `ctypes`/`importlib`, not cupy. Workers still get
  everything preloaded except cupy: `colour` is listed in `_FORKSERVER_PRELOAD`, but cupy is
  deliberately not (a forked CUDA context is unusable in the child; the GPU service is spawned and
  resolves the device itself — or, in per-worker mode, each GPU worker does). `tests/unit/test_cli_startup.py` fails if building the parser imports any of
  them again (`HEAVY` includes `"cupy"`). Outputs unchanged bit-for-bit (verified on real scans:
  invert, export, batch, contact).
- **Cut for now, deliberately**: ColorChecker calibration tier, a denoise stage, and a real (not
  naive-average) B&W negative mode. Not oversights — out of scope until asked for.

## Interactive GUI testing

The GUI has no automated test coverage for actual rendering/interaction (only `gui/sampling.py`'s
pure logic is unit-tested) — verify real interaction changes with a virtual display and synthetic
mouse input rather than trusting code review alone, especially for anything with no prior precedent
to compare against (a new widget, a new popup/dialog, a new interaction pattern). `Xvfb`, `xdotool`,
and `import` (screenshot capture) are installed in this environment:

```bash
Xvfb :99 -screen 0 1280x1024x24 &
DISPLAY=:99 QT_QPA_PLATFORM=xcb .venv/bin/python -m halide.cli.main calibrate <test.tif> &
# find/activate the window, then drive it:
xdotool search --name halide windowactivate
xdotool mousemove --window <id> <x> <y>
xdotool click 1
# capture and actually look at the result (don't just check "no crash"):
import -window <id> /tmp/check.png    # or -root for the whole virtual screen
```

For a pure static render (no interaction needed — e.g. checking a theme/layout change), Qt's
`QT_QPA_PLATFORM=offscreen` plus a widget's own `.grab().save(path)` is faster and doesn't need
Xvfb at all; reach for Xvfb + xdotool specifically when the thing being verified is an interaction
(a click, a hover, a popup opening) rather than a static look.

Always tear down both the app process and the `Xvfb` process afterward — neither self-terminates.
This is naturally background/forked-subagent work given the volume of trial-and-error interaction
involved; keep the raw tool-call trace out of the main conversation and report back a tight
pass/fail summary instead.

## Working with the user

They are experienced at film photography and darkroom/printing concepts (paper contrast grades,
enlarger exposure, characteristic curves) but not necessarily at reading Python — explanations that
lean on the darkroom analogy land well. They want to understand *why* a fix works, not just that it
does, and they push back (correctly) on hand-wavy claims — verify against real data/real scans
before asserting something is fixed. Two real test scans usable for this live in the repo root but
are gitignored (`IMG_0156.tif`/`IMG_0156-nowb.tif`, `IMG_0158.tif`) — ask the user before assuming
new ones are available, and never commit them.
