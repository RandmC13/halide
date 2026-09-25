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
```
zsh Tab completion sets itself up on the first run from a zsh terminal (see "Decisions and why").
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
               # ToneCurveParams — both frozen dataclasses).
  io/          # tiff.py (read/write + dtype normalization), icc.py (validate + color-manage
               # the embedded ICC profile), raster.py (ACEScg -> sRGB delivery export),
               # lut.py (.cube reader, used by tone_render.py at runtime AND by golden tests).
  calibration/ # auto.py (statistical fallback calibration), profile_store.py (named,
               # reusable DensityProfile JSON files under ~/.config/halide/profiles/), anchors.py
               # (the picker's neutral-point model: scan-gain normalisation, agreement, gates).
  batch/       # orchestrator.py (ProcessPoolExecutor over processing.py, one bad frame doesn't
               # abort the batch), progress.py (terminal rendering only, no math).
  cli/         # main.py + commands/*.py (argparse), _calibration_args.py (shared flag
               # definitions/resolution used by both invert and batch).
  gui/         # PySide6/Qt app. Pure, tested, no Qt: sampling.py (picking math), roll.py (the
               # session model), render.py (negative/positive display). Widgets: main_window.py
               # (the picker), filmstrip.py, step_wedge.py, point_list.py, drawers.py,
               # proof_window.py (contact sheet window), loaders.py (background loading), theme.py.
  processing.py # The glue layer: read -> validate ICC -> convert to working space -> calibrate
               # -> run_pipeline -> write. Both the single-file CLI command and the batch worker
               # call this same function rather than duplicating the chain.
  banding.py   # map_in_bands: runs per-pixel core/ functions over ~4 MiB bands of rows into a
               # buffer the caller owns — how every full-resolution path stays near 1 frame of
               # memory. Bit-identical to the whole-array call (see "Decisions and why").
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
    *not* done, because each would change output: fusing the two ICC matrices into one float32
    matrix (~2 s/frame faster, output moves ~1e-7), replacing colour-science, measuring the fit or
    auto calibration on a downsampled frame, float16 buffers. Known, pre-existing and untouched:
    export PNGs / contact-sheet JPEGs aren't byte-reproducible between runs even on unchanged code —
    Pillow stamps its synthesized sRGB ICC profile with the creation time (pixels are identical).
  - **Worker pool after that pass**: `K = 3` (baseline 150 MiB) for inversion and `K = 2` (100 MiB)
    for export/contact, refit from real worker processes' peaks (worst: `--auto-density` 509 MiB RSS
    on a 182 MiB scan) — see the constants' comment in `batch/orchestrator.py`. exiftool needs no
    term: it streams (67 MiB peak on a 130 MiB output) and `process_scan` frees the frame before
    running it. Workers come from a forkserver with `halide.processing` preloaded (they share numpy/
    colour-science pages copy-on-write: 4 idle workers 294 -> 73 MiB proportional memory). The old
    fixed `min(cpu_count, 6)` cap is now one worker per *physical* core (`_cpu_cap`, the user's
    choice) — a hyperthread sibling adds little to a numpy-bound worker but costs a frame of memory.
    Net effect: 4 GiB free now runs 5 workers (was 1), 7 GiB runs 9 (was 3). Measured on 16 real
    frames in the dev sandbox (5.5 GiB free, 16 cores), each version at its own default: old code 2
    workers, 46.4 s, 3.83 GiB total PSS; new code 8 workers, 16.9 s, 2.49 GiB — outputs 16/16
    bit-identical. Scaling flattens past ~4 workers there (1: 66 s, 2: 35 s, 4: 22 s, 6: 20 s, 8:
    17 s) — probably disk writes (~2 GB of TIFFs per run) and memory bandwidth, not re-tuned from
    one sandbox; re-check on the user's own machine before lowering the physical-core cap.
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
- **zsh tab completion installs itself; there is deliberately no `halide completion` command**
  (`cli/completion.py`, called from `run_cli` after the command, so tests calling `main()` never
  trigger it). The user uses zsh and wanted no extra commands. It's a static script from `shtab`,
  generated from `build_parser()`, rather than `argcomplete`: argcomplete re-runs halide on every
  Tab, and importing halide takes ~1.1 s (mostly colour-science). On every run from a zsh terminal
  (`$SHELL` is zsh and stdin/stdout are ttys), halide rewrites `$XDG_DATA_HOME/halide/zsh/_halide`
  if it changed (6 ms). The first time, it also appends a marked block to `$ZDOTDIR/.zshrc` and
  prints one note; the user chose this over printing the line for them to paste. The block uses
  `compdef` because it lands after `compinit`, and it runs `compinit` itself if the zshrc never
  does. A stamp file makes the edit one-time (a removed block stays removed);
  `HALIDE_NO_COMPLETION=1` disables it all; failures are swallowed. Value completers are set by
  argument `dest` in `_COMPLETERS` (TIFFs, folders, or saved profile names, which zsh lists from
  the profiles folder itself), so a new file/profile argument needs an entry there. Verified by
  driving a real `zsh -i` in a pty. bash/fish: not done.
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
