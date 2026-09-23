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
  (invert/batch also take --match-scan-exposure [--scan-reference FRAME])
halide export <positive.tif> <delivery.png>   # ACEScg TIFF -> delivery-ready sRGB PNG/JPEG
halide profile list|show|rename|delete|set-scan-reference
halide calibrate [negative.tif]                # Dear PyGui: anchor-frame picker + live preview;
                                                # auto-loads the given TIFF if a path is passed
```
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
               # reusable DensityProfile JSON files under ~/.config/halide/profiles/).
  batch/       # orchestrator.py (ProcessPoolExecutor over processing.py, one bad frame doesn't
               # abort the batch), progress.py (terminal rendering only, no math).
  cli/         # main.py + commands/*.py (argparse), _calibration_args.py (shared flag
               # definitions/resolution used by both invert and batch).
  gui/         # PySide6/Qt app. sampling.py is pure/tested (no Qt import); app.py (entry point),
               # main_window.py (the picker), preview_popup.py, theme.py hold the widget code.
  processing.py # The glue layer: read -> validate ICC -> convert to working space -> calibrate
               # -> run_pipeline -> write. Both the single-file CLI command and the batch worker
               # call this same function rather than duplicating the chain.
```

The internal working color space is **ACEScg**, chosen (not just used) — the blog is explicit that
the inversion math's result depends on which working space it runs in, so the working-space choice
is not an implementation detail. `core/` never touches color management directly; `io/icc.py`
converts into ACEScg on read, output is always written back tagged with a real ACEScg ICC profile
(vendored from Elle Stone's `elles_icc_profiles`, cross-validated against `colour-science`'s own
ACEScg definition in `tests/unit/test_icc.py` — not hand-authored).

Calibration is three-tier, all producing the same `DensityProfile`: ColorChecker (not yet built),
anchor-frame manual picking (`gui/main_window.py`, or `--rm/--bm/--rs/--bs` on the CLI), and
statistical auto-detection (`calibration/auto.py`, `--auto-density`/`--auto-density-roll`).
Profiles are meant to be solved once per film-stock/process/scanner combination and reused
(`--save-profile-as`, `halide profile`), not re-solved per image.

## Decisions and why (don't re-litigate these without new evidence)

- **TIFF-in, not RAW-in.** The user's darktable/RawTherapee export already handles
  demosaic/crop/dust-removal/lens-correction correctly; reimplementing that would duplicate work
  for no accuracy gain. `io/icc.py` owns validating that the color-management handoff is correct.
- **Hand-rolled ICC parser in `io/icc.py`, not Pillow.** Pillow's `ImageCms` only exposes
  profile-level metadata (name, color space, rendering intent) — no way to check "is this a linear
  matrix-shaper profile," which is exactly what's needed here. The matrix/TRC subset of the ICC
  spec this project reads is small and precisely documented; no PyPI package does this narrow job.
  Pillow is still a dependency, but only for `io/raster.py`'s PNG/JPEG writing — an unrelated job.
- **The default tone-render curve is a vendored real paper response curve** (`assets/tone_curves`,
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
  (profiles now carry a "scan" sidecar; `halide profile set-scan-reference` for older ones; batch
  falls back to the roll's most common setting with a warning). Deliberately **not** corrected:
  per-frame raw white balance (a per-channel multiply in the camera's own colour space, before the
  raw converter's camera matrix — not invertible from the export without that matrix, so it's
  reported, never approximated) and active tone/colour modules in darktable's embedded history
  (the export isn't linear). Found on the same roll: the camera was metering each frame (1/25-1/60)
  and "as shot" white balance was the camera's auto WB (14 distinct values, R ±5%, B ±7%); one
  frame had shadows & highlights active. Not yet validated against a real two-exposure scan of one
  frame (see TONE_OUTPUT_PLAN.md follow-ups).
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
- **`mode="linear"` (`--output flat`, alias `--linear-output`) output is scaled, not a bare
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
- **`--auto-density-roll` selects neutral candidates per frame, then pools candidates — never pools
  raw pixels across frames first.** The per-channel median used to judge "how neutral is this
  pixel" (`calibration/auto.py::_saturation`) is only a valid proxy for the film's own systematic
  imbalance when computed from one frame's own pixels; computed from pixels pooled across frames
  with different scene content, it blends in each frame's own scene-color average too, which isn't
  shared across a roll the way the film base is. This was a real bug (found via testing on two real
  same-roll scans), now fixed — but did not fully resolve a residual color cast, see the limitation
  noted below.
- **`calibration/auto.py::_saturation` judges neutrality against a density-local reference
  (`_density_reference`), not one frame-wide median.** A single global median is only a valid stand-
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
  under-selected even when it genuinely is neutral. `_density_reference` fixes this half of the
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
    common, not an edge case — anchor-frame manual calibration (`gui/main_window.py`, and its
    auto-detected-candidate overlay is now a meaningfully better, though still imperfect, sanity
    check) or a ColorChecker are the reliable options, not further heuristic tuning of the auto tier.
- **The Calibrate picker's point-picking labels state explicitly which way brightness is
  inverted, not just "dark"/"bright."** The screen displays the RAW, uninverted negative
  (`gui/sampling.py::preview_stretch`) — on a raw negative, brightness is backwards from the real
  scene: a real highlight (bright in life) is a dense, DARK area on the raw negative, and a real
  shadow (dark in life) is a thin, LIGHT area. The old labels ("Shadow (dark neutral)" / "Highlight
  (bright neutral)") didn't say which brightness they meant, and a user hunting for a dark-looking
  spot on the displayed raw negative would click a real highlight instead — a plausible source of
  genuinely wrong calibration, not just confusing wording. Fixed by stating both framings explicitly
  in the radio labels and instructional text. Don't revert to shorter/vaguer labels for aesthetics.
- **The Calibrate picker solves a preview profile automatically once both points are picked**,
  shown on request via a Preview popup (`gui/preview_popup.py`) rather than requiring a profile name
  + "Save" click before seeing anything. Saving to a named, reusable profile is a fully separate,
  optional action — per the user's real use case: sometimes you just want to see how a pick looks,
  not build a reusable profile. The main window also shows a hover magnifier (a small always-on-top
  `QWidget` that follows the cursor, `gui/main_window.py::Magnifier`) and draws picked-point markers
  directly in `ImageView.paintEvent`, plus a collapsible "Details" section to visualize
  `calibration/auto.py`'s own neutral-candidate selection (`_neutral_candidate_mask`) so a manual
  pick can be cross-checked against the independent statistical method. Verified working via real
  interactive testing (Xvfb + xdotool, see "Interactive GUI testing" below), not just code review.
- **`halide invert --pick` opens a standalone calibration picker inline for a single, one-shot
  run** (`gui/quick_pick.py::run_quick_pick`, wired into `cli/_calibration_args.py::
  resolve_density_profile` as a fourth mutually-exclusive calibration source alongside
  `--profile`/manual/`--auto-density`). Addresses a real workflow gap: nothing connected a
  first-time user to `halide calibrate`, and a user who just wanted to manually pick values for
  one image had to go through the full profile-naming app. Reuses `gui/main_window.py::MainWindow`
  as-is via its `is_pick_session=True` mode (hides the load controls since the path is already
  known from the CLI arg, and relabels the primary button "Develop") — the only new code is the
  blocking entry point itself, which runs a local `QEventLoop` instead of `QApplication.exec()` so
  it can return the picked `(DensityProfile, ToneCurveParams | None)` (or `None` if the window was
  closed without picking) to its caller instead of running until the process exits. This is new
  territory for this codebase's GUI code (every other entry point runs the full event loop and
  exits the process) — verified via real interactive testing, not just code review, including a
  full real `halide invert --pick` run through to a written output file. Saving a named profile
  (`--save-profile-as`) still works unmodified in combination with `--pick` — using a calibration
  for this run and persisting it for later remain orthogonal, as everywhere else in this project.
- **A saved profile can optionally carry an exposure/contrast override alongside its density
  calibration** (`calibration/profile_store.py`'s `tone` sidecar in the profile JSON, written by
  the Calibrate picker's Preview popup's opt-in "Fine-tune" controls). Deliberately kept as a
  sidecar rather than a field on `DensityProfile` itself (`core/types.py` stays untouched) — density
  calibration and tone-rendering choices are conceptually different things, and `ToneCurveParams`'s
  own docstrings already explain why exposure defaults to per-image auto-computation rather than a
  fixed value. Precedence in `cli/_calibration_args.py::resolve_tone_params`: an explicit CLI flag
  wins, then a saved override, then the built-in default. Linear-output mode is deliberately
  **never** part of this override — it stays a CLI-flag/popup-preview-only choice, since silently
  changing a future run's output format based on a saved profile felt like the wrong kind of thing
  for a profile to do by default.
- **The GUI moved from dearpygui to PySide6/Qt** (a full rewrite, not an incremental port) after the
  first dearpygui-based redesign still felt "thrown together" — its default auto-stacked widget flow
  made a reasonably-sized window need scrolling to see everything, which the user explicitly didn't
  want, and the two-tab structure (Calibrate/Preview) read as confusing rather than purposeful.
  Qt's real floating windows (the Preview popup, save-name dialog), QSS theming, and hand-positioned
  layout gave a genuinely small (fixed size, ~25% of screen, computed from
  `QGuiApplication.primaryScreen()`), non-scrolling, purpose-built window instead. `gui/sampling.py`
  (all the actual pixel-math/coordinate logic) needed zero changes across this rewrite — it was
  already framework-independent. Note for anyone testing a fresh environment: this sandbox needed
  several system libraries installed (`libegl1`, `libxcb-cursor0`, `libxkbcommon-x11-0`,
  `libxcb-icccm4`, `libxcb-keysyms1`, `libxcb-shape0`, `libxcb-xkb1`) before Qt would even import —
  a normal desktop Linux machine almost certainly already has these.
- **Deferred, deliberately**: a skeuomorphic "enlarger controller" skin for the GUI's buttons (grey
  rounded body panels, chunky black secondary buttons, a large red circular primary button, a small
  red LED-style digit readout) modeled on a reference photo (`enlarger-controller.png`, repo root,
  gitignored) of a real Durst enlarger timer. The current theme (`gui/theme.py`) only takes the
  cheap, immediate step of reserving red for one primary button per window as a nod to this — the
  full custom-painted-widget version needs its own separate plan.
- **The "no calibration source" error names the actual missing step** instead of just listing
  flags: it now says to run `halide calibrate --save-profile-as NAME` to produce a `--profile`
  source, and (only on `invert`, where it's wired up) mentions `--pick` as the immediate one-shot
  alternative. `batch`'s version of the same error deliberately doesn't mention `--pick` — see
  below, it isn't implemented there yet.
- **`--pick` is invert-only for now, not implemented on `batch`, deliberately.** `add_calibration_
  arguments(parser, allow_pick=...)` defaults to `False`; only `invert_cmd.py` passes `True`. A
  real `batch --pick` needs more than a straight port: unlike `invert`, `batch` has no single input
  image to pre-load a picker with — a roll's worth of frames — so it will likely need a GUI
  frame-picker (choose which image in the roll to calibrate against) before the existing
  shadow/highlight picker makes sense to open at all. Don't wire up `--pick` on `batch` without
  designing that piece first.
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
