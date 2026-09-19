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
halide batch <in_dir> <out_dir> [--auto-density-roll] [--save-profile-as NAME]
halide export <positive.tif> <delivery.png>   # ACEScg TIFF -> delivery-ready sRGB PNG/JPEG
halide profile list|show|rename|delete
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
  gui/         # Dear PyGui app. sampling.py is pure/tested (no dearpygui import); app.py,
               # calibrate_screen.py, preview_screen.py hold the actual widget/callback code.
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
anchor-frame manual picking (`gui/calibrate_screen.py`, or `--rm/--bm/--rs/--bs` on the CLI), and
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
  "faithful over flashy" priority. `ToneCurveParams.contrast` (default 0.5, not the curve's native
  1.0) and `exposure` (default `None` = auto-computed per image, see
  `core/tone_render.py::estimate_exposure`) both exist because of real bugs found by testing
  against actual scans, not speculative options — see their docstrings for the specific failure
  each one fixes. Don't "simplify" them back to fixed constants.
- **`mode="linear"` output is scaled, not a bare unbounded passthrough** (`estimate_linear_scale`
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
    common, not an edge case — anchor-frame manual calibration (`gui/calibrate_screen.py`, and its
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
- **The Calibrate picker now solves and renders a live preview automatically once both points are
  picked** (`CalibrateScreen._maybe_render_live_preview`/`_render_preview` in
  `gui/calibrate_screen.py`), instead of requiring a profile name + "Solve & Save" click before
  showing anything. Saving to a named, reusable profile is now a fully separate, optional action —
  per the user's real use case: sometimes you just want to invert one image with clearly visible
  neutrals to see how it looks, not build a reusable profile. The screen also shows a hover
  magnifier (a fixed-size `dpg.add_dynamic_texture` updated via `dpg.set_value` in place, not
  delete+recreate — the click-driven main image and preview textures stay on delete+recreate since
  they only change on discrete events, not every mouse-move) and draws picked-point markers on a
  `dpg.drawlist` overlay, plus an optional toggle to visualize `calibration/auto.py`'s own
  neutral-candidate selection (`_neutral_candidate_mask`) so a manual pick can be cross-checked
  against the independent statistical method. All verified working via real interactive testing
  (Xvfb + xdotool, see "Interactive GUI testing" below), not just code review — this was new DPG
  territory for this codebase (no prior drawlist/hover-handler/dynamic-texture usage existed).
- **`calibrate_screen.py` never deletes+recreates a GPU texture except when a genuinely new,
  differently-sized image is loaded — everything else updates an existing texture's pixels via
  `dpg.set_value` in place.** Found via a real segfault (`segmentation fault (core dumped)`) on the
  user's actual machine, in two spots: toggling the auto-candidate overlay checkbox, and clicking
  around the image to re-pick points. Both used to call `dpg.delete_item`+`dpg.add_raw_texture` on
  every single interaction, not just on image load — repeatedly destroying and recreating a
  GPU-backed texture from Python is a known class of instability in this dearpygui version
  (couldn't reproduce it under Xvfb's software rendering even with heavy stress-testing, which
  points at the real GPU/driver-backed rendering path specifically). Fixed: `toggle_auto_overlay`
  now calls `_refresh_main_image` (`set_value` only) instead of `_upload_main_image`
  (delete+recreate); `_render_preview` creates its texture once on the first successful pick pair
  and `set_value`s it on every re-pick after that. `_upload_main_image` itself still legitimately
  delete+recreates, but only from `load_image` — a new image can be a different size, so that one
  case genuinely needs it. Don't reintroduce delete+recreate on a path that fires from routine
  interaction (a checkbox, a click) rather than a new file being loaded.
- **`halide invert --pick` opens a standalone calibration picker inline for a single, one-shot
  run** (`gui/quick_pick.py::run_quick_pick`, wired into `cli/_calibration_args.py::
  resolve_density_profile` as a fourth mutually-exclusive calibration source alongside
  `--profile`/manual/`--auto-density`). Addresses a real workflow gap: nothing connected a
  first-time user to `halide calibrate`, and a user who just wanted to manually pick values for
  one image had to go through the full profile-naming app. Reuses `calibrate_screen.build`/
  `CalibrateScreen` as-is (`build(show_path_input=False)` hides the now-redundant TIFF-path field
  since the path is already known from the CLI arg) — the only new code is the blocking entry
  point itself, which runs a manual `dpg.render_dearpygui_frame()` loop instead of
  `dpg.start_dearpygui()` so it can return the picked `DensityProfile` (or `None` if the window
  was closed without picking) to its caller instead of running until the process exits. This is
  new territory for this codebase's GUI code (every other entry point runs the full event loop
  and exits the process) — verified via real interactive testing, not just code review. Saving a
  named profile (`--save-profile-as`) still works unmodified in combination with `--pick` — using
  a calibration for this run and persisting it for later remain orthogonal, as everywhere else in
  this project.
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
mouse input rather than trusting code review alone, especially for anything DPG-specific
(drawlists, hover handlers, dynamic textures) that has no prior precedent to compare against.
`Xvfb`, `xdotool`, and `import` (screenshot capture) are installed in this environment:

```bash
Xvfb :99 -screen 0 1280x1024x24 &
DISPLAY=:99 .venv/bin/python -m halide.cli.main calibrate <test.tif> &
# find/activate the window, then drive it:
xdotool search --name halide windowactivate
xdotool mousemove --window <id> <x> <y>
xdotool click 1
# capture and actually look at the result (don't just check "no crash"):
import -window <id> /tmp/check.png    # or -root for the whole virtual screen
```

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
