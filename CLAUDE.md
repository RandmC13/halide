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
halide invert <negative.tif> <positive.tif> [--profile NAME | --rm/--bm/--rs/--bs | --auto-density]
halide batch <in_dir> <out_dir> [--auto-density-roll] [--save-profile-as NAME]
halide export <positive.tif> <delivery.png>   # ACEScg TIFF -> delivery-ready sRGB PNG/JPEG
halide profile list|show|rename|delete
halide calibrate                               # Dear PyGui: anchor-frame picker + live preview
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
- **Cut for now, deliberately**: ColorChecker calibration tier, a denoise stage, and a real (not
  naive-average) B&W negative mode. Not oversights — out of scope until asked for.

## Working with the user

They are experienced at film photography and darkroom/printing concepts (paper contrast grades,
enlarger exposure, characteristic curves) but not necessarily at reading Python — explanations that
lean on the darkroom analogy land well. They want to understand *why* a fix works, not just that it
does, and they push back (correctly) on hand-wavy claims — verify against real data/real scans
before asserting something is fixed. Two real test scans usable for this live in the repo root but
are gitignored (`IMG_0156.tif`/`IMG_0156-nowb.tif`, `IMG_0158.tif`) — ask the user before assuming
new ones are available, and never commit them.
