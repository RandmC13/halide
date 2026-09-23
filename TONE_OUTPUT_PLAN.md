# Plan: two output modes ("print" and "flat") and a stand-alone print step

Status: **implemented** on this branch (2026-09-23). Decisions taken: D1 print is the default,
D2 99.9th-percentile highlight anchor, D3 `estimate_exposure` removed, D4 per-frame grade in batch
(the optional `--grade roll` and `--linear-scale roll` flags were *not* built), D5 black-point
compensation deferred. Written against `c9fa7de`.

This plan covers three things you raised:

1. Does stretching halide's current output to true black/white in darktable distort the tones,
   given that the tone curve has already been applied?
2. Two output modes: a finished-looking **print**, and a **flat** linear positive with as little
   bias as possible, meant for your own editing.
3. A command that applies *only* the print/tone stage, so a flat positive can be edited in
   darktable and then printed by halide. That only works if the order of operations is right.

It changes **none** of the colour science: `core/density.py`, `core/invert.py`,
`calibration/auto.py`, and `io/icc.py` are untouched. Everything here happens in the tone-render
stage (the "printing" step) and in the CLI around it.

---

## 1. What is happening today (measured, not guessed)

All three real scans were run through the current default (`paper` mode, `contrast=0.5`,
auto-exposure). The sRGB 8-bit values are luminance at the given percentile of pixels:

| scan | darkest 0.1% | median | brightest 0.1% | paper's own black / white |
|---|---|---|---|---|
| IMG_0151 | 46 | 79 | 221 | ≈11 / 255 |
| IMG_0156 | 44 | 99 | 220 | ≈11 / 255 |
| IMG_0158 | 45 | 80 | 233 | ≈11 / 255 |

(These runs used `--auto-density` profiles because no saved profile was available in this
environment. The tone-range numbers depend very little on which calibration is used.)

**Why.** In print terms, the paper curve (Endura, `assets/tone_curves/paper_endura.cube`) goes from
highlight detail to deep shadow over about **0.82 log units** of print exposure (computed in §3).
That is the paper's *exposure scale* (in ISO terms, its "paper range"). A typical negative's scene occupies
**about 0.94–1.05 log units** of density. `contrast=0.5` halves the negative's density range
before it reaches the curve, so each frame covers only curve inputs of about 1.17 → 1.69. That is
roughly half the paper. On top of that, `estimate_exposure` anchors only the *shadow* end
(1st percentile → a fixed point on the curve), so nothing pulls the highlights toward paper white.

In darkroom terms, every frame is printed on a much softer paper grade than the negative calls for
(about half of what would fill the paper), and the exposure is set from a single shadow reading. The result is a muddy, flat
print: the paper's black and white are never used. The colour is correct; the *printing* is what
falls short.

Nothing here is a clipping or colour-science bug. `contrast=0.5` was a deliberate fix for colour
casts at `contrast=1.0` (see `ToneCurveParams`' docstring). The flatness is a side effect of using
one fixed grade for every negative.

## 2. Your question: does stretching the current output in darktable distort the tones?

**Yes, somewhat, and it is the wrong place to fix the problem.** The details:

- The current TIFF is a *finished print*: the curve has already done its toe/shoulder work.
  Setting a black point and white point on top of that is a second, non-physical tone curve
  stacked on the first:
  - **The white point** is a multiply in linear light. That is harmless: it equals printing
    slightly lighter.
  - **The black point** subtracts a constant in linear light. That is not something a darkroom
    can do. In density terms it steepens the deep shadows much more than the midtones, so it
    reshapes the paper's toe rather than choosing a different paper. It looks like more contrast,
    but it does not match any real paper grade.
- **Colour stays safe only if the levels are linked** (the same black/white point on R, G, and B).
  Linked levels keep neutrals neutral: equal R=G=B stays equal. **Per-channel ("auto") levels
  would overwrite your density balance.** They set each channel's black and white point
  independently, which is exactly the per-channel correction halide's calibration exists to make
  properly. Avoid that entirely.
- The correct fix belongs *before* the curve, using the two controls a printer actually has:
  **exposure** (how far along the curve the negative sits) and **paper grade** (how much of the
  curve the negative's density range covers). Matching these to each negative fills the paper from
  its white to its black **through the curve's own toe and shoulder**. There is no hard clip and no
  post-curve stretch, which is the order-of-operations problem you identified. Section 3 describes
  how.

## 3. Output mode A: **print** (the default, finished-looking)

### The idea: match the paper grade to the negative, as a printer does, but measured

Darkroom sensitometry has a standard answer to "which grade suits this negative": pick the paper
whose **exposure range** matches the negative's **density range**. ISO 6846 (written for B&W paper,
but the definition only needs a characteristic curve, which we have) defines a paper's range as the log-exposure span between:

- the exposure giving density **0.04 above paper white** (the first visible tone, i.e. highlight
  detail), and
- the exposure giving **90% of the paper's D-max** (the last separable shadow tone).

Both points can be **computed directly from the vendored curve file** (no hand tuning). For
`paper_endura.cube`: D-max = 2.46, highlight point at curve input 1.786, shadow point at 0.962,
range R = 0.824.

### The fit (two measurements, two unknowns, solved exactly)

1. Measure this frame's negative density range from its luminance (`log10` of the inverted
   positive's ACEScg Y): `D_lo` = 0.1th percentile, `D_hi` = 99.9th percentile. These are the same
   robust-percentile choices, for the same dust/clamp-artifact reasons, already documented in
   `estimate_linear_scale` and `calibration/auto.py`.
2. `contrast = R / (D_hi − D_lo)`. This is the grade that makes the negative's range fill the
   paper's range.
3. `exposure` is solved so that `D_hi` lands on the paper's highlight point (print for the
   highlights, the normal rule for printing negatives). With (2) in place, `D_lo` then lands on the
   shadow point automatically.
4. Everything outside the 0.1–99.9% range goes into the curve's toe or shoulder: it is
   **compressed, never clipped**, because the curve is asymptotic at both ends.

**Why this cannot introduce a colour cast.** Exposure and contrast are single scalars applied
identically to R, G, and B before the same channel-identical curve. A pixel that is neutral after
density balance stays exactly neutral. The fit only decides *where on the paper* the image
sits, never *what colour* anything is.

### Guard rails

- **Grade cap: `contrast ≤ 1.0`**, the untouched, measured paper. A genuinely low-contrast scene
  (fog, overcast) is then **not** stretched to full black and white. It prints as soft as it
  really was, which is faithful, rather than being normalised into a punchy image. This is the one
  clamp, and it has a physical meaning: "never harder than the real paper we emulate". No lower
  clamp: a very contrasty negative simply gets a softer grade.
- **Pinned values still win**, with existing precedence unchanged (CLI flag > saved profile
  override > automatic):
  - `--contrast X` alone: fit exposure only, using the two-point rule with the given grade
    (highlight-anchored).
  - `--exposure X --contrast Y`, or a saved profile's `tone` override: no fitting at all, exactly
    as today. Existing saved profiles keep working unchanged.
- **The CLI reports what it chose**, e.g. `print: grade 0.88, exposure +0.15`. The printing
  decision is then visible and reproducible, not hidden. `--exposure/--contrast` can pin it for
  the rest of a roll.
- **Roll consistency (batch):** `--grade roll` fits one grade per roll (median of the per-frame
  fits, from the same downsampled pre-pass `estimate_roll_density_profile` already does) and fits
  only exposure per frame. Frames then keep their real contrast differences relative to each other.
  Per-frame fitting remains the default because it matches what a printer does per negative.

### Prototype result on your scans (throwaway script, same data as §1)

| scan | fitted grade | darkest 0.1% | median | brightest 0.1% |
|---|---|---|---|---|
| IMG_0151 | 0.88 | 18 | 41 | 245 |
| IMG_0156 | 0.87 | 18 | 68 | 244 |
| IMG_0158 | 0.79 | 18 | 39 | 245 |

I viewed side-by-side renders, not just these numbers. The fitted versions look like real prints:
real blacks, clean highlights, and no visible clipping. **Two things you must judge yourself
before this becomes the default:**

1. **Midtones get darker on highlight-heavy frames.** On IMG_0151, the sunlit white facade sets the
   highlights, so the crowd in shade drops from 79 to 41. That is correct "print for the
   highlights" behaviour and what a straight darkroom print would do. It may still be darker than
   you would print it. If so, the fix is to discuss the *anchor* (for example 99.5% instead of
   99.9%), deliberately and against several frames, not to tune constants until one image looks
   right.
2. **A higher grade amplifies calibration error.** Going from grade 0.5 to about 0.85 makes
   any residual density error about 1.7× larger in the print. On IMG_0156 with an *auto* profile, a faint cool
   cast appears in the shadows. This is the exact effect that led to `contrast=0.5` in the first
   place. It must be verified with **your manual/picked profiles**, not auto ones. If a cast
   survives with a good manual profile, the honest options are a lower grade cap or better
   calibration (for example the ColorChecker tier), not hiding the cast with a soft grade.

### Paper black is not zero, deliberately

The print's black is the paper's D-max (linear 0.0035, sRGB ≈ 11), like a real print, not display
black 0. Mapping paper black to display black is **black-point compensation** (standardised in
ISO 18619, used by ICC print-to-screen proofing). If you want it, it belongs in `halide export` as
an opt-in flag, since it concerns the display medium, not the print. I would defer it unless you
ask.

## 4. Output mode B: **flat** (minimal-bias linear positive for your own editing)

This mode already exists as `--linear-output` (`mode="linear"`), and its maths is already right:
white balance → density balance → invert → **one global multiply** (`estimate_linear_scale`).
The blog is explicit that a flat positive stays faithful only under *exposure and white balance*;
any other operation "would disrupt the linearity of the negative". A single scale is exposure,
so there is nothing to remove.

What it is (worth documenting in the command help):

- It is **the film's own recorded contrast**. Colour negative film has a gamma of about 0.6, so
  the positive is proportional to scene exposure^0.6. That is why it looks very flat: the blog says
  as much ("it records a flat and very low contrast image"). Undoing that gamma would require
  knowing it, which only a ColorChecker with known patch reflectances could measure. That is a
  natural extension for when the ColorChecker tier is built, not something to guess now.
- **Its black is the film base (D-min), not zero.** The darkest value (≈ sRGB 77–84 in §1) is the
  least-exposed density the film can record: real data, not headroom to trim. Subtracting it would
  be a black-point choice (flare/fog correction), which is an editorial decision and yours to make.

Proposed changes (small):

1. **CLI naming:** add `--output print|flat`, with `--linear-output` kept as an alias for
   `--output flat`. The GUI popup's "Linear output" checkbox gets the same label.
2. **Provenance metadata:** write the applied scale factor, the profile's numbers, and the halide
   version into the TIFF (`ImageDescription` as JSON). This is for reproducibility and for exact
   round trips when darktable is *not* in between (see §5). darktable will likely drop it, so
   nothing depends on it.
3. **Optional `--linear-scale roll`** in batch: one scale for the whole roll instead of per frame, so
   bracketed or differently exposed frames keep their true relative brightness. The per-frame
   default is unchanged.

## 5. The stand-alone print step: `halide print` (tone curve only)

### Why "stretch it myself, then apply the curve" can't work, and what does

Your instinct is correct. If you set black and white points by hand in darktable and then apply the
paper curve, the curve's toe and shoulder operate on a range that has already been rearranged. Two
of those operations (the black-point subtraction especially) are not exposure/grade changes, so
the curve then places shadows on the wrong part of the paper. **Dynamic range is the print step's
job, not the editor's.** The workflow therefore splits cleanly in two:

| step | where | what may happen | why |
|---|---|---|---|
| 1. develop the negative | `halide invert --output flat` | WB, density balance, invert, one scale | the science |
| 2. edit the scene | darktable, **linear, scene-referred** | crop, rotate/perspective, lens, spot removal/retouch, denoise, **global exposure**; white-balance tweaks allowed but better done in halide's calibration | these either don't change tone at all or are a global multiply, which the print fit absorbs |
| 3. print | `halide print flat_edited.tif out.tif` | fit exposure + grade, paper curve | the same print stage as mode A |
| 4. deliver | `halide export` | ACEScg → sRGB PNG/JPEG | unchanged |

**Not allowed in step 2** (all of them re-shape tone or reset range before the paper sees it):
filmic / sigmoid / AgX, tone curve, base curve, rgb levels / black point, local contrast,
tone equalizer, and "color balance rgb" contrast. darktable's scene-referred workflow may apply
filmic/sigmoid and an exposure boost by default; for a TIFF input these must be switched off
(verify this during implementation, since it depends on your darktable version and preferences).

**darktable export settings for step 2 → 3:** TIFF, **32-bit float**, output profile
**linear** (either "linear Rec2020 RGB", or the same `ACEScg-elle-V4-g10.icc` halide uses, which
can be installed in `~/.config/darktable/color/out/`), with the profile embedded. `halide print`
validates this with the same `io/icc.py` checks as `invert` and rejects anything gamma-encoded,
with the existing specific errors.

**Why a darktable exposure change is harmless to the print:** exposure is a multiply in linear
light, which is a constant offset in density, and the fitted `exposure` absorbs it exactly. With
automatic fitting, `invert --output flat` then `print` gives **the same result as `invert
--output print`** regardless of what global exposure darktable applied. This will be a test:
the round trip must match to float rounding noise.

**The one exception: pinned absolute exposure.** A saved profile's pinned `exposure` means
"this many density units at this scan's scale", so it is not scale-invariant. If `halide print`
is given a pinned exposure, it either (a) reads the scale factor from the provenance metadata (when
darktable preserved it), or (b) warns that the pinned value can't be reproduced exactly after
external editing and falls back to fitting exposure. Pinned *contrast* (grade) has no such
problem.

### Other things `halide print` accepts

The same tone flags as `invert`: `--exposure`, `--contrast`, `--profile NAME` (for its saved tone
override only; density numbers are ignored because they are already applied). Directory input
works like `export` (bulk, with the existing worker-pool/memory logic reused).

## 6. Implementation steps (in order, each separately testable)

1. **`core/tone_render.py`** (pure; the only file with new maths):
   - `paper_exposure_range(curve) -> (x_lo, x_hi)`: the ISO 6846 points computed from the curve.
   - `fit_print(positive, curve, contrast=None) -> (exposure, contrast)`: the §3 fit, with the
     ≤ 1.0 cap.
   - `tone_render` calls the fit when `params.exposure is None` (and `params.contrast is None`).
   - `estimate_exposure` (shadow-anchored) is kept, but only for the explicit legacy case of a
     pinned contrast with no pinned exposure, if you prefer the old behaviour there. Otherwise it
     is removed together with the GUI's use of it (decision D3 below).
2. **`core/types.py`**: `ToneCurveParams.contrast: float | None = None` (None = fit). Docstring
   updated with the history of why 0.5 existed and why a fitted grade replaces it. Existing
   saved profiles store explicit numbers, so they are unaffected.
3. **`cli/_calibration_args.py`**: `--output print|flat` (+ `--linear-output` alias), `--grade roll`
   (batch only), and reporting of the chosen exposure/grade.
4. **`processing.py`**: `print_scan(input, output, tone_params)`, which reuses
   `load_working_space_image` → `tone_render` → `write_tiff` with the ACEScg tag. Also the
   provenance metadata writer for flat outputs.
5. **`cli/commands/print_cmd.py`** + registration in `cli/main.py` (single-file and directory).
6. **`batch`**: pass the roll-grade pre-pass through the existing orchestrator; no change to the
   worker-memory logic (the fit is two percentiles on a luminance array, far below the existing
   per-worker estimate).
7. **GUI (`gui/preview_popup.py`)**: auto mode shows the fitted exposure/grade on the sliders.
   "Fine-tune" still saves pinned values as today.
8. **`CLAUDE.md`**: update the "Decisions and why" entries for `contrast` / `exposure` and the
   linear mode, and add the §5 workflow rules.

### Tests

- Unit: `paper_exposure_range` against hand-computed points from the cube; `fit_print` recovers a
  known exposure/grade on a synthetic image with a known density range; the grade cap engages on a
  low-contrast synthetic image; **neutral in → exactly neutral out** for fitted params; fit is
  invariant to a global multiply of the input (the property the darktable round trip relies on).
- Integration: `invert --output flat` → `print` equals `invert --output print` (float noise
  only); `print` rejects a gamma-encoded TIFF with the existing ICC error; pinned-exposure warning
  path.
- Existing golden LUT regression tests stay as they are (they test the curve lookup, which does
  not change). Tests that assert today's default `contrast=0.5` output are updated deliberately,
  with the reason in the commit.

### Verification against real data (before calling anything done)

- Re-render all real scans with **your own saved/picked profiles**, current vs. new, side by side,
  plus the percentile table. You judge the midtone and cast questions (§3).
- A real darktable round trip: flat export → darktable (exposure change + crop only) → 32-bit
  float linear TIFF → `halide print`, compared against the direct print.
- An Xvfb check of the GUI popup changes (per the CLAUDE.md procedure).

## 7. Decisions I need from you

- **D1. Default mode.** I recommend `print` (fitted) as the default for `invert`/`batch`, with
  `flat` one flag away. Alternatively, keep the current look as default and make fitting opt-in.
- **D2. Highlight anchor.** 99.9th percentile (recommended to start, matching the rest of the
  project) vs. something lower, *decided after you have looked at a set of real frames*.
- **D3. Old shadow-anchored `estimate_exposure`.** Remove it (recommended: the two-point fit
  replaces it and its docstring already admits its single target constant was a scene-dependent
  guess) or keep it as a fallback.
- **D4. Batch grade default.** Per-frame (recommended: what a printer does) or `--grade roll` by
  default.
- **D5. Black-point compensation in `export`.** Defer (recommended) or include now.

## What this plan deliberately does not do

- It does not touch white balance, density balance, inversion, auto-calibration, or ICC handling.
- It adds no invented curve, "look", saturation, or per-channel adjustment in either mode.
- It does not undo film gamma in flat mode (that needs a ColorChecker to measure honestly).
- It does not attempt to make the print step accept arbitrary darktable-graded images: the
  contract is "linear, scene-referred, tone untouched", enforced by validation where possible and
  documented where it can't be (darktable module choices).
