# Can `halide check` suggest the best frame to calibrate a roll from?

**Status (2026-09-24).** A finished investigation, kept for its evidence and for one idea not yet
built.
- **What it led to:** the multi-point, multi-frame calibration picker (`docs/plans/multipoint-picker.md`,
  CLAUDE.md "Manual calibration fits any number of user-picked neutral points"). Its object-error
  finding is why points are fitted together with per-point agreement, and its lever-arm finding is
  the step wedge's coverage bar.
- **Still open:** `halide check --suggest-anchor` (see "Recommendation, if this is built" below),
  and a grey card or ColorChecker frame per roll (the unbuilt ColorChecker tier).
- **Deleted:** the prototype scripts and the `Roll16-anchor-comparison/` sheets, profiles and A/B
  strips mentioned below were throwaway and have been removed. Their numbers are recorded here.

Investigation only: no code in `src/` was changed. The prototype scripts read a 1/8-scale, linear,
ACEScg, scan-gain-matched cache of each frame. All numbers below come from the real 37-frame Roll 16
(`Roll16-Testing/`). There's no ground truth: no saved profile or recorded manual picks exist in
the dev sandbox.

## Short answer

**Partly.** Pixels can reliably measure one thing: whether a frame has a large, uniform,
near-grey object dense enough to cover the roll's own highlights. That's worth reporting. Pixels
**can't** tell whether that object is actually grey, or whether it was lit by daylight. That's
the dominant source of calibration error, and it's the same wall `calibration/auto.py` hit (see
CLAUDE.md). So the useful feature is a short list of candidate frames, each with its candidate
objects marked, for the photographer to confirm. It shouldn't be a single "best frame" verdict.

## What makes an anchor frame good (the physics)

- **Shadow pick:** barely matters which frame it comes from. Every frame has dark near-neutral
  content right at the toe, and toe convergence pulls it onto the film-base axis anyway. In the
  prototype, the lowest qualifying patch in *every* frame sat at the toe cutoff (D_G ≈ 0.77).
- **Highlight pick:** sets the density-balance slope, so it carries two kinds of error.
  1. **Object error.** Is the clicked object truly grey, and under daylight? This dominates.
  2. **Extrapolation.** `solve_density_balance` fits a straight line through the two picks. Any
     highlight on another frame that's denser than the highlight pick is extrapolated, and the
     pick's error grows by `(D - D_shadow) / (D_highlight - D_shadow)`.

## What was tried, in order

1. **A roll-wide "neutral axis" plus a tight grey tolerance (±0.03 D).** At each density, the
   roll's pixels form a long, thin warm-to-cool streak (the daylight locus: sunlit stone through
   to sky). Across the streak the axis is well pinned; along it the axis is ambiguous. Plausible
   axes differ by ~0.06 D in B−G at D_G 1.35 (~CC15). Examples: the current `--auto-density-roll`
   profile, which is sky-biased (blue sky dominates D_G 1.1–1.4), versus a line from the film
   base through the second, smoother ridge. At ±0.03 D, the frame ranking flipped completely
   between axes (Spearman 0.17), and the masks were scattered specks, not objects.
2. **A looser tolerance (±0.06 D).** Rankings agreed across axes (0.7–0.99), but the mask
   included skin, cream and pink facades, hair and foliage. That measures "not very saturated",
   not "grey".
3. **Highlight-focused, axis-robust band.** Within ±0.03 D of *either* plausible axis: tight in
   the green–magenta direction, as wide as the genuine warm–cool uncertainty. Plus a uniformity
   test at click scale (3×3 neighbouring 8-px blocks agree to 0.015 D; a within-block test was
   dominated by grain, median 0.023 D), and patches of at least 64 blocks. The chosen highlight
   objects were mostly real: clouds (0144, 0147, 0159), overcast sky (0157), a white T-shirt
   (0175), the white trike body (0158).
4. **Validation: does the score predict calibration quality?** Each frame's calibration was
   simulated from its own shadow and highlight patches with the real `solve_density_balance`,
   then compared with the consensus of the top half:

   | evaluated at D_G      | 1.2   | 1.4   | 1.6   | 1.7   |
   |-----------------------|-------|-------|-------|-------|
   | Spearman(lever, dev.) | -0.07 | -0.15 | -0.27 | -0.22 |
   | median dev., top half | 0.023 | 0.034 | 0.040 | 0.041 |
   | median dev., bottom   | 0.034 | 0.037 | 0.055 | 0.057 |

   Most frames' calibrations land 0.01–0.06 D apart, whatever their score: that's object error.
   The lever arm only shows up where short-lever frames must extrapolate. At the roll's
   brightest highlights (99.5th pct D_G = 1.55), their error is ~40% larger.
5. **Can non-daylight frames be flagged from pixels?** Only the extremes. A frame's median offset
   along the warm–cool direction put the tungsten frames 0141 (−0.22) and 0150 (−0.20) at the
   warm end. But daylight 0175 (−0.15, green hedges) came next, and the dusk frame 0153 (−0.09)
   sat among the daylight street scenes. Scene content and illuminant can't be separated.

## Resulting ranking on Roll 16

**Coverage** is how far the densest large, uniform, near-grey patch reaches between the toe
(D_G 0.77) and the roll's highlight end (99.5th pct, D_G 1.55). **Choice** is the near-grey area
within 0.1 D of that patch. Frames are ranked by coverage, then choice.

- **Top:** 0159 (clouds, beside the white Alhambra wall), 0153, 0144, 0147, 0141, 0157, 0143,
  0175.
  - 0153 (dusk, a backlit shop window) and 0141 (a restaurant plate under tungsten) are **wrong
    answers** the metric can't detect.
- **Bottom:** the wide landscapes 0172, 0168, 0161, full of white clouds but exposed so the
  clouds sit at only ~1.25 D. Also 0158, 0154, and 0150 (tungsten; nothing near-grey above the
  toe).
- Coverage tracks the frame's own 99.5th-percentile density closely (Spearman 0.87). The grey
  test's value is in the exceptions: 0155, 0142 and 0154 have dense highlights that *aren't*
  near-grey.

## Follow-up: the user's real profile (Roll16-Profile1, anchored on IMG_0158)

The user anchored Roll16-Profile1 on IMG_0158 and felt some frames printed "a little too orange".
The profile file isn't in the sandbox, so it was **reconstructed from the contact sheet it
rendered** (`recon.py`). The print stage applies one paper curve, grade and exposure to all three
channels, so within a frame the rendered green is a monotone function of D_G alone. That map is
learned per frame and applied to rendered R and B. Regressing the result on the negative's own
D_R / D_B gives density_scale (slope) and white balance (intercept).

- **Consistent across frames:** 36/37 frames agree on R scale 1.135 (IQR 1.128–1.144) and
  B scale 0.788 (IQR 0.780–0.793), with residuals 0.002–0.006 D. (0150 is too washed out.)
- **Checked end to end:** re-rendering the roll through real `halide batch --contact-sheet` with
  the reconstructed values (`--rm 0.7376 --bm 1.0679 --rs 1.1348 --bs 0.7884`, scan-matched to
  IMG_0158's 1/50) matches the original sheet to ~2/255 mean absolute per frame. The mean signed
  bias is +0.4 R / +0.6 B, which is JPEG-vs-PNG level.
- **Confirmed against the real profile** (supplied later by the user): WB
  (0.7569, 1, 1.0996), density scale (1.1454, 1, 0.7937). The reconstruction's scales are within
  ~1%. Its WB multipliers differ by 2.5–3%, but WB and scale trade off: the two profiles put
  neutral within ±0.005 D of each other (~CC1) across D_G 0.8–1.8. Rendering the roll with the
  exact values reproduces the original sheet with a mean signed bias of −0.04 R / +0.01 G /
  −0.06 B (per 255). The reconstructed render differs from it by only +0.5 R / +0.65 B. This also
  confirms the sheet was scan-matched to IMG_0158's 1/50.
- **Where Profile1 is neutral:** at the film base and at IMG_0158's white trike (D_G ~1.4, within
  0.02 D). Denser verified whites on other frames (clouds, white T-shirt, overcast sky,
  D_G 1.53–1.65) render slightly *cool*: +0.02–0.04 D blue, except one cloud in 0147. So the
  anchor's shorter reach does drift in the highlights, but not toward orange.
- **Raw white balance ruled out:** every current export carries one fixed raw WB (1.875 / 1 /
  1.810), and the sheet was rendered from these files.
- **A/B against a cloud-anchored profile:** same neutral film base, highlight = IMG_0159's cloud
  (D_G 1.57): `--rm 0.7293 --bm 1.0484 --rs 1.1203 --bs 0.7783`. Relative to that, the real
  Profile1 prints on average 2.9/255 redder, 1.8 less green and 1.1 bluer. (The reconstructed
  profile gave 3.4 / 1.7 / 1.7.) That's a faint red-magenta
  warmth, about CC3–5, visible mainly as salmon on light stone and pale walls (0138, 0152, 0169);
  skin, sky and greens barely change. Neither profile can be called *correct* without a known
  neutral. The difference is small, but it's in the direction the user described. Sheets and A/B
  strips are in `Roll16-anchor-comparison/` (repo root, untracked).
- **Warm-lit frames are orange either way:** 0141/0150 (tungsten), 0153 (dusk) and 0174
  (sandstone lit by warm bounced light) look orange under both profiles. That's what daylight
  film recorded, not a calibration error.

### Candidate anchors for re-calibrating Roll 16

Each candidate keeps Profile1's shadow at the film base and takes its highlight from one white
object, solved with the real `solve_density_balance` (`candidates.py`, profile JSONs in
`Roll16-anchor-comparison/`). Each JSON carries IMG_0158's 1/50 scan sidecar.

| anchor (highlight)       | D_G  | R scale | B scale | look across the roll |
|--------------------------|------|---------|---------|----------------------|
| Profile1: trike, 0158    | ~1.4 | 1.145   | 0.794   | warmest; light stone reads peach/salmon |
| T-shirt, 0175            | 1.48 | 1.106   | 0.765   | stone cream; skin, greens, sky barely change |
| cloud, 0159              | 1.57 | 1.120   | 0.778   | ≈ T-shirt |
| white wall, 0159         | 1.16 | 1.029   | 0.829   | over-corrected: stone grey-blue, sky cyan |

- **The trike and the wall disagree** by ~CC15 at the same density (under Profile1 the wall
  renders R +0.06, B −0.03 to −0.06). The wall-anchored print turns everything blue, so the
  Alhambra plaster is evidently warm cream. That's a concrete case of the dominant error: an
  object that "looks white" isn't necessarily a neutral.
- **Two independent whites agree:** the T-shirt and the cloud, on different frames, land within
  ~0.015 in scale of each other, both slightly cooler than Profile1. The choice between them and
  Profile1 is the user's to make by eye (`compare-A.jpg`, `compare-B.jpg`).

## Recommendation, if this is built

- Add an **opt-in** `halide check --suggest-anchor`. `check` is headers-only and instant today.
  This option needs pixels: ~25 s for 37 frames at 1/8 scale, reusing
  `processing.load_working_space_image` and the `--match-scan-exposure` gain.
- Report the top ~5 frames by coverage, then choice, each with where the candidate highlight
  objects are. Write a proof sheet (the contact-sheet code) with those objects outlined.
- Say plainly what the list means: "these frames reach the roll's highlights with something
  near-grey". The user confirms which object they *know* is white or grey, and that it was lit
  by daylight.
- Don't auto-pick one frame. Don't claim neutrality. Don't try to detect the illuminant.
- The only way to make the object error go away is a known neutral: a grey card or ColorChecker
  frame shot on the roll (the unbuilt ColorChecker tier). Ideally it's exposed so the card's
  white patch sits near the roll's highlight density.

Open parameters, fit to one roll only: toe margin 0.15 D, grey band ±0.03 D, uniformity
0.015 D, minimum patch 64 blocks (~64×64 full-res px), coverage target = roll 99.5th percentile.
