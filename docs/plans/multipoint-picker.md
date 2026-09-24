# Multi-point, multi-frame calibration picker (+ real-contact-sheet proofs)

**Status: implemented** (2026-09-24, merged into claude-rewrite). Later changes the user asked for
while using it: "Proof roll" became "Build contact sheet" (with a Save button), "Roll details" became
"Extra information", and Clear frame / Clear all were added under the point list. The agreement hint fires
from CC 5 (amber), not CC 10, on the Roll 16 evidence. `invert`/`batch` also offer saved profiles
when no calibration is given.

## Context

The anchor-frame investigation (`ANCHOR_FRAME_INVESTIGATION.md`, branch
`worktree-anchor-frame-investigation`) found the biggest calibration error isn't frame choice. It's
whether the one highlight object you click is truly neutral. On Roll 16:

- Roll16-Profile1 (trike, IMG_0158) printed ~CC3 warm.
- A T-shirt (0175) and a cloud (0159) agreed with each other.
- The "white" Alhambra wall was really cream (~CC15 off).

Two-point calibration can't show any of that. The fix is to calibrate from **many user-confirmed
neutral points, across frames**, fitted by least squares. Each point shows how far it sits from
the fit, so a bad object is visible the moment it's clicked.

This is a permanent change to `halide calibrate` and a restructure of the Qt GUI. It must keep
the original GUI's integrity (`ENLARGER_SKIN_SPEC.md`; original plan `okay-i-really-like-pure-castle.md`):

- fixed, non-scrolling, purpose-built window;
- a few big labelled controls;
- amber secondaries, **one red primary per window**;
- the inversion wording stays explicit;
- bounded collapsibles.

Every UI decision below was chosen by the user; none is assumed. The output format is unchanged
(`DensityProfile`, one WB + one density scale per channel), so invert/batch/print are unaffected.

## Decisions (user-chosen)

- **No shadow/highlight roles:** every click adds a "neutral point".
- **Multiple frames from day one:**
  - "Load roll…" loads a folder;
  - `halide calibrate <dir>` works too (single files still work);
  - frames appear in a **filmstrip** that develops in the background.
- **Landscape window:**
  - filmstrip across the top;
  - image on the left;
  - a control panel on the right, like an enlarger control face.
- **Negative | Positive switch** above the image (no Preview popup any more):
  - Positive uses the **auto-density estimate** until 2+ points exist, then **the live fit**, updating as points change;
  - a standing note says which, and that the estimate is rough and not the final output;
  - picks always sample the raw negative;
  - **filmstrip thumbnails follow the switch**.
- **Point list:**
  - compact rows showing number, frame, density, agreement (CC) and ✕;
  - **fixed height, scrolls** when long;
  - no point names;
  - clicking a row jumps to its frame and flashes the marker.
- **Variety:**
  - a **step-wedge coverage bar**, film base → roll highlights, with a tick per point;
  - gentle nudges ("close to ③ in tone and colour — a different object adds more");
  - nothing is refused.
- **Agreement:** markers and rows are colour-coded by CC distance from the fit, with an "is this
  really neutral?" hint above ~CC10. Shown only from 3 points (with 2 the line passes through both).
  - **How it's written:** "CC 15 Y" means a colour-printing filter-pack value (Kodak CC: density ×
    100, so CC10 = 0.10) plus the direction the point is off (the nearest of R/G/B/C/M/Y, from the
    point's R−G / B−G residual). The direction tells a warm cream wall from a cool shaded white.
  - **Where it's explained:** in the point list header's tooltip and in Details. The explanation
    says the value is measured on the density-balanced negative, before the paper curve.
- **Marker click:** clicking on a marker selects it; Delete or ✕ removes it.
- **Film base:** never added automatically.
- **Fine-tune:** a collapsible **Print** drawer (exposure/grade sliders starting from the per-frame
  fit, flat-output preview check, "Back to fitted"). Active in Positive with a fit. Saved to the
  tone record as today.
- **Roll details:** a collapsible drawer (film stock, process, scanner, notes) that feeds the proof
  sheet live, pre-fills Save, and is restored when a profile is reopened. The panel's drawers
  (Roll details / Print / Details) are an **accordion**: opening one closes the others, so the
  window never overflows.
- **Proof roll…:** opens a non-modal, **zoom/pan contact-sheet window** of the whole roll with the
  current calibration:
  - an instant **draft** from the filmstrip-resolution frames, marked "draft";
  - then the **full-resolution render** (identical to `batch --contact-sheet`) runs in the
    background with a progress bar, replacing frames as they finish.
- **Contact sheets everywhere get a real-contact-sheet look** (one renderer for the proof window,
  `halide contact` and `batch --contact-sheet`), after the user's reference (`contactsheet.png`):
  - black rebate and gaps, frames butted in neat aligned strips of six;
  - orange edge print: top edge carries the frame number + film stock ("INVERTED BY HALIDE" if
    unset); bottom edge carries the frame number, "A" half-frame numbers and barcode-style dashes;
  - each frame's printing decision stays as a small dim line, so sheets remain self-describing;
  - sequential frame numbers 1…N (the file name shows on hover in the proof window);
  - sheets stay marked so `halide contact` skips them.
- **Reopen a profile's points:** profiles record their picks. Opening one restores the points (and
  the roll, if its frames are still there). Points keep their stored RGB, so they still count in
  the fit even when a frame has moved; they're listed as "frame missing".
- **Port first:** bring the unmerged `notes` field + `halide profile edit` + `--notes` (commit
  a3a94bb) across. Its old dearpygui Notes box is superseded by the Roll details drawer.
- **Stays deferred:** the enlarger skin, including the LED readout. `ENLARGER_SKIN_SPEC.md` gains
  the frame counter / point count as candidate homes for it.
- **Out of scope:** `batch --pick`.

## Window layout

```
┌─ halide · calibration picker ───────────────────────────────────────────┐
│ Roll16-Testing · 37 frames              [Open profile…] [Load roll…]    │
│ ▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫ │
│ ▯ ▯ ▯[▮]▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ▯ ›   (counts under frames)    │
│ ▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫▫ │
│ (Negative|Positive)  Positive · your fit (5 pts) │ COVERAGE             │
│ ┌──────────────────────────────────────────────┐ │ ▕░▒▓██████████▏      │
│ │                                              │ │   ⑤  ④    ①②③        │
│ │            IMG_0159 (numbered markers)       │ │ NEUTRAL POINTS   5   │
│ │                                              │ │ ① 0158 D1.41 ●CC2 ✕ ▲│
│ └──────────────────────────────────────────────┘ │ ② 0175 D1.48 ●CC1 ✕ ▼│
│ caption: inversion note / what to click          │ ⓘ nudge line         │
│                                                  │ [ Proof roll… ]      │
│                                                  │ ▸ Roll details       │
│                                                  │ ▸ Print              │
│                                                  │ ▸ Details            │
│ status…                                          │ [● Save profile  ]   │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Window size:** fixed and non-resizable, computed from `availableGeometry()` as roughly 55% of
  screen width × 60% of height (clamped). The image column takes what's left after the fixed-width
  panel; the image box still shrink-wraps the frame with uniform padding (existing
  `_compute_display_budget` logic).
- **`--pick` one-shot (invert):** one frame, so no filmstrip, no Proof roll and no Open/Load. The
  panel is otherwise identical, with the red button reading "Develop".

## Implementation

Work happens in a new worktree off `claude-rewrite`. I pause after each step with test results
and, for GUI steps, Xvfb screenshots.

1. **Port notes + profile edit.**
   - Cherry-pick a3a94bb's non-GUI parts: `core/types.py` notes, `calibration/profile_store.py`,
     `cli/commands/profile_cmd.py` edit, `--notes` in `cli/_calibration_args.py`.
   - Resolve conflicts, keep its tests passing.
2. **Math (pure, tested).**
   - `core/density.py::fit_density_balance(neutral_rgbs)`: per-channel least-squares line
     D_c = a_c + b_c·D_G in log density; scale = 1/b_c, WB = 10^a_c. With exactly 2 points it must
     equal `solve_density_balance` (existing, same file) to float precision.
   - `neutral_residuals(profile, rgbs)`: each point's balanced-density deviation per channel, in CC
     (×100). A small pure helper turns it into a magnitude plus a direction letter (R/G/B/C/M/Y).
   - Reuse `MIN_DENSITY_SEPARATION` from `calibration/auto.py` for the Save gate: ≥2 points
     spanning ≥0.1 D.
3. **Picking model (pure, no Qt): new `calibration/anchors.py`.**
   - `NeutralPoint(frame_path, x, y, rgb, scan)`.
   - Scan-gain normalisation: each point is scaled to the roll's reference scan exposure via
     `calibration/scan_consistency.py::scan_gain` / `most_common_settings` and
     `io/scan_metadata.py::read_scan_metadata`. The reference is recorded as the profile's `scan`
     sidecar, so `--match-scan-exposure` works unchanged.
   - Wedge range: 0.5th / 99.5th pct D_G over the filmstrip frames (99.5 matches `fit_print`'s
     anchor).
   - Coverage, the near-duplicate test (ΔD_G < ~0.05 and similar chroma), and agreement bands
     (≤CC5 calm, 5–10 amber, >10 red).
   - `profile_store`: optional `anchors` + `roll` sidecars (like `tone`/`scan`), plus a loader.
4. **GUI structure (replaces `preview_popup.py`):**
   - `gui/main_window.py`: landscape layout, state and wiring.
   - New widgets, each small and single-purpose: `gui/filmstrip.py` (background thumbnail loading
     via a small process pool, following the switch), `gui/step_wedge.py` (custom-painted),
     `gui/point_list.py` (fixed-height scroll list), `gui/drawers.py` (accordion: Roll details,
     Print, Details).
   - Rendering helpers move from `preview_popup.py` into `gui/render.py`: estimate =
     `calibration/auto.py::auto_density_balance`, fit = `core/pipeline.py::run_pipeline`,
     `io/raster.py::to_srgb_8bit`.
   - `gui/sampling.py` is reused unchanged (snap, magnifier, coordinate mapping, stretch).
   - Magnifier kept. Markers become numbered, agreement-coloured rings; `theme.py` swaps the
     shadow/highlight colours for agreement colours.
5. **Entry points.**
   - `cli/commands/calibrate_cmd.py` + `gui/app.py`: accept a folder, files, or `--profile NAME`
     to reopen.
   - `gui/quick_pick.py`: single-frame variant; still returns `(DensityProfile, tone | None)`.
   - Save dialog pre-filled from Roll details; saves notes/stock/etc. plus the anchors, roll,
     scan and tone sidecars.
6. **Contact-sheet restyle (`io/contact_sheet.py`, one renderer).**
   - The new look above.
   - `Tile` gains a frame number; `render_sheet` takes `film_stock`.
   - `processing.py::provenance_json` records `film_stock` so `halide contact` can print it;
     `batch` passes the profile's stock.
   - Update `tests/unit/test_contact_sheet.py` (marker skip, captions).
   - After the new look has been compared against the reference, delete `contactsheet.png` from
     the repo root, as the user asked.
7. **Proof window (`gui/proof_window.py`).**
   - Non-modal; a `QGraphicsView` with wheel-zoom, drag-pan and double-click to fit.
   - The draft renders immediately from filmstrip frames. A `QThread` runs `batch/orchestrator.py::
     run_batch` on jobs with `output_path=None` + a temp `thumbnail_path` (the existing
     contact-sheet preview path); `on_result` drives the progress bar and swaps each frame in
     place. The temp dir is always deleted; closing the window cancels the run.
8. **Docs.** Update `CLAUDE.md` (GUI decisions, multi-point calibration, contact-sheet look, the
   ported notes) and `ENLARGER_SKIN_SPEC.md` (the new widgets to skin, candidate LED homes).

## Verification

- `pytest tests/ -q`, plus new unit tests:
  - the 2-point fit is identical to `solve_density_balance`;
  - a known axis is recovered from noisy N points;
  - residuals flag an injected off-axis point;
  - scan-gain normalisation (mixed 1/25–1/60 frames give the same fit as matched frames);
  - anchors/roll/notes sidecar round-trip and old profiles still load;
  - contact-sheet marker/caption behaviour;
  - the ported `profile edit` tests.
- **Real data (Roll 16):** in the picker, rebuild a profile from trike + T-shirt + cloud + a dark
  neutral.
  - The scales should land near candidates B/C, with all points within ~CC5.
  - Adding the 0159 wall should show it red at ~CC15.
  - Save it, reopen it, and confirm the points and roll details come back.
  - Run `batch --contact-sheet` with it and compare to the earlier sheets.
- **GUI, live under Xvfb + xdotool + screenshots** (CLAUDE.md workflow), every state:
  - empty;
  - roll loading;
  - Negative and Positive (estimate, then fit);
  - many points (the list scrolls, the window doesn't grow);
  - select/delete;
  - row → frame jump;
  - nudge shown;
  - each drawer and the accordion behaviour;
  - the proof window (draft → progress → full, zoom/pan, close mid-run cleans up);
  - Save and reopen;
  - `invert --pick` end to end.
  Screen sizes 1366×768 and 1920×1080 to check the fixed landscape window fits.
- **Contact-sheet look:** render Roll 16 via `halide contact` and `batch --contact-sheet`, and
  compare visually against the user's reference before deleting it.
