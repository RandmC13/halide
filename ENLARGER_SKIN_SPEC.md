# halide GUI: session recap and the enlarger-skin spec

This document has two jobs: record why the GUI was rebuilt this session (for anyone — human or a
future Claude session — reading the history cold), and specify the one piece of the original ask
that was deliberately deferred: giving the GUI's controls the look of a real darkroom enlarger
timer. Read this before starting that follow-up work rather than re-deriving it from the chat log.

## Why this session happened

The GUI had already been given a first pass (dearpygui-based: theme colors, sectioning, wording)
in an earlier session, but it still felt "thrown together": dearpygui's default auto-stacked widget
flow made a reasonably-sized window need scrolling to see everything, and a two-tab
(Calibrate/Preview) structure read as confusing rather than purposeful. The user wanted a real
rebuild — a small, fixed, purpose-built window with a handful of big, clearly-labeled controls, a
mouse-anchored zoomed magnifier, and the live-preview moved into its own popup — and was explicit
that changing the GUI dependency entirely was on the table. After a tradeoff discussion (staying on
dearpygui vs. PySide6/Qt vs. a pywebview/HTML approach), **PySide6/Qt was chosen**, for real floating
windows, QSS theming, and — relevantly for this document — a much higher ceiling for exactly the
kind of custom-painted, skeuomorphic control skin specified below.

## What this session delivered

A full rewrite of `src/halide/gui/`, not an incremental port:

- `theme.py` — a QSS stylesheet: charcoal background, amber for secondary controls, **red**
  reserved for exactly one primary action button per window. This red reservation is the *only*
  piece of enlarger-inspired theming done so far — see below for the rest.
- `main_window.py` — top bar (filename + "Load negative..."), an image box that shrink-wraps
  whatever image is loaded with a uniform margin, a mouse-anchored floating magnifier, a "Neutral
  Points" panel with Shadow/Highlight toggle buttons colored to match the markers they draw, a
  collapsible fixed-height scrollable Details section, and one big red primary button ("Save
  calibration profile" / "Develop" in the `--pick` one-shot flow via `is_pick_session=True`).
- `preview_popup.py` — a non-modal live-render popup, live-synced to further picking on the main
  window, with an opt-in "Fine-tune" reveal for exposure/contrast/linear controls.
- `quick_pick.py` — rewritten for the `--pick` flow on top of the same `MainWindow`.
- Non-GUI addition (flagged since it reaches outside `gui/`): `calibration/profile_store.py` gained
  an optional exposure/contrast "tone" sidecar on saved profiles; `cli/_calibration_args.py`
  resolves it with precedence CLI flag > saved tone > built-in default. Linear-output mode is
  deliberately excluded from this persistence.
- `gui/sampling.py` needed **zero changes** — it was already framework-independent and is reused
  as-is; its own tests still pass unmodified.

Along the way, several real bugs were found through live Xvfb testing and the user's own runs on
their actual machine (not just code review) — an invalid QSS pseudo-state that silently broke the
Shadow/Highlight checked-state styling, a Wayland popup-parent crash-adjacent bug in the magnifier,
two rounds of image-box sizing/centering bugs (the box being sized from a theoretical minimum
window size instead of the real per-machine size, then a fixed-aspect-ratio box producing
non-uniform padding for any image whose shape didn't match it), and a Details-panel layout-overlap
bug now fixed via a bounded scroll area. All 179 tests pass. Full detail is in the commit that
introduced this rewrite — `git log --oneline -- src/halide/gui/` from this point.

## What's deliberately not done: the enlarger-controller skin

The user asked for the GUI's controls to eventually look like a real darkroom enlarger's control
panel, and explicitly asked that this be deferred to its own follow-up plan rather than attempted
in the same session as the structural rewrite above. A reference photo is at `enlarger-controller.png`
(repo root, gitignored — ask the user if it's missing) — a real Durst enlarger timer
(maker's marks "durst" and "Labotim" embossed into the body). Described precisely, since whoever
picks this up won't necessarily have the image in front of them:

- **Body**: warm grey/putty-colored plastic, a rounded rectangular main housing with a distinct
  circular lobe bulging out on one side.
- **Digital readout**: a red 7-segment LED display, recessed into a chrome/metallic bezel.
- **Secondary controls**: small, flat, black circular buttons (three visible) — two sit as a
  vertical pair (a "+"/"-" increment pair, by the look of it) beside the display, and a third sits
  lower, distinguished by a chrome/metallic ring collar around it, suggesting a different function
  class (e.g. mode select) from the plain pair.
- **Small indicator icons**: two small pictogram icons between the black buttons and the red button,
  each paired with a tiny red LED dot beneath it (status/mode indicators).
- **Primary control**: one large, glossy, domed red circular button, sitting on its own raised
  circular lobe, clearly bigger and more tactile than every other control — the unmistakable "press
  this to act" button (a real exposure/start button on the original device).

The visual hierarchy is the important thing to carry over, not literal fidelity to this exact
device: **one obviously-primary, large, glossy, red, circular action control; several small, flat,
recessed, dark secondary controls; a warm neutral body material; a red digital-readout accent.**
That hierarchy already exists at the *color* level in this session's work (one red button per
window, amber secondary controls) — the follow-up work is making the *shapes and materials* carry
it too, not just color.

### What needs designing/building

1. **A custom-painted primary-button widget** (`QPushButton` subclass, painted via `QPainter`, not
   just QSS) styled after the reference's red button: circular or heavily rounded, a radial-gradient
   highlight suggesting a domed glossy surface, a subtle drop shadow/bevel for physical depth,
   pressed-state visually "pushing in" (shift the highlight, darken slightly). This replaces the
   current flat-red `role="primary"` QSS button (`Save calibration profile` / `Develop`).
2. **A custom-painted secondary-button widget** for the Shadow/Highlight toggle buttons and the
   Preview button: flatter, darker, a subtle inset/recessed look (inner shadow at the top edge,
   suggesting the button sits slightly below the panel surface) rather than the current flat QSS
   rounded-rect. Needs a distinct "one is chosen from a set" visual (the chrome-ringed button in the
   reference is a plausible model for a checked/active state).
3. **A body-panel treatment** for the "Neutral Points" frame (and possibly the whole window
   background) — currently a flat charcoal `QFrame` with a 1px border; the reference's warm grey
   material could translate to a subtly textured or gradient-shaded panel background, not necessarily
   a literal grey (should stay legible against the rest of the app's now-established dark theme —
   this needs a real design pass, not just recoloring to grey, which could clash).
4. **A digit-readout treatment**, if a suitable use is found for it — the reference's red LED display
   is visually striking but this app doesn't currently have an obvious numeric readout that begs for
   it (the Fine-tune sliders show plain numbers via standard Qt slider labels). Worth deciding
   whether to (a) skip this element entirely since there's no natural home for it, (b) add one
   specifically for the Fine-tune exposure/contrast values, styled as a small LED-digit readout next
   to each slider, or (c) find another use (e.g. a frame counter in a future batch-picking flow).
   Don't force it in just because the reference has it.
5. **Slider skinning** to match, if sliders end up kept in the Fine-tune popup — a chunkier,
   more physical-looking groove/handle consistent with the rest of the skin.

### Technical approach — open question for that session

Two real options, with a real tradeoff, to weigh with the user before writing code:

- **QSS-only, using `qlineargradient`/`qradialgradient`** — Qt stylesheets support gradients
  directly; a domed-red-button look is plausibly achievable without any custom `paintEvent` code at
  all. Lower effort, lower risk, easier to maintain, but a real ceiling on how convincing the
  "physical object" illusion gets (no true shadow falloff, no easy pressed-state depth animation).
- **Custom-painted `QPainter` widgets** — full control (real radial gradients with precise stops,
  drop shadows via `QGraphicsDropShadowEffect` or hand-painted, smooth pressed-state transitions),
  but meaningfully more code, and every such widget needs its own hit-testing/sizing/accessibility
  handled manually instead of getting it for free from a styled standard widget.

Recommend starting with the QSS-gradient approach for the primary button specifically (highest
visual impact, lowest effort) and only reaching for custom painting if that genuinely doesn't get
close enough — but this is exactly the kind of call to make together with the user, ideally with a
couple of quick visual prototypes (even just static mockup screenshots) to compare before committing
either way, rather than assuming.

### Suggested scope for that follow-up session

1. Prototype 2-3 visual directions for just the primary button (cheapest to iterate on, highest
   visual payoff) and get the user's reaction before building anything else.
2. Once a direction is picked, apply it to the primary button for real across all three places it
   appears (main window's Save/Develop button, and anywhere else a `role="primary"` button exists).
3. Secondary-button (Shadow/Highlight/Preview) treatment.
4. Body-panel treatment for the Neutral Points frame.
5. Decide on and (if warranted) build the digit-readout treatment.
6. Full Xvfb-verified pass across every screen (main window, Preview popup, save-name modal,
   `--pick` flow) to confirm nothing about the new skin breaks readability, contrast, or the
   fixed-window-size/no-scrolling constraint this session was built around.

Treat this the same way this session was run: as its own plan (likely its own worktree), verified
live via Xvfb/xdotool/screenshots at each step, not assumed correct from code alone — visual/tactile
design specifically benefits from actually looking at it, more than most of this codebase's other
work.

## Update: the multi-point picker changed what there is to skin

The picker was restructured (branch `worktree-multipoint-picker`, plan `hidden-popping-crown.md`):
calibration now uses any number of neutral points across a roll, and the window is landscape with a
filmstrip. Some controls this spec names are gone, and there are new ones. The skin work itself is
still deferred (the user's choice). When it's picked up, skin these instead:

- **Primary (red, domed):** unchanged role — "Save calibration profile" / "Develop".
- **Secondary (flat, dark, recessed):** "Open profile…", "Load roll…", "Build contact sheet…",
  "Clear frame" / "Clear all", and in the contact sheet window "Rebuild contact sheet" and "Save
  contact sheet…". The **Shadow/Highlight toggle buttons and the Preview button no longer exist.**
- **"One chosen from a set" (the chrome-ringed button):** the Negative | Positive view switch is
  now the natural fit.
- **Body panel:** the right-hand control panel as a whole — step wedge, point list, drawers, red
  button — reads like an enlarger's control face beside the baseboard (the image).
- **LED digit readout:** it now has natural homes, which this spec said not to force until there
  were some: the current frame number over the filmstrip (like a frame counter), and the neutral
  point count. The Print drawer's exposure and grade values are a third candidate.
- **Filmstrip, step wedge and contact sheets** already use the film's own palette (black rebate,
  orange edge print, `theme.EDGE_PRINT`). Keep them as they are, not the controller's materials.

## Where things stand as of this document

- Branch `worktree-gui-qt-rewrite` (merged into `claude-rewrite`) has the full Qt rewrite described
  above.
- `gui/theme.py` has the red/amber color hierarchy in place; no custom-painted widgets exist yet.
- `enlarger-controller.png` (repo root, gitignored) is the reference photo for the work above.
- 179/179 tests pass; nothing about the enlarger-skin work should need new test coverage beyond a
  visual Xvfb check, since it's presentation-only.
