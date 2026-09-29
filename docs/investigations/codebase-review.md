# halide codebase and UX review (2026-09-28)

The review the plan `docs/plans/codebase-review.md` describes, run against `main` at `b52d6ab`.
It checks three things:

- whether halide meets what the user has asked for
- whether the code is correct and consistent
- how the CLI and GUI feel to a film photographer picking halide up for the first time

The review changed no source code. The per-task findings, the requirements list and the
matrices behind this report are in `codebase-review-evidence/`. Finding IDs in brackets (for
example [2.3-1]) point into those files.

## 1. Summary

**The core of halide is in very good shape.**

- **Colour maths.** It matches both reference implementations exactly. abpy's own
  `density levels.py`, run unmodified, gives the numbers halide's test pins, and the paper curve
  is byte-identical to abpy's.
- **Memory and speed.** Measured on the real scans, both match `CLAUDE.md`'s recorded figures
  to within about 4% (invert 372 MiB vs 377 recorded, `--auto-density` 511 vs 521).
- **Round trip.** Flat → `print` matches a direct `invert` to 4.2e-7. A +0.6 EV copy with its
  metadata stripped matches to exactly 1 sRGB step, as recorded.
- **Tests.** 772 pass. Coverage is 72% overall and at least 75% everywhere outside `gui/` and
  `check`.
- **Requirements.** Of the 89 requirements drawn from the transcripts, `CLAUDE.md` and `docs/`,
  every one that can be checked in this sandbox is met in the code. The exceptions are the
  UX-level ones (R-011 to R-016), which the findings below speak to.

The review's first-impression screens (the `halide` banner, the run sheet, the batch
contact-sheet animation, the picker's live negative-to-positive, the contact sheet window) all
carry the film/darkroom identity the user asked for, without tipping into gimmick. Section 3
lists what works, so it isn't "fixed" away.

**The weak spots are all at the edges.** The pipeline itself is sound; what's weak is the code
around it:

1. **Data safety.** halide can overwrite a user's original scans or saved profiles without a
   word (F01-F04). This is the one area that needs fixing before anyone else uses halide.
2. **Hostile or unusual input.** NaN pixels, broken ICC profiles, RGBA/greyscale files, macOS
   `._` files and unwritable folders either crash with library jargon or go silently wrong
   (F05, F08, F09).
3. **CLI polish.** A wrong completion message on every batch, no `NO_COLOR`/pipe support, no
   `--version`, no examples in `--help`, and small wording inconsistencies (F10, F20-F22).
   Individually minor, but together they're what separates "designed" from "nearly designed".

**Counts** (after merging duplicates across tasks and re-rating): 142 raw entries became 32
consolidated findings: 4 S1 (F01-F04), 8 S2 (F05-F09, F11-F13), and the rest S3-S5, plus ideas.

**What the review corrected about its own agents:**
- **Refuted:** "the picker never quits when its window is closed" [6-C]. The agent's
  `xdotool windowclose` destroys the X window outright. A real close request (`WM_DELETE_WINDOW`)
  exits cleanly, workers included.
- **Resolved:** two audits disagreed on whether `invert` guards against overwriting. It does in a
  terminal, and doesn't in a script (F01).
- Details: `codebase-review-evidence/verified.md`.

## 2. Decisions the user needs to make

These can't be settled by the review alone:

- **D-1 (F02): overwrite policy for existing outputs.** Options:
  - (a) Refuse, and add `--overwrite`.
  - (b) Prompt in a terminal, refuse in a script, and add `--overwrite`.
  - (c) Skip existing outputs by default (resumable batches), and add `--overwrite`.

  Recommendation: (b), plus a `--skip-existing` flag for resuming an interrupted roll.
  Output = input is a hard error in every mode.
- **D-2 (F07): the manual-calibration Save gate.** This reopens a settled decision (0.1 D minimum
  separation). A simulation says a fit through points only 0.1 D apart can be CC 8-15 off at the
  ends of the tonal range, even when the points agree with each other. That is a simulation, not
  a measurement on the user's picks. Options:
  - Raise the gate.
  - Keep the gate, but warn and show "fit reliable over D x-y".
  - Leave it.

  Recommendation: the warning, which is informative and changes no output.
- **D-3 (F31): the picker's 1400x900 size cap.** `CLAUDE.md` says "~55% x 70% of the screen"
  without mentioning this ceiling. On a 2560x1440 screen the window is about 55% x 63%. Keep it or
  lift it?
- **D-4: `IMG_0158-positive.tif`.** A 127 MB untracked output sits in the repo root [3a-3]. Delete
  it, or is it a reference you're keeping?

## 3. What works (keep these)

**CLI.**
- **The bare `halide` screen.** A sprocket rule scaled to the terminal, the spaced wordmark, and
  two curated first workflows instead of argparse's error. It's the best first impression in
  the tool.
- **The run sheet.** One aligned, wrapped block that reads well at 60 to 200 columns. The
  scan-consistency warning in it is legible and actionable.
- **The batch contact-sheet animation.** Restrained frames that pulse and fade, with an ETA. It
  stays tidy at every width tested.
- **Ctrl-C** in `invert`, `batch` and the prompt: exit 130, an accurate line, no partial output,
  no orphan processes.
- **ICC error messages** for honest-but-wrong profiles. For example: "this profile is
  gamma-encoded … Re-export using your raw processor's linear gamma option". The typo error for
  profile names is equally good. These are the model for fixing F09.
- **Paths.** Non-ASCII paths and paths with spaces work everywhere.

**GUI.**
- **Layout.** The fixed, non-scrolling layout holds at 1366x768, 1920x1080, 2560x1440 and 2x
  scaling.
- **The live negative → positive** after two good points is the "wow" moment.
- **The agreement hint.** It names the right culprit even when another point's own reading is
  worse, exactly as designed (seen live: "Point 4 is furthest from the others (CC 9 R)").
- **Safety rails.** The auto-candidate overlay excludes the orange jacket. Save/Develop stays
  disabled for clustered points. The Save dialog guards overwriting. "Clear all" asks first.
  Reopening a profile and re-finding a moved roll both work as documented.
- **The contact sheet** (window, `halide contact`, batch preview) is the best-looking thing in the
  tool and reads as a real contact print.

## 4. Findings, ranked

Severity: **S1** = can lose a user's file or silently give a wrong colour. **S2** = crash or
visibly wrong result. **S3** = a user gets stuck or confused. **S4** = inconsistency.
**S5** = polish. Effort: S = under an hour, M = a few hours, L = a day or more.

### Data safety

**F01 · S1 · S — A scan can be overwritten by its own output** [2.3-1, 2.5-1, 4-3; verified]
- `halide batch roll roll` (no `--suffix`) replaced both negatives with positives. It exited 0 and
  printed nothing.
- `halide invert s.tif s.tif` in a script (no terminal) does the same.
- In a terminal, `invert` asks a generic "already exists — overwrite?" that doesn't say the file
  is the input scan.
- The same applies to `print` in bulk.

Fix: output = input is a hard error everywhere, checked before any work starts.

**F02 · S1 · M — Bulk commands silently overwrite existing outputs** [2.3-2, 4-4]
- `batch`, `print --bulk` and `export --bulk` never check.
- Single-file commands prompt in a terminal but overwrite silently in scripts
  (`console.confirm_overwrite`).

Needs D-1.

**F03 · S1 · M — Saved profiles can be overwritten, deleted or mis-saved without a word**
[2.2-1, 2.2-2, 2.2-3, 2.2-8, 2.5-2, 2.6-1, 2.6-5]
A profile can hold an evening's worth of picked points.
- `--save-profile-as NAME` overwrites an existing profile silently. The GUI's Save dialog does
  guard this, so the two disagree.
- `profile delete` doesn't confirm.
- Names are never validated: `../x` writes or deletes outside the profiles folder. A leading `-`
  saves fine but can't then be addressed.
- In the GUI, a name with `/` (or any disk error) makes Save fail silently: the dialog just stays
  open.

Fix: one `validate_profile_name` shared by the CLI and the GUI; a confirm/`--force` for
overwrite and delete; the GUI shows the error.

**F04 · S1 · S — Writes aren't atomic** [2.3-3, 2.3-4, 2.2-5]
- A disk-full or killed write leaves a truncated TIFF (or profile JSON) under the real file
  name, which looks finished.
- A later `--profile NAME` then fails with a raw JSON parser error.
- If exiftool fails after a good TIFF write, it's reported as an "unexpected error".

Fix: write to a temp file in the same folder, then `os.replace`. That changes no bytes of the
output.

### Accuracy and robustness

**F05 · S2 · S — Non-finite and non-positive pixels** [2.1-2, 2.2-6; extended in verified.md]
- **One NaN or inf pixel:**
  - `invert` crashes with "index -2147483648 is out of bounds".
  - `--output flat` writes an all-NaN file, reports "✓ Developed", and writes `NaN` into the
    provenance JSON.
  - `--auto-density` returns an all-NaN profile.
- **Non-positive pixels** are ranked as the *most neutral* candidates by auto calibration. On a
  synthetic frame, 0.1% of pixels at −0.001 moved the solved density scale from (1.28, 0.79) to
  (0.95, 1.05), silently.
- **The user's scans are safe today.** Measured fractions are 1e-7 to 1e-6 (IMG_0158: 18 pixels),
  below the level where it matters. A noisier camera or export could cross it with no warning.

Fix: mask non-finite and non-positive pixels out of the statistics, map non-finite values once
at the working-space boundary with a counted warning, and set `allow_nan=False` in the
provenance. This is bit-identical on clean input.

**F06 · S2 · S — An interrupted GPU-service request can write an undeveloped negative** [2.4-1]
- If Ctrl-C lands during a request, the client isn't marked dead. The worker's next queued
  frame then receives the previous frame's reply, and its undeveloped negative is written under a
  normal-looking provenance.
- Reproduced at the client level. In a real batch the window is milliseconds wide.

Fix: treat any `BaseException` in `_request` as a dead connection.

**F07 · S2 · M — Calibration from points close in density extrapolates badly, and nothing says
so** [2.1-1] REOPENS the 0.1 D decision. See D-2.

**F08 · S2 · M — ICC validation gaps** [2.1-3, 2.1-4, 2.1-5]
- **Linearity check.** It uses an absolute 5e-3 tolerance, which accepts a 0.004 black offset
  (+0.15 D at transmittance 0.01) and gamma 1.013.
- **Truncated or malformed profiles** either crash ("unpack requires a buffer") or are accepted
  with a garbage matrix.
- **White point.** Nothing checks that the primaries sum to D50, so an unadapted profile gives
  an unnoticed cast.

Fix: a relative or density-space tolerance, bounds checks on every tag read, and a white-sum
check.

**F09 · S2/S3 · M — Common mistakes produce library jargon** [2.1-6, 2.3-5, 2.3-7, 2.5-7,
2.5-9, 4-16, 4-17, 2.2-4, 2.2-5]
- **Wrong input files.** RGBA or greyscale TIFFs give "matmul … size 3 is different from 4". A
  JPEG or a fake `.NEF` passes raw tifffile log lines through. So does a directory given where a
  file is expected.
- **Output folder problems.** A missing or unwritable output folder is found only *after* the
  frame is developed, and reported as `[Errno 13]`, once per frame in a batch.
- **macOS `._IMG_*.tif` files** are queued as frames, so a clean roll "fails".
- **Profiles.** One unreadable profile crashes `profile list`, contradicting its docstring.

Fix: check paths and the output folder up front; skip hidden and AppleDouble files; give each
input failure the plain-language treatment the ICC errors already get.

**F10 · S3 · S — "Contact sheet complete" after every batch, export and print** [2.3-6, 2.5-3,
4-1, 4-2, 7.1-1]
`GridProgressRenderer.finish()` hard-codes it. With failures, it also mis-states the success
count. It's the most visible single bug, and a one-line fix plus a test.

**F11 · S2 · M — Contact sheet window: rebuild problems** [2.6-2 confirmed live, 2.6-3, 2.6-4]
- After the first "Rebuild contact sheet", the window no longer notices when it goes out of date.
- The rebuild blocks the UI thread.
- Scrubbing the filmstrip starts unbounded concurrent full-resolution loads.

**F12 · S2 · S — Leftovers after abnormal exits** [2.4-2, 2.4-9]
- Closing the terminal (SIGHUP), or a whole-group kill, during a GPU batch leaves frames in
  `/dev/shm` (RAM) until reboot: about 1 GiB per run on the user's machine.
- Ctrl-C in a batch's first second prints pages of forkserver tracebacks.

**F13 · S2 · S — Dependency versions are unbounded** [8x-2, 2.7-4]
`io/icc.py` and `io/raster.py` rebuild colour-science's internal matrix steps (D1). A future
colour-science or numpy release could change halide's colour or break it on a fresh install.

Fix: lower bounds, an upper bound on colour-science and numpy, and the tested versions noted in
the README.

**F14 · S3 · S — Out-of-range numbers are accepted** [2.1-7, 2.1-8, 2.1-11]
- `--contrast -1` and `--rm 0` "succeed".
- A near-flat channel gives a density scale of 50 to 1e15, which nothing rejects.
- (Internal) core functions give wrong results on integer arrays.

**F15 · S3 · S — Re-inverting a halide output isn't caught** [2.3-8, downgraded from S2]
`print` checks provenance to refuse the wrong input; `invert`/`batch` don't. Pointing `batch` at
last run's output folder re-inverts positives without a warning.

**F16 · S3 · S — Missing EXIF is treated as "consistent"** [2.2-9, 2.2-10, 2.1-10]
- `halide check` and batch's Scans row report a roll with no EXIF anywhere as fine.
- A picked point from a frame without EXIF enters the fit unnormalised, silently.

Fix: say "not verifiable" instead.

**F17 · S3 · S — exiftool edge cases** [2.4-5, 2.4-13]
- The last-resort exiftool call has no timeout, so a hung exiftool still hangs the batch. That
  contradicts `CLAUDE.md`'s "never hangs".
- A missing exiftool drops metadata without a word. One line on the run sheet would do it.

**F18 · S3 · S — `describe_cast` can name the wrong filter** [2.1-9]
A (+0.1, 0, −0.1) cast reads "CC 20 R" when the dichroic pack is 20Y + 10M. A printer who
dials the reading gets the wrong correction.

### GPU messaging

**F19 · S3 · M** [2.4-3, 2.4-4, 2.4-6, 2.4-7, 2.4-8, 2.4-10]
- **Service death.** If the service dies at frame 2, all 35 remaining frames print the same
  warning, worded as a Python error ("BrokenPipeError: [Errno 32]"). One warning is enough.
- **Bad input.** A bad input file is reported as "the GPU failed", then fails again on the CPU.
- **Broken CuPy.** A broken CuPy install falls back silently, and `halide gpu` prints the reason
  as "None".
- **Wording.** Fallback wording differs between paths.
- **CUDA in the parent** (likely, by reading). The batch parent imports CuPy and keeps a CUDA
  context just to probe, which contradicts "CUDA only exists in the service" and costs about a
  worker's worth of RAM. This needs a measurement on the RTX 3070.
- **Uninstall.** The uninstall advice from `gpu --install` leaves most of the ~1 GB behind.

### CLI experience and consistency

**F20 · S3 · M — Terminal etiquette** [2.5-4, 4-7, 4-6, 2.5-8, 4-5, 4-19]
- **Colour.** halide's own colours ignore `NO_COLOR`, `TERM=dumb` and non-terminal output. The
  argparse half of the same output respects them.
- **Piped output.** The animated grid has no plain-lines fallback when piped. The only option is
  `--quiet`, which is all or nothing.
- **Log order.** stdout is block-buffered while stderr isn't, so in a logged run the error prints
  before the context that explains it.
- **`--quiet` success.** `--quiet` on a successful batch prints nothing, not even a one-line
  summary.

**F21 · S3/S4 · M — Help and discoverability** [2.5-11, 4-8, 4-9, 4-10, 4-11, 4-12, 2.1-12,
2.5-16]
- **No `--version`.**
- **No examples.** Subcommand `--help` never shows one, although the bare banner does.
- **`profile` subcommands.** `list`, `show`, `rename` and `delete` have almost no help text,
  while `profile edit` has a paragraph.
- **Formatting slips.** `calibrate --help` cuts its `--profile` text off mid-sentence. The
  banner's last example is misaligned by 3 columns.
- **Old name.** "Fine-tune", the drawer's old name, is still in `print --help`.
- **Stated defaults** are uneven: `--rs`/`--bs` state theirs, `--rm`/`--bm` don't, and
  `--workers` is explained at four different depths.
- **`--debug`** after the subcommand fails with a bare argparse error.

**F22 · S4/S5 · S — Wording and style drift** [2.5 observations, 4-13, 4-14, 4-15, 2.5-13]
- "point(s)", "frame(s)" never pluralise properly.
- `check`'s heading and its rows use different nouns.
- "Scanned at" vs "digitized at".
- Two "clear this field" conventions in `profile edit` (`''` vs `-`).
- Ctrl-C looks different in batch and at the prompt.
- Full stops are used inconsistently.
- `strings.tsv` (155 rows) is the working list for one style pass.

**F23 · S3/S4 · S — Tab completion** [5-5, 2.5-5, 2.5-6]
- `HALIDE_NO_COMPLETION=1` still runs the psutil parent-shell probe: about 25 ms, roughly a
  quarter of `profile list`'s run time.
- A failed rc edit is stamped as done and never retried.
- Writing the fish script replaces a symlink with a plain file, which breaks chezmoi or
  home-manager dotfiles.

**F24 · S4 · S — `profile` subcommand inconsistencies** [2.2-7, 2.5-12]
- There's no "did you mean" for typos, which `--profile` gets.
- `rename`, `delete` and `edit` reject file paths that `show` accepts, without saying why.

**F25 · S4 · S — Skipped files go unmentioned** [4-18]
A mixed folder silently drops non-TIFF files; the run sheet should say "N other files ignored".

### GUI

**F30 · S4/S5 · M** [2.6-7, 6-E, 6-F, 6-G, 6-D, 2.6-8, 2.6-6]
- **HiDPI.** No devicePixelRatio handling in the custom painting. The code gap is real; blur
  wasn't visible at 2x in Xvfb.
- **Keyboard focus** has no visible indicator.
- **Status line.** Its text runs off the window edge.
- **Stuck status.** "measuring the roll…" stays after "Continue without".
- **Repaint glitch.** A transient mis-painted label ("JTRAL POINTS") behind the Roll-not-found
  dialog.
- **Failed frames.** A frame that fails to load shows no reason until you click into it.
- **Theme.** Two hex colours in `filmstrip.py` are hard-coded outside the theme.

**F31 · S5 — The window's 1400x900 maximum** [6-A]. D-3.

### Cross-platform (macOS)

**F35 · S4 · S-M** [8x-1, 2.2-12, 2.4-11]
- **Fonts.** The contact sheet's edge print and the GUI use DejaVu or Liberation, which macOS
  doesn't have. There the edge print loses its bold film lettering. Vendoring one font makes the
  look the same everywhere (R-016).
- **Config location.** No macOS-native config path, and nothing guards against profile names
  that differ only by case on a case-insensitive disk.
- **Shared memory (latent).** Shared-frame names are 55 characters, over macOS's 31-character
  limit. Unreachable today, because the service needs CUDA.

### Code quality

**F40 · S4/S5 · M** [3a-1, 2.5-10, 2.2-11, 2.5-14, 2.3-9, 2.5-17, 2.6-9, 2.1-13, 2.1-14,
2.4-12, 2.5-15, 7.1-3, 7.1-4, 7.1-5]
- **The `core/` rule.** `core/tone_render.py` reads the `.cube` file itself, which breaks the
  "`core/` = no file I/O" rule.
- **Duplication:**
  - TIFF folder discovery is copied three times, and the copies have already drifted apart.
  - The "manual calibration given?" check is copied four times.
  - Mutual exclusion of calibration flags is done by hand rather than by argparse.
- **Large modules.** `processing.py` (785 lines), `orchestrator.py` (1037), `console.py` (615) and
  `main_window.py` (1059) each mix separable jobs. Worth splitting only when next touched.
- **Stale or wrong docstrings** in `core/`, `roll_auto_density_balance` (describes the old
  reference), `save_profile` (names a missing file) and `core/pipeline.py`'s module docstring.
- **Other small items:**
  - The `.cube` reader ignores `LUT_1D_SIZE`.
  - The `gpu --install` prompt doesn't catch EOF.
  - Assorted small items in the GPU plumbing.

### Tests

**F41 · S5 · M** [0-1, 0-2, 3b-1, 3b-2, 3b-3, 3a-5, 2.1-15]
- **Coverage gaps.** `gui/` is 14-39% covered and `check` 22%. Every hostile input above lacks a
  test.
- **Environment-dependent tests.** Seven `gpu --install` tests depend on the host having pip.
- **Flaky teardown.** Multiprocessing semaphores leak between test files and trip the `/dev/shm`
  teardown check when files run together.
- **Unpinned requirements.** R-054, R-077, R-090, R-092, R-096 and R-105 have no test pinning
  them.
- **Picker smoke test.** A headless test (offscreen Qt: open a roll, add points, save, close) would
  cover most of section H cheaply.

### Documentation

**F42 · S3/S4 · M** [7.1-2, 7.1-6, 7.1-7, 7.1-8, 7.1-9, 3a-2]
- **Missing README sections** a newcomer needs:
  - menu-by-menu darktable and RawTherapee export steps for "linear Rec.2020 TIFF with embedded
    ICC"
  - a troubleshooting section
  - what a finished output looks like
- **Stale docs.** `docs/README.md` and `docs/plans/gpu-acceleration.md` still say the GPU work is
  on its own branch.
- **Auto-calibration advice.** The CLI never says manual picking is the reliable option when auto
  calibration is used (R-050).
- **IDEA:** `CLAUDE.md` is ~600 lines. Its evidence could move to `docs/`, leaving a decision line
  plus a link (list by heading in 7.1.md).

### Performance

**F43 · IDEA — no regressions found**
- Start-up of every command is ~0.1 s, as claimed, apart from F23's 25 ms.
- In a CPU batch, TIFF compression is ~60% of CPU time and the colour maths ~17%. A level-1
  deflate test showed no clear win.
- Anything more needs a real-roll benchmark on the user's machine.

## 5. Feature ideas (not argued for; from Task 4.4)

- `--dry-run`: show what a batch would do without writing.
- `--overwrite` / `--skip-existing` (see D-1).
- Output name templates.
- Recursive roll folders.
- `--log FILE`.
- Per-command examples (`--examples`, or in `--help`).
- A richer end-of-batch summary: a per-frame grade and exposure recap, the data the contact
  sheet already captions.

Open follow-ups inherited from earlier work, all still accurate:
- scan-exposure matching not yet validated on a real two-exposure scan
- `check --suggest-anchor`
- ColorChecker, denoise and B&W modes
- the enlarger skin
- the untested Windows branches
- the picker keeping its CUDA context for the whole session
- the `release_memory` traceback caveat
- the Compute line not saying which cap set the worker count
- `--grade roll`

## 6. What wasn't checked

- **GPU paths on real hardware.** Reviewed by reading, and with the fake device through a real
  spawned service. The user should run `pytest -m gpu` on the RTX 3070 against this branch.
  F19's CUDA-context claim needs a memory reading there.
- **macOS and Windows.** Everything about them is by reading only.
- **GUI details** a sandbox can't show: the row → marker flash, memory while scrubbing a
  37-frame filmstrip, true compositor HiDPI.
- **Transcripts.** The session transcripts only cover 27-28 September. Earlier requirements come
  from `CLAUDE.md` and `docs/`. The user added R-016 (visual identity) at the ledger checkpoint.

## 7. Suggested order for fixing

1. **Data safety:** F01, F02 (after D-1), F03, F04. All small; do these first.
2. **Robustness:** F05, F06, F08, F09, F12, F13, F17.
3. **Visible polish:** F10, F20, F21, F22, F23, F24, F25, F18, F16, F14, F15.
4. **GUI:** F11, F30.
5. **Docs and README:** F42, and F35's fonts.
6. **Housekeeping:** F40, F41, F19, and any F07 change (after D-2).

Each group can be its own task in `docs/plans/review-fixes.md` once the user has marked findings
fix / won't fix / later.
