# Codebase and UX Review Plan

> **For agentic workers:** this is a *review* plan, not a feature build. Each task produces findings,
> not code. Use superpowers:dispatching-parallel-agents for the independent audit tasks (Phase 2),
> and follow the model choice given for each task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, evidence-backed review of halide as it stands on `main` (b52d6ab). It checks
three things: (1) the code meets every requirement and design principle the user has stated,
(2) the code is correct, efficient and consistent, (3) the CLI and GUI feel like one designed,
professional tool to a film photographer picking it up for the first time.

**Working files:** `.review/` in the repo root, excluded from git through `.git/info/exclude`
(so `.gitignore` is untouched). It holds the agents' finding files, the ledger, captures and the
throwaway venv. Delete it once the report is committed.

**Deliverable:** `docs/investigations/codebase-review.md` on branch `review-2026-09`: one ranked
list of findings plus a requirements matrix. **No source changes during the review.** The user
chooses which findings get fixed. The fixes then get their own plan (`docs/plans/review-fixes.md`).

**Scope decisions (user, 2026-09-28):**
- Report first, then fix.
- Requirements come from the earlier session transcripts as well as CLAUDE.md, docs/ and git.
- Audience: a film photographer on Linux or macOS who can follow a README and use pip/pipx.
  Windows problems are noted but rated low priority.
- Data: the four repo-root scans (IMG_0151, IMG_0156, IMG_0156-nowb, IMG_0158) plus
  `Roll16-Testing/` (37 frames). No GPU in the sandbox: GPU code is reviewed by reading it, and
  the user runs `pytest -m gpu` at the end.

## Global Constraints

- CLAUDE.md's "Decisions and why" are settled. A finding may challenge one only with **new
  evidence**, and it must be labelled `REOPENS: <decision>`.
- The priority is colorimetric faithfulness. A finding that trades accuracy for looks is out of
  scope; a finding that threatens accuracy is top severity.
- CPU output is bit-identical today. Any optimisation proposal must say whether it keeps output
  bit-identical, and if it doesn't, what tolerance applies (D1/D2).
- Don't add a linter, formatter or new dependency during the review. Tools used only to review
  (coverage, py-spy) go into a throwaway venv in `.review/`, never into `pyproject.toml`.
- Never commit scans or outputs. Large outputs (TIFFs) are deleted straight after they're checked.
  `/tmp` is RAM on the user's machine, so nothing large stays in `.review/` either.
- Token economy (memory `token-economy`): Opus only for colour-math, concurrency and synthesis.
  Sonnet for audits and walkthroughs, Haiku for extraction and running commands. Agents write
  findings to files and return a short summary, never raw logs.

## Finding format (every task uses this)

Each agent writes `.review/<task-id>.md` as a list of entries:

```
### <task-id>-<n>: <one-line claim>
- Severity: S1 accuracy/data-loss | S2 bug | S3 UX friction | S4 inconsistency | S5 polish | IDEA
- Where: src/halide/…:line (or command + terminal size, or screenshot path)
- Evidence: the command run and its output excerpt, or the code quoted. "Looks wrong" is not evidence.
- Requirement: R-nnn from the ledger, if one applies
- Suggestion: what to do, and roughly how much work it is (S/M/L)
- Confidence: confirmed (reproduced) | likely (read, not reproduced)
```

Severity guide: **S1** can silently give a wrong colour or lose or overwrite a user's file.
**S2** crashes, or gives a wrong result in a way the user would notice. **S3** makes a
reasonable user stuck, confused or slowed down. **S4** is a place where two parts of halide
behave or look differently for no reason. **S5** is cosmetic. **IDEA** is a missing feature,
recorded but not argued for.

---

## Phase 0: Setup and baseline (inline, Haiku for the runs)

### Task 0.1: Branch, tools and baseline
- [ ] Branch `review-2026-09` (created).
- [ ] Install the tools CLAUDE.md expects but this sandbox lacks: `xvfb`, `xdotool`, `imagemagick`
      (`import`), `libimage-exiftool-perl`, plus the Qt system libraries CLAUDE.md lists.
- [ ] Throwaway venv `.review/venv` with halide installed editable plus `coverage` and
      `py-spy`.
- [ ] Baseline: `.venv/bin/python -m pytest tests/ -q --durations=25`. Record pass/skip counts,
      total time and the slowest tests. Then run coverage per module (review venv).
- [ ] Record `halide --version` (if there is one), Python version and dependency versions.
- **Output:** `.review/0-baseline.md`.

## Phase 1: Requirements ledger

### Task 1.1: Extract requirements (Haiku, one agent per transcript, parallel)
- [ ] Preprocess with a script (no model): for each of the 7 earlier transcripts in
      `~/.claude/projects/-home-sam-Documents-halide/*.jsonl`, keep only the **user's own
      messages** and the assistant's final summary messages, with timestamps. This cuts ~12 MB of
      tool output down to readable text. Write `.review/transcripts/<session>.txt`.
- [ ] One Haiku agent per transcript lists every requirement, preference, complaint, promise
      ("I'd like…", "don't…", "make sure…") and every open follow-up. Each item gets a quote,
      the session and the date.
- [ ] Also extract from: CLAUDE.md (every "Decisions and why" bullet, "Working with the user"),
      each `docs/` write-up's open follow-ups and "not yet validated" notes, and commit messages
      that mention the user.

### Task 1.2: Merge into a ledger (Opus, inline)
- [ ] De-duplicate into `.review/requirements-ledger.md`: `R-001…`, each with its
      statement, sources, category (philosophy, accuracy, feature, CLI UX, GUI UX, performance,
      process/docs) and **how to verify it** (test name, command, or code to read).
- [ ] Mark requirements that were later changed on purpose (for example the dearpygui → Qt move)
      as superseded, so they aren't reported as unmet.
- [ ] **Checkpoint with the user:** show the ledger's categories and any requirements that
      conflict with each other. The user confirms that nothing is missing before Phase 3 checks
      against it.

## Phase 2: Code audit (parallel agents, independent of Phase 1)

Every audit agent gets: the finding format, the Global Constraints, the relevant CLAUDE.md
section *by line range* (not the whole file), and this checklist:

> correctness and edge cases · error handling (is the message useful to a photographer?) ·
> resource cleanup (files, processes, shared memory, threads) · dead or duplicated code ·
> docstrings and comments that no longer match the code · naming consistency · the
> architecture rules (core/ pure, no I/O; heavy imports stay lazy) · possible speed-ups (measured,
> not guessed) · tests: what's missing, what's brittle, what tests the implementation rather than
> the behaviour.

### Task 2.1: Colour and print math — `core/`, `io/icc.py`, `io/raster.py`, `io/lut.py`, `calibration/density`-side (Opus)
- [ ] Check the pipeline against Buchler's method and the two reference repositories step by step
      (white balance → density balance → invert → print curve, in ACEScg). Note every deliberate
      difference and whether a decision covers it.
- [ ] ICC parser against the ICC spec subset it claims (tag types, s15Fixed16, parametric vs.
      table TRCs, `chad` handling, what "linear" is allowed to mean). Try hostile profiles.
- [ ] `fit_print`, `estimate_linear_scale`, the `MIN_TRANSMITTANCE` floor, NaN/inf/negative
      paths, integer input dtypes, single-channel or alpha inputs.
- [ ] `fit_density_balance` / `anchors.py`: numerical conditioning, leave-one-out maths, the
      scan-gain normalisation.

### Task 2.2: Calibration and profiles — `calibration/` (Sonnet)
- [ ] `auto.py` against its documented limits; `profile_store.py`: atomic writes, corrupt or
      hand-edited JSON, old-format profiles, names with `/`, spaces or unicode, two processes
      saving at once, `$XDG_CONFIG_HOME` and macOS paths.
- [ ] `scan_consistency.py`: missing or odd EXIF, mixed cameras.

### Task 2.3: Processing, banding, batch — `processing.py`, `banding.py`, `batch/` (Sonnet)
- [ ] `processing.py` is 785 lines and `orchestrator.py` 1037: is each doing one job? Propose
      splits only where they clearly help.
- [ ] Output safety: overwriting existing files, output path equal to input, output folder
      missing or read-only, disk full part-way through a write (is a half-written TIFF left
      behind?), Ctrl-C mid-batch (orphan workers, leftover temp folders or `/dev/shm` segments,
      terminal left in a bad state by the progress display).
- [ ] Worker-count maths against the constants CLAUDE.md records.

### Task 2.4: GPU, service and shared memory — `device.py`, `gpu_service.py`, `shared_frames.py`, `io/exiftool.py` (Opus)
- [ ] Read-only review of the concurrency: races, timeouts, cleanup on every exit path, the
      authkey, what a second halide run at the same time does, macOS behaviour (no `/dev/shm`,
      spawn start method, no parent-death signal).
- [ ] Every fallback path runs under the fake device in the unit tests. Confirm that each one
      tells the user the same thing in the same words.

### Task 2.5: CLI code — `cli/`, `console.py`, `batch/progress.py`, `completion.py` (Sonnet)
- [ ] Argument definitions: flag naming consistency across commands, defaults stated in help,
      mutually exclusive flags actually enforced, exit codes (0 / 1 / 2 used consistently),
      `--quiet` honoured everywhere.
- [ ] Build an **inventory of every user-facing string** (a grep of `print`, `console.`,
      `warn`, `RunSheet`, argparse `help=`, exception messages that reach the user) into
      `.review/strings.tsv`: command, kind (heading/row/warning/error/help/hint),
      text. Task 4.2 uses this inventory.

### Task 2.6: GUI code — `gui/` (Sonnet)
- [ ] `main_window.py` (1059 lines): structure, signal and thread lifetimes, what happens on a
      failed frame load, a roll of one frame, a roll of 200 frames, a non-TIFF file in the roll,
      a portrait frame, HiDPI.

### Task 2.7: Packaging and project hygiene (Haiku)
- [ ] Build a wheel and an sdist in the review venv. Install the wheel into a **clean** venv
      and check that it includes the assets (ICC profiles, tone curves, fonts) and that the
      entry point runs.
- [ ] `pyproject.toml`: dependency pins and ranges, the Python version floor versus the code
      actually used (the GPU service needs 3.13; the floor says 3.11), extras, license metadata
      of vendored assets, `legacy/` still present, stray `__pycache__` in `docs/plans/`.

## Phase 3: Requirements verification (Sonnet, after Phases 1–2)

### Task 3.1: Check every ledger item
- [ ] For each R-nnn: **Met / Partial / Unmet / Regressed / Superseded**, with evidence (a test
      that pins it, a command run, or code quoted). Where a requirement has no test pinning it,
      say so; that's a test-coverage finding in its own right.
- [ ] Split by category across 2–3 agents. Items that need real scans go to Task 5.1 rather
      than each agent running its own.
- **Output:** `.review/requirements-matrix.md`.

## Phase 4: CLI experience walkthrough (Sonnet, one agent, serial)

The agent plays a photographer who has never seen halide, with only the README.

### Task 4.1: First run
- [ ] In a fresh venv, follow `README.md` word for word. Every step that fails, is unclear or
      assumes knowledge is a finding.
- [ ] `halide`, `halide --help`, `halide <cmd> --help` for every command. Check: does help say
      what the command is *for*, show an example, and state defaults? Is there `--version`?
- [ ] The first real task a new user would try: invert one frame, then a roll. Run it both with
      no profile (the interactive prompt, driven through a pty) and without a terminal.

### Task 4.2: Output design audit
- [ ] Capture every command's output (normal, `--quiet`, piped/not a tty, `NO_COLOR=1`,
      `TERM=dumb`) at 60, 80, 120 and 200 columns, replayed into a virtual terminal (the
      technique CLAUDE.md records for the progress display). Save as text or screenshots.
- [ ] Using those and `strings.tsv`, check that the whole tool reads as one design: vocabulary
      (frame/scan/negative/positive/print/roll used consistently), capitalisation, punctuation,
      symbols (⚠, ✓, bullets), colours, how warnings versus errors look, verb tense ("Developing"
      and "Exporting"), number and unit formatting, how paths are shown, and run-sheet rows that
      appear in one command but not in a similar one.
- [ ] Noise: count lines printed for a clean 37-frame batch and for a failing one. Is every
      line earning its place? Does anything important scroll away?

### Task 4.3: Mistakes a user will make
- [ ] Wrong inputs: a gamma-encoded TIFF, a TIFF without ICC, 8-bit, 16-bit, float, CMYK, with
      alpha, greyscale, a JPEG or a RAW file, an empty folder, a folder of mixed files, a
      profile name typo, a missing output folder, output = input, a non-ASCII path, a path with
      spaces. For each: is the error in plain photographer's language, and does it say what to
      do next?
- [ ] Ctrl-C during `invert`, `batch`, the roll estimate and the interactive prompt.
- [ ] Re-running a batch into a folder that already has outputs.

### Task 4.4: Missing features (IDEA findings only)
- [ ] Compare against what a user of Negative Lab Pro, Grain2Pixel and darktable's negadoctor
      would expect (dry run, overwrite control, output naming, recursive folders, a log file,
      shell completion, a man page or `--examples`, progress ETA, a summary at the end). Record
      each as IDEA with who'd miss it. No argument for building any of them yet.

## Phase 5: Real-data verification (Haiku runs, Sonnet reads)

### Task 5.1: Real scans and Roll 16
- [ ] Run each README workflow on the real data: `check Roll16-Testing`, `batch` with
      `--auto-density-roll` and with a saved profile, `--contact-sheet`, `export`, the flat →
      `print` round trip, `profile` subcommands. Note peak RSS and time per frame and compare
      with the numbers CLAUDE.md records (a drift is a finding).
- [ ] Run the verification items Phase 3 routed here.
- [ ] Delete every TIFF output once checked; keep contact sheets and text.

### Task 5.2: Performance profile (Sonnet)
- [ ] `py-spy` or `cProfile` on one `invert` and a 37-frame CPU batch. Rank the hot spots.
- [ ] Startup time of every command (`python -X importtime`).
- [ ] Propose an optimisation only with a measured cost behind it, and say whether it keeps
      output bit-identical.

## Phase 6: GUI experience walkthrough (Sonnet, one agent, Xvfb)

### Task 6.1: Drive the picker as a new user
- [ ] `halide calibrate Roll16-Testing` at 1366x768, 1920x1080 and 2560x1440, and with
      `QT_SCALE_FACTOR=2`. Look at every screenshot, don't just check that nothing crashed.
- [ ] The full job: pick points on several frames, read the agreement readings, save a profile,
      reopen it with `--profile`, move the roll folder and reopen it, build and save a contact
      sheet, close the window mid-load. Then `halide invert --pick`.
- [ ] Check: can a first-time user tell what to do next without the README (empty states,
      tooltips, the negative caption)? Keyboard use (Tab order, Delete, Esc, shortcuts), what
      error dialogs say, and whether the GUI and CLI use the same words for the same things.
- **Output:** findings plus the screenshots worth showing, in `.review/gui/`.

## Phase 7: Documentation review (Haiku)

### Task 7.1: Docs match the tool
- [ ] Run every command in `README.md` verbatim. Compare its descriptions with `--help` and with
      the real behaviour.
- [ ] CLAUDE.md and `docs/`: statements that are now stale (for example, `docs/README.md` still
      says GPU work is "on branch `gpu-acceleration`"), broken references to files, functions or
      constants (grep each name). Also: CLAUDE.md is now very long, and its evidence could move
      to `docs/` so the guide stays readable. That's a suggestion for the user, not a change.
- [ ] Is anything a new user needs (installation on macOS, what "linear Rec.2020 with an ICC
      profile" means in darktable/RawTherapee step by step, what a profile is) missing from
      the README?

## Phase 8: Synthesis (Opus, inline)

### Task 8.1: Write the report
- [ ] Verify every S1/S2 finding (reproduce it, or quote the code proving it). Mark the rest
      *confirmed* or *likely*, as reported.
- [ ] Merge duplicates across tasks, and rank.
- [ ] Write `docs/investigations/codebase-review.md`:
  1. Summary: the state of halide in a few paragraphs, and the ten findings that matter most.
  2. Requirements matrix (summary table; the full matrix in an appendix).
  3. Findings by area (accuracy, correctness, robustness, CLI UX, GUI UX, performance, code
     quality, tests, docs, packaging), each ranked.
  4. IDEAs, kept apart from the defects.
  5. What wasn't checked, and why (GPU on real hardware, macOS, Windows).
- [ ] Add a line to `docs/README.md`, commit on `review-2026-09`.
- [ ] Ask the user to run `pytest -m gpu` on the RTX 3070 against the branch.
- [ ] **Hand-off:** the user marks findings fix / won't fix / later. Those marked "fix" become
      `docs/plans/review-fixes.md`.

## Review Focus

The inputs most likely to bite a new photographer, which today's tests probably don't cover.
Tasks 4.3 and 2.3 must exercise each one through the real CLI:

1. **A TIFF exported with the wrong settings** (gamma-encoded, sRGB, 16-bit display-referred)
   from darktable or RawTherapee. Expect a clear error that names the export setting to change.
2. **Re-running into a folder that already has outputs, or output = input.** Expect no silent
   overwrite of a user's scan.
3. **Ctrl-C mid-batch.** Expect no orphan processes, no leftover temp folders or `/dev/shm`
   segments, no half-written TIFF that looks complete, and a sane terminal afterwards.
4. **A folder that isn't a clean roll** (mixed sizes, portrait and landscape, stray JPEGs, a
   contact sheet, hidden `._` macOS files). Expect them skipped with one clear line each, not a
   crash.
5. **Paths with spaces or non-ASCII characters, and the macOS config location.** Expect them to
   work in the CLI, the profile store, exiftool and the GUI.

## Execution order and cost

Phase 0 → (Phase 1 ∥ Phase 2) → checkpoint with the user on the ledger → Phase 3 ∥ 4 ∥ 5 ∥ 6
∥ 7 → Phase 8. That's about 14 agents: 7 Haiku extraction runs, 3 Opus (2.1, 2.4, and synthesis
inline), and the rest Sonnet. The review changes no source code, so a mistake in it costs a
wrong finding, which the S1/S2 verification in Task 8.1 catches.
