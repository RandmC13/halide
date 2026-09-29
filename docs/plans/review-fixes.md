# Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding (F01-F43) of `docs/investigations/codebase-review.md`, apply the
user's four decisions, and make the two contact-sheet changes the user asked for.

**Architecture:** 18 tasks grouped by theme, data safety first. Each task says which findings it
closes, which files it touches, the exact behaviour (including user-facing wording), and the
tests that pin it. Code is spelled out where the design matters. For the rest, the behaviour
and the tests are the contract, and the implementer matches the surrounding code's style.

**Tech Stack:** Python 3.11+, numpy, tifffile, colour-science, PySide6, Pillow, pytest.

**Spec:** `docs/investigations/codebase-review.md` (the findings) plus its evidence in
`docs/investigations/codebase-review-evidence/`. Each finding's file:line, reproduction and
suggestion are there, under the IDs in brackets.

**Branch:** `review-fixes`, created from `review-2026-09`. Merge only with the user's explicit
approval.

## User decisions (2026-09-28)

- **D-1 overwrite policy = (b).**
  - In a terminal: ask before overwriting existing outputs.
  - Not in a terminal: refuse.
  - `--overwrite` replaces existing outputs. `--skip-existing` develops only the missing ones.
  - Output = input is always a hard error, whatever the flags.
- **D-2 calibration Save gate.** Keep the 0.1 D gate. Add a warning when the fit is poorly
  constrained, and show the density range it's reliable over.
- **D-3 picker window.** The size on the user's 1920x1080 screen must not change: today that's
  1056x756 (55% x 70%, under the 1400x900 cap). The logic may be tidied, but a test must pin that
  size.
- **D-4.** `IMG_0158-positive.tif` has been deleted (done).
- **Contact sheet, user's notes.**
  - The half-frame number gets an arrow on its left, `→3A`, on the lower edge only.
  - The edge-code bars become a real DX-style two-track code. The reference is
    `contactsheet-barcoderef.jpg` (repo root, gitignored in Task 17), used **only** for the
    barcode. Do **not** copy its red rectangles or any other oddities.

## Global Constraints

- **Output must not move.** On clean input, CPU output stays **bit-identical** to `main`
  (`b52d6ab`). Any task that touches pixels proves it on the four real scans with Task 0's
  `compare_outputs.py`. The one allowed exception is Task 4's `--auto-density` change: measure it, report it,
  and the user must OK it.
- **Settled decisions.** CLAUDE.md "Decisions and why" stays settled apart from D-1/D-2 above.
  Record each new decision there in the same style (Task 17).
- **Dependencies.** No new runtime dependency. A vendored font file with its licence is an asset,
  not a dependency. No linter or formatter.
- **Start-up imports.** CLI start-up must not import numpy, tifffile, Pillow, colour or cupy
  (`tests/unit/test_cli_startup.py` enforces this).
- **Messages.** Plain photographer's language: what happened, then what to do next. Follow the
  existing ICC errors ("this profile is gamma-encoded … Re-export using your raw processor's
  linear gamma option"). Use the existing `console.error/warning/success` helpers and `RunSheet`
  rows. No new symbols or colours.
- **Test data.** Tests never touch the real `~/.config/halide`; use `tmp_path` and a monkeypatched
  profiles directory. Real scans are gitignored: never commit them. `/tmp` is RAM on the user's
  machine: delete large outputs straight away.
- **GUI verification.** Use Xvfb. Close windows with a real `WM_DELETE_WINDOW` message
  (`.review/scratch/verify/wmdelete.py`, or the same ctypes snippet), **not** `xdotool
  windowclose`, which destroys the X window and looks like a hang.
- **Models** (token economy): Tasks 5 and 10 on Opus (concurrency, calibration maths). Task 12's
  GUI verification and Task 16 on Sonnet. Everything else on Sonnet. Mechanical doc edits on
  Haiku.
- **Commits.** Commit after each task. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

Inputs a photographer will hit that no single task's tests exercise end to end. Task 18 runs
each one through the real CLI.

1. **Re-running a batch into last time's output folder.** In a terminal: one question for the
   whole roll. In a script: refusal naming `--overwrite`/`--skip-existing`. Never a scan
   overwritten, never a positive re-inverted without a warning.
2. **A roll copied from a Mac** (`._*.tif`, `.DS_Store`). Junk is skipped with one run-sheet line
   and no frame fails.
3. **A killed or interrupted run** (Ctrl-C, closed terminal, disk full). No truncated file under
   a real output name, no `/dev/shm` segments left, and the next run behaves normally.
4. **A profile name typed with a slash, a leading dash or different capitals.** A clear refusal
   or a "did you mean", the same in the CLI and the GUI.
5. **Piped or logged output** (`halide batch … > log.txt 2>&1`, `NO_COLOR=1`). Readable plain
   lines in the right order, no escape codes.

---

## Shared helpers created early (used by later tasks)

Task 1 creates `src/halide/cli/_output_policy.py`, Task 2 creates `src/halide/io/atomic.py` and
Task 3 extends `profile_store.py`. Later tasks import these rather than re-implementing them.

### Task 0: Branch and the output-comparison script

- [ ] **Branch.** `git checkout review-2026-09 && git checkout -b review-fixes`.
- [ ] **Write `.review/compare_outputs.py`** (not committed). It runs `main`'s halide (a
  `git worktree` of `b52d6ab` in `.review/main-wt`, run with that worktree's `src` first on
  `PYTHONPATH`) and the branch's halide on the four real scans, with `--device cpu` (plus
  `--overwrite` on the branch, once Task 1 exists), for each of:
  - `invert --rm 0.9 --bm 1.1 --rs 1 --bs 1`
  - `--output flat`
  - `--auto-density`
  - `print` of the flat file
  - `export`

  It compares pixels (tifffile arrays for TIFFs, PIL arrays for PNG) and reports "identical" or
  the max difference per output. It takes `--scans` and `--modes` to run a subset, and deletes
  every output as it goes.
- [ ] **Sanity check.** Run it once before any change: every row must say identical (the branch
  is `main` plus docs).

### Task 1: Output safety policy (F01, F02, D-1)

**Files:**
- Create: `src/halide/cli/_output_policy.py`
- Modify: `src/halide/cli/commands/invert_cmd.py`, `batch_cmd.py`, `print_cmd.py`,
  `export_cmd.py`, `contact_cmd.py`, `src/halide/cli/console.py` (`confirm_overwrite`),
  `src/halide/batch/orchestrator.py` (`discover_jobs` gains no policy; the CLI applies it)
- Test: `tests/unit/test_output_policy.py`, and additions to `tests/integration/test_invert_cli.py`,
  `test_batch_cli.py`, `test_export_cli.py`

**Interfaces:**
- Produces:
  - `add_output_policy_arguments(parser) -> None`: a mutually exclusive `--overwrite` /
    `--skip-existing` group.
  - `OutputPolicy` (enum: `ASK`, `OVERWRITE`, `SKIP_EXISTING`), and
    `policy_from_args(args) -> OutputPolicy`.
  - `check_not_input(pairs: list[tuple[Path, Path]]) -> None`: raises `SystemExit` with the
    message below.
  - `resolve_existing(pairs, policy, *, interactive: bool) -> list[tuple[Path, Path]]`: returns
    the pairs to process, or raises `SystemExit`.

- [ ] **Step 1: Write the failing tests** (`tests/unit/test_output_policy.py`):

```python
def test_output_equal_to_input_is_refused_even_with_overwrite(tmp_path):
    scan = tmp_path / "a.tif"; scan.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        check_not_input([(scan, tmp_path / "." / "a.tif")])
    assert "is the scan itself" in str(exc.value.code)

def test_output_equal_to_input_via_symlink_is_refused(tmp_path): ...   # os.path.samefile

def test_existing_outputs_non_interactive_refuses_and_names_both_flags(tmp_path):
    # two pairs, one output exists; policy ASK, interactive=False
    # -> SystemExit mentioning "1 of 2", "--overwrite" and "--skip-existing"

def test_existing_outputs_skip_existing_drops_them(tmp_path): ...     # returns only the missing pair
def test_existing_outputs_overwrite_keeps_all(tmp_path): ...
def test_existing_outputs_interactive_asks_once_for_the_whole_roll(tmp_path, monkeypatch):
    # monkeypatch console.confirm to record prompts; 3 existing outputs -> exactly one prompt,
    # whose text contains "3 of 3"
def test_interactive_decline_writes_nothing(tmp_path, monkeypatch): ...  # SystemExit, exit code 1
def test_no_existing_outputs_never_prompts(tmp_path, monkeypatch): ...
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `.venv/bin/python -m pytest tests/unit/test_output_policy.py -q`
  Expected: FAIL (the module doesn't exist).

- [ ] **Step 3: Implement.**

```python
"""What halide does when an output file already exists (the user's decision D-1, recorded in
CLAUDE.md): ask in a terminal, refuse in a script, --overwrite / --skip-existing to choose.
Writing over the scan being developed is refused in every mode."""
import enum, os, sys
from pathlib import Path
from halide.cli import console

class OutputPolicy(enum.Enum):
    ASK = "ask"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip-existing"

def add_output_policy_arguments(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true",
                       help="replace output files that already exist (and a saved profile of the same name)")
    group.add_argument("--skip-existing", action="store_true",
                       help="develop only the frames whose output doesn't exist yet - resumes an interrupted roll")

def policy_from_args(args) -> OutputPolicy:
    if getattr(args, "overwrite", False):
        return OutputPolicy.OVERWRITE
    if getattr(args, "skip_existing", False):
        return OutputPolicy.SKIP_EXISTING
    return OutputPolicy.ASK

def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a.resolve() == b.resolve()

def check_not_input(pairs) -> None:
    clashes = [out for src, out in pairs if _same_file(src, out)]
    if clashes:
        first = clashes[0]
        more = f" (and {len(clashes) - 1} more)" if len(clashes) > 1 else ""
        raise SystemExit(console.error(
            f"{first} is the scan itself{more} - halide never writes over a scan. "
            "Choose a different output folder, or add --suffix."))

def resolve_existing(pairs, policy: OutputPolicy, *, interactive: bool):
    existing = [(s, o) for s, o in pairs if o.exists()]
    if not existing or policy is OutputPolicy.OVERWRITE:
        return list(pairs)
    if policy is OutputPolicy.SKIP_EXISTING:
        return [(s, o) for s, o in pairs if not o.exists()]
    n, total = len(existing), len(pairs)
    where = existing[0][1] if n == 1 else existing[0][1].parent
    if interactive:
        if console.confirm(f"{n} of {total} outputs already exist in {where} - overwrite them?"):
            return list(pairs)
        raise SystemExit(console.error("Nothing was written. (--skip-existing develops only the new frames.)"))
    raise SystemExit(console.error(
        f"{n} of {total} outputs already exist in {where}. Add --overwrite to replace them, "
        "or --skip-existing to develop only the new frames."))
```

  For a single file, `n == 1` gives "1 of 1 outputs already exist in <path>". Word it for one
  file: "`<path>` already exists - overwrite it?". The plural form above is for bulk commands.
  Keep the "(s)" out of the wording (F22).

- [ ] **Step 4: Wire it into every command that writes files.** Call it after parsing and
  **before** any decoding, run sheet or worker start:
  - `invert` (one pair)
  - `batch` (from `discover_jobs`, including `--contact-sheet`'s sheet path when given)
  - `print` (single and bulk)
  - `export` (single and bulk)
  - `contact` (the sheet path)

  `interactive = sys.stdin.isatty() and sys.stdout.isatty()`. Remove
  `console.confirm_overwrite`'s silent non-interactive `return True`, and make its callers use
  `resolve_existing`. Delete `confirm_overwrite` if nothing else calls it. With
  `--skip-existing`, the run sheet shows a row: `Skipping    3 frames already developed`.

- [ ] **Step 5: Integration tests.** Build a synthetic roll with the helper the existing
  integration tests use:
  - `test_batch_into_its_own_folder_is_refused_before_any_work`: md5 unchanged, exit 1, message
    contains "is the scan itself".
  - `test_invert_output_equal_to_input_refused_non_interactive`.
  - `test_batch_rerun_non_interactive_refuses_then_skip_existing_develops_only_new`.
  - `test_export_bulk_rerun_refuses_without_overwrite`.

- [ ] **Step 6: Run the full suite.** `.venv/bin/python -m pytest tests/ -q`. Expected: all
  pass. Update any existing test that relied on silent overwriting to pass `--overwrite`, and
  say so in the commit message.

- [ ] **Step 7: Commit.** `git commit -m "Never write over a scan; ask/refuse/--overwrite/--skip-existing for existing outputs (F01, F02, D-1)"`

### Task 2: Atomic writes (F04)

**Files:**
- Create: `src/halide/io/atomic.py`
- Modify: `src/halide/processing.py` (every output write: develop, print, export),
  `src/halide/io/contact_sheet.py` (`write_sheet`), `src/halide/calibration/profile_store.py`
  (every JSON write)
- Test: `tests/unit/test_atomic.py`, additions to `tests/unit/test_profile_store.py`

**Interfaces:** Produces `atomic_output(path: Path) -> ContextManager[Path]`.

- [ ] **Step 1: Failing tests.**
  - `test_success_replaces_target_and_leaves_no_temp`.
  - `test_exception_inside_leaves_target_untouched_and_removes_temp`: the target holds old
    bytes before, and still holds them after `raise RuntimeError` inside the block.
  - `test_temp_is_hidden_same_folder_same_suffix`: the name starts with ".", ends with ".tif"
    (exiftool picks the file type from the extension), and sits in the target's folder.
  - `test_keyboardinterrupt_also_cleans_up`.
  - Profile store: `test_interrupted_profile_save_keeps_old_profile` (monkeypatch `json.dump`
    to raise).

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.**

```python
@contextmanager
def atomic_output(path: str | Path):
    """Yield a hidden temporary path beside `path`; on success move it over `path` in one step
    (os.replace), so a killed or failed write never leaves a truncated file under the real name.
    The temp keeps the real suffix (exiftool picks the file type from it) and starts with "."
    (roll discovery skips hidden files, so a leftover is never mistaken for a frame)."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.halide-partial-{os.getpid()}{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: Route every output through it.** The TIFF write, `set_description` and
  `copy_exif_metadata` all run on the temp path, then the temp is replaced over the real name.
  If exiftool fails after a good write (2.3-4), that isn't fatal:
  - Catch `ExifToolError`, keep the file, and move it into place.
  - The frame's result carries a warning: "⚠ `<name>`: developed, but its camera metadata
    couldn't be copied (`<exiftool's reason>`)."
  - A batch lists these at the end, like other warnings.

  Also atomic: export PNG/JPEG, contact sheets and profile JSON.

- [ ] **Step 5: Bit-identity.** Output bytes must not change. Run Task 0's
  `compare_outputs.py` on IMG_0156 (invert, flat, export). Expected: identical.

- [ ] **Step 6: Full suite, then commit.**
  `git commit -m "Write every output atomically; exiftool failure after a good write is a warning (F04)"`

### Task 3: Profile store safety (F03, F24, 2.2-4, 2.2-5, 2.2-8, 2.6-1, 2.6-5, case-insensitive half of 2.2-12)

**Files:**
- Modify: `src/halide/calibration/profile_store.py`, `src/halide/cli/commands/profile_cmd.py`,
  `src/halide/cli/_calibration_args.py` (`--save-profile-as`),
  `src/halide/gui/main_window.py` (`SaveProfileDialog`)
- Test: `tests/unit/test_profile_store.py`, `tests/integration/test_profile_cli.py`

**Interfaces (produces):**
- `class ProfileError(ValueError)`, with subclasses `ProfileNameError` and
  `ProfileExistsError(name, existing_path)`.
- `validate_profile_name(name: str) -> str`: returns the stripped name or raises
  `ProfileNameError`.
- `find_profile(name, profiles_dir=None) -> Path | None`: an exact match first, then a
  case-insensitive (`casefold`) match.
- `save_named_profile(..., overwrite: bool = False)` raises `ProfileExistsError` when the name
  (compared case-insensitively) exists and `overwrite` is False.
- `list_profiles()` returns `(name, DensityProfile | None, problem: str | None)` triples:
  unreadable or damaged files are listed with a reason and never crash the listing. Update
  every caller.
- `suggest_profile_name(name, profiles_dir=None) -> str | None`: move the existing "did you mean"
  helper from `_calibration_args.py` here so every profile command uses it.

Name rules (and their messages):

| Input | Message |
|---|---|
| empty or only spaces | "Profile names can't be empty." |
| contains `/`, `\` or NUL | "Profile names can't contain / or \\ - they name a file in halide's profiles folder." |
| starts with `.` or `-` | "Profile names can't start with '.' or '-'." |
| longer than 64 characters | "Profile names can be at most 64 characters." |
| `.` or `..` | (covered by the leading-`.` rule) |

- [ ] **Step 1: Failing tests.** One test per rule above. Then:
  - `test_dotdot_name_cannot_touch_files_outside_profiles_dir`
  - `test_saving_existing_name_raises_unless_overwrite`
  - `test_saving_name_differing_only_by_case_counts_as_existing`
  - `test_list_profiles_reports_unreadable_file_without_crashing` (chmod 000 skipped under root)
  - `test_damaged_profile_json_gives_plain_message` ("profile 'X' is damaged … re-save it from
    `halide calibrate --profile X`, or delete it")
  - `test_profile_delete_non_interactive_requires_yes`
  - `test_profile_delete_interactive_confirms` (the prompt names the profile, its points count
    and its date)
  - `test_profile_rename_onto_existing_refused`
  - `test_profile_show_typo_suggests` (same for rename/edit/delete)
  - `test_rename_with_a_path_explains_names_only` ("`profile rename` takes a profile name - file
    paths work with `profile show` and `--profile`")
  - `test_save_profile_as_existing_non_interactive_needs_overwrite`, checked **before** any frame
    is developed.

- [ ] **Step 2: Implement.**
  - `profile delete` gets `--yes`. `--save-profile-as` uses Task 1's `--overwrite`, or a terminal
    prompt: "A profile named 'X' already exists - replace it?". The name is validated at
    argument-parsing time.
  - `profile list` shows problem entries dimmed, as `name   (unreadable: <reason>)`.
  - Every profile write goes through `atomic_output` (Task 2).

- [ ] **Step 3: GUI Save dialog** (2.6-1).
  - Validate as the user types: the Save button is disabled and a one-line reason shows under
    the field, in the theme's warning colour.
  - Wrap the save in `try/except (ProfileError, OSError)` and show a `QMessageBox.warning` with
    the plain message.
  - The existing overwrite confirmation stays. It now keys on `find_profile`, so a name differing
    only by case also asks.

- [ ] **Step 4: Full suite, then commit.**
  `git commit -m "Validate profile names, confirm overwrite/delete, never crash on a damaged profile (F03, F24)"`

### Task 4: Non-finite and non-positive pixels (F05)

**Files:**
- Modify:
  - `src/halide/io/icc.py` or `processing.py` (the banded working-space conversion in
    `load_working_space_image`)
  - `src/halide/calibration/auto.py` (`_saturation` and `_neutral_candidate_mask`)
  - `src/halide/processing.py` (`provenance_json`)
  - `src/halide/core/tone_render.py` (`estimate_linear_scale` / `fit_print` guards)
- Test: `tests/unit/test_nonfinite.py`

**Behaviour:**
- **Non-finite pixels.** At the working-space boundary, count non-finite values per band and
  replace them with 0 using `np.nan_to_num(band, copy=False, nan=0.0, posinf=0.0, neginf=0.0)`.
  Apply this **only if the band has any**, so clean input stays bit-identical. The reciprocal's
  existing `MIN_TRANSMITTANCE` floor then prints them as the brightest white, and the percentile
  fits are robust to a few.
- **Warning:** "⚠ `<name>`: N pixels weren't valid numbers (NaN/inf) and were treated as clear
  film. If this is more than a handful, check the raw converter's export." Use a batch-result
  warning in batch and a warning line in invert.
- **Too many:** if more than 1% of pixels are non-finite, fail that frame with "`<name>`: X% of
  its pixels aren't valid numbers - this export looks broken; re-export it from the raw
  converter."
- **Auto calibration:** non-positive or non-finite pixels are **excluded** from neutral
  candidates (they're masked out; no longer given saturation 0).
- **Guards:** `estimate_linear_scale` and `fit_print` raise a clear `ValueError` if a statistic
  comes out non-finite (defence in depth). `provenance_json` uses `allow_nan=False`.

- [ ] **Step 1: Failing tests.**
  - `test_one_nan_pixel_invert_succeeds_and_warns`
  - `test_one_inf_pixel_flat_output_is_finite`
  - `test_over_one_percent_nonfinite_fails_frame_with_plain_message`
  - `test_auto_density_ignores_nonpositive_pixels`: reuse verified.md's synthetic frame (known s =
    1.3/1/0.78). With 0.1% of pixels at −0.001, the solved scale is within 0.01 of the clean
    solve.
  - `test_auto_density_single_nan_gives_finite_profile`
  - `test_provenance_rejects_nan`
  - `test_clean_input_bit_identical_to_before`: hash a synthetic frame's invert output before and
    after the change.

- [ ] **Step 2: Implement.** Then run the tests. Expected: PASS.

- [ ] **Step 3: Real scans.**
  - The manual-calibration and profile paths must be bit-identical on all four scans
    (`compare_outputs.py`).
  - For `--auto-density`, report the old and new solved profile for each scan (the scans have
    2-18 non-positive pixels, so a change of ~1e-4 or less is expected), plus the max pixel
    difference.
  - **Stop and show the user these numbers before committing** (Global Constraints exception).

- [ ] **Step 4: Commit.**
  `git commit -m "Handle NaN/inf and non-positive pixels: warn, stay finite, keep them out of auto calibration (F05)"`

### Task 5: Concurrency and exit paths (F06, F12, 2.4-9, F17 timeout) — Opus

**Files:**
- Modify: `src/halide/gpu_service.py` (`ServiceClient._request`),
  `src/halide/batch/orchestrator.py` (`_run_pool`, signal handling, start-of-run sweep),
  `src/halide/shared_frames.py` (sweeping stale prefixes), `src/halide/io/exiftool.py` (the
  one-shot fallback timeout)
- Test: `tests/unit/test_gpu_service.py`, `tests/unit/test_shared_frames.py`,
  `tests/unit/test_exiftool.py`, `tests/unit/test_orchestrator.py`

**Behaviour:**
- **F06:** `_request` treats **any** `BaseException` during send/poll/recv as a dead connection:
  `self._give_up(..., host_touched=True)`, then re-raise the original. Regression test modelled on
  `.review/scratch/2.4/desync.py` (in the evidence folder's 2.4.md). Interrupt request A with a
  `KeyboardInterrupt`, then assert request B raises `ServiceUnavailable` rather than returning A's
  reply.
- **F12, SIGHUP/SIGTERM:** the batch parent turns SIGHUP and SIGTERM into the same orderly
  cancel path as Ctrl-C: stop the pool, stop the service, run `sweep(prefix)`. A whole-group
  SIGKILL can't be handled, so at the **start** of every GPU batch, sweep segments whose
  `batch_prefix` pid is no longer alive (`psutil.pid_exists`). Test: create a fake segment with a
  dead pid in its prefix, start a batch, and assert it's gone.
- **2.4-9:** Ctrl-C in the first second of a batch prints the normal "Cancelled" line and no
  forkserver tracebacks. Suppress `KeyboardInterrupt` output in the workers and forkserver
  children (a worker initializer that sets `signal.SIGINT` to `SIG_IGN`; the parent owns
  cancelling). Test through `subprocess` with a SIGINT sent 0.3 s after start, asserting
  "Traceback" is not in stderr.
- **F17:** the one-shot exiftool fallback gets the same timeout as a session request. On
  timeout, kill it and raise `ExifToolError("exiftool didn't answer within Ns")`, which Task 2
  turns into the "metadata couldn't be copied" warning. Test with a fake exiftool script that
  sleeps forever.

- [ ] Steps: write the tests, see them fail, implement, run `pytest tests/unit/test_gpu_service.py
  tests/unit/test_shared_frames.py tests/unit/test_exiftool.py tests/unit/test_orchestrator.py -q`,
  then the full suite, then commit:
  `"Interrupt-safe GPU service client, cleanup on SIGHUP/SIGTERM, no hanging exiftool (F06, F12, F17)"`.

### Task 6: ICC validation (F08)

**Files:** Modify `src/halide/io/icc.py`. Test `tests/unit/test_icc.py`.

**Behaviour:**
- **Linearity (2.1-3).** Judge the TRC in **density**, not linear light. The profile's curve
  must map transmittance t to t within 0.005 D (`abs(log10(curve(t)/t)) <= 0.005`) for t in
  [0.001, 1]. This rejects the 0.004 black offset and gamma 1.013 from the finding. Real
  linear profiles (the four scans, halide's own ACEScg, RawTherapee's and darktable's linear
  Rec.2020) must still pass: add each as a fixture if a copy exists in the tests' assets;
  otherwise the real scans cover it.
- **Malformed profiles (2.1-4).** Bounds-check every tag offset and size against the profile's
  length before unpacking. Anything out of bounds raises
  `ScanColorError("the embedded colour profile is damaged (<tag> runs past the end); re-export the scan")`.
  It never crashes with a `struct.error` and never reads the next tag's bytes.
- **White point (2.1-5).** The sum of the rXYZ/gXYZ/bXYZ columns must equal the D50 PCS white
  within 0.002 in each of X, Y and Z. Otherwise:
  `ScanColorError("the embedded colour profile isn't adapted to D50 as ICC requires, so its colours would come out with a cast; re-export with a standard profile")`.
- **Tests:** a truncated profile at every tag boundary (parametrised); a gamma-1.013 curve; a
  black offset of 0.004; unadapted primaries; the real scans still accepted.
- **Commit:** `"Tighter, damage-proof ICC validation (F08)"`.

### Task 7: Input handling, folder discovery and the re-invert guard (F09, F15, F16, F25, 2.5-10)

**Files:**
- Create: `src/halide/io/roll.py`, holding **the** one folder-discovery function. It replaces
  the three copies in `batch/orchestrator.py::discover_jobs`, `cli/commands/contact_cmd.py`,
  `print_cmd.py`/`export_cmd.py` and `gui/roll.py`; grep for `iterdir()` and `.tif` suffix
  checks.
- Modify: `src/halide/processing.py` (`_read_scan`, the develop request), `cli/main.py` (the
  error boundary), `cli/commands/check_cmd.py`, `cli/_run_sheet.py`
- Test: `tests/unit/test_roll_discovery.py`, integration tests

**Interfaces (produces):**
`list_scans(folder: Path) -> tuple[list[Path], Skipped]`, where `Skipped` holds counts by
reason: hidden or AppleDouble, a halide contact sheet, not a TIFF, and a subfolder.
- Accepts `.tif`, `.tiff`, `.TIF`, `.TIFF`.
- Skips names starting with `.` (this covers `._*` and Task 2's partial files).
- Sorted by name, the same order everywhere.

**Behaviour:**
- **Skipped files.** The run sheet (batch, print, export, contact, check) shows one row when
  anything was skipped, for example: `Skipped     2 hidden macOS files, 1 JPEG, 1 folder`.
  It never shows one line per file.
- **Pre-flight checks** before any work, each a plain error naming the path and what to do:
  - the input exists and is a file (or a folder, for bulk commands)
  - the output folder exists and is writable (`os.access`); a missing output folder for batch is
    created, as today; check what each command does now and keep it
- **Input errors.** Wrap `read_tiff`/validation errors in `ScanInputError` with plain messages:
  - Not a TIFF, or unreadable: "`<name>` isn't a TIFF halide can read (it looks like a JPEG /
    raw file). halide develops linear TIFFs exported from darktable or RawTherapee - see the
    README's 'Exporting your scans'."
    Detect the kind from magic bytes: JPEG `FF D8`; raw formats by extension (`.nef .cr2 .cr3
    .arw .raf .dng .orf .rw2`).
  - RGBA: "`<name>` has an alpha (transparency) channel; export without it."
  - Greyscale: "`<name>` is greyscale; halide needs an RGB scan of a colour negative."
  - Other channel counts: "`<name>` has N channels; halide needs RGB."

  The generic "unexpected error" boundary in `cli/main.py` stays for real bugs only.
- **F15.** `invert`/`batch` read the provenance (as `print_scan` already does). A file that is
  already a halide output fails with: "`<name>` is already a halide positive (made on `<date>`).
  To re-print it use `halide print`; to develop again, point halide at the original scan." A
  batch reports it per frame.
- **F16.** `halide check` and batch's Scans row show frames without scan EXIF as "not
  verifiable (no camera EXIF in N frames)", as a ⚠ row, never as "consistent". In manual
  calibration, a picked point from a frame without scan EXIF, on a roll whose other frames have
  it, gets a note in the point list and on the CLI: "point N: frame has no scan exposure data,
  used unnormalised" (2.1-10).
- **Tests:** discovery (every skip reason, case of the suffix, sort order); each input error
  message; `test_reinverting_a_halide_output_is_refused`;
  `test_check_reports_no_exif_as_not_verifiable`;
  `test_unwritable_output_folder_fails_before_developing` (skipped when running as root).
- **Commit:** `"One roll discovery, plain-language input errors, refuse re-inverting a positive (F09, F15, F16, F25)"`.

### Task 8: Terminal etiquette, the finish line, GPU messages, wording pass (F10, F20, F19, F22)

This task also does the F19 and F22 items listed under "Coverage map" at the end of this plan,
because they share the console helpers.

**Files:**
- Modify: `src/halide/cli/console.py` (`Style`, the helpers), `src/halide/batch/progress.py`
  (`GridProgressRenderer.finish` plus a plain renderer), `src/halide/cli/main.py`
  (line-buffered stdout)
- Test: `tests/unit/test_console.py`, `tests/unit/test_progress.py`

**Behaviour:**
- **F10.** `finish()` takes the command's noun. It prints "Developed 36 of 37 frames (1 failed)
  in 42s", "Exported 12 files in 8s", "Printed …", and "Contact sheet complete" only for contact
  sheets. The success count must be right when there are failures (4-2).
  Test: `test_finish_message_per_command_and_with_failures`.
- **Colour.** `console.use_color()` is `False` when `NO_COLOR` is set (any value), when
  `TERM=dumb`, or when stdout isn't a terminal. `FORCE_COLOR` overrides. Every `Style` use goes
  through it.
  Tests: `NO_COLOR=1` gives no `\033` in any output; piped output has no `\033`.
- **Non-terminal progress.** When stdout isn't a terminal (or `TERM=dumb`), batch, export, print
  and contact use a plain renderer: one line per finished frame (`✓ IMG_0138.tif  grade 0.88
  exp +0.39  4.1s`) and the same end line. No cursor movement.
- **Log order.** `sys.stdout.reconfigure(line_buffering=True)` at CLI start, so a logged run
  keeps the order.
- **`--quiet`.** It keeps warnings and errors, and prints the one end line (4-19).
- **Commit:** `"Right finish message, NO_COLOR/pipes/TERM=dumb, plain progress when logged (F10, F20)"`.

### Task 9: Help, version, examples and number checks (F14, F21)

**Files:** `src/halide/cli/main.py`, `cli/commands/*.py`, `cli/_calibration_args.py`,
`cli/console.py` (`help_banner`). Test `tests/unit/test_cli_help.py`.

**Behaviour:**
- **`--version`** prints `halide 0.1.0` (from `importlib.metadata`, falling back to the package
  `__version__`).
- **Examples.** Every subcommand's `--help` ends with an "examples:" epilog of 2-3 real lines,
  using `RawDescriptionHelpFormatter` and the banner's style. Write them from the README's
  workflows.
- **`profile` subcommands** each get a one-sentence description and an example (4-10).
- **Formatting slips.** Fix `calibrate --help`'s truncated `--profile` text (4-11) and the
  banner's last-line misalignment (4-12). Replace "Fine-tune" with "Print" everywhere
  (2.1-12; grep).
- **Defaults.** State them consistently: `--rm`/`--bm` "(default: 1.0)" like `--rs`/`--bs`.
  `--workers` gets identical help text on every command, from one shared definition.
- **`--debug`** is accepted after the subcommand too (2.5-16).
- **F14 range checks** (argparse `type=` functions with plain messages):
  - `--contrast` in (0, 2]
  - `--exposure` finite
  - `--rm/--bm/--rs/--bs` > 0 and finite
  - `--workers` ≥ 1

  In `fit_density_balance`, reject any fitted scale outside [0.2, 5] with a plain `ValueError`
  (2.1-8): "the neutral points don't pin down the <colour> layer; pick points further apart in
  density".
- **Tests:**
  - the `--version` output
  - every subcommand's help contains "examples:"
  - each range error
  - `test_near_flat_channel_rejected`
  - a parametrised test that every command's `--workers` help text is identical
- **Commit:** `"--version, examples in every --help, consistent defaults, range checks (F14, F21)"`.

### Task 10: Calibration honesty (F07/D-2, F18, R-050) — Opus

**Files:** `src/halide/calibration/anchors.py`, `core/density.py` (`fit_density_balance`
returning its uncertainty), `gui/point_list.py`/`step_wedge.py`/`main_window.py`,
`cli/_calibration_args.py`. Test `tests/unit/test_anchors.py`.

**Behaviour:**
- **D-2.** Keep `MIN_DENSITY_SEPARATION = 0.1` as the Save gate. Add
  `fit_reliability(anchors, wedge_range) -> Reliability(worst_cc_at_ends: float,
  reliable_range: tuple[float, float])`:
  - Propagate an assumed pick error of 0.005 D per channel through the least-squares fit. Use
    the lstsq covariance: `sigma² (AᵀA)⁻¹`, then the predicted error at D_G.
  - Report the predicted worst-end CC error over the roll's wedge range, and the D_G range where
    it stays ≤ CC 5.
  - Reproduce 2.1-1's table as a test: 3 points spanning 0.1 D gives a predicted worst-end
    error > CC 5; spanning 0.6 D gives < CC 5.
- **Where the warning shows** (when the worst-end prediction exceeds CC 5, the amber band's
  threshold):
  - GUI: an amber note under the step wedge, "Fit reliable over D 0.9-1.3 only - add a point
    in the shadows or highlights for the ends of the roll", and the reliable range drawn on the
    wedge as a bracket.
  - CLI `--save-profile-as` with manual picks: the same text as a ⚠ run-sheet row.
  - Saving is still allowed.
- **F18.** Rewrite `describe_cast` to give a real filter pack. Express the cast as CMY densities
  and remove neutral density (the minimum component). Name the one or two filters left, sized
  in CC units: (+0.1, 0, −0.1) → "CC 20Y + 10M". Dichroic convention: no filter of the colour
  itself for R/G/B; use Y/M/C pairs, as darkroom printers dial them. Table-driven tests covering
  all six hues, a two-filter case and a neutral case ("neutral").
- **R-050.** When `--auto-density` or `--auto-density-roll` is used, the run sheet's Calibration
  row adds, dimmed: "automatic estimate - for the most faithful colour, pick neutral points with
  `halide calibrate`".
- **Commit:** `"Warn when a calibration is only reliable over part of the roll; real filter-pack casts (F07, F18)"`.

### Task 11: Tab completion fixes (F23)

**Files:** `src/halide/cli/completion.py`. Test `tests/unit/test_completion.py`.

- **Opt-out first.** `HALIDE_NO_COMPLETION` and the tty checks run **before** `detect_shell()`
  and before psutil is imported. Test: with the variable set, `psutil` isn't imported
  (`sys.modules`).
- **Stamp after success.** The per-shell stamp is written only after the rc edit succeeds. A
  failed edit prints its note once per run until it succeeds, or until `HALIDE_NO_COMPLETION`
  is set.
- **Symlinks.** `_write_if_changed` follows a symlink and writes through it
  (`path.resolve()`), so managed dotfiles stay managed. Test with a symlinked fish file.
- **Commit:** `"Completion: opt-out skips all work, retry failed rc edits, keep symlinks (F23)"`.

### Task 12: GUI fixes (F11, F30, F31/D-3)

**Files:** `src/halide/gui/main_window.py`, `proof_window.py`, `filmstrip.py`, `theme.py`,
`step_wedge.py`, `loaders.py`. Test: `tests/unit/test_gui_*.py` (offscreen Qt), plus Xvfb checks.

- **F11:**
  - Keep a strong reference to the current proof window, and compare identity rather than
    relying on the deferred `destroyed` signal. Test: rebuild twice, change a point, and the
    window is marked out of date.
  - The rebuild never calls `QThread.wait()` on the UI thread: stop the old loader
    asynchronously and ignore its late signals by generation number.
  - Full-resolution frame loads are capped at 1 in flight plus 1 queued (latest wins) while the
    filmstrip is scrubbed.
- **F30:**
  - HiDPI: custom painting multiplies pixmap sizes by `devicePixelRatioF()` and calls
    `setDevicePixelRatio` on the result.
  - A visible focus ring on focusable widgets (theme QSS `:focus`, using the existing accent
    colour).
  - The status line elides with `QFontMetrics.elidedText`, and the full text goes in its tooltip.
  - "measuring the roll…" clears after "Continue without".
  - Repaint the panel after a modal dialog closes (the "JTRAL POINTS" glitch).
  - A frame that fails to load shows a ⚠ over its filmstrip thumbnail, with the reason as a
    tooltip.
  - Move the two hard-coded hex colours in `filmstrip.py` into `theme.py`.
- **F31/D-3:**
  - Keep `_WINDOW_MIN`/`_WINDOW_MAX` and the 55%/70% rule. Add
    `test_window_size_on_1920x1080_is_1056x756` (it calls the sizing function with a
    1920x1080 available rect) and one for 2560x1440 recording today's result.
  - Comment the cap: "on large screens the picker stays a comfortable working size rather than
    filling the screen".
  - Record the cap in CLAUDE.md (Task 17).
- **Xvfb check** at 1366x768, 1920x1080 and 2560x1440 (plus `QT_SCALE_FACTOR=2`):
  - screenshots of the rebuild flow and the focus ring
  - close via `WM_DELETE_WINDOW`
  - look at each screenshot
- **Commit:** `"GUI: contact sheet rebuild tracking, non-blocking loads, HiDPI, focus ring, status elide (F11, F30, F31)"`.

### Task 13: Contact sheet look (user's notes, F35 fonts)

**Files:**
- Modify: `src/halide/io/contact_sheet.py` (`_barcode`, the edge-print layout around lines
  266-282, `_font`), `src/halide/gui/theme.py`
- Create: `src/halide/assets/fonts/` (DejaVu Sans, Sans Bold and Sans Condensed Bold `.ttf`,
  plus `LICENSE-DejaVu.txt`)
- Test: `tests/unit/test_contact_sheet.py`

**Behaviour:**
- **Arrow.** Lower-edge half-frame numbers read `→3A` (an arrow, a thin space, then the number),
  and the top edge has no arrow. That matches real film: the arrow points along the film towards
  the next frame. Draw the arrow as a small filled triangle plus a shaft in the edge-print
  colour, not a font glyph, so it doesn't depend on font coverage.
- **DX-style two-track edge code**, replacing `_barcode`'s random rectangles:
  - **Clock track** (the lower ~55% of the code height): narrow bars of one width `u`, spaced
    evenly with gaps of `u`. That gives the regular comb visible in the reference.
  - **Data track** (the upper ~45%): one cell per clock bar **and** per gap (2x the clock
    resolution). A set cell draws a block over that cell's width, in the data track only.
    Adjacent set cells merge into one wide block. Where a data block sits over a clock bar, the
    two join into a tall bar, which is what produces the `m`, `Y`, `n` and `h` shapes in the
    reference.
  - **Start mark:** a data block spanning the first 3 clock bars (the wide `m`).
  - **End mark:** a data block joining the last 2 clock bars (`n`/`h`).
  - The data bits between them are deterministic per frame (the existing seed), with roughly 40%
    of cells set, and never more than 3 set cells in a row (the reference never shows a solid
    run).
  - Code blocks sit between the numbers, in the reference's order along one frame's lower edge:
    `N  [code]  →NA  [code]  N+1`.
  - Keep `_EDGE_PRINT`'s orange and the black rebate. No red rectangles, and nothing else from
    the reference.
- **Fonts.** `_font` and the GUI load the vendored DejaVu by path
  (`importlib.resources.files("halide.assets") / "fonts" / …`), falling back to the current
  behaviour only if the file is missing. Check `pyproject`'s package-data includes `*.ttf`: build a wheel into
  `.review/` and list it, and confirm the `.ttf` files and the licence are inside.
- **Tests:**
  - `test_half_frame_number_has_arrow_on_lower_edge_only`: render a 2-frame sheet and check the
    arrow's pixels exist left of the `A` number's box. Expose the box positions from a layout
    helper so the test doesn't scan pixels blindly.
  - `test_edge_code_has_clock_track_and_merged_data_blocks`: render the code into a small image,
    and check that its lower band has a regular period of 2u and its upper band has runs of
    widths that are multiples of u.
  - `test_edge_code_deterministic_per_frame`.
  - `test_fonts_load_from_package`.
- **Visual check.** Render a sheet from the real contact-sheet preview (`batch --contact-sheet`
  on 6-12 Roll16 frames). Crop the lower edge at 4x and compare side by side with the crops of
  `contactsheet-barcoderef.jpg` (method in the plan history: `convert -crop … -filter point
  -resize 400%`). **Show the user the before/after crop before committing.**
- **Commit:** `"Contact sheet: arrowed half-frame numbers, DX-style two-track edge code, vendored font"`.

### Task 14: Dependencies and macOS readiness (F13, rest of F35)

**Files:** `pyproject.toml`, `src/halide/calibration/profile_store.py`
(`default_profiles_dir`), `src/halide/shared_frames.py` (`batch_prefix`, frame names), `README.md`.

- **Version bounds.** Pin to what's installed, with the next major excluded: read the installed
  versions from `.venv/bin/pip freeze`. For example `numpy>=2.1,<3`, `colour-science>=0.4.6,<0.5`,
  `tifffile>=2024.8`, `PySide6>=6.7,<7`, `pillow>=10.1`. Use the real installed versions as lower
  bounds. The README lists the tested versions.
- **macOS config.** `default_profiles_dir()` on macOS (`sys.platform == "darwin"`) uses
  `~/Library/Application Support/halide/profiles` unless `XDG_CONFIG_HOME` is set, in which case
  it follows XDG as today. Linux is unchanged. Test by monkeypatching `sys.platform`.
- **Shared-frame names** are at most 30 characters: `hl` + 6-character base36 pid + 4-character
  token + `-` + frame index. Test: the longest possible name is ≤ 30.
- **Commit:** `"Bound dependency versions; macOS profile folder; short shared-frame names (F13, F35)"`.

### Task 15: Code quality (F40)

**Files:** as listed in F40. Behaviour-neutral; bit-identical outputs.

- **`core/` file I/O.** `core/tone_render.py` no longer reads files. `load_1d_cube` moves to its
  callers (`processing.py` and the GUI's render), which pass the loaded `Cube1D` into
  `tone_render` functions. Keep a module-level cache in `io/lut.py` so the file is read once per
  process. Test: `core/` contains no `open(`, `Path(…).read`, `load_` from `io`, or `print(`
  (a grep-based test).
- **One copy of each shared check.** The "manual calibration given?" check becomes one function
  in `_calibration_args.py`, used by all four callers. Calibration-source exclusivity moves to
  an argparse mutually exclusive group where the flags allow it; keep the hand check only where
  argparse can't express it, with a comment saying why.
- **Split large files where there's a seam.** `console.py` splits into `console/style.py`
  (colours, symbols, rules), `console/prompts.py` (confirm, menu, prompt_line) and
  `console/runsheet.py`, with `console/__init__.py` re-exporting the current names so imports
  don't change. Leave `processing.py`, `orchestrator.py` and `main_window.py` whole unless a
  seam is obvious and small. Don't split for its own sake.
- **Fix docstrings:**
  - stale ones (2.1-13)
  - `roll_auto_density_balance` (7.1-4)
  - `save_profile`'s path (7.1-3)
  - `core/pipeline.py`'s module docstring (7.1-5)
- **Other small items:**
  - The `.cube` reader honours `LUT_1D_SIZE`, and gives plain errors for a bad or empty file
    (2.1-14).
  - The `gpu --install` prompt catches `EOFError` (2.5-15).
  - The GPU plumbing items in 2.4-12.
- **Verify.** Full suite, then `compare_outputs.py` on all four scans must come out identical.
- **Commit:** `"Code quality: core/ does no I/O, one copy of shared checks, console split, docstrings (F40)"`.

### Task 16: Tests (F41)

**Files:** `tests/`.

- **`test_gpu_cmd.py` (0-1).** Fake the pip-availability probe, so the tests pass in a venv
  without pip. Verify in `.review/venv`.
- **Leaking semaphores (3b-1).** Find which test leaks multiprocessing semaphores into `/dev/shm`
  (run `test_orchestrator.py` then `test_gpu_service.py`) and fix the leak: close the pools or
  managers it opens. Make the cleanliness fixture ignore `sem.mp-*` only if the leak is in
  Python itself; say which in the commit.
- **`halide check` integration tests (0-2)** on synthetic rolls: consistent; mixed exposure; mixed
  white balance; no EXIF (Task 7's wording).
- **Picker smoke test** (offscreen Qt, 3b-3): open a 3-frame synthetic roll, add two points
  programmatically, switch Negative/Positive, open each drawer, save a profile, close. Then
  assert the profile's contents and that no thread is still running.
- **Pin the unpinned requirements** (R-054, R-077, R-090, R-092, R-096, R-105), one small test
  each (see requirements-matrix-a/b for what each needs).
- **Hostile-input tests** from 2.1-15 that Tasks 4, 6 and 7 didn't already add.
- **Commit:** `"Tests: host-independent gpu_cmd, no semaphore leak, check and picker coverage (F41)"`.

### Task 17: Documentation (F42) and the decision record

**Files:** `README.md`, `docs/README.md`, `docs/plans/gpu-acceleration.md`, `CLAUDE.md`,
`.gitignore`.

- **README additions:**
  - "Exporting your scans": menu-by-menu steps for darktable (export module: TIFF, 32-bit float,
    profile "linear Rec2020 RGB", no tone/colour modules active in the history) and RawTherapee
    (Color Management → Output profile "RTv4_Rec2020" with linear TRC, 32-bit float TIFF).
    **Check each menu name in the current versions' documentation**, and mark anything
    unverified.
  - "What you get": print vs flat, what the provenance records, one line on the contact sheet.
  - "Troubleshooting": wrong-export errors, missing exiftool, Qt libraries, the GPU fallback,
    where profiles live on Linux and macOS.
  - Profile subcommands in `--help` order.
- **Stale branch mentions.** `docs/README.md` and `docs/plans/gpu-acceleration.md` no longer
  say "on branch gpu-acceleration".
- **`.gitignore`:** add `contactsheet-barcoderef.jpg` next to `enlarger-controller.png`.
- **CLAUDE.md:** add or adjust "Decisions and why" entries for:
  - D-1, the overwrite policy, and the atomic writes
  - D-2's reliability warning
  - D-3's window cap
  - NaN handling and the auto-calibration exclusion
  - one roll discovery
  - `NO_COLOR` and the plain progress renderer
  - the vendored font and the DX edge code
  - the macOS profile folder
  - the dependency bounds

  Update the "Commands" block (`--overwrite`, `--skip-existing`, `--version`, `profile delete
  --yes`). **Don't** restructure CLAUDE.md's length now (the 7.1 IDEA): ask the user
  separately.
- **Commit:** `"Docs: export walkthrough, troubleshooting, decisions for the review fixes (F42)"`.

### Task 18: Final verification and hand-off

- [ ] **Run Task 0's `compare_outputs.py`** on all four scans. Everything must be identical
  except `--auto-density`, whose difference was approved in Task 4. Provenance JSON may differ
  only in keys added on purpose.
- [ ] **Full suite** in `.venv` and in `.review/venv` (the no-pip environment). Both are green.
- [ ] **Review Focus 1-5** run through the real CLI on synthetic rolls and 3 real frames. Record
  the outputs in the final summary.
- [ ] **Xvfb run of the picker**, per Task 12.
- [ ] **Whole-branch review:** `superpowers:requesting-code-review` (one reviewer, most capable
  model).
- [ ] **Ask the user** to run `pytest -m gpu` on the RTX 3070, and one real `halide batch` of
  Roll 16 on their machine.
- [ ] **Delete `.review/`** after the user confirms. Offer merge options
  (`superpowers:finishing-a-development-branch`).

## Coverage map

| Finding | Task |
|---|---|
| F01, F02 | 1 |
| F03, F24 | 3 |
| F04 | 2 |
| F05 | 4 |
| F06, F12, F17 | 5 |
| F07, F18 | 10 |
| F08 | 6 |
| F09, F15, F16, F25 | 7 |
| F10, F19, F20, F22 | 8 |
| F11, F30, F31 | 12 |
| F13, F35 | 14 (fonts in 13) |
| F14, F21 | 9 |
| F23 | 11 |
| F40 | 15 |
| F41 | 16 (plus every task's own tests) |
| F42 | 17 |
| F43 | no change needed (no regression found) |
| Contact sheet notes | 13 |

**F19 (GPU messaging) and F22 (wording) items, done in Task 8:**
- **F19:**
  - After the GPU service dies, print one warning for the rest of the batch: "⚠ The GPU service
    stopped (`<reason in words>`); developing the remaining frames on the CPU."
  - A bad input file is reported as an input error, not as "the GPU failed". Check
    `ScanInputError` before the device fallback.
  - `halide gpu` shows CuPy's actual import error instead of "None".
  - The fallback wording comes from one helper, so every path says the same thing.
  - The `gpu --install` removal advice names the pip cache and the NVIDIA wheels to remove.
  - 2.4-7 (the batch parent holding a CUDA context): make the parent's probe lazy, or run it in a
    subprocess, so CUDA only exists in the service. **Measure the parent's RSS before and after
    on the RTX 3070** (ask the user) before claiming the saving.
- **F22:**
  - A `plural(n, "frame")` helper replaces every "(s)".
  - "digitized at" becomes the only term for scan exposure ("Scanned at" goes).
  - `check`'s heading uses the same noun as its rows.
  - One "clear this field" convention for `profile edit`: `-`, in both the flag and the prompt
    forms (the flag also accepts `''` for compatibility, undocumented).
  - One Ctrl-C treatment: "✗ Cancelled - N of M frames developed; nothing half-written."
  - Full stops: none at the end of one-line messages, used within multi-sentence ones.
  - Work through `docs/investigations/codebase-review-evidence/strings.tsv` row by row.
