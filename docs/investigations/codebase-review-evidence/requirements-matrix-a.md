# Requirements matrix — sections A, B, C, D, I (Task 3.1)

Verified against branch `review-2026-09`, `.venv/bin/python -m pytest` (see command per row).
"Judged in Phases 4/6/8" = out of this task's scope per the ledger's own Verify column.

| ID | Status | Evidence | Untested? |
|---|---|---|---|
| R-001 | Met | No non-physical "look" adjustment found (grep for saturation/vibrance/punch turned up nothing in `core/`); CLAUDE.md decisions consistently favor measured curves over invented ones (paper curve, ISO-range fit) | untested (philosophy, not code-testable) |
| R-002 | Met | `core/tone_render.py::fit_print` fits exposure/grade automatically; `choose_calibration_source` offers auto-estimate with no manual input | tested: `tests/unit/test_tone_render.py::test_fit_print_*` |
| R-003 | Met | `core/pipeline.py::negative_to_positive` = white_balance → density_balance → invert; `develop()` adds print/tone stage | tested: `tests/unit/test_pipeline.py`, `test_density.py`, `test_tone_render.py` |
| R-004 | Met | `tests/unit/test_icc.py::test_bundled_output_profile_matches_colour_sciences_acescg_definition` passes | tested (ran: 1 passed) |
| R-005 | Met | Baseline suite: 772 passed / 4 skipped, 72% coverage (`.review/0-baseline.md`, `.review/0-coverage.txt`) vs. legacy script's zero coverage | tested (suite itself is the evidence) |
| R-006 | Met | `grep -rn "halide_v1" src/ tests/` → only a docstring reference in `core/density.py:5`, no import | confirmed by grep |
| R-007 | **Partial** | See finding 3a-1: `core/tone_render.py` performs file I/O (`load_1d_cube` → `Path(path).read_text()`) despite CLAUDE.md's "core/ = PURE functions only: no file I/O" | untested (architectural rule, no test enforces it) |
| R-008 | Met | `core/pipeline.py::run_pipeline` is the one entry point; `processing.py` (`process_scan`, `develop_request`, `print_scan`, `export_delivery_image`) is the one glue layer used by both CLI and batch worker | confirmed by reading; no duplicate chain found |
| R-009 | Met | CLAUDE.md's "Decisions and why" entries each cite the user's own say-so before changing a constant (e.g. 99.5 percentile, per-frame grade) | untested (process discipline, not code) |
| R-010–R-016 | — | Judged in Phases 4/6 (per task instructions) | — |
| R-020 | Met | `tests/unit/test_icc.py::test_rejects_gamma_encoded_trc`, `test_rejects_lut_based_profile`, `test_rejects_non_rgb_color_space`, `test_rejects_missing_required_tag`, `test_rejects_the_real_gamma_encoded_rec2020_profile` all pass | tested |
| R-021 | Met | `io/icc.py:7` docstring explains Pillow's `ImageCms` is metadata-only; hand-rolled parser in same file | confirmed by reading |
| R-022 | Met | `processing.py::_read_scan` (lines 158-165) skips conversion when `scan.icc_profile == output_profile_bytes()`; round trip pinned by `tests/integration/test_print_cli.py::test_untouched_flat_then_print_equals_direct_print` (passes, synthetic image) | tested (synthetic); real-scan magnitude (~4e-7) is Needs-real-data, see below |
| R-023 | Met | `src/halide/assets/tone_curves/paper_endura.cube` + `LICENSE-paper_endura.txt` (MIT) present and shipped (`.review/2.7.md` confirms wheel packaging) | confirmed by file presence |
| R-024 | Met | `ToneCurveParams` defaults `exposure=None, contrast=None`; `--exposure`/`--contrast` CLI flags exist (`cli/_calibration_args.py`) | tested: `test_tone_render.py::test_fit_print_*` |
| R-025 | Met | `core/tone_render.py::fit_print` (lines 142-176): grade solved from negative/paper density ranges *before* the curve is applied, capped at `MAX_PRINT_CONTRAST=1.0`, returns two scalars applied identically to every channel by construction | tested: `test_fit_print_fills_the_paper_range_from_the_negatives_own_range`, `test_fit_print_caps_the_grade_at_the_real_paper_for_a_low_contrast_negative` |
| R-026 | Met | `core/tone_render.py:73`: `_PRINT_HIGHLIGHT_PERCENTILE = 99.5`, comment cites the user's own greyscale-proof-sheet decision | confirmed by reading |
| R-027 | Met | `fit_print` operates per call (per frame); `batch/orchestrator.py:8` docstring: "computes its own per-frame automatic profile"; no roll-median grade code path exists | confirmed by grep (no "roll grade" implementation found) |
| R-028 | **Needs-real-data** | Mechanism (provenance JSON in `ImageDescription`) is Met and tested synthetically: `tests/integration/test_print_cli.py::test_outputs_record_their_printing_decision` passes; `describe_resolved_tone()` is printed by `invert_cmd.py:170`. Real-scan console check not yet run. | Run: `.venv/bin/python -m halide.cli.main invert --device cpu --rm 0.9 --bm 1.1 --rs 1 --bs 1 IMG_0156.tif .review/scratch/out.tif` then inspect stdout + `exiftool -ImageDescription .review/scratch/out.tif` (delete output after) |
| R-029 | Met | `core/tone_render.py::estimate_linear_scale` (lines 183-211): percentile-based headroom scale, docstring explains dust/clamp-artifact rationale; film base/gamma untouched (no black-point or gamma operation in the linear path) | tested: `tests/unit/test_tone_render.py` (linear-scale tests present, suite passed) |
| R-030 | **Needs-real-data** | Mechanism confirmed on synthetic data: `test_untouched_flat_then_print_equals_direct_print` and `test_externally_exposure_adjusted_flat_still_prints_identically` both pass. CLAUDE.md's ~4e-7 / 1-step figures are from real scans, not re-verified here. | Run Task 5.1's round trip: `invert --output flat` then `halide print` on IMG_0156.tif vs direct `invert --output print`, diff pixels; also with metadata stripped + exposure-adjusted copy |
| R-031 | Met | `io/raster.py::to_srgb_8bit`, `processing.py::export_delivery_image`; `tests/integration/test_export_cli.py` and `tests/unit/test_raster.py` pass (in baseline suite) | tested |
| R-032 | **Needs-real-data** | Exiftool copy mechanism tested with real exiftool binary on synthetic TIFFs: `tests/integration/test_exiftool_real.py` (both tests). Not yet checked against a real scan's actual EXIF payload. | Run: `.venv/bin/python -m halide.cli.main invert --device cpu --rm 0.9 --bm 1.1 --rs 1 --bs 1 IMG_0156.tif .review/scratch/out.tif` then `exiftool -G1 -a -s IMG_0156.tif .review/scratch/out.tif` and diff the tag sets (delete output after) |
| R-033 | Met | `calibration/profile_store.py::load_tone_override` hardcodes `mode="paper"` regardless of stored data (never reads a `mode` field) | tested: `tests/unit/test_profile_store.py::test_load_tone_override_never_restores_linear_mode` |
| R-040 | Met | `core/types.py::DensityProfile` is one dataclass with a `source` field; manual (`fit_density_balance`), auto (`solve_density_balance`/`roll_auto_density_balance`) both construct it; ColorChecker deliberately absent (F-3) | confirmed by reading |
| R-041 | Met | `calibration/profile_store.py::save_named_profile`/`load_named_profile`; `halide profile` CLI commands exist | tested: `tests/unit/test_profile_store.py` (passed) |
| R-042 | Met | `core/density.py::fit_density_balance` (lines 80-99): least-squares line per channel, any N ≥ 2 points, docstring notes it reproduces `solve_density_balance` exactly for N=2 | tested: `tests/unit/test_density.py`, `tests/unit/test_anchors.py` |
| R-043 | Met | `calibration/anchors.py::normalised_rgb` applies `scan_gain(point.scan, reference)` before fitting | tested: `tests/unit/test_anchors.py` (passed) |
| R-044 | Met | `anchors.py::agreement` uses `leave_one_out_residuals`, reports `describe_cast` as "CC n.n <dir>"; gated by `MIN_DENSITY_SEPARATION = 0.1` (line 90-95) | tested: `tests/unit/test_anchors.py` |
| R-045 | Met | `anchors.py:37-38`: `AGREEMENT_AMBER_CC = 5.0`, `AGREEMENT_RED_CC = 10.0`; `worst()` (line 141) picks the point whose removal helps the others most, hint fires at amber | tested: `tests/unit/test_anchors.py` (15 tests, includes worst/band logic) |
| R-046 | Met | No code path adds a film-base point automatically; `gui/step_wedge.py` only *labels* film base on the coverage bar, never injects a point | confirmed by grep (no auto-add call found) |
| R-047 | Met (code); GUI dialog untested by automation | `anchors.py::absolute`/`recorded_path` (lines 188-201) implement absolute-path storage and as-recorded fallback; `gui/main_window.py:614-653` has "Roll not found" / "Find roll…" / "Continue without" / "Frames not found" dialogs | tested (path logic): `test_a_relative_frame_is_stored_absolute`, `test_an_unfound_relative_frame_is_kept_as_recorded_not_guessed`; GUI dialogs untested (no GUI test harness touches them — expected per CLAUDE.md, judged in Phase 6) |
| R-048 | Met | `calibration/auto.py::roll_auto_density_balance` (lines 231-260): candidates selected per-frame first (`_neutral_candidates` per image), pooled only afterward; docstring documents the rejected alternative | tested: `tests/unit/test_auto_calibration.py` (passed) |
| R-049 | Met | `auto.py::_density_local_saturation` (line 64) bins by density before scoring saturation | tested: `tests/unit/test_auto_calibration.py::test_saturation_matches_the_old_where_form_including_degenerate_pixels` and others |
| R-050 | **Partial** | See finding 3a-2: near-degenerate warning (`_check_density_separation`) is Met and tested (`test_auto_density_balance_warns_on_low_density_separation`), but the CLI never explicitly recommends manual calibration as more reliable — the only in-product hint is `--auto-density` being labelled "a quick approximate guess" (`cli/_calibration_args.py:290`) | partially tested |
| R-051 | Met | `profile_store.py::update_profile`/`rename_profile` edit raw JSON, explicitly to preserve sidecars (lines 173-174, 200-201) | tested: `test_update_profile_keeps_tone_and_scan_sidecars`, `test_rename_profile` |
| R-052 | Met | `cli/_calibration_args.py::resolve_tone_params` (lines 158-168): CLI flag → saved tone → `None` (fitted); linear mode is CLI-flag-only | confirmed by reading; covered indirectly by `test_profile_store.py` tone tests |
| R-053 | Met | `choose_calibration_source` gates on `sys.stdin.isatty()` (line 200); `resolve_density_profile` raises a SystemExit naming the missing step otherwise | tested: `tests/integration/test_invert_cli.py::test_missing_input_file_errors_before_calibration_check` and mutual-exclusion tests confirm the error path exists; the interactive-prompt itself is Task 4.1's pty-driven job |
| R-054 | Met | `invert_cmd.py:43` is the only call site passing `allow_pick=True`; `batch_cmd.py:58` does not | **untested** — no test asserts `batch --pick` is rejected/absent (grep of `tests/integration/test_batch_cli.py` for "pick" finds nothing) |
| R-060 | **Needs-real-data** | `cli/commands/check_cmd.py` and batch's auto-invocation exist; unit-level logic (`ScanConsistencyReport`) is tested synthetically in `tests/unit/test_scan_consistency.py` (passed) | Run: `.venv/bin/python -m halide.cli.main check Roll16-Testing/` |
| R-061 | **Needs-real-data** | `scan_consistency.py::most_common_settings` fallback exists; synthetic tests pass (`tests/unit/test_scan_consistency.py`) | Run: `halide batch Roll16-Testing/ .review/scratch/out --match-scan-exposure` (delete outputs after) |
| R-062 | **Needs-real-data** | `io/scan_metadata.py` parses raw white balance + darktable XMP history; `scan_consistency.py` reports (never corrects) both — confirmed by code and synthetic tests | Run `halide check Roll16-Testing/` and inspect the white-balance/tonal-module lines against known Roll 16 frames |
| R-063 | **Needs-real-data** | `scan_consistency.py:96`: "...evened out by --match-scan-exposure..." still routed through the warning path; pinned synthetically by `tests/unit/test_scan_consistency.py:81` (`"evened out by --match-scan-exposure" in corrected[0]`) | Run `halide batch Roll16-Testing/ .review/scratch/out --match-scan-exposure` and confirm the ⚠ row still appears on the run sheet |
| R-064 | Met | CLAUDE.md explicitly states this as an open, unvalidated item ("Not yet validated against a real two-exposure scan of one frame"); the requirement is that this limitation is *documented*, which it is | untested by design (documents an untested claim) |
| R-120 | Met | `docs/README.md` indexes every file under `plans/`, `specs/`, `investigations/`; verified all files present match the index (`plans/codebase-review.md`, `gpu-acceleration*.{md,py}`, `multipoint-picker.md`, `tone-output.md`; `specs/enlarger-skin.md`; `investigations/anchor-frame.md`, `gpu-batch-throughput.md`) | confirmed by `ls` + diff against README |
| R-121 | Met | `pyproject.toml` has no ruff/flake8/black/mypy/isort/pylint entry; `dev = ["pytest"]` only | confirmed by grep |
| R-122 | Met | `git ls-files \| grep -iE "\.tif$\|\.tiff$"` → no matches; `.gitignore` has `*.tif`/`*.tiff` | confirmed |
| R-123 | Met | Current branch is `review-2026-09`, not `main`; `main`'s tip (`b52d6ab`) is a merge commit, i.e. work landed via merge rather than direct pushes to main | confirmed via `git branch --show-current`, `git log` |
| R-124 | — | Judged in Phase 8 (ledger's own Verify column names it explicitly) | — |
| R-125 | **Partial** | See finding 3a-1... see finding 3a-3: a 126.7 MB stray file `IMG_0158-positive.tif` (dated Sep 18, before this review) sits untracked in the repo root — not one of the four documented test scans, never cleaned up. Task 2.7's hygiene review (`.review/2.7.md`) did not catch it. | confirmed (`ls -la`, `git status --ignored`) |
| R-126 | — | Process only, not reported (per ledger) | — |

## Findings

### 3a-1: `core/tone_render.py` violates the documented "core/ = no file I/O" rule
- Severity: S4
- Where: `src/halide/core/tone_render.py:34` (`from halide.io.lut import Cube1D, load_1d_cube`), `:42-44` (`_load_curve` calling `load_1d_cube(path)`, which does `Path(path).read_text()` in `src/halide/io/lut.py:82`), used at `:220` and `:230`
- Evidence: CLAUDE.md's Architecture section states `core/ # PURE functions only: no file I/O, no print, no globals, no argparse`. `tone_render.py` is inside `core/` and reads a `.cube` LUT file from disk (cached via `lru_cache`, but still a real filesystem read triggered from within `core/`).
- Requirement: R-007
- Suggestion: Either move curve loading into `io/` or `processing.py` and pass the parsed `Cube1D` into `apply_tone`/`resolve_tone` as a parameter, or amend CLAUDE.md's rule to note this one deliberate, cached exception. S (small, mechanical) either way.
- Confidence: confirmed

### 3a-2: R-050's "manual picking is more reliable" guidance isn't actively surfaced by the CLI
- Severity: S5
- Where: `src/halide/cli/_calibration_args.py:211-212, 290`
- Evidence: The only in-product signal that auto-density is less reliable is the phrase "a quick automatic estimate"/"a quick approximate guess" next to the `--auto-density` menu option and help text; there is no message pointing a user toward `halide calibrate` specifically because auto's highlight estimate is structurally weaker (as CLAUDE.md's "Auto calibration's documented limits" section spells out in detail). The near-degenerate-pair warning (`_check_density_separation`) IS implemented and tested, so this finding is about the softer "recommend manual as reliable" half of R-050, not the warning half.
- Requirement: R-050
- Suggestion: Add one line to `--auto-density`'s help text or the post-run warning, e.g. "for a frame whose best neutrals aren't at the tonal extremes, `halide calibrate` is more reliable." S.
- Confidence: confirmed

### 3a-3: Stray 126.7 MB output file left in the repo root
- Severity: S5
- Where: repo root, `IMG_0158-positive.tif`
- Evidence: `ls -la IMG_0158-positive.tif` → 126,725,546 bytes, dated `Sep 18 20:58`. `git status --porcelain --ignored` shows `!! IMG_0158-positive.tif` (ignored by the blanket `*.tif` rule, so it won't appear in a normal `git status`, but it is not one of the four scans CLAUDE.md/AUDIT-BRIEF name as legitimate fixtures — it looks like a leftover manual-invert output from before this review started). Task 2.7's hygiene pass (`.review/2.7.md`) checked `.gitignore` coverage and legacy/ tracking but did not flag this file.
- Requirement: R-125
- Suggestion: Delete it (confirm with the user first, since it's outside `.review/scratch/`). Trivial.
- Confidence: confirmed

### 3a-4: `.coverage` is untracked and not gitignored
- Severity: S5
- Where: repo root `.gitignore`
- Evidence: `git status` (session start) shows `?? .coverage`; `.gitignore` has no `.coverage`/`.coverage.*` entry despite covering `.pytest_cache/` and other test artifacts. `pytest-cov` isn't even a declared `dev` dependency in `pyproject.toml`, so this file is produced by an undeclared tool.
- Requirement: R-125 (project-folder hygiene)
- Suggestion: Add `.coverage` (and `.coverage.*`) to `.gitignore`. Trivial.
- Confidence: confirmed

### 3a-5: `--pick` being invert-only (R-054) has no regression test
- Severity: S5
- Where: `tests/integration/test_batch_cli.py`
- Evidence: `grep -n "pick" tests/integration/test_batch_cli.py` returns nothing — nothing asserts `halide batch --pick` is rejected by argparse (it currently would be, since `add_calibration_arguments(parser)` in `batch_cmd.py:58` omits `allow_pick=True`, but a future edit could silently add it back without any test failing).
- Requirement: R-054
- Suggestion: One test: `main(["batch", ..., "--pick"])` raises `SystemExit` (unrecognized argument). S.
- Confidence: confirmed
