# Requirements ledger

Sources: **U** = the user's own words (transcripts 27-28 Sep, or today's review request),
**C** = CLAUDE.md, **P** = a docs/ plan/spec/investigation, **G** = git history.
Transcripts only cover 27-28 Sep (mostly the GPU work). Earlier requirements (June-Sep) are
known only as recorded in C and P.
Status is filled in by Phase 3. "Verify" says how.

## A. Philosophy and principles

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-001 | Colorimetric faithfulness to what the film recorded comes first. No "punchy" look, no trading accuracy for looks | C (What this project is) | audit 2.1; grep for any non-physical adjustment |
| R-002 | Getting a good result must not need per-image manual eyeballing | C | the default invert/batch path needs no per-frame input; print fit is automatic |
| R-003 | Implement Buchler's method: white balance, density balance (per-channel power), reciprocal, print curve, in a linear wide-gamut working space | C | audit 2.1 vs references |
| R-004 | Working space is ACEScg, chosen on purpose; output tagged with a real (vendored, cross-validated) ACEScg ICC profile | C | test_icc; read an output's ICC |
| R-005 | Consistent clipping, real colour management, statistically sound calibration, real test coverage (the reasons for the rewrite) | C | audits 2.1-2.3; coverage |
| R-006 | legacy/halide_v1.py is kept for reference only and never extended | C | nothing imports legacy/ |
| R-007 | core/ holds pure functions only: no file I/O, printing, globals or argparse | C | grep core/ |
| R-008 | One composed pipeline entry point (run_pipeline) and one glue chain (processing.py) shared by the single-file CLI and batch | C | read; no duplicated chains |
| R-009 | Don't re-tune decisions (99.5 anchor, per-frame grade, CC 5 band…) without the user's say-so | U/C | git history since each decision |
| R-010 | The tool should be usable by anyone, not only the user, and public on GitHub | U 27 Sep | Phases 4, 6, 7 |
| R-011 | It should feel professional and designed, not "vibe coded": reliable, intuitive, enjoyable | U today | Phases 4, 6 |
| R-012 | CLI verbose in the right ways but never spamming the terminal | U today | Task 4.2 (line counts) |
| R-013 | All CLI output aesthetically consistent, so the tool looks designed | U today | Task 4.2 + strings.tsv |
| R-014 | No missing features or help a user would benefit from | U today | Tasks 4.1, 4.4 |
| R-015 | Explanations and messages can lean on darkroom language (grade, exposure, CC filters) | C (Working with the user) | strings.tsv, GUI strings |
| R-016 | halide has its own visual identity: styling that serves the experience, drawn from film and the darkroom (filmstrip rules, sprocket edges, animations matching darkroom steps). Never cringey, bloated, ugly or flashy, and never generic: a bespoke tool made by photographers for photographers. Applies to CLI and GUI alike | U today (ledger checkpoint) | Tasks 4.2, 6.1: judge every screen against it |

## B. Input, colour management and output

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-020 | TIFF in, not RAW. Input must be linear with an embedded ICC profile; anything else is rejected with a specific error | C | Task 4.3 wrong-input runs |
| R-021 | Hand-rolled ICC parser that can tell a linear matrix-shaper profile apart | C | audit 2.1 |
| R-022 | A byte-identical halide ACEScg profile on input skips conversion (exact round trips) | C | test; flat -> print round trip |
| R-023 | Default print curve is the vendored real paper curve (MIT, abpy) | C | assets + licence present (2.7) |
| R-024 | Exposure and grade are fitted per image by default, and can be pinned | C | fit_print tests; --exposure/--contrast |
| R-025 | The print fit sets grade before the curve (never a post-curve stretch), fills the paper's ISO range, grade capped at 1.0, the same scalars on every channel | C | audit 2.1 |
| R-026 | Highlight anchor at the 99.5th percentile | C/U decision | read constant |
| R-027 | Per-frame grade, not per-roll | C/U decision | read |
| R-028 | Every output records its printing decision as provenance JSON; invert prints it | C | real run |
| R-029 | Flat (`--output flat`) output is one global multiply only, scaled by a robust highlight percentile with headroom; film base black and film gamma preserved | C | audit 2.1 |
| R-030 | `halide print` re-prints a flat file edited elsewhere; the print fit is invariant to a global multiply; pinned exposure reproduced only when provenance survives, otherwise warn and fit | C/P | Task 5.1 round trip |
| R-031 | `halide export` makes delivery sRGB PNG/JPEG from the ACEScg TIFF | C | Task 5.1 |
| R-032 | EXIF is carried over to outputs (exiftool) | C | real run |
| R-033 | Linear-output mode is never saved in a profile | C | profile_store |

## C. Calibration

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-040 | Three calibration tiers produce the same DensityProfile: ColorChecker (deliberately not built), manual picking, statistical auto | C | read |
| R-041 | Profiles are solved once per stock/process/scanner and reused (`--save-profile-as`, `halide profile`) | C | CLI run |
| R-042 | Manual calibration fits any number of neutral points across any frames by least squares | C/P | test_anchors |
| R-043 | Points are normalised to the roll's reference scan exposure | C | test |
| R-044 | Agreement is shown leave-one-out, in CC filter units with a direction; no reading while the others span < 0.1 D | C/P | test; GUI |
| R-045 | Bands: <= CC 5 calm, 5-10 amber, > 10 red; the hint fires from amber and names the one point whose removal helps most | C/U decision | read + GUI |
| R-046 | Film base is never added automatically | C/U decision | read |
| R-047 | Profiles record picks and the roll folder (absolute paths); old relative paths are kept as recorded; missing roll -> "Roll not found" dialog with Find roll… / Continue without; missing frames warned | C/U decision | 2.2, Phase 6 |
| R-048 | `--auto-density-roll` selects candidates per frame, then pools them | C | read |
| R-049 | Auto calibration judges neutrality against a density-local reference | C | test |
| R-050 | Auto calibration's documented limits are communicated: warn on near-degenerate pairs; manual picking recommended as the reliable option | C | CLI output wording |
| R-051 | Profiles carry free-text details (stock, process, scanner, notes), and editing never drops sidecars (tone, scan, anchors, roll) | C | test_profile_store |
| R-052 | A profile may carry an exposure/contrast override; precedence: CLI flag > saved > default | C | read |
| R-053 | With no calibration source in a terminal, invert/batch ask up front (profiles newest first, picking, auto); without a terminal, an error naming the missing step | C | Task 4.1 pty run |
| R-054 | `--pick` is invert-only | C/U decision | help |

## D. Scan consistency

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-060 | `halide check` checks roll consistency from headers only, and batch does it automatically at start | C | real run on Roll 16 |
| R-061 | `--match-scan-exposure` corrects exposure differences from EXIF against the profile's recorded scan settings; batch falls back to the roll's most common setting with a warning | C | real run |
| R-062 | Raw white-balance differences and active darktable tone modules are reported, never approximated | C | real run on Roll 16 |
| R-063 | An exposure spread being corrected is still shown as a ⚠ warning | C/U | real run |
| R-064 | OPEN: scan-exposure matching not yet validated on a real two-exposure scan | C/P | report as open follow-up |

## E. Batch, memory and speed

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-070 | One bad frame (or a crashed worker) never aborts the batch; each failure gets an actionable message | C | 2.3 |
| R-071 | Worker count sized from free RAM and physical cores; an explicit `--workers` is respected but warned when over budget | C | real run |
| R-072 | Full-resolution paths work band by band, ~1 frame of memory, bit-identical to the whole-array result | C | test_banding; RSS on real scans vs recorded numbers |
| R-073 | CPU output stays bit-identical across refactors (D1's matrix fusion is the only accepted tolerance, 2 ULP) | C/U | tests |
| R-074 | CLI start-up imports no numpy, tifffile, Pillow, colour or cupy (~0.1 s) | C | test_cli_startup; importtime |
| R-075 | exiftool kept open per process on Linux only; one-shot elsewhere; a broken exiftool never hangs a batch | C/U | 2.4 |
| R-076 | Test temp files deleted per test; the suite forces HALIDE_DEVICE=cpu | C/U | pyproject, conftest |
| R-077 | Large outputs never left in /tmp (RAM on the user's machine) | U | code: temp folders used by contact sheet preview, tests |

## F. GPU

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-080 | GPU on by default if it's faster, and able to be turned off (`--device auto|cpu|gpu`, `$HALIDE_DEVICE`) | U (D3, D4) | 2.4 |
| R-081 | CuPy is an optional ~1 GB install; users with an NVIDIA card are told it exists and given an easy way to add it (`halide gpu --install`); nobody without a card is pitched it | U | 2.4, Task 4.1 |
| R-082 | GPU vs CPU within D2 (1e-5 relative, <= 1 code value on 8-bit), recorded in provenance as the device used | U (D2) | user's pytest -m gpu |
| R-083 | Any GPU failure falls back to the CPU for that frame, with a warning; never fails the batch | C | 2.4 |
| R-084 | A GPU batch shares one GPU service; falls back to per-worker mode (Python < 3.13, no /dev/shm room, service won't start, `HALIDE_GPU_SERVICE=0`) with the reason shown | C/U | 2.4, run with 64 MB /dev/shm |
| R-085 | `halide contact` never initialises the GPU unless asked explicitly | C | read |

## G. CLI experience

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-090 | Everything batch/export/print decides before developing is shown as one aligned run sheet | C | Task 4.2 |
| R-091 | The progress display is a static contact sheet of the roll, frames pulse and fade; the layout never wraps or leaves stale rows | C/U | Task 4.2 pty capture at several widths |
| R-092 | `--quiet` still prints warnings | C | Task 4.2 |
| R-093 | Tab completion for zsh, bash and fish installs itself on first terminal run; no `completion` command; one note; `HALIDE_NO_COMPLETION=1` turns it off | C/U | 2.5 |
| R-094 | Contact sheets: a real contact print look (black rebate, strips of 6, orange edge print, stock name when every frame agrees), captioned with each frame's printing decision, marked and skipped by `halide contact` | C/U | Task 5.1 |
| R-095 | `batch --contact-sheet` without an out_dir keeps no full-size TIFFs; temp folder always deleted | C | 2.3 |
| R-096 | Export's console verb is "Exporting" | C | strings |

## H. GUI experience

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-100 | Qt picker: landscape, fixed size (~55% x 70% of the screen, 900x700 floor), never scrolls except the point list | C/U | Phase 6 at several resolutions |
| R-101 | Sprocket filmstrip of the roll with point counts; previews "develop in" | C/U | Phase 6 |
| R-102 | Negative / Positive switch; the positive is labelled a rough auto estimate until two separated points exist; no greyscale positive | C/U | Phase 6 |
| R-103 | The caption says brightness is reversed on the negative, in those plain words | C/U | Phase 6 |
| R-104 | Right panel: step wedge coverage bar, scrolling point list, Clear frame / Clear all, Build contact sheet…, drawers Extra information / Print / Details, one red Save (Develop in `--pick`) | C/U | Phase 6 |
| R-105 | Only one red primary button per window; agreement "red" is a different, warmer colour | C/P | Phase 6 |
| R-106 | Clicking a marker selects it (Delete removes); clicking a row jumps to its frame and flashes the marker | C | Phase 6 |
| R-107 | Contact sheet window: draft at once, full-quality frames swap in, zoom/pan, out-of-date marking, save | C | Phase 6 |
| R-108 | Closing any window stops its workers at once and never aborts on a running thread | C | Phase 6 + 2.6 |
| R-109 | `invert --pick` opens the same picker for one frame | C | Phase 6 |
| R-110 | Enlarger-controller skin deferred (spec exists) | U/P | not checked |

## I. Process and docs

| ID | Requirement | Source | Verify |
|---|---|---|---|
| R-120 | Write-ups live in docs/ (plans, specs, investigations), indexed in docs/README.md; the repo root stays uncluttered | C/U | Phase 7 |
| R-121 | No linter or formatter added without being asked | C | pyproject |
| R-122 | Real scans never committed | U | git ls-files |
| R-123 | Work on branches other than main; nothing merged without explicit approval | U | process |
| R-124 | Claims backed by real data and real scans, not hand-waving | C/U | Phase 8 verification |
| R-125 | Clean up the project folder after work (no stray probe scripts, outputs) | U 27 Sep | Task 2.7 hygiene |
| R-126 | Token economy: Opus plans, Sonnet/Haiku implement, few agents | U 28 Sep | process only, not reported |

## Open follow-ups inherited from earlier work (reported, not judged)

- F-1 Scan-exposure matching not validated on a real two-exposure scan (R-064).
- F-2 `check --suggest-anchor` designed, not built (investigations/anchor-frame.md).
- F-3 ColorChecker tier, denoise stage, real B&W mode: cut deliberately.
- F-4 Enlarger skin deferred (specs/enlarger-skin.md).
- F-5 Windows: `nvcuda.dll` probe untested; exiftool one-shot only; GPU service untested.
- F-6 Picker holds a CUDA context for the whole session (could be lazy).
- F-7 GPU fallback path: `release_memory()` doesn't free because the traceback holds the frame.
- F-8 Per-worker-mode Compute line doesn't say which cap set the worker count.
- F-9 tone-output plan: whether `--grade roll` would be useful was left to the user.
