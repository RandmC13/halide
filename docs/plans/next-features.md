# Next features (after the review fixes)

**Status:** queued. Not designed yet. Work starts once `docs/plans/review-fixes.md` is merged.
Each feature goes through superpowers:brainstorming with the user, then gets its own plan,
because these involve choices only the user can make.

**Source:** `docs/investigations/codebase-review.md` §5 (from the review's CLI walkthrough,
Task 4.4). Two ideas from that list are already in the fix plan: `--overwrite`/`--skip-existing`
(Task 1) and examples in every `--help` (Task 9).

**Suggested order:** the end-of-batch summary and the log file first (small, and they build on
Task 8's plain progress renderer), then dry run, then recursive folders, then naming templates.

## 1. A richer end-of-batch summary

- **What.** After a batch, a short table of what happened to each frame: grade, exposure, scan
  gain, and any warning (for example "developed, metadata not copied"). The same data the contact
  sheet already captions. Failures are grouped with their reasons.
- **Who it's for.** Anyone reviewing a roll's consistency without opening every file, and
  anyone coming from Negative Lab Pro's roll view.
- **Open questions:**
  - Always on, or behind a `--summary` flag? It shouldn't turn into spam (R-012).
  - Should frames whose grade sits at the 1.0 cap, or far from the roll's median, be flagged?
- **Size:** S-M. **Depends on:** Task 8's renderer.

## 2. A log file (`--log FILE`)

- **What.** A plain-text record of the run next to the outputs: the run sheet, each frame's
  printing decision and warnings, halide's version, and the full command line.
- **Open questions:**
  - Write it by default into the output folder (`halide-log.txt`), or only when asked?
  - Plain text or JSON lines? (JSON is easier to compare between runs.)
  - Does it replace some of what the provenance already records, or add to it?
- **Size:** S. **Depends on:** Task 8.

## 3. Dry run (`--dry-run`)

- **What.** Show everything the run sheet would decide (frames found, frames skipped, the
  calibration, scan-exposure matching, workers, the outputs that would be written or skipped)
  without developing anything.
- **Open questions:**
  - Should it also run the print fit and report each frame's grade and exposure? Accurate, but it
    has to decode every frame, so it's no longer "instant".
  - How does it interact with the interactive calibration prompt?
- **Size:** S-M.

## 4. Recursive roll folders

- **What.** Develop a folder of rolls, or a roll organised into subfolders, keeping the folder
  structure in the output.
- **Open questions:**
  - Is each subfolder its own roll? That matters for `--auto-density-roll` and for
    scan-consistency checks.
  - One contact sheet per subfolder?
  - A flag (`--recursive`), or detect it automatically?
- **Size:** M. **Depends on:** Task 7's single roll discovery.

## 5. Output naming templates

- **What.** Something like `--name "{roll}_{frame:02}_{stem}"` instead of only `--suffix`.
  Fields might include the roll folder name, frame number, original stem, date and film stock.
- **Open questions:**
  - Which fields matter to a film photographer's archive?
  - Should the frame number come from the file order or from EXIF?
  - How do templates interact with Task 1's overwrite checks when two frames map to one name?
- **Size:** M.

## Also waiting (earlier work, recorded in CLAUDE.md and docs/)

- `halide check --suggest-anchor` (designed in `docs/investigations/anchor-frame.md`).
- Validate `--match-scan-exposure` on a real two-exposure scan of one frame (the user makes the
  scans).
- The ColorChecker calibration tier, a denoise stage, and a real B&W negative mode (cut
  deliberately until asked for).
- The enlarger-controller GUI skin (`docs/specs/enlarger-skin.md`).
- Whether `--grade roll` would be useful (`docs/plans/tone-output.md`).
- Moving `CLAUDE.md`'s evidence into `docs/` so the guide stays short
  (`codebase-review-evidence/7.1.md` lists what by heading).
