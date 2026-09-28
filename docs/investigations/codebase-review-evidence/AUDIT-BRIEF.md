# Brief for every halide review agent

Repo: /home/sam/Documents/halide (branch review-2026-09). Plan: docs/plans/codebase-review.md — read
the "Global Constraints", "Finding format" and "Review Focus" sections and YOUR task's section.
Project guide: CLAUDE.md (very long — grep it for the parts relevant to your area and read only
those line ranges; the "Decisions and why" entries are settled unless you have NEW evidence, and
then label the finding `REOPENS: <decision>`).

Rules:
- REVIEW ONLY. Do not edit anything under src/, tests/, docs/ or any tracked file. Do not commit.
- Write findings to the file your task names under /home/sam/Documents/halide/.review/, using the
  plan's finding format exactly. Number them <task-id>-<n>.
- Evidence or it doesn't count: quote code (file:line) or show the command you ran and the
  relevant output lines. Mark each finding `confirmed` (you reproduced it) or `likely` (read only).
  You MAY write and run throwaway scripts under .review/scratch/<task-id>/ to reproduce things.
- Python: /home/sam/Documents/halide/.venv/bin/python. Tests: `.venv/bin/python -m pytest tests/ -q`.
- Real scans (gitignored, never commit, never modify): IMG_0151.tif, IMG_0156.tif, IMG_0156-nowb.tif,
  IMG_0158.tif in the repo root, and Roll16-Testing/ (37 frames). Any output TIFF you create must go
  under .review/scratch/ and be DELETED as soon as you've checked it (~130-180 MB each, RAM is limited).
- No GPU here. Do not try to install CuPy.
- Your final reply to the orchestrator: <= 15 lines — count of findings by severity, the 3 most
  important in one line each, and anything you could not check. No raw logs.
- Quality bar: the user is a film photographer who pushes back on hand-wavy claims. Prefer 10
  solid findings over 40 speculative ones, but DO report small inconsistencies (S4/S5) — the
  user explicitly wants polish and a tool that feels designed.
