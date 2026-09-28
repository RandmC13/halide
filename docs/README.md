# halide docs

Long-form write-ups that don't belong in the code or in `CLAUDE.md` (the project's guide and its
"Decisions and why" record, which stays at the repo root). `CLAUDE.md` holds the decisions
themselves; these hold the reasoning, evidence and designs behind them.

- **`plans/`**: approved designs for features, kept after they're built as the record of what was
  decided and why.
  - `tone-output.md`: the "print" and "flat" output modes, the per-frame print fit and
    `halide print`. Implemented; lists open follow-ups.
  - `multipoint-picker.md`: the calibration picker rebuilt for many neutral points across a roll,
    with every UI decision the user made. Implemented.
  - `gpu-acceleration.md`: where a frame's time goes, and the CuPy-based `--device auto|cpu|gpu`
    design for running the pipeline on an NVIDIA GPU. Implemented on branch `gpu-acceleration`
    (see `CLAUDE.md`'s "Decisions and why" for what was actually built), verified on the user's
    RTX 3070 (all 46 `pytest -m gpu` tests pass; the benchmark results are in the plan's §7).
    `gpu-acceleration-probe.py` is the one-off measurement script that produced the plan's first
    §7 numbers; `gpu-acceleration-bench.py` times the real `invert`/`batch` on CPU vs. GPU and
    measures a GPU worker's memory — rerun it to refit the worker-count constants on new hardware.
  - `gpu-batch-throughput.md`: one exiftool kept open per worker, and one GPU process shared by
    CPU-only workers, to make batch faster. Implemented and verified on the user's RTX 3070
    (63/63 `pytest -m gpu`; GPU batch 0.69 -> 0.45 s/frame against the per-worker mode, which is
    now the fallback). Results in the plan's Task B0/B4 sections.
  - `codebase-review.md`: the plan for a full review of halide (requirements from every earlier
    session, code audit, CLI and GUI experience) before new features. Report goes to
    `investigations/codebase-review.md`.
- **`specs/`**: designs agreed but not built yet.
  - `enlarger-skin.md`: the deferred skeuomorphic "enlarger controller" look for the GUI's
    controls.
- **`investigations/`**: questions looked into, with the evidence, including what didn't work.
  - `anchor-frame.md`: can halide suggest the best frame to calibrate from? It led to the
    multi-point picker. The `check --suggest-anchor` design in it is still unbuilt.
  - `codebase-review.md`: the full review of halide before new features (requirements from
    every session, code audit, CLI and GUI experience): 32 ranked findings, what works, and the
    decisions left to the user. Evidence (per-task findings, requirements matrix, string
    inventory) in `codebase-review-evidence/`.
  - `gpu-batch-throughput.md`: why GPU batch is only ~20% faster (the GPU is idle ~90% of the
    time; exiftool and TIFF I/O dominate), and what would speed it up. Acted on in
    `plans/gpu-batch-throughput.md` (built; GPU batch ~1.5x faster).

New write-ups go in the matching folder, not the repo root.
