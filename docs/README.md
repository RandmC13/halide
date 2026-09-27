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
    (see `CLAUDE.md`'s "Decisions and why" for what was actually built) — **pending the user's
    verification on real GPU hardware** (`pytest -m gpu`, the bench script below); this dev sandbox
    has no usable GPU. `gpu-acceleration-probe.py` is the one-off measurement script that produced
    the plan's §7 numbers; `gpu-acceleration-bench.py` times the real `invert`/`batch` on CPU vs.
    GPU and measures a GPU worker's memory, for refitting the worker-count constants once real
    numbers are in.
- **`specs/`**: designs agreed but not built yet.
  - `enlarger-skin.md`: the deferred skeuomorphic "enlarger controller" look for the GUI's
    controls.
- **`investigations/`**: questions looked into, with the evidence, including what didn't work.
  - `anchor-frame.md`: can halide suggest the best frame to calibrate from? It led to the
    multi-point picker. The `check --suggest-anchor` design in it is still unbuilt.

New write-ups go in the matching folder, not the repo root.
