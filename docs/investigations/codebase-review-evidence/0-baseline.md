# 0 Baseline (main b52d6ab, 2026-09-28)
- Python 3.14.4; sandbox 16 cores, 7 GB RAM, /dev/shm 64 MB, no GPU.
- Suite (review venv, under coverage): 772 passed, 7 failed, 4 skipped, 1 warning, 62 s.
  - All 7 failures in tests/unit/test_gpu_cmd.py::test_install_*; the same file passes 40/40 in the
    project .venv. Cause: the review venv has no pip module, and these tests read the real
    environment's pip availability instead of faking it -> finding 0-1 (S5 test isolation).
- Coverage 72% overall. Lowest: gui/main_window 14%, point_list 21%, check_cmd 22%, drawers 23%,
  filmstrip 26%, step_wedge 28%, loaders 31%, proof_window 35%, quick_pick 39%, gpu_cmd 70%.
  Everything outside gui/ and check_cmd is >= 75%. Full table: 0-coverage.txt.
- Slowest tests ~3.4 s (GPU-service timeout tests); suite is not slow.

### 0-1: test_gpu_cmd install tests depend on the host environment having pip
- Severity: S5
- Where: tests/unit/test_gpu_cmd.py (test_install_*)
- Evidence: 7 fail in a venv created --without-pip ("halide can't install GPU support automatically here"), 40/40 pass in .venv.
- Suggestion: fake the pip-availability probe in these tests (S).
- Confidence: confirmed
### 0-2: halide check has 22% test coverage
- Severity: S5 (test coverage)
- Where: src/halide/cli/commands/check_cmd.py
- Evidence: coverage report.
- Suggestion: integration test of `halide check` on synthetic rolls with consistent / inconsistent exposure (S).
- Confidence: confirmed
