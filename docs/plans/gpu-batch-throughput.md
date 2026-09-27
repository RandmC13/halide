# Faster Batch: One exiftool per Worker, One GPU for All Workers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: **Part A implemented; B0 measured (go, see Task B0); B1-B5 in progress (2026-09-27).** Branch
`gpu-acceleration` (not merged; the user wants this done first).

**Goal:** Make `halide batch` (and bulk `print`/`export`, contact sheets) substantially faster,
above all on the GPU, without changing any output byte.

**Architecture:** Part A keeps one exiftool process running per worker (`-stay_open`) instead of
starting one per output file. Part B replaces "every GPU worker has its own CUDA context" with
one GPU service process shared by CPU-only worker processes. The workers read, decode, compress,
write and tag files; the service develops frames that the workers hand over through shared
memory.

**Tech Stack:** Python 3.14 stdlib (`subprocess`, `multiprocessing.shared_memory`,
`multiprocessing.connection`, `threading`), tifffile, CuPy (optional, as today), exiftool (optional,
as today).

**Spec:** the user's request (2026-09-27): process many frames at once on the GPU instead of 4
workers that each load their own copy of NVIDIA's libraries. Evidence and corrections:
`docs/investigations/gpu-batch-throughput.md`. It must be read first. It explains why "load
tens of frames into VRAM" is not the design: the GPU is idle ~90% of the time, one frame
already fills it, and the time is CPU-side file work.

## Global Constraints

- **No output byte changes.** CPU path: TIFF/PNG/JPEG pixels, provenance and EXIF metadata
  byte-identical to HEAD before this plan (`d35c476`), except Pillow's ICC creation timestamp,
  which already varies run to run (CLAUDE.md). GPU service path: bit-identical to today's
  in-process GPU path, because it runs the same device code on the same card. Checked with
  `pytest -m gpu` on the user's machine.
- Every existing guarantee stays:
  - one bad frame never loses the rest of the batch;
  - any GPU failure redoes that frame on the CPU, with a warning;
  - CUDA is never initialised in the parent or the forkserver of a worker pool;
  - closing a GUI window stops its worker processes (and now the GPU service too);
  - building the CLI parser imports nothing heavy.
- Never write frames to `/tmp` (RAM on the user's machine). Shared-memory frames live in
  `/dev/shm`, which is also RAM. They are the working buffers themselves, not extra copies, and
  must be counted in the RAM budget and unlinked as soon as the frame is done.
- exiftool stays optional (halide works without it, as today). CuPy stays optional.
- Linux and Windows. No fork-only tricks: the GPU service uses the `spawn` start method.

## Review Focus

1. **exiftool dies or hangs mid-batch** (killed, crashes on one odd file). The next frames must
   still get their metadata, via a restarted session or the one-shot fallback, with no batch-wide
   hang. (Task A1: fake exiftool that exits mid-protocol; one that never answers → timeout.)
2. **Paths exiftool's argument file can't carry** (a newline in a file name; non-ASCII on
   Windows). Newline → one-shot fallback; non-ASCII → `-charset filename=utf8`. (Task A1 test
   with a non-ASCII path through the real exiftool, skipped where exiftool is absent.)
3. **The GPU service crashes mid-batch.** Every frame after that is developed on the CPU in its
   worker with a warning; nothing hangs waiting for a dead service. (Task B2: kill the service
   process during a batch.)
4. **`/dev/shm` too small or unavailable** (small Docker containers default to 64 MiB). This
   must not crash: fall back to today's per-worker GPU mode with a warning on the run sheet.
   (Task B1: shared-memory creation raising.)
5. **A worker killed while it holds a shared-memory frame** (OOM killer). There must be no
   `/dev/shm` leak: Python's resource tracker unlinks segments a dead process created, and the
   service attaches with `track=False` so it doesn't double-own them. (Task B1 test: kill a child
   holding a segment, check the segment is gone after the pool shuts down.)

---

## Part A — one exiftool per worker

Measured (investigation, Corrections): a fresh exiftool per output takes ~0.66 s. A kept-open one
takes ~0.29 s per file after the first, and its output files are byte-identical. That's
~0.37 s/frame saved on both CPU and GPU paths, with no change to what's written.

### Task A1: `ExifToolSession`, used by `copy_exif_metadata`

**Files:**
- Create: `src/halide/io/exiftool.py`
- Modify: `src/halide/io/tiff.py` (`copy_exif_metadata` delegates to it)
- Test: `tests/unit/test_exiftool.py` (fake exiftool), `tests/integration/test_exiftool_real.py`
  (real exiftool, skipped when absent)

**Interfaces:**
- Produces:
  ```python
  class ExifToolSession:
      def __init__(self, executable: str = "exiftool", timeout: float = 120.0) -> None
      def run(self, args: list[str]) -> str          # one command; raises ExifToolError on failure
      def close(self) -> None
  class ExifToolError(RuntimeError): ...
  def session() -> ExifToolSession | None            # this process's shared session; None if exiftool isn't installed
  ```
  `copy_exif_metadata(source, dest, drop_icc=False) -> bool` keeps its signature and return value
  (False = exiftool not installed), and builds exactly the same argument list as today.

- [x] **Step 1: Failing tests.** Use a fake exiftool: a small Python script written to `tmp_path`
  and made executable, which implements the part of the `-stay_open True -@ -` protocol used:
  - read argument lines until `-executeNNN`;
  - append a marker line to a log file named in an environment variable;
  - print `    1 image files updated` or an error line;
  - print `{readyNNN}`.

  Behaviour switches come from environment variables: `FAIL_ON=<n>`, `DIE_AFTER=<n>`, `HANG=1`.

  Tests:
  - many copies use **one** process (log shows one PID);
  - args are exactly today's list;
  - `0 image files updated` / `Error:` output raises `ExifToolError`, and `copy_exif_metadata`
    lets it propagate, exactly as today's `check=True` does;
  - a session whose process died is restarted once, then falls back to one-shot;
  - a hung process times out, and the frame falls back to one-shot;
  - a path containing `\n` uses one-shot;
  - no exiftool on PATH → `copy_exif_metadata` returns False and starts nothing;
  - `close()` ends the process, and an `atexit` hook closes it at interpreter exit.
- [x] **Step 2: Run, verify they fail.**
- [x] **Step 3: Implement.**

  ```python
  # src/halide/io/exiftool.py
  class ExifToolSession:
      """One exiftool process reused for many commands (its -stay_open mode). Most of a fresh
      exiftool's ~0.66 s per file is its own start-up and tag-table loading, not the file
      rewrite: kept open it is ~0.29 s per file, with byte-identical output
      (docs/investigations/gpu-batch-throughput.md)."""

      def __init__(self, executable="exiftool", timeout=120.0):
          self._proc = subprocess.Popen(
              [executable, "-stay_open", "True", "-@", "-", "-common_args", "-charset", "filename=utf8"],
              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              encoding="utf-8", bufsize=1,
          )
          self._counter = itertools.count(1)
          self._lock = threading.Lock()
          ...
  ```

  Read the reply on a helper thread (or with `selectors` on POSIX / a reader thread on Windows)
  so the timeout works on both platforms. Success is `N image files updated` with N ≥ 1 and no
  `Error:` line, the same condition exiftool's non-zero exit encodes today. `session()` holds one
  per process (module global plus `threading.Lock`), created on first use and registered with
  `atexit`. Worker processes each get their own automatically, because they're separate
  processes. The session must be created lazily inside the worker, never inherited over a fork.

  First confirm that `-common_args -charset filename=utf8` doesn't change a single output byte
  on Linux, against the real exiftool (Step 5). If it does, pass `-charset filename=utf8` only on
  Windows.
- [x] **Step 4: Run tests; full suite.**
- [x] **Step 5: Real-exiftool acceptance** (runs where exiftool is installed; the implementer
  fetches exiftool's source into their scratch folder, the same way the investigation did):
  - for all four real scans and 3 Roll16 frames, output files written through the session are
    **byte-identical** to the one-shot path;
  - timing: first file and median of the next five, one-shot vs session;
  - record both in the task report.
- [x] **Step 6: Commit.**

**Done** (`97425d9`, `a6c26ef`): kept-open session on **Linux only** (ruling: a `-stay_open`
exiftool never exits on end of input, and only Linux's parent-death signal guarantees it dies with
a terminated worker; elsewhere one-shot as before). Measured: one-shot 0.67 s first / 0.66 s
median vs session 0.57 s first / 0.23 s median of the next five; 15/15 real outputs byte-identical.

---

## Part B — one GPU service for all workers

### Task B0: Measure the ceiling first (the user runs it) — go/no-go for B1-B5

**Why:** today's GPU batch already moves ~13 GB through the disk in ~27 s (~0.5 GB/s), and the
user's scans originally live under a path named `hdd`. If the disk is the limit, B1-B5 buy
nothing, and the plan stops after Part A.

**Files:** Create `docs/plans/gpu-batch-throughput-probe.py`. Its output goes to a file on disk
beside the roll, never `/tmp`; scratch data goes in a folder beside the roll, deleted in a
`finally`.

The probe measures, and prints as plain tables:
1. **Disk:** sequential write speed on the roll's filesystem with `fsync`, so the page cache
   can't hide the disk (~1 GiB, deleted after).
2. **One frame's file work, step by step, on this machine:** read + decode a scan; compress +
   write an output; exiftool one-shot; exiftool kept open (Part A's session, once it exists; the
   probe imports it).
3. **The ceiling:** that whole per-frame cycle (read → write → exiftool, no developing at all) in
   1, 2, 4 and 8 processes over 16 real frames of the roll, as frames per second. This is the
   fastest any batch design can run on this machine. Compare with today's GPU batch: 1.4 frames/s.
4. **With CuPy installed:** host RSS and GPU memory of one process at each stage:
   - bare Python;
   - after `import cupy`;
   - after creating the context;
   - after an elementwise kernel;
   - after a matrix multiply (which loads cuBLAS).

   This tells B2 what the one service process will hold.

Build it with `--devices cpu` checks in this sandbox (steps 1-3 need no GPU), then hand it to the
user.

- [ ] Write the probe; validate steps 1-3 here on 2-4 Roll16 frames; commit.
- [x] **User runs it; decision recorded in this plan:**
  - **Go** if the 8-process ceiling is at least ~2x today's GPU batch (≥ 2.8 frames/s);
    otherwise B1-B5 are dropped. Also record the actual ceiling, which becomes B5's target.
  - If the disk is the limit, record it and stop. Options for a separate plan: a lower zlib level
    or no compression for the output TIFF (bigger files), or writing outputs to a faster disk.
    Both are user decisions.

**B0 as actually run (2026-09-27): go, on RAM, not on the probe above.** The probe script was
not written (a first attempt was cut off by a safety filter). The same questions were answered with
the existing bench (`gpu-acceleration-bench.py`, rerun after Part A), `dd` and `vmstat`/
`nvidia-smi dmon` during real GPU batches on the user's machine (37 frames of Roll 16):
- **Disk: not the limit.** The project's NVMe writes 1.3 GB/s with `fsync` (the HDD holding the
  original rolls: 180 MB/s); a batch needs ~0.45 GB/s.
- **Part A worked where one frame runs at a time** (GPU, 1 worker: 1.44 -> 1.14 s/frame), but
  multi-worker GPU batch didn't speed up (2 workers 0.92 s/frame before and after).
- **GPU batch is limited by host RAM.** Per worker, a frame took 1.19 s alone, 1.61 s with 2
  workers and 3.24 s with 4 (wall 44.2 / 29.9 / 29.9 s — 2 and 4 workers identical). During the
  4-worker run the machine swapped out ~167 MB/s (swap in use 5.0 -> 10.4 GB), 29% of CPU time went
  to the kernel, the CPU was still 34% idle and the GPU was busy only ~30% of the time. 1.2 GiB per
  GPU worker (vs ~0.4 GiB per CPU-only worker) is what Part B removes, so Part B goes ahead.
- **CPU batch** is limited by cores and memory bandwidth, not disk (dev sandbox, 16 frames, vmstat:
  iowait 0, 8 workers keep all 8 physical cores busy; Part A took 8 workers from 17 s to 14.3 s).
  Part B doesn't change the CPU path.
- Not measured (was probe item 4): the service's own host memory. B3 starts from the measured GPU
  worker overhead (`_GPU_HOST_OVERHEAD_BYTES`, 1 GiB) plus one frame's working memory, and B4's
  bench refits it.
- One odd result, not reproduced: in the bench's first CPU run (1 worker) 7 of 37 input scans failed
  to decode (`LIBDEFLATE_BAD_DATA`) and read fine in every other run, on both machines; the files
  are unmodified.

### Task B1: Frames in shared memory; decode straight into them

**Files:**
- Create: `src/halide/shared_frames.py`
- Modify: `src/halide/io/tiff.py` (`read_tiff(path, out=None)`), `src/halide/processing.py`
  (`_read_scan(path, out=None)`)
- Test: `tests/unit/test_shared_frames.py`, additions to `tests/unit/test_tiff_io.py`

**Interfaces:**
- Produces:
  ```python
  @contextmanager
  def new_frame(shape: tuple[int, ...], dtype) -> Iterator[SharedFrame]   # creates, yields, unlinks
  @contextmanager
  def attach_frame(name: str, shape, dtype) -> Iterator[np.ndarray]       # service side; track=False
  @dataclass(frozen=True)
  class SharedFrame:
      name: str; shape: tuple[int, ...]; dtype: str; array: np.ndarray
  class SharedMemoryUnavailable(RuntimeError): ...
  def read_tiff(path, out: np.ndarray | None = None) -> RawScan   # decodes into `out` when given
  ```

- [ ] **Step 1: Failing tests.**
  - `read_tiff(path, out=buf)` returns pixels identical to `read_tiff(path)` and writes them into
    `buf` (same object) for float32 scans. For uint8/16/32 scans it normalises into `buf`,
    matching today's values exactly.
  - `new_frame` removes its segment on normal exit and on exception.
  - `attach_frame` sees the creator's data and doesn't unlink.
  - `new_frame` raises `SharedMemoryUnavailable` when creation fails (monkeypatch
    `SharedMemory` to raise `OSError`).
  - A child process that creates a frame and is killed leaves no segment after the resource
    tracker runs (Linux only; skip elsewhere).
- [ ] **Step 2: Run, verify they fail.**
- [ ] **Step 3: Implement.** For float32 input, `tifffile.TiffFile.asarray(out=...)` decodes
  directly into the buffer with no extra copy. For other dtypes, decode then normalise into
  `out` with `np.divide(..., out=)`, using the same arithmetic as today. The header (shape and
  dtype) is read first to size the frame. `SharedMemory(..., track=False)` when attaching
  (Python ≥ 3.13).
- [ ] **Step 4: Full suite, CPU outputs bit-identical** (test_banding etc. unchanged).
- [ ] **Step 5: Commit.**

### Task B2: The GPU service process

**Files:**
- Create: `src/halide/gpu_service.py`
- Modify: `src/halide/processing.py`. Make each device job a module-level function with
  picklable arguments, so the service can run exactly what `_run_on_device`'s closures run today:
  `develop_request(frame, request, band_bytes)`, `print_request(...)`, `export_request(...)`, with
  request dataclasses holding the profile, tone params, stage, scan gain and source ICC matrix.
  The in-process device path then calls these same functions, so there is one implementation.
- Test: `tests/unit/test_gpu_service.py`

**Interfaces:**
- Produces:
  ```python
  @contextmanager
  def running_service(device_kind: str) -> Iterator[ServiceAddress]   # "gpu" normally; "cpu" in tests
  class ServiceClient:
      def __init__(self, address: ServiceAddress) -> None
      def develop(self, frame: SharedFrame, request: DevelopRequest) -> DevelopReply   # raises ServiceUnavailable
      def print_(self, frame: SharedFrame, request: PrintRequest) -> PrintReply
      def export(self, frame: SharedFrame, out: SharedFrame, request: ExportRequest) -> None
  class ServiceUnavailable(RuntimeError): ...   # dead/unreachable service → caller develops on the CPU
  ```
  Transport: `multiprocessing.connection.Listener`/`Client`. That's a Unix socket on Linux and a
  named pipe on Windows, with a random `authkey`. Each worker holds one connection, and the
  service reads each connection on its own thread.

  The service is started with the **spawn** context, so CUDA is initialised only inside it. One
  compute thread takes requests from a queue and runs them one at a time on the GPU (~0.14 s per
  frame; enough for ~7 frames/s). The frame buffer stays the worker's shared-memory segment
  throughout:
  - the service attaches and uploads;
  - it runs the same device code as today;
  - it downloads into the same segment and replies with the `ResolvedTone`/profile.

  An error inside a request (out of memory etc.) is returned as a reply. The worker then
  develops that frame on the CPU, exactly as `_run_on_device` does today, including the
  re-read-from-disk rule when the failure was during download.
- [ ] **Step 1: Failing tests** (no GPU needed):
  - A `"cpu"`-kind service, in real separate processes, develops, prints and exports frames
    **bit-identical** to calling the CPU functions directly. This proves the IPC and
    shared-memory plumbing across real processes.
  - A service running the strict fake device in a thread (in-process, `register_namespace` as
    in the existing fake-device tests) is bit-identical too. This proves the device code runs
    unchanged.
  - A request error comes back as a reply and doesn't kill the service.
  - Killing the service process makes the next `develop` raise `ServiceUnavailable` within a
    bounded time, with no hang.
  - Leaving `running_service` stops the process even when requests are in flight.
  - The module and the service's own imports keep cupy out of the parent
    (`"cupy" not in sys.modules` in the parent after starting a `"gpu"` service with a fake).
- [ ] **Step 2-4:** run → implement → run, full suite.
- [ ] **Step 5: Commit.**

### Task B3: Workers use the service; the worker count stops being VRAM-bound

**Files:** Modify `src/halide/batch/orchestrator.py` (`run_batch`, `run_print_batch`,
`run_export_batch`, `default_worker_count`, the worker functions), `src/halide/gui/proof_window.py`
(its pool), `src/halide/processing.py` (`process_scan`/`print_scan`/`export_delivery_image`
accept `service: ServiceClient | None`); tests in `tests/unit/test_orchestrator.py`.

- [ ] **Step 1: Failing tests.**
  - On a GPU device, `run_batch` starts one service and the workers get its address, not
    `("gpu", pool_limit)`. No worker resolves a CUDA device: every worker reports no cupy
    import, checked with a real pool and a `"cpu"`-kind service.
  - The GPU worker count is `min(physical cores, RAM cap, jobs)`, where the RAM cap uses the CPU
    worker constants plus one shared frame each, minus the service's own host memory (a
    constant, fitted from B0 item 4). There is no VRAM cap per worker any more: the service
    holds one context and one frame's working memory.
  - Service unavailable at start (e.g. `SharedMemoryUnavailable`, or the service fails to start)
    → today's per-worker GPU mode, with a run-sheet warning saying why.
  - Service dies mid-batch → the remaining frames are developed on the CPU with a warning each;
    the batch completes.
  - The GUI proof window starts and stops the service with its pool, including when the window
    is closed mid-render.
  - The CPU path is untouched: identical worker args and outputs.
- [ ] **Step 2-4:** run → implement → run, full suite.
- [ ] **Step 5: Commit.**

### Task B4: Real-GPU verification and the benchmark

**Files:** `tests/gpu/test_gpu_parity.py` (add: service path vs in-process GPU path,
**bit-identical**, on synthetic images and the real scans, deleting outputs after each test, as
today), `docs/plans/gpu-acceleration-bench.py` (a service-mode GPU row: workers auto, plus 4 and 8,
reporting the service's and the workers' host memory and VRAM).

- [ ] Write the tests and bench changes; validate what can be validated here (`--devices cpu`,
  and the `"cpu"`-kind service).
- [ ] **User runs `pytest -m gpu` and the bench.** Acceptance:
  - all GPU tests pass;
  - the service-mode GPU batch is faster than today's GPU batch (0.73 s/frame) by a margin worth
    its complexity. The target is the B0 ceiling; below ~1.5x today, discuss with the user
    before keeping it.
  - The numbers are recorded in this plan.

### Task B5: Record it

- [ ] `CLAUDE.md`: a "Decisions and why" entry covering:
  - why one GPU service rather than more frames in VRAM, or threads (with the investigation's
    numbers);
  - the exiftool session;
  - the shared-memory hand-off, and the `/dev/shm`-is-RAM caveat;
  - failure handling;
  - the new worker-count rule and its fitted constant;
  - B0's and B4's measurements.
- [ ] Update the Architecture tree (`io/exiftool.py`, `shared_frames.py`, `gpu_service.py`),
  `docs/README.md`, this plan's status, and the investigation ("acted on").
- [ ] Commit.

---

## Out of scope

- More than one frame on the GPU at a time (CUDA streams). The service runs frames one at a
  time, with capacity ~7 frames/s; revisit only if B4 shows the GPU becoming the queue.
- PNG export's own compression (Pillow zlib, ~4.9 s per PNG; JPEG ~0.04 s). It's a real cost in
  `halide export`, but it isn't about the GPU.
- Changing output compression or format. That changes files, so it's the user's call, and only
  relevant if B0 shows the disk is the limit.
