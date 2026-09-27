# Why GPU batch isn't much faster, and what would make it faster

Asked 2026-09-27, after the RTX 3070 benchmark (`docs/plans/gpu-acceleration.md` §7) showed GPU
batch only ~19% faster than CPU batch (0.73 vs 0.90 s/frame). The question: an 8 GiB card and
~150 MB scans should hold many frames at once. Why only 4 GPU workers, each loading its own copy
of NVIDIA's libraries? Why not load as many frames as fit into VRAM and process them all in
parallel?

## Short answer

The GPU is idle about 90% of the time. The time goes to work that happens on the CPU around each
frame. Putting more frames on the card at once would add waiting, not speed. The per-worker copies
of the CUDA libraries are real waste, though: they cost ordinary RAM, and RAM is what limits how
many frames can be read and written at once. Fixing that (one GPU process shared by CPU-only
workers), and a second cost (a fresh exiftool per output file), is where the speed is — unless the
disk is already the limit, which hasn't been measured yet.

## Where one GPU batch frame's time goes

One GPU worker on the user's machine: **1.44 s/frame** (benchmark). Broken down:

| step | where | time per frame | source |
|---|---|---:|---|
| read + decode the scan TIFF (deflate) | CPU | ~0.26 s | measured, dev sandbox, IMG_0158 |
| upload to the GPU | PCIe | 0.066 s | user's probe |
| all the arithmetic (ICC, balance, invert, print fit, paper curve) | GPU | ~0.045 s | user's probe |
| download | PCIe | 0.026 s | user's probe |
| write + compress the output TIFF (zlib 6) | CPU | ~0.26 s | measured, dev sandbox |
| **exiftool copying EXIF onto the output** | CPU + disk | **~0.65 s** | measured, dev sandbox (see "Corrections" below for what that time is) |
| process/pool overhead | CPU | remainder | |

The sandbox timings are from a different CPU, but they add up to ~1.3 s against the user's
measured 1.44 s. **The GPU-side work is ~0.14 s of 1.44 s, about 10%.**

## Why "more frames on the GPU at once" wouldn't help

- **One frame already fills the GPU.** Each step is an operation on 47 million numbers, and the
  RTX 3070 has 5,888 cores; a single frame is thousands of times more parallel work than the
  card can run at once. The arithmetic is limited by memory bandwidth (448 GB/s). One full
  pass over a frame takes ~0.85 ms, and the whole develop is ~53 such passes (45 ms). Two frames
  at once take twice as long, not the same time. Batching frames on a GPU pays off when each item
  is too small to fill the card; a 16-megapixel frame isn't.
- **The card isn't the queue.** At 0.14 s of GPU time per frame, one GPU could keep up with ~7
  frames per second. The batch runs at ~1.4 frames per second, because each frame needs ~1.2 s of
  CPU work (read, write, exiftool) before and after its 0.14 s on the card.
- In darkroom terms: the enlarger exposure takes one second. The bottleneck is developing,
  washing and drying the prints. A second enlarger, or a bigger one, doesn't print faster.

## What the per-worker duplication does cost

Each GPU worker is its own process, with its own CUDA context and libraries:

- **VRAM:** ~168 MiB each (measured). Small, and not what limits the count.
- **Ordinary RAM:** ~840 MiB each on top of the ~0.4 GiB a CPU worker needs (measured: largest GPU
  worker ~1.2 GiB, largest CPU worker ~0.4 GiB). This *is* what limits GPU batch to 4 workers on
  the user's 7.6 GiB. Since the time is CPU work (read/write/exiftool), and the machine has 8
  cores, 4 workers leave half the CPU idle during a GPU batch.

## Corrections, from prototypes (same day)

Two things above this section were assumptions, and both turned out wrong when tested:

- **exiftool's cost is mostly exiftool itself, not rewriting the file.** On a 1-pixel TIFF the
  same copy still takes ~0.49 s; Perl start-up is ~0.06 s; the full-size output adds only ~0.17 s.
  So the idea of letting exiftool tag a tiny TIFF and then appending the pixel data after its
  bytes was prototyped: all 382 tags identical to today's output (including Canon's MakerNote,
  which stores absolute file offsets), identical exiftool validation, identical pixels — but no
  faster (1.1 s vs 0.9 s), so it was dropped. **What works: one exiftool kept running**
  (`-stay_open`): 0.58 s for the first file, then **0.28-0.31 s per file**, with output files
  **byte-for-byte identical** to today's.
- **Threads can't share the file work in one process.** tifffile decodes the scans strip by strip
  (3,266 one-row strips per scan) and encodes 817 strips per output, with Python work per strip,
  so it holds Python's global lock most of the time. Measured entirely in memory: decode 0.33 s
  per frame on 1 thread, 0.47-0.58 s per frame on 2-8 threads (worse); encode 0.32 -> 0.25-0.28 s.
  tifffile's own `maxworkers` gave nothing either (0.26 -> 0.23 s). A single process with a
  thread pool would therefore do the file work about one frame at a time. The file work needs
  separate *processes*, so sharing one GPU means one GPU process serving several CPU-only worker
  processes, with frames passed through shared memory.
- **The disk may already be the limit.** A GPU batch today reads ~4.6 GB and writes ~9 GB for 37
  frames (exiftool writes each output twice) in ~27 s — about 0.5 GB/s. The user's scans
  originally live under a path named `hdd`. If that disk is near its limit, no batch design is
  much faster. This has to be measured before the bigger change is built.

## What would actually help, in order

1. **Keep one exiftool running per worker** (`-stay_open`) instead of starting one per output:
   0.66 -> ~0.29 s per frame, output files byte-for-byte identical (see Corrections). Helps the
   CPU path equally. The metadata matters (EXIF, darktable's XMP, IPTC, Canon's MakerNote), so
   "byte-identical to today" is the bar, and it's met.
2. **One GPU process for a GPU batch, instead of one per worker** (the user's idea, aimed at the
   real bottleneck). A single process holds one CUDA context and develops frames one at a time;
   CPU-only worker processes (threads won't do — see Corrections) read, decode, compress, write
   and tag, handing frames to it through shared memory. Each extra worker then costs ~0.4 GiB of
   RAM plus its frame, not ~1.2 GiB, so all 8 cores can do file work, and the GPU (~0.14 s of work
   per frame) takes frames from a short queue.
   - Expected: unknown until the disk is measured (Corrections, last point). If the disk isn't the
     limit, the file work (~0.8 s/frame after step 1) over 8 processes instead of 4 suggests
     roughly twice today's GPU batch speed; if it is, little.
   - Costs: a GPU service process, shared-memory frame hand-off, and keeping today's guarantees
     (one bad frame never loses the rest; any GPU failure redoes that frame on the CPU).
3. Measured and not worth pursuing: tifffile's multithreaded decode/encode (see Corrections), and
   splicing exiftool's metadata onto a separately written file (correct, but not faster).

## Not investigated

- The user's disk throughput, the real ceiling for step 2. A GPU batch writes ~9 GB for a
  37-frame roll today (exiftool still rewrites each output after step 1, so that doesn't change).
- exiftool's cost on the user's machine (only measured in the sandbox, with exiftool 13.36 from its
  GitHub source).
