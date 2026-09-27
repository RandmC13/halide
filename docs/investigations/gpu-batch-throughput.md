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
many frames can be read and written at once. Fixing that, and a second cost (exiftool rewriting
every output file), is where the speed is.

## Where one GPU batch frame's time goes

One GPU worker on the user's machine: **1.44 s/frame** (benchmark). Broken down:

| step | where | time per frame | source |
|---|---|---:|---|
| read + decode the scan TIFF (deflate) | CPU | ~0.26 s | measured, dev sandbox, IMG_0158 |
| upload to the GPU | PCIe | 0.066 s | user's probe |
| all the arithmetic (ICC, balance, invert, print fit, paper curve) | GPU | ~0.045 s | user's probe |
| download | PCIe | 0.026 s | user's probe |
| write + compress the output TIFF (zlib 6) | CPU | ~0.26 s | measured, dev sandbox |
| **exiftool copying EXIF onto the output** | CPU + disk | **~0.65 s** | measured, dev sandbox (Perl start-up is only 0.08 s; the rest is rewriting the whole 120 MiB file, because inserting metadata into a TIFF means writing a new file) |
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

## What would actually help, in order

1. **Stop exiftool rewriting every output file** (~0.65 s/frame, about half the CPU work per
   frame, and half the disk writes). This helps the CPU path equally. Options to investigate:
   - write the EXIF tags into the TIFF when halide first writes it, in the same pass. tifffile
     writes main-IFD tags (`extratags`); an EXIF sub-IFD needs checking, and correctness of every
     copied tag has to be verified against exiftool's output on real scans;
   - keep exiftool but give it less to do (e.g. one long-running `-stay_open` process: saves only
     the 0.08 s start-up, not the rewrite, so a small win).
   The metadata matters: `halide check` and `--match-scan-exposure` read the scan's exposure from
   it. So this has to be byte-for-byte faithful to what exiftool copies today.
2. **One GPU process for a GPU batch, instead of one per worker** (the user's idea, aimed at the
   real bottleneck). A single process holds one CUDA context. A pool of threads reads and decodes
   scans, then hands each frame to the one GPU, and a second pool compresses, writes and tags the
   results. tifffile/imagecodecs, zlib and exiftool all work outside Python's lock, so threads
   run truly in parallel. Each extra frame in flight then costs ~0.2-0.4 GiB of RAM (its buffers),
   not ~1.2 GiB (a whole process). All 8 cores can work on I/O, and the GPU takes frames one at a
   time from a short queue. Two or three frames on the card is enough, and it's never the wait.
   - Expected: the CPU work (~1.2 s/frame today, ~0.55 s after step 1) spread over 8 cores instead
     of 4. The ceiling is then disk speed: 37 frames read ~4.6 GB and write ~4.4 GB (twice that
     while exiftool rewrites). Without numbers from the user's disk, "about 2x" is a guess, not a
     promise.
   - Costs: a new batch runner for GPU mode (the CPU keeps its process pool); a crash in one frame
     must still not lose the rest (threads can't be OOM-killed individually the way processes
     can); the per-frame CPU fallback and warnings stay as they are.
3. Smaller, measured-only: tifffile's multithreaded decode/encode gave nothing here (decode 0.26
   -> 0.23 s; encode with more strips got slower), so it isn't worth pursuing on its own.

## Not investigated

- The user's disk throughput, the real ceiling for step 2. A GPU batch rewrites ~9 GB for a
  37-frame roll today, ~4.5 GB after step 1.
- exiftool's cost on the user's machine (only measured in the sandbox, with exiftool 13.36 from its
  GitHub source).
