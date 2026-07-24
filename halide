#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2
import concurrent.futures
from multiprocessing import Manager
import random


# ==========================================
# Terminal Aesthetics & State
# ==========================================
class T:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    @staticmethod
    def CURSOR_UP(n):
        return f"\033[{n}A"  # Terminal Variables for batch processing


# The visual block character
BLOCK = "██"

# Dictionary mapping state integer to a colored block
STATE_COLORS = {
    0: f"{T.DIM}{BLOCK}{T.RESET}",  # Pending (Dimmed)
    1: f"{T.CYAN}{BLOCK}{T.RESET}",  # Computing The Maths
    2: f"{T.MAGENTA}{BLOCK}{T.RESET}",  # Inverting / Compressing (Magenta)
    3: f"{T.GREEN}{BLOCK}{T.RESET}",  # Complete (Green)
}

# Global flag to mute verbose output during batch processing
QUIET_MODE = False


def print_step(msg):
    if not QUIET_MODE:
        print(f"{T.CYAN}[ * ]{T.RESET} {msg}")


def print_success(msg):
    if not QUIET_MODE:
        print(f"{T.GREEN}[ ✔ ]{T.RESET} {msg}")


def print_warning(msg):
    if not QUIET_MODE:
        print(f"{T.YELLOW}[ ! ]{T.RESET} {msg}")


def progress_bar(iteration, total, prefix="", length=40):
    """Dynamic terminal progress bar."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{T.CYAN}{bar}{T.RESET}| {percent}% ")
    sys.stdout.flush()
    if iteration == total:
        print()


# ==========================================
# Logic: Interactive Picker
# ==========================================
def interactive_density_balance(img_array, title_override=None):
    if not QUIET_MODE:
        print_step("Opening interactive picker...")
        print(f"      {T.DIM}(Waiting for user input via Matplotlib...){T.RESET}")

    display_img = img_array[::4, ::4, :].copy()
    display_img = np.maximum(display_img, 0.0001)  # Avoid division by 0

    display_img = 0.01 / display_img

    # Normalize back to 0-1 range for the plot
    d_min, d_max = display_img.min(), display_img.max()
    display_img = (display_img - d_min) / (d_max - d_min)

    display_img = np.clip(display_img, 0, 1) ** (1 / 2.2)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(display_img)

    title = (
        title_override
        if title_override
        else "1. Click a NEUTRAL SHADOW (Dmax)\n2. Click a NEUTRAL HIGHLIGHT (Dmin)"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.axis("off")

    coords = []

    def onclick(event):
        if event.inaxes is None:
            return
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode != "":
            return

        coords.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, "r+", markersize=12, markeredgewidth=2)
        fig.canvas.draw()

        if len(coords) == 2:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=True)

    if len(coords) != 2:
        print_warning(
            "You must click exactly two points. Falling back to 1.0 defaults."
        )
        return 1.0, 1.0, 1.0, 1.0

    x1, y1 = int(coords[0][0] * 4), int(coords[0][1] * 4)
    x2, y2 = int(coords[1][0] * 4), int(coords[1][1] * 4)

    t2_raw = np.median(img_array[y1 - 1 : y1 + 2, x1 - 1 : x1 + 2], axis=(0, 1))
    t1_raw = np.median(img_array[y2 - 1 : y2 + 2, x2 - 1 : x2 + 2], axis=(0, 1))

    t1_raw = np.maximum(t1_raw, 0.00002)
    t2_raw = np.maximum(t2_raw, 0.00001)

    r1_raw, g1_raw, b1_raw = t1_raw[0], t1_raw[1], t1_raw[2]
    r2_raw, g2_raw, b2_raw = t2_raw[0], t2_raw[1], t2_raw[2]

    r1_log, g1_log, b1_log = [np.log10(1 / c) for c in (r1_raw, g1_raw, b1_raw)]
    r2_log, g2_log, b2_log = [np.log10(1 / c) for c in (r2_raw, g2_raw, b2_raw)]

    rs = 1 / (r2_log - r1_log) if (r2_log - r1_log) != 0 else 1
    gs = 1 / (g2_log - g1_log) if (g2_log - g1_log) != 0 else 1
    bs = 1 / (b2_log - b1_log) if (b2_log - b1_log) != 0 else 1

    rs_norm, bs_norm, gs_norm = rs / gs, bs / gs, 1.0
    rd, bd, md = r1_log * rs_norm, b1_log * bs_norm, g1_log * gs_norm

    ra = (md - rd) / rs_norm if rs_norm != 0 else 0
    ba = (md - bd) / bs_norm if bs_norm != 0 else 0

    rm, bm = 1 / (10**ra), 1 / (10**ba)
    red_gamma, blue_gamma = 1 / rs_norm, 1 / bs_norm

    return rm, bm, red_gamma, blue_gamma


def auto_density_balance(img_array):
    if not QUIET_MODE:
        print_step("Calculating statistical neutral points...")

    # t2 is the film base (Dmin of the negative, highest transmittance)
    t2_raw = np.percentile(img_array, 99.9, axis=(0, 1))

    # t1 is the densest highlight (Dmax of the negative, lowest transmittance)
    t1_raw = np.percentile(img_array, 0.1, axis=(0, 1))

    t1_raw = np.maximum(t1_raw, 0.00002)
    t2_raw = np.maximum(t2_raw, 0.00001)

    r1_raw, g1_raw, b1_raw = t1_raw[0], t1_raw[1], t1_raw[2]
    r2_raw, g2_raw, b2_raw = t2_raw[0], t2_raw[1], t2_raw[2]

    r1_log, g1_log, b1_log = [np.log10(1 / c) for c in (r1_raw, g1_raw, b1_raw)]
    r2_log, g2_log, b2_log = [np.log10(1 / c) for c in (r2_raw, g2_raw, b2_raw)]

    rs = 1 / (r2_log - r1_log) if (r2_log - r1_log) != 0 else 1
    gs = 1 / (g2_log - g1_log) if (g2_log - g1_log) != 0 else 1
    bs = 1 / (b2_log - b1_log) if (b2_log - b1_log) != 0 else 1

    rs_norm, bs_norm, gs_norm = rs / gs, bs / gs, 1.0
    rd, bd, md = r1_log * rs_norm, b1_log * bs_norm, g1_log * gs_norm

    ra = (md - rd) / rs_norm if rs_norm != 0 else 0
    ba = (md - bd) / bs_norm if bs_norm != 0 else 0

    rm, bm = 1 / (10**ra), 1 / (10**ba)
    red_gamma, blue_gamma = 1 / rs_norm, 1 / bs_norm

    return rm, bm, red_gamma, blue_gamma


def roll_analysis_density_balance(files):
    # 2^20 bins provides extreme precision for 32-bit floating point images
    # while keeping the histogram array size under 25MB in RAM.
    NUM_BINS = 1048576
    global_hist = np.zeros((3, NUM_BINS), dtype=np.int64)
    total_pixels = 0

    for i, f in enumerate(files):
        with tifffile.TiffFile(f) as tif:
            raw_img = tif.asarray()[::8, ::8, :]

            # Expanded read logic to explicitly handle 32-bit formats
            if raw_img.dtype == np.uint16:
                img_array = raw_img.astype(np.float32) / 65535.0
            elif raw_img.dtype == np.uint8:
                img_array = raw_img.astype(np.float32) / 255.0
            elif raw_img.dtype == np.uint32:
                img_array = raw_img.astype(np.float32) / 4294967295.0
            else:
                # Assumes np.float32 (often 0.0-1.0 HDRi scans).
                # Clipped to prevent out-of-bounds histogram binning.
                img_array = np.clip(raw_img.astype(np.float32), 0.0, 1.0)

            for channel in range(3):
                hist, _ = np.histogram(
                    img_array[:, :, channel], bins=NUM_BINS, range=(0.0, 1.0)
                )
                global_hist[channel] += hist

            total_pixels += img_array.shape[0] * img_array.shape[1]

            sys.stdout.write(
                f"\r      {T.DIM}Analyzing frame {i + 1}/{len(files)}...{T.RESET}"
            )
            sys.stdout.flush()

    print()

    t1_raw = np.zeros(3)
    t2_raw = np.zeros(3)

    for channel in range(3):
        cdf = np.cumsum(global_hist[channel]) / total_pixels

        p001_idx = np.searchsorted(cdf, 0.001)
        p999_idx = np.searchsorted(cdf, 0.999)

        t1_raw[channel] = p001_idx / float(NUM_BINS - 1)
        t2_raw[channel] = p999_idx / float(NUM_BINS - 1)

    t1_raw = np.maximum(t1_raw, 0.00002)
    t2_raw = np.maximum(t2_raw, 0.00001)

    r1_raw, g1_raw, b1_raw = t1_raw[0], t1_raw[1], t1_raw[2]
    r2_raw, g2_raw, b2_raw = t2_raw[0], t2_raw[1], t2_raw[2]

    r1_log, g1_log, b1_log = [np.log10(1 / c) for c in (r1_raw, g1_raw, b1_raw)]
    r2_log, g2_log, b2_log = [np.log10(1 / c) for c in (r2_raw, g2_raw, b2_raw)]

    rs = 1 / (r2_log - r1_log) if (r2_log - r1_log) != 0 else 1
    gs = 1 / (g2_log - g1_log) if (g2_log - g1_log) != 0 else 1
    bs = 1 / (b2_log - b1_log) if (b2_log - b1_log) != 0 else 1

    rs_norm, bs_norm, gs_norm = rs / gs, bs / gs, 1.0
    rd, bd, md = r1_log * rs_norm, b1_log * bs_norm, g1_log * gs_norm

    ra = (md - rd) / rs_norm if rs_norm != 0 else 0
    ba = (md - bd) / bs_norm if bs_norm != 0 else 0

    rm, bm = 1 / (10**ra), 1 / (10**ba)
    red_gamma, blue_gamma = 1 / rs_norm, 1 / bs_norm

    return rm, bm, red_gamma, blue_gamma


# ==========================================
# Logic: Edge-Preserving Denoise
# ==========================================
def denoise_negative(img_array, intensity=3):
    """
    Applies a Bilateral Filter to kill scanner noise and film grain
    while preserving sharp photographic edges.
    """
    print_step(f"Applying bilateral noise reduction (Level {intensity})...")

    # OpenCV's bilateral filter works natively on 32-bit floats.
    # d=0 allows the algorithm to calculate radius based on sigmaSpace
    # sigmaColor: How much colors can mix (higher = more smoothing)
    # sigmaSpace: How far pixels can mix (higher = wider blur)

    sigma = intensity / 100.0  # Scales the UI intensity to float math
    denoised = cv2.bilateralFilter(img_array, d=0, sigmaColor=sigma, sigmaSpace=5)

    return denoised


# ==========================================
# Logic: Auto-Exposure
# ==========================================
def global_auto_exposure(img_array, headroom=0.02):
    print_step("Calculating global auto-exposure (Histogram Stretch)...")
    global_min = np.percentile(img_array, 0.05)
    global_max = np.percentile(img_array, 99.95)

    new_max = 1 - headroom
    new_min = headroom

    if global_max > global_min:
        scale_factor = (new_max - new_min) / (global_max - global_min)
        img_array -= global_min
        img_array *= scale_factor
        img_array += new_min

        if not QUIET_MODE:
            print(f"      {T.DIM}-> Black point stretched to {headroom:.3f}{T.RESET}")
            print(
                f"      {T.DIM}-> White point stretched to {1.0 - headroom:.3f}{T.RESET}"
            )

        return np.clip(img_array, 0.0, 1.0, out=img_array)

    print_warning("Could not calculate valid stretch range. Skipping auto-exposure.")
    return img_array


# ==========================================
# Core Processing Pipeline
# ==========================================
def process_core(
    input_file,
    output_file,
    args,
    override_density=None,
    file_index=None,
    shared_grid=None,
):
    """The central math engine, decoupled from terminal UI prints."""
    with tifffile.TiffFile(input_file) as tif:
        raw_array = tif.asarray()
        icc_tag = tif.pages[0].tags.get(34675)
        icc_profile = icc_tag.value if icc_tag else None

    if raw_array.dtype == np.uint16:
        print_step("Normalizing 16-bit integer to 32-bit float.")
        img_array = raw_array.astype(np.float32) / 65535.0
    elif raw_array.dtype == np.uint8:
        img_array = raw_array.astype(np.float32) / 255.0
    else:
        img_array = raw_array.astype(np.float32)

    # --- Apply Denoising ---
    if args.denoise:
        img_array = denoise_negative(img_array, intensity=args.denoise_level)

    # B&W Pre-Processing
    if args.bw:
        print_step("Black & White mode active. Applying grayscale to image...")
        img_gray = np.mean(img_array, axis=2)
        img_safe = np.maximum(img_gray, 0.0000001)

    else:
        # Density Balance
        if not args.invert_only:
            if override_density:
                rm, bm, rg, bg = override_density
            elif args.rm is not None and args.bm is not None:
                rm, bm, rg, bg = args.rm, args.bm, args.rg, args.bg
            elif args.auto_density:
                rm, bm, rg, bg = auto_density_balance(img_array)
            else:
                title = f"Picker: {Path(input_file).name}\n1. Shadow | 2. Highlight"
                rm, bm, rg, bg = interactive_density_balance(
                    img_array, title_override=title
                )

            if not QUIET_MODE:
                print(f"\n{T.CYAN}    ┌── Density Constants ───────────┐{T.RESET}")
                print(
                    f"{T.CYAN}    │{T.RESET} R_Mult: {T.BOLD}{rm:6.4f}{T.RESET} {T.CYAN}│{T.RESET} R_Gam: {T.BOLD}{rg:6.4f}{T.RESET} {T.CYAN}│{T.RESET}"
                )
                print(
                    f"{T.CYAN}    │{T.RESET} B_Mult: {T.BOLD}{bm:6.4f}{T.RESET} {T.CYAN}│{T.RESET} B_Gam: {T.BOLD}{bg:6.4f}{T.RESET} {T.CYAN}│{T.RESET}"
                )
                print(f"{T.CYAN}    └────────────────────────────────┘{T.RESET}\n")

            print_step("Applying exposure multipliers and gamma curves...")
            img_array[:, :, 0] *= rm
            img_array[:, :, 2] *= bm

            img_safe = np.maximum(img_array, 0.0000001)

            img_safe[:, :, 0] = img_safe[:, :, 0] ** (1.0 / rg)
            img_safe[:, :, 2] = img_safe[:, :, 2] ** (1.0 / bg)

            img_safe = np.maximum(img_safe, 0.0000001)

        else:
            print_step("Skipping Density Balance (--invert-only active).")
            img_safe = np.maximum(img_array, 0.0000001)

    # Inversion
    # Trigger state 2: Inversion
    if shared_grid is not None and file_index is not None:
        shared_grid[file_index] = 2

    if not args.density_only:
        print_step("Executing 0.01/x mathematical inversion...")
        final_array = 0.01 / img_safe
        # Prevent the small pixel values from leading to impossible luminances
        final_array = np.clip(final_array, 0.0, 1.0)

        if args.auto_expose:
            final_array = global_auto_exposure(final_array)
    else:
        print_step("Skipping Inversion (--density-only active).")
        final_array = img_safe

    # Save
    print_step(f"Writing to: {T.BOLD}{output_file}{T.RESET}")
    write_kwargs = {
        "compression": "zlib",
        "compressionargs": {
            "level": 6
        },  # Compression level 1-9 (6 is the optimal balance of speed/size)
        "predictor": 3,
    }
    if icc_profile and not args.bw:
        write_kwargs["extratags"] = [(34675, "B", len(icc_profile), icc_profile)]

    tifffile.imwrite(output_file, final_array, **write_kwargs)

    try:
        # Dynamically build the ExifTool command
        exif_cmd = [
            "exiftool",
            "-TagsFromFile",
            input_file,
            "-all:all",
            "--ExifImageWidth",  # Drops the redundant 16-bit EXIF width
            "--ExifImageHeight",  # Drops the redundant 16-bit EXIF height
        ]

        # If B&W, explicitly forbid ExifTool from copying the RGB color space
        if args.bw:
            exif_cmd.append("--icc_profile")

        exif_cmd.extend(["-overwrite_original", output_file])

        subprocess.run(
            exif_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print_warning("ExifTool not found. Metadata skipped.")


# ==========================================
# Workflow Managers
# ==========================================
def process_single(input_path, output_path, args):
    """Standard verbose single-file processor."""
    start_time = time.time()
    print(f"\n{T.BOLD}{T.MAGENTA}=== Alchemy Linear Pipeline (Single) ==={T.RESET}\n")
    print_step(f"Reading: {T.BOLD}{input_path}{T.RESET}")
    process_core(input_path, output_path, args)
    elapsed = time.time() - start_time
    print(
        f"\n{T.GREEN}{T.BOLD}SUCCESS:{T.RESET} Process completed in {elapsed:.2f} seconds.\n"
    )


# Worker task for multiprocessing
def worker_task(
    file_index,
    shared_grid,
    file_path,
    out_directory,
    suffix,
    arguments,
    anchor_density,
    per_frame_dict,
    quiet_flag,
):

    global QUIET_MODE
    QUIET_MODE = quiet_flag

    # Trigger State 1: Maths
    if shared_grid is not None and file_index is not None:
        shared_grid[file_index] = 1

    if suffix:
        new_fname = f"{file_path.stem}{suffix}{file_path.suffix}"
    else:
        new_fname = file_path.name

    output_file = out_directory / new_fname

    # Determine the correct density to use
    frame_density = anchor_density
    if file_path.name in per_frame_dict:
        frame_density = per_frame_dict[file_path.name]

    # Run the core math
    process_core(
        str(file_path),
        str(output_file),
        arguments,
        override_density=frame_density,
        file_index=file_index,
        shared_grid=shared_grid,
    )

    # Trigger State 3: Done
    if shared_grid is not None and file_index is not None:
        shared_grid[file_index] = 3

    return file_path.name


def process_batch(input_dir, output_dir, args):
    """Quiet, progress-bar driven batch processor."""
    global QUIET_MODE

    # Restrict underlying C libraries to single-threaded mode to prevent thrashing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print(f"\n{T.BOLD}{T.MAGENTA}=== Alchemy Linear Pipeline (Batch) ==={T.RESET}\n")

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
    ]
    if not files:
        print_warning(f"No TIFF files found in {input_dir}")
        return

    print(f"Found {T.BOLD}{len(files)}{T.RESET} image(s) to process.\n")

    anchor_density = None
    per_frame_densities = {}

    # Batch Density Balance Prompting
    if not args.invert_only and (args.rm is None or args.bm is None) and not args.bw:
        print(f"{T.CYAN}Density Balance Selection Method:{T.RESET}")
        print(
            "  [1] Anchor Frame (Pick neutral points once, apply math to entire roll)"
        )
        print(
            "  [2] Per-Frame    (Density is calculating individually for every image)"
        )
        print(
            "  [3] Roll-Analysis    (Guesses the best neutral points by analysing every image)"
        )

        while True:
            choice = input(f"\nSelect mode (1-3): ").strip()
            if choice in ["1", "2", "3"]:
                break
            print("Invalid choice.")

        if choice == "1":
            print(f"\n{T.DIM}Available Frames:{T.RESET}")
            for i, f in enumerate(files):
                print(f"  [{i + 1:02d}] {f.name}")

            while True:
                try:
                    idx = int(input(f"\nEnter the number of your Anchor Frame: ")) - 1
                    if 0 <= idx < len(files):
                        break
                except ValueError:
                    pass
                print("Invalid index.")

            anchor_file = files[idx]
            print(f"\n{T.CYAN}Loading Anchor Frame: {anchor_file.name}{T.RESET}")

            # Briefly unmute to load the anchor frame normally
            QUIET_MODE = False
            with tifffile.TiffFile(anchor_file) as tif:
                raw_anchor = tif.asarray()

                if raw_anchor.dtype == np.uint16:
                    anchor_array = raw_anchor.astype(np.float32) / 65535.0
                elif raw_anchor.dtype == np.uint8:
                    anchor_array = raw_anchor.astype(np.float32) / 255.0
                else:
                    anchor_array = raw_anchor.astype(np.float32)

            if args.auto_density:
                anchor_density = auto_density_balance(anchor_array)
            else:
                anchor_density = interactive_density_balance(
                    anchor_array, title_override=f"ANCHOR: {anchor_file.name}"
                )
            print(f"{T.GREEN}Anchor density locked. Applying to batch.{T.RESET}\n")

        elif choice == "2":
            print(f"\n{T.CYAN}Calculating Per-Frame...{T.RESET}")
            QUIET_MODE = True
            for f in files:
                with tifffile.TiffFile(f) as tif:
                    raw_img = tif.asarray()
                    if raw_img.dtype == np.uint16:
                        img_array = raw_img.astype(np.float32) / 65535.0
                    elif raw_img.dtype == np.uint8:
                        img_array = raw_img.astype(np.float32) / 255.0
                    else:
                        img_array = raw_img.astype(np.float32)

                if args.auto_density:
                    per_frame_densities[f.name] = auto_density_balance(img_array)
                else:
                    per_frame_densities[f.name] = interactive_density_balance(
                        img_array, title_override=f"PICKER: {f.name}"
                    )
            print(f"{T.GREEN}All densities locked. Applying to batch.{T.RESET}\n")

        elif choice == "3":
            print(f"\n{T.CYAN}Analysing Roll...{T.RESET}")
            QUIET_MODE = True
            anchor_density = roll_analysis_density_balance(files)
            print(f"{T.GREEN}Roll Analysis Complete. Applying to batch.{T.RESET}\n")

    # --- Filename Suffix Prompt ---
    print(f"{T.CYAN}Output Formatting:{T.RESET}")
    filename_suffix = input(
        "  Enter a suffix to append to filenames (press Enter to skip): "
    ).strip()
    print()  # Add a clean newline before the progress bar starts

    # Start Batch Processing Loop
    QUIET_MODE = True
    start_time = time.time()

    safe_ram_workers = 6
    optimal_workers = min(os.cpu_count() or 1, safe_ram_workers)

    # --- GRID MATRIX MATH ---
    total_files = len(files)
    columns = 9
    rows = (total_files + columns - 1) // columns

    # Calculate exactly how many lines to jump up for the redraw
    total_lines_to_draw = (rows * 2) + 1

    # Pre-print empty space to ensure the terminal doesn't scroll wildly
    for _ in range(total_lines_to_draw):
        print()

    with Manager() as manager:
        # Initialize all blocks to State 0
        shared_grid = manager.dict({i: 0 for i in range(total_files)})

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=optimal_workers
        ) as executor:
            futures = []

            # Shuffle the order of the files to make the animation more aesthetic
            submission_order = list(enumerate(files))
            random.shuffle(submission_order)

            # Submit all files to the workers
            for i, f in submission_order:
                futures.append(
                    executor.submit(
                        worker_task,
                        i,
                        shared_grid,
                        f,
                        out_dir,
                        filename_suffix,
                        args,
                        anchor_density,
                        per_frame_densities,
                        QUIET_MODE,
                    )
                )

            # THE RENDER LOOP
            completed = 0
            while completed < total_files:
                completed = sum(1 for f in futures if f.done())

                # Snap cursor to the top of our grid space
                sys.stdout.write(T.CURSOR_UP(total_lines_to_draw))

                # Draw the Matrix
                for r in range(rows):
                    row_str = ""
                    for c in range(columns):
                        idx = r * columns + c
                        if idx < total_files:
                            state = shared_grid[idx]
                            row_str += STATE_COLORS[state] + " "

                    sys.stdout.write(f"{T.CLEAR_LINE}    {row_str}\n")

                    # Print vertical gap between rows
                    if r < rows - 1:
                        sys.stdout.write(f"{T.CLEAR_LINE}\n")

                # Draw Progress Footer
                sys.stdout.write(f"{T.CLEAR_LINE}\n")
                sys.stdout.write(
                    f"{T.CLEAR_LINE}    [ Processed {completed}/{total_files} Frames ]\n"
                )

                sys.stdout.flush()
                time.sleep(0.05)  # Refresh at 20fps

    elapsed = time.time() - start_time
    QUIET_MODE = False
    print(
        f"\n{T.GREEN}{T.BOLD}BATCH SUCCESS:{T.RESET} Processed {len(files)} files in {elapsed:.2f} seconds.\n"
    )


# ==========================================
# CLI Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert Linear Film Negatives.")
    parser.add_argument("input", help="Input TIFF file OR directory of TIFFs")
    parser.add_argument("output", help="Output TIFF file OR directory")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--invert-only", action="store_true", help="Skip density balance"
    )
    mode_group.add_argument(
        "--density-only", action="store_true", help="Skip inversion"
    )

    # Add black and white flag
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Black & White mode (collapses channels, skips density balance)",
    )

    parser.add_argument(
        "--auto-expose",
        action="store_true",
        help="Automatically stretches the histogram to the full dynamic range (not recommended for production)",
    )

    parser.add_argument(
        "--auto-density",
        action="store_true",
        help="Automatically selects neutral points (results vary in quality depending on the image)",
    )

    # Denoise Controls (Now Opt-In)
    parser.add_argument(
        "--denoise", action="store_true", help="Enable bilateral noise reduction"
    )
    parser.add_argument(
        "--denoise-level",
        type=int,
        default=3,
        help="Intensity of grain/noise reduction if --denoise is active (1-10, default: 3)",
    )

    parser.add_argument("--rm", type=float, help="Red Exposure Multiplier")
    parser.add_argument("--bm", type=float, help="Blue Exposure Multiplier")
    parser.add_argument("--rg", type=float, default=1.0, help="Red Gamma")
    parser.add_argument("--bg", type=float, default=1.0, help="Blue Gamma")

    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        process_batch(args.input, args.output, args)
    elif in_path.is_file():
        process_single(args.input, args.output, args)
    else:
        print_warning(f"Input path not found: {args.input}")
