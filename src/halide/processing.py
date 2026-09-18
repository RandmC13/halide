"""Ties I/O, color management, calibration, and the core pipeline together into "process one
negative scan" — the single place that logic lives, so the single-file CLI command and the batch
orchestrator (which calls this once per file inside a process pool) don't duplicate it.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from halide.calibration.auto import auto_density_balance, roll_auto_density_balance
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.pipeline import run_pipeline
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.icc import (
    UnsupportedICCProfileError,
    convert_to_working_space,
    output_profile_bytes,
    parse_linear_rgb_profile,
)
from halide.io.tiff import copy_exif_metadata, read_tiff, write_tiff

IDENTITY_PROFILE = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))


class Stage(Enum):
    FULL = "full"  # white balance + density balance + invert + tone render
    INVERT_ONLY = "invert_only"  # skip density balance (identity profile), still invert + tone render
    DENSITY_ONLY = "density_only"  # white balance + density balance only, no invert/tone render


class ScanColorError(Exception):
    """Raised when a scan's embedded ICC profile is missing or unsupported."""


def load_working_space_image(path: str | Path) -> np.ndarray:
    """Read a TIFF and convert it into the internal ACEScg working space, validating its embedded
    ICC profile along the way. Raises ScanColorError with a specific, actionable message if the
    profile is missing or unsupported."""
    scan = read_tiff(path)
    if scan.icc_profile is None:
        raise ScanColorError(f"{path}: no embedded ICC profile found; cannot verify color space")
    try:
        source_profile = parse_linear_rgb_profile(scan.icc_profile)
    except UnsupportedICCProfileError as exc:
        raise ScanColorError(f"{path}: unsupported color profile — {exc}") from exc
    return convert_to_working_space(scan.image, source_profile)


def process_scan(
    input_path: str | Path,
    output_path: str | Path,
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
) -> None:
    """Process one negative scan end to end and write the result.

    `density_profile=None` means "compute a per-frame automatic profile from this image" — not
    valid combined with `stage=Stage.INVERT_ONLY`, which always uses the identity profile.
    """
    working_image = load_working_space_image(input_path)

    if stage is Stage.INVERT_ONLY:
        profile = IDENTITY_PROFILE
    elif density_profile is not None:
        profile = density_profile
    else:
        profile = auto_density_balance(working_image)

    if stage is Stage.DENSITY_ONLY:
        result = apply_density_balance(apply_white_balance(working_image, profile), profile)
    else:
        result = run_pipeline(working_image, profile, tone_params)

    write_tiff(output_path, result, icc_profile=output_profile_bytes())
    # Output is always ACEScg, a different profile than the source — exiftool must not clobber
    # the ACEScg tag we just wrote with the source's own ICC bytes.
    copy_exif_metadata(str(input_path), str(output_path), drop_icc=True)


def estimate_roll_density_profile(input_paths: list[str | Path], stride: int = 8) -> DensityProfile:
    """Estimate one shared density-balance profile from a whole roll's worth of input files, for
    --auto-density-roll batch mode. Reads every file (downsampled by `stride` for speed/memory —
    a roll's worth of full-resolution scans held in memory at once would be wasteful for what is
    just a statistical estimate) and combines them via calibration.auto.roll_auto_density_balance.

    Runs in the main process, ahead of (and outside) the per-worker try/except in
    batch.orchestrator — so a single unreadable/malformed file here must not abort the whole roll
    estimate any more than it aborts the rest of the batch. Unreadable files are skipped with a
    warning; the actual per-file error still surfaces normally when that file is processed for
    real in the worker pool.
    """
    images = []
    for path in input_paths:
        try:
            images.append(load_working_space_image(path)[::stride, ::stride, :])
        except ScanColorError as exc:
            print(f"Warning: skipping {path} while estimating roll density balance ({exc})")
    if not images:
        raise ScanColorError("no readable frames found to estimate a roll density-balance profile from")
    return roll_auto_density_balance(images)
