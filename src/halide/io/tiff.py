"""TIFF read/write: dtype normalization, embedded ICC tag access, EXIF passthrough.

Color management (validating/converting the embedded ICC profile) is deliberately a separate
concern — see halide.io.icc — this module only moves bytes and pixel values in and out of the
0-1 linear-transmittance convention the core pipeline expects.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

_ICC_TAG = 34675  # TIFF InterColorProfile / ICC Profile tag


@dataclass(frozen=True)
class RawScan:
    image: np.ndarray  # float32/64, normalized to [0, 1] in the *source* color space
    icc_profile: bytes | None


def read_tiff(path: str | Path) -> RawScan:
    """Read a TIFF and normalize its pixel values to a 0-1 float range. Does not touch color
    space — the returned image is still in whatever RGB space the file's ICC tag (if any)
    describes; pass it through halide.io.icc before feeding it to the core pipeline."""
    with tifffile.TiffFile(path) as tif:
        raw = tif.asarray()
        icc_tag = tif.pages[0].tags.get(_ICC_TAG)
        icc_profile = icc_tag.value if icc_tag is not None else None

    if raw.dtype == np.uint8:
        image = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        image = raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint32:
        image = raw.astype(np.float32) / 4294967295.0
    elif np.issubdtype(raw.dtype, np.floating):
        # copy=False: the normal case (RawTherapee/darktable's linear-Rec.2020 export) already
        # decodes as float32 — avoid a second full-size copy on top of tifffile's own decode.
        image = raw.astype(np.float32, copy=False)
    else:
        raise ValueError(f"{path}: unsupported TIFF sample dtype {raw.dtype}")

    return RawScan(image=image, icc_profile=icc_profile)


def write_tiff(path: str | Path, image: np.ndarray, icc_profile: bytes | None = None) -> None:
    """Write a float32 TIFF, optionally embedding an ICC profile tag."""
    write_kwargs: dict = {
        "photometric": "rgb",
        "compression": "zlib",
        "compressionargs": {"level": 6},
        "predictor": 3,
    }
    if icc_profile is not None:
        write_kwargs["extratags"] = [(_ICC_TAG, "B", len(icc_profile), icc_profile)]

    tifffile.imwrite(path, image.astype(np.float32, copy=False), **write_kwargs)


def copy_exif_metadata(source_path: str | Path, dest_path: str | Path, drop_icc: bool = False) -> bool:
    """Copy EXIF metadata from source_path onto dest_path via exiftool, if available.
    Returns False (without raising) if exiftool isn't installed — metadata is a nice-to-have,
    not a correctness requirement of the pipeline."""
    command = [
        "exiftool",
        "-TagsFromFile",
        str(source_path),
        "-all:all",
        "--ExifImageWidth",
        "--ExifImageHeight",
    ]
    if drop_icc:
        command.append("--icc_profile")
    command.extend(["-overwrite_original", str(dest_path)])

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
