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

# How much compressed strip data tifffile reads at once. Its default (256 MiB) exceeds a whole real
# scan (~120 MiB on disk — float data barely compresses), so the entire file sat in memory next to
# the decoded frame. 16 MiB keeps decoding multithreaded and costs ~0.15 s per full-resolution frame
# for ~55 MiB less peak memory (single-threaded decoding saved only 17 MiB more, for 0.6 s).
# Decoding is deterministic: the pixels are identical either way.
_READ_BUFFER_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class RawScan:
    image: np.ndarray  # float32/64, normalized to [0, 1] in the *source* color space
    icc_profile: bytes | None
    description: str | None = None  # TIFF ImageDescription (where halide's provenance JSON lives)


def read_tiff(path: str | Path) -> RawScan:
    """Read a TIFF and normalize its pixel values to a 0-1 float range. Does not touch color
    space — the returned image is still in whatever RGB space the file's ICC tag (if any)
    describes; pass it through halide.io.icc before feeding it to the core pipeline."""
    with tifffile.TiffFile(path) as tif:
        raw = tif.asarray(buffersize=_READ_BUFFER_BYTES)
        icc_tag = tif.pages[0].tags.get(_ICC_TAG)
        icc_profile = icc_tag.value if icc_tag is not None else None
        description = tif.pages[0].description or None

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

    return RawScan(image=image, icc_profile=icc_profile, description=description)


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


def read_tiff_description(path: str | Path) -> str | None:
    """The first page's ImageDescription only, without decoding any pixels."""
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].description or None


def set_description(path: str | Path, text: str) -> None:
    """Overwrite the first page's ImageDescription in place. Done as a separate step *after*
    copy_exif_metadata, which would otherwise copy the source scan's own description over it."""
    tifffile.tiffcomment(path, text)


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
