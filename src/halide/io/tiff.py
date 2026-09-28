"""TIFF read/write: dtype normalization, embedded ICC tag access, EXIF passthrough.

Color management (validating/converting the embedded ICC profile) is deliberately a separate
concern — see halide.io.icc — this module only moves bytes and pixel values in and out of the
0-1 linear-transmittance convention the core pipeline expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from halide.io import exiftool

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


def read_tiff(path: str | Path, out: np.ndarray | None = None) -> RawScan:
    """Read a TIFF and normalize its pixel values to a 0-1 float range. Does not touch color
    space — the returned image is still in whatever RGB space the file's ICC tag (if any)
    describes; pass it through halide.io.icc before feeding it to the core pipeline.

    `out`, if given, must already be a float32, C-contiguous array of the file's own (H, W, C)
    shape (checked up front — raises ValueError rather than silently reallocating or copying, since
    `out` is meant to be a caller-owned buffer, e.g. a shared-memory frame from halide.shared_frames
    that another process already knows the identity of). The returned RawScan.image is then `out`
    itself. For a file that's already float32, tifffile decodes straight into `out` — no extra host
    copy, the same fast path `copy=False` below already gives the no-`out` caller. For any other
    sample dtype, the file is decoded into a temporary buffer first and then normalized into `out` —
    see `_normalize_into` for why that's not simply `np.divide(raw, scale, out=out)`."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        icc_tag = page.tags.get(_ICC_TAG)
        icc_profile = icc_tag.value if icc_tag is not None else None
        description = page.description or None

        # Shape/dtype come from series[0], not pages[0]: `tif.asarray()` below decodes the series
        # (tifffile's own unit for "the array this file represents"), which for some files isn't
        # simply the first page's own raw shape/dtype (e.g. a file whose series stacks multiple
        # pages). They agree for every real scan this project reads, but validating/sizing `out`
        # against the page instead of the series could pass a header check here and then still
        # mismatch what asarray() actually decodes.
        series = tif.series[0]
        if out is not None:
            _validate_out(out, series.shape)

        if out is not None and series.dtype == np.dtype(np.float32):
            tif.asarray(out=out, buffersize=_READ_BUFFER_BYTES)
            raw = None
        else:
            raw = tif.asarray(buffersize=_READ_BUFFER_BYTES)

    if raw is None:
        image = out
    elif raw.dtype == np.uint8:
        image = _normalize_into(raw, 255.0, out)
    elif raw.dtype == np.uint16:
        image = _normalize_into(raw, 65535.0, out)
    elif raw.dtype == np.uint32:
        image = _normalize_into(raw, 4294967295.0, out)
    elif np.issubdtype(raw.dtype, np.floating):
        if out is not None:
            # Not the direct-decode path above (raw.dtype isn't float32) but still floating (e.g.
            # float64) — cast into `out`, matching the no-`out` copy=False cast below.
            out[:] = raw
            image = out
        else:
            # copy=False: the normal case (RawTherapee/darktable's linear-Rec.2020 export) already
            # decodes as float32 — avoid a second full-size copy on top of tifffile's own decode.
            image = raw.astype(np.float32, copy=False)
    else:
        raise ValueError(f"{path}: unsupported TIFF sample dtype {raw.dtype}")

    return RawScan(image=image, icc_profile=icc_profile, description=description)


def _validate_out(out: np.ndarray, page_shape: tuple[int, ...]) -> None:
    if out.dtype != np.dtype(np.float32):
        raise ValueError(f"out must be float32, got dtype {out.dtype}")
    if tuple(out.shape) != tuple(page_shape):
        raise ValueError(f"out has shape {out.shape}, but the file's shape is {tuple(page_shape)}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out must be C-contiguous")


def _normalize_into(raw: np.ndarray, scale: float, out: np.ndarray | None) -> np.ndarray:
    """`raw.astype(np.float32) / scale`, or the same two operations run into `out` in place when
    given. Deliberately cast-then-divide in both cases, not `np.divide(raw, scale, out=out)`: dividing
    straight from the integer input computes (at least internally) in a wider/different type than
    the float32-array-by-python-float division `raw.astype(np.float32) / scale` performs (NEP 50
    keeps that division in float32 throughout), so the two can round differently in the last bit —
    see the B1 task brief's ruling. Casting first and dividing second, in both branches, keeps the
    `out=` path bit-identical to today's no-`out` result."""
    if out is None:
        return raw.astype(np.float32) / scale
    out[:] = raw
    out /= scale
    return out


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


def read_tiff_shape(path: str | Path) -> tuple[int, ...]:
    """The pixel shape (e.g. (H, W, C)) `read_tiff` will decode, without decoding any pixel data —
    for sizing a shared-memory frame (halide.shared_frames.new_frame) before decoding a scan into it
    with `read_tiff(path, out=...)`. Reads `series[0].shape`, matching what `tif.asarray()` (what
    `read_tiff` actually decodes) reports — not `pages[0].shape`, which isn't guaranteed to agree
    (see `read_tiff`'s own comment on this). Similar header-only idea to
    halide.batch.orchestrator._decoded_pixel_bytes, which reads series[0].shape/series[0].dtype for
    the same reason (sizing the worker pool without paying for a decode) but only ever needs a byte
    count, not an exact shape to allocate."""
    with tifffile.TiffFile(path) as tif:
        return tuple(tif.series[0].shape)


def set_description(path: str | Path, text: str) -> None:
    """Overwrite the first page's ImageDescription in place. Done as a separate step *after*
    copy_exif_metadata, which would otherwise copy the source scan's own description over it."""
    tifffile.tiffcomment(path, text)


def copy_exif_metadata(source_path: str | Path, dest_path: str | Path, drop_icc: bool = False) -> bool:
    """Copy EXIF metadata from source_path onto dest_path via exiftool, if available.
    Returns False (without raising) if exiftool isn't installed — metadata is a nice-to-have,
    not a correctness requirement of the pipeline. On Linux it runs on this process's kept-open
    exiftool (halide.io.exiftool), ~0.4 s per file faster than starting one and writing the same
    bytes; elsewhere it starts one per call; a failure exiftool reports for the file raises (ExifToolError, or CalledProcessError
    where the command had to run one-shot)."""
    args = [
        "-TagsFromFile",
        str(source_path),
        "-all:all",
        "--ExifImageWidth",
        "--ExifImageHeight",
    ]
    if drop_icc:
        args.append("--icc_profile")
    args.extend(["-overwrite_original", str(dest_path)])
    return exiftool.execute(args, written=dest_path)
