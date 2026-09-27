"""Delivery-format export: ACEScg -> sRGB -> 8-bit PNG/JPEG.

This is a genuinely different job from io/icc.py's ICC handling: here we are *writing* a known,
standard profile (sRGB) rather than validating an arbitrary unknown one, so Pillow's ImageCms is
the right tool (it directly supports synthesizing a standard sRGB profile — the same limitation
that ruled it out for io/icc.py's job, only supporting LAB/XYZ/sRGB, is exactly what we need here).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from halide.banding import map_in_bands

_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@functools.cache
def srgb_matrix() -> np.ndarray:
    """colour-science's own float64 ACEScg -> sRGB matrix (`colour.matrix_RGB_to_RGB`, Bradford
    adaptation), applied as one matrix multiply per band instead of colour.RGB_to_RGB's per-pixel
    broadcast — see to_srgb_8bit and CLAUDE.md, "D1"."""
    import colour

    return np.asarray(
        colour.matrix_RGB_to_RGB(
            colour.RGB_COLOURSPACES["ACEScg"], colour.RGB_COLOURSPACES["sRGB"], "Bradford"
        ),
        dtype=np.float64,
    )


def to_srgb_8bit(acescg_image: np.ndarray) -> np.ndarray:
    """Convert a linear ACEScg image (the output of core.pipeline.run_pipeline) into gamma-encoded
    8-bit sRGB. Values outside sRGB's displayable range are clipped here — unlike the tone-render
    clipping bug this project fixed, this clip is expected and correct: an 8-bit delivery raster
    cannot represent out-of-gamut or HDR values, and by this stage we are deliberately committing
    to a bounded display format rather than preserving scene-referred data (that's what the ACEScg
    TIFF intermediate is for)."""
    import colour  # imported here, not at module level: it's most of the CLI's startup time

    # colour.RGB_to_RGB with apply_cctf_decoding=False (its default, unchanged here) is exactly
    # `vecmul(matrix_RGB_to_RGB(...), acescg)` — the same per-pixel-broadcast cost this project
    # already fixed once for the ICC step (see icc.py::working_space_matrices). One BLAS matmul per
    # band does the same maths; measured on a real scan, the matrix multiply was the small half of
    # this function's cost (~0.3 s of ~1.7 s) — colour's own cctf_encoding call (applied unchanged
    # below, so this step is bit-identical) is the rest. Still worth doing: it's free, and it's the
    # same fix already validated for icc.py.
    srgb_linear = acescg_image.astype(np.float64) @ srgb_matrix().T
    encoded = colour.RGB_COLOURSPACES["sRGB"].cctf_encoding(srgb_linear)
    clipped = np.clip(encoded, 0.0, 1.0)
    return (clipped * 255).round().astype(np.uint8)


def write_delivery_image(path: str | Path, acescg_image: np.ndarray, quality: int = 95) -> None:
    """Write a PNG or JPEG (chosen by `path`'s extension) with an embedded sRGB ICC profile."""
    path = Path(path)
    # Band by band into one 8-bit buffer (a quarter of the float frame): colour-science converts in
    # float64 internally, and doing the whole frame at once held several float64 copies of it — the
    # single heaviest step halide had (~2.6 GiB on a real scan). Per-pixel, so bit-identical.
    srgb_8bit = map_in_bands(acescg_image, to_srgb_8bit, out=np.empty(acescg_image.shape, dtype=np.uint8))
    image = Image.fromarray(srgb_8bit, mode="RGB")

    suffix = path.suffix.lower()
    if suffix == ".png":
        image.save(path, format="PNG", icc_profile=_SRGB_ICC_BYTES)
    elif suffix in (".jpg", ".jpeg"):
        image.save(path, format="JPEG", icc_profile=_SRGB_ICC_BYTES, quality=quality)
    else:
        raise ValueError(f"{path}: unsupported export format {suffix!r} (use .png, .jpg, or .jpeg)")
