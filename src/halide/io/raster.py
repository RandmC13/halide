"""Delivery-format export: ACEScg -> sRGB -> 8-bit PNG/JPEG.

This is a genuinely different job from io/icc.py's ICC handling: here we are *writing* a known,
standard profile (sRGB) rather than validating an arbitrary unknown one, so Pillow's ImageCms is
the right tool (it directly supports synthesizing a standard sRGB profile — the same limitation
that ruled it out for io/icc.py's job, only supporting LAB/XYZ/sRGB, is exactly what we need here).
"""

from __future__ import annotations

from pathlib import Path

import colour
import numpy as np
from PIL import Image, ImageCms

_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def to_srgb_8bit(acescg_image: np.ndarray) -> np.ndarray:
    """Convert a linear ACEScg image (the output of core.pipeline.run_pipeline) into gamma-encoded
    8-bit sRGB. Values outside sRGB's displayable range are clipped here — unlike the tone-render
    clipping bug this project fixed, this clip is expected and correct: an 8-bit delivery raster
    cannot represent out-of-gamut or HDR values, and by this stage we are deliberately committing
    to a bounded display format rather than preserving scene-referred data (that's what the ACEScg
    TIFF intermediate is for)."""
    srgb_linear = colour.RGB_to_RGB(
        acescg_image,
        input_colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        output_colourspace=colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=True,
    )
    clipped = np.clip(srgb_linear, 0.0, 1.0)
    return (clipped * 255).round().astype(np.uint8)


def write_delivery_image(path: str | Path, acescg_image: np.ndarray, quality: int = 95) -> None:
    """Write a PNG or JPEG (chosen by `path`'s extension) with an embedded sRGB ICC profile."""
    path = Path(path)
    srgb_8bit = to_srgb_8bit(acescg_image)
    image = Image.fromarray(srgb_8bit, mode="RGB")

    suffix = path.suffix.lower()
    if suffix == ".png":
        image.save(path, format="PNG", icc_profile=_SRGB_ICC_BYTES)
    elif suffix in (".jpg", ".jpeg"):
        image.save(path, format="JPEG", icc_profile=_SRGB_ICC_BYTES, quality=quality)
    else:
        raise ValueError(f"{path}: unsupported export format {suffix!r} (use .png, .jpg, or .jpeg)")
