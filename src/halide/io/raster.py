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
from halide.core._xp import array_namespace

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


def encode_srgb(linear):
    """The IEC 61966-2-1:1999 sRGB inverse EOTF (a.k.a. its "gamma" encoding), written generically
    over `xp` so it can run on a device array — colour's own `cctf_encoding` (used by the CPU
    branch of `to_srgb_8bit` below, unchanged) only accepts numpy. Mirrors
    `colour.models.rgb.transfer_functions.srgb.eotf_inverse_sRGB` exactly, constants included:
    `V = where(L <= 0.0031308, 12.92*L, 1.055*spow(L, 1/2.4) - 0.055)`, where colour's `spow`
    ("signed power", `colour.algebra.common.spow`) is `sign(a) * |a|**p` rather than a plain
    `a**p`, to avoid NaN for a negative `L` raised to a fractional power. That branch is always
    discarded by `xp.where` for a negative `L` here (0.0031308 > 0, so every negative input already
    takes the linear branch) — spow is mirrored anyway so the discarded branch doesn't raise a
    device-side NaN/warning and to keep this a faithful, general copy of colour's formula rather
    than one that happens to work only because of where this project calls it from."""
    xp = array_namespace(linear)
    powered = xp.sign(linear) * xp.power(xp.abs(linear), 1.0 / 2.4)
    return xp.where(linear <= 0.0031308, linear * 12.92, 1.055 * powered - 0.055)


def to_srgb_8bit(acescg_image):
    """Convert a linear ACEScg image (the output of core.pipeline.run_pipeline) into gamma-encoded
    8-bit sRGB. Values outside sRGB's displayable range are clipped here — unlike the tone-render
    clipping bug this project fixed, this clip is expected and correct: an 8-bit delivery raster
    cannot represent out-of-gamut or HDR values, and by this stage we are deliberately committing
    to a bounded display format rather than preserving scene-referred data (that's what the ACEScg
    TIFF intermediate is for).

    `acescg_image` may be a numpy array or a device (CuPy) array (halide.core._xp.array_namespace
    picks the branch). The CPU branch is unchanged from before this function became
    namespace-generic — colour's own `cctf_encoding` stays the authority on the numbers there, so
    CPU output stays bit-identical. The device branch can't call into colour (numpy-only), so it
    uses `encode_srgb` above instead and keeps the whole computation in the image's own dtype (one
    float32 matrix multiply, per docs/plans/gpu-acceleration.md §3.3) rather than colour's float64
    conversion — held to the D2 tolerance against the CPU path, not to bit-identity."""
    xp = array_namespace(acescg_image)
    if xp is np:
        import colour  # imported here, not at module level: it's most of the CLI's startup time

        # colour.RGB_to_RGB with apply_cctf_decoding=False (its default, unchanged here) is exactly
        # `vecmul(matrix_RGB_to_RGB(...), acescg)` — the same per-pixel-broadcast cost this project
        # already fixed once for the ICC step (see icc.py::working_space_matrices). One BLAS matmul
        # per band does the same maths; measured on a real scan, the matrix multiply was the small
        # half of this function's cost (~0.3 s of ~1.7 s) — colour's own cctf_encoding call (applied
        # unchanged below, so this step is bit-identical) is the rest. Still worth doing: it's free,
        # and it's the same fix already validated for icc.py.
        srgb_linear = acescg_image.astype(np.float64) @ srgb_matrix().T
        encoded = colour.RGB_COLOURSPACES["sRGB"].cctf_encoding(srgb_linear)
    else:
        # The (host, float64) matrix is cast down and transposed *before* upload — matching
        # icc.py::convert_to_working_space's pattern — so no device array ever needs a `.T`.
        matrix = xp.asarray(srgb_matrix().astype(acescg_image.dtype).T)
        srgb_linear = acescg_image @ matrix
        encoded = encode_srgb(srgb_linear)
    clipped = xp.clip(encoded, 0.0, 1.0)
    return (clipped * 255).round().astype(xp.uint8)


def srgb_8bit_from_acescg(acescg_image: np.ndarray) -> np.ndarray:
    """`to_srgb_8bit`, banded into a fresh host uint8 buffer — the host-only half of
    `write_delivery_image` below, split out so `processing.py::export_delivery_image` can instead
    supply an 8-bit array it already computed on the device (see that function)."""
    # Band by band into one 8-bit buffer (a quarter of the float frame): colour-science converts in
    # float64 internally, and doing the whole frame at once held several float64 copies of it — the
    # single heaviest step halide had (~2.6 GiB on a real scan). Per-pixel, so bit-identical.
    return map_in_bands(acescg_image, to_srgb_8bit, out=np.empty(acescg_image.shape, dtype=np.uint8))


def write_srgb_8bit_image(path: str | Path, srgb_8bit: np.ndarray, quality: int = 95) -> None:
    """Write an already-encoded 8-bit sRGB array as a PNG or JPEG (chosen by `path`'s extension)
    with an embedded sRGB ICC profile. The save half of `write_delivery_image`, split out so a
    caller that converted to 8-bit itself (the device export path) doesn't convert twice."""
    path = Path(path)
    image = Image.fromarray(srgb_8bit, mode="RGB")

    suffix = path.suffix.lower()
    if suffix == ".png":
        image.save(path, format="PNG", icc_profile=_SRGB_ICC_BYTES)
    elif suffix in (".jpg", ".jpeg"):
        image.save(path, format="JPEG", icc_profile=_SRGB_ICC_BYTES, quality=quality)
    else:
        raise ValueError(f"{path}: unsupported export format {suffix!r} (use .png, .jpg, or .jpeg)")


def write_delivery_image(path: str | Path, acescg_image: np.ndarray, quality: int = 95) -> None:
    """Write a PNG or JPEG (chosen by `path`'s extension) with an embedded sRGB ICC profile."""
    write_srgb_8bit_image(path, srgb_8bit_from_acescg(acescg_image), quality=quality)
