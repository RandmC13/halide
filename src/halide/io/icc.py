"""Validate and convert the embedded ICC profile of an input TIFF into this project's internal
linear working space (ACEScg).

This project needs to answer one narrow question precisely: "is the embedded profile a *linear*,
*matrix-shaper* RGB profile (the kind RawTherapee/darktable's 'linear Rec.2020' export produces),
and if so, what is its RGB->XYZ matrix?" No off-the-shelf Python library answers that at the tag
level — Pillow's ImageCms wraps LittleCMS but only exposes profile-level metadata (name, color
space, rendering intent), not raw colorant/TRC tags or a matrix-shaper check, and no dedicated ICC
tag-parsing package exists on PyPI for this. The matrix-shaper subset of the ICC.1:2010 spec this
module reads (ss. 9.2.10 header, 10.18 curveType, 10.24 parametricCurveType, 10.27 XYZType) is
small, stable, and precisely documented, so it is implemented directly here rather than trusted to
a CMM's less transparent rendering-intent handling — verified in tests/unit/test_icc.py against
both a real-world profile (a gamma-encoded Rec.2020 ICC, correctly rejected) and synthetic
profiles constructed byte-for-byte from the spec.

ICC matrix/TRC profiles store their rXYZ/gXYZ/bXYZ/wtpt tags relative to the PCS illuminant, which
by spec is always D50 — regardless of the profile's own "native" white point (e.g. D65 for
Rec.2020). That D50-relative matrix is used as-is; the chromatic adaptation to the working space's
own white point (ACEScg's is close to D60) is delegated to `colour-science`, which is designed and
tested specifically for that step.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import colour
import numpy as np

_LUT_TAG_SIGNATURES = frozenset({"A2B0", "A2B1", "A2B2", "B2A0", "B2A1", "B2A2"})
_LINEARITY_TOLERANCE = 5e-3
_D50_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]

# The output working-space profile: Elle Stone's community-authored linear ACEScg ICC profile
# (CC BY-SA 3.0, see assets/icc_profiles/LICENSE-elles_icc_profiles.txt), the exact profile the
# source blog post recommends. Cross-checked in tests/unit/test_icc.py: parsing this file with our
# own reader and independently computing ACEScg's D50-adapted matrix via colour-science agree to
# within ICC s15Fixed16 quantization (~2e-4) — i.e. two independent sources of truth agree, rather
# than us hand-authoring a profile with no independent way to verify it here.
OUTPUT_PROFILE_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "icc_profiles" / "ACEScg-elle-V4-g10.icc"
)


def output_profile_bytes() -> bytes:
    """The ICC profile bytes to embed on any output written from the internal ACEScg working
    space (i.e. anything that has been through core.pipeline.run_pipeline)."""
    return OUTPUT_PROFILE_PATH.read_bytes()


class UnsupportedICCProfileError(ValueError):
    """Raised when an embedded ICC profile is not a linear matrix-shaper RGB profile."""


@dataclass(frozen=True)
class LinearRGBProfile:
    """A validated linear matrix-shaper RGB profile's colorimetry.

    `rgb_to_pcs_xyz` maps the profile's own linear RGB values to CIE XYZ relative to the ICC PCS
    illuminant (D50, per spec) — not relative to the profile's native white point.
    """

    rgb_to_pcs_xyz: np.ndarray  # shape (3, 3)


def _read_header(data: bytes) -> tuple[bytes, bytes]:
    if len(data) < 132:
        raise UnsupportedICCProfileError("not a valid ICC profile (file too small)")
    color_space = data[16:20]
    pcs = data[20:24]
    return color_space, pcs


def _read_tag_table(data: bytes) -> dict[str, tuple[int, int]]:
    (n_tags,) = struct.unpack(">I", data[128:132])
    tags: dict[str, tuple[int, int]] = {}
    for i in range(n_tags):
        offset = 132 + i * 12
        sig, tag_offset, tag_size = struct.unpack(">4sII", data[offset : offset + 12])
        tags[sig.decode("latin1")] = (tag_offset, tag_size)
    return tags


def _read_xyz_tag(data: bytes, tags: dict[str, tuple[int, int]], signature: str) -> np.ndarray:
    if signature not in tags:
        raise UnsupportedICCProfileError(f"missing required '{signature}' tag")
    offset, size = tags[signature]
    tag_type = data[offset : offset + 4]
    if tag_type != b"XYZ ":
        raise UnsupportedICCProfileError(f"'{signature}' tag is type {tag_type!r}, expected XYZType")
    x, y, z = struct.unpack(">iii", data[offset + 8 : offset + 20])
    return np.array([x, y, z], dtype=np.float64) / 65536.0


def _eval_parametric_curve(function_type: int, params: list[float], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    def powf(base: np.ndarray, exponent: float) -> np.ndarray:
        return np.power(np.clip(base, 0.0, None), exponent)

    if function_type == 0:
        (g,) = params
        return powf(x, g)
    if function_type == 1:
        g, a, b = params
        threshold = -b / a if a != 0 else 0.0
        return np.where(x >= threshold, powf(a * x + b, g), 0.0)
    if function_type == 2:
        g, a, b, c = params
        threshold = -b / a if a != 0 else 0.0
        return np.where(x >= threshold, powf(a * x + b, g) + c, c)
    if function_type == 3:
        g, a, b, c, d = params
        return np.where(x >= d, powf(a * x + b, g), c * x)
    if function_type == 4:
        g, a, b, c, d, e, f = params
        return np.where(x >= d, powf(a * x + b, g) + e, c * x + f)
    raise UnsupportedICCProfileError(f"unsupported parametricCurveType functionType {function_type}")


def _eval_sampled_curve(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(table)
    if n == 0:
        return np.asarray(x, dtype=np.float64)  # count == 0 means identity, per spec
    if n == 1:
        gamma = table[0] / 256.0  # u8Fixed8Number
        return np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, None), gamma)
    normalized = table / 65535.0
    idx = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * (n - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = idx - lo
    return normalized[lo] * (1 - frac) + normalized[hi] * frac


def _check_trc_is_linear(data: bytes, tags: dict[str, tuple[int, int]], signature: str) -> None:
    if signature not in tags:
        raise UnsupportedICCProfileError(f"missing required '{signature}' tag")
    offset, size = tags[signature]
    tag_type = data[offset : offset + 4]
    sample_points = np.linspace(0.0, 1.0, num=11)

    if tag_type == b"curv":
        (count,) = struct.unpack(">I", data[offset + 8 : offset + 12])
        table = np.array(
            struct.unpack(f">{count}H", data[offset + 12 : offset + 12 + 2 * count]),
            dtype=np.float64,
        )
        response = _eval_sampled_curve(table, sample_points)
    elif tag_type == b"para":
        function_type, _reserved = struct.unpack(">HH", data[offset + 8 : offset + 12])
        n_params = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function_type)
        if n_params is None:
            raise UnsupportedICCProfileError(
                f"'{signature}': unsupported parametricCurveType functionType {function_type}"
            )
        raw = struct.unpack(
            f">{n_params}i", data[offset + 12 : offset + 12 + 4 * n_params]
        )
        params = [p / 65536.0 for p in raw]
        response = _eval_parametric_curve(function_type, params, sample_points)
    else:
        raise UnsupportedICCProfileError(f"'{signature}' tag is type {tag_type!r}, expected curv/para")

    if not np.allclose(response, sample_points, atol=_LINEARITY_TOLERANCE):
        raise UnsupportedICCProfileError(
            f"'{signature}' is not a linear tone curve — this profile is gamma-encoded (e.g. for "
            f"display use). Re-export using your raw processor's linear gamma/tone-curve option."
        )


def parse_linear_rgb_profile(icc_bytes: bytes) -> LinearRGBProfile:
    """Validate that `icc_bytes` is a linear matrix-shaper RGB ICC profile and extract its
    RGB->PCS(D50)XYZ matrix. Raises UnsupportedICCProfileError with a specific, actionable message
    otherwise."""
    color_space, pcs = _read_header(icc_bytes)
    if color_space != b"RGB ":
        raise UnsupportedICCProfileError(f"profile color space is {color_space!r}, expected RGB")
    if pcs != b"XYZ ":
        raise UnsupportedICCProfileError(f"profile connection space is {pcs!r}, expected XYZ")

    tags = _read_tag_table(icc_bytes)

    lut_tags_present = _LUT_TAG_SIGNATURES & tags.keys()
    if lut_tags_present:
        raise UnsupportedICCProfileError(
            f"profile contains LUT-based tag(s) {sorted(lut_tags_present)} — only pure "
            f"matrix-shaper profiles (rXYZ/gXYZ/bXYZ + linear TRC) are supported"
        )

    for signature in ("rTRC", "gTRC", "bTRC"):
        _check_trc_is_linear(icc_bytes, tags, signature)

    r_xyz = _read_xyz_tag(icc_bytes, tags, "rXYZ")
    g_xyz = _read_xyz_tag(icc_bytes, tags, "gXYZ")
    b_xyz = _read_xyz_tag(icc_bytes, tags, "bXYZ")
    _read_xyz_tag(icc_bytes, tags, "wtpt")  # required by spec; not otherwise needed here

    matrix = np.stack([r_xyz, g_xyz, b_xyz], axis=1)  # columns = primaries
    return LinearRGBProfile(rgb_to_pcs_xyz=matrix)


def convert_to_working_space(img: np.ndarray, profile: LinearRGBProfile) -> np.ndarray:
    """Convert a linear-RGB image (in the color space described by `profile`) into this project's
    internal linear ACEScg working space."""
    pcs_xyz = img @ profile.rgb_to_pcs_xyz.T
    return colour.XYZ_to_RGB(
        pcs_xyz,
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=_D50_XY,
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False,
    )
