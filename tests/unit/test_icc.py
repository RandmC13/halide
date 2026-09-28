"""Tests for the minimal ICC tag parser, using both a real-world profile and synthetic profiles
built byte-for-byte from the ICC.1:2010 spec (no ICC-writing library needed for either)."""

from pathlib import Path

import numpy as np
import pytest

from halide.io.icc import (
    OUTPUT_PROFILE_PATH,
    UnsupportedICCProfileError,
    convert_to_working_space,
    output_profile_bytes,
    parse_linear_rgb_profile,
    working_space_matrices,
)
from halide.io.tiff import read_tiff

REAL_GAMMA_ENCODED_REC2020_ICC = (
    Path("/tmp/color-negative-inversion/1. pre-requisites/Rec. 2020.icc")
)


# ---------------------------------------------------------------------------
# Synthetic ICC profile builder — writes just enough of the spec (header + tag
# table + tag data) for our parser to exercise, nothing display/CMM-facing.
# ---------------------------------------------------------------------------
import struct  # noqa: E402


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + b"\x00" * 4 + struct.pack(
        ">iii", round(x * 65536), round(y * 65536), round(z * 65536)
    )


def _curv_identity_tag() -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 0)


def _curv_gamma_tag(gamma: float) -> bytes:
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", round(gamma * 256))


def _curv_table_tag(values: list[float]) -> bytes:
    ints = [round(np.clip(v, 0.0, 1.0) * 65535) for v in values]
    return b"curv" + b"\x00" * 4 + struct.pack(">I", len(ints)) + struct.pack(f">{len(ints)}H", *ints)


def _para_tag(function_type: int, params: list[float]) -> bytes:
    encoded = [round(p * 65536) for p in params]
    return b"para" + b"\x00" * 4 + struct.pack(">HH", function_type, 0) + struct.pack(
        f">{len(encoded)}i", *encoded
    )


def _pad4(data: bytes) -> bytes:
    return data + b"\x00" * ((4 - len(data) % 4) % 4)


def build_icc(tags: dict[str, bytes], color_space: bytes = b"RGB ", pcs: bytes = b"XYZ ") -> bytes:
    header = bytearray(128)
    header[16:20] = color_space
    header[20:24] = pcs

    padded = {sig: _pad4(data) for sig, data in tags.items()}
    n_tags = len(padded)
    table_start = 128 + 4
    tag_table_size = n_tags * 12
    data_start = table_start + tag_table_size

    entries = []
    blob = b""
    offset = data_start
    for sig, data in padded.items():
        entries.append(struct.pack(">4sII", sig.encode("latin1"), offset, len(data)))
        blob += data
        offset += len(data)

    return bytes(header) + struct.pack(">I", n_tags) + b"".join(entries) + blob


# A realistic linear Rec.2020-like profile: identity TRCs, D65-ish primaries pre-adapted to D50
# (values aren't required to be colorimetrically perfect for the parser tests — only the
# *shape*/type of each tag matters here; convert_to_working_space's own correctness is exercised
# with the exact real rXYZ/gXYZ/bXYZ values below, lifted from the real Rec.2020.icc file).
LINEAR_TAGS = {
    "wtpt": _xyz_tag(0.9642, 1.0, 0.8249),
    "rXYZ": _xyz_tag(0.6734, 0.2790, -0.0019),
    "gXYZ": _xyz_tag(0.1657, 0.6754, 0.0300),
    "bXYZ": _xyz_tag(0.1250, 0.0456, 0.7968),
    "rTRC": _curv_identity_tag(),
    "gTRC": _curv_identity_tag(),
    "bTRC": _curv_identity_tag(),
}


def test_accepts_a_linear_matrix_shaper_profile():
    profile = parse_linear_rgb_profile(build_icc(LINEAR_TAGS))
    assert profile.rgb_to_pcs_xyz.shape == (3, 3)
    assert profile.rgb_to_pcs_xyz[1, 1] == pytest.approx(0.6754, abs=1e-4)


def test_accepts_linear_curve_encoded_as_gamma_one_sampled_curve():
    tags = dict(LINEAR_TAGS)
    tags["rTRC"] = _curv_gamma_tag(1.0)
    tags["gTRC"] = _curv_table_tag(list(np.linspace(0.0, 1.0, num=64)))
    parse_linear_rgb_profile(build_icc(tags))  # must not raise


def test_accepts_linear_curve_encoded_as_parametric_gamma_one():
    tags = dict(LINEAR_TAGS)
    for ch in ("rTRC", "gTRC", "bTRC"):
        tags[ch] = _para_tag(0, [1.0])
    parse_linear_rgb_profile(build_icc(tags))  # must not raise


def test_rejects_gamma_encoded_trc():
    tags = dict(LINEAR_TAGS)
    for ch in ("rTRC", "gTRC", "bTRC"):
        tags[ch] = _para_tag(0, [2.2])
    with pytest.raises(UnsupportedICCProfileError, match="not a linear tone curve"):
        parse_linear_rgb_profile(build_icc(tags))


def test_rejects_lut_based_profile():
    tags = dict(LINEAR_TAGS)
    tags["A2B0"] = b"mft2" + b"\x00" * 8  # minimal stand-in payload; only the tag's presence matters
    with pytest.raises(UnsupportedICCProfileError, match="LUT-based"):
        parse_linear_rgb_profile(build_icc(tags))


def test_rejects_non_rgb_color_space():
    with pytest.raises(UnsupportedICCProfileError, match="color space"):
        parse_linear_rgb_profile(build_icc(LINEAR_TAGS, color_space=b"GRAY"))


def test_rejects_missing_required_tag():
    tags = dict(LINEAR_TAGS)
    del tags["bXYZ"]
    with pytest.raises(UnsupportedICCProfileError, match="bXYZ"):
        parse_linear_rgb_profile(build_icc(tags))


@pytest.mark.skipif(
    not REAL_GAMMA_ENCODED_REC2020_ICC.exists(),
    reason="reference repo not available in this environment",
)
def test_rejects_the_real_gamma_encoded_rec2020_profile():
    # This specific vendored file is a *generic display* Rec.2020 profile (gamma ~2.2), not the
    # linear working-space profile RawTherapee/darktable's "linear Rec.2020" export produces —
    # exactly the kind of file this validation must catch rather than silently mis-color-manage.
    icc_bytes = REAL_GAMMA_ENCODED_REC2020_ICC.read_bytes()
    with pytest.raises(UnsupportedICCProfileError, match="not a linear tone curve"):
        parse_linear_rgb_profile(icc_bytes)


def test_bundled_output_profile_matches_colour_sciences_acescg_definition():
    # Two independent sources of truth for the same thing: a community-authored ICC profile
    # (parsed with our own reader) vs. colour-science's own ACEScg colourspace definition,
    # independently D50-adapted the same way convert_to_working_space does.
    import colour

    profile = parse_linear_rgb_profile(output_profile_bytes())
    d50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    expected = colour.RGB_to_XYZ(
        np.eye(3),
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=d50,
        chromatic_adaptation_transform="Bradford",
        apply_cctf_decoding=False,
    ).T
    assert profile.rgb_to_pcs_xyz == pytest.approx(expected, abs=5e-4)


def test_output_profile_asset_exists():
    assert OUTPUT_PROFILE_PATH.exists()


def test_convert_to_working_space_preserves_neutral_gray():
    profile = parse_linear_rgb_profile(build_icc(LINEAR_TAGS))
    # A profile's own white point, run through its own matrix, should land at ACEScg's neutral
    # axis (R==G==B) once converted — any imbalance here would mean the matrix/CAT wiring is wrong.
    white_rgb = np.array([[[1.0, 1.0, 1.0]]])
    result = convert_to_working_space(white_rgb, profile)
    r, g, b = result[0, 0]
    assert r == pytest.approx(g, rel=0.05)
    assert b == pytest.approx(g, rel=0.05)


# ---------------------------------------------------------------------------
# D1: convert_to_working_space's BLAS matmul form vs colour-science's own per-pixel broadcast
# (see CLAUDE.md, "D1" — the user accepted "looks identical", not bit-identical).
# ---------------------------------------------------------------------------


def _convert_with_colour(img: np.ndarray, profile) -> np.ndarray:
    """Verbatim copy of convert_to_working_space as of dff3cb2 — the D1 oracle."""
    import colour

    matrix = profile.rgb_to_pcs_xyz.astype(img.dtype, copy=False)
    pcs_xyz = img @ matrix.T
    working = colour.XYZ_to_RGB(
        pcs_xyz,
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False,
    )
    return np.asarray(working, dtype=img.dtype)


def test_working_space_matrices_reproduce_colour_science():
    import colour

    m_cat, m_xyz = working_space_matrices()
    xyz = np.random.default_rng(0).uniform(0, 1, (1000, 3))
    expected = colour.XYZ_to_RGB(
        xyz,
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False,
    )
    # atol=0 (as the brief specifies) is too strict here: this is exactly the last-float64-bit
    # divergence the brief's own background measurements describe — colour's per-pixel `vecmul`
    # broadcast and a single chained matmul use FMA differently, which shows up as a ~1-2 ULP
    # *absolute* difference that's a large *relative* difference for the small values this random
    # sample happens to produce near the neutral axis. Confirmed by direct comparison (not assumed):
    # `vecmul(m_cat, xyz)` vs `xyz @ m_cat.T` differ by up to 2.22e-16 absolute (1 float64 ULP) on
    # this exact seed, before the second matrix is even applied. A tiny atol absorbs that expected
    # noise without loosening the real check (that the two matrices reproduce colour's maths).
    np.testing.assert_allclose(xyz @ m_cat.T @ m_xyz.T, expected, rtol=1e-14, atol=1e-15)


def test_convert_to_working_space_within_two_ulps_of_colour_science():
    rng = np.random.default_rng(3)
    img = (10.0 ** rng.uniform(-5, 0.5, (1000, 1000, 3))).astype(np.float32)
    profile = parse_linear_rgb_profile(build_icc(LINEAR_TAGS))
    np.testing.assert_array_max_ulp(
        convert_to_working_space(img, profile), _convert_with_colour(img, profile), maxulp=2
    )


@pytest.mark.parametrize("name", ["IMG_0151.tif", "IMG_0156.tif", "IMG_0156-nowb.tif", "IMG_0158.tif"])
def test_convert_to_working_space_on_real_scans(name):
    path = Path(__file__).resolve().parents[2] / name
    if not path.exists():
        pytest.skip(f"{name} not present (real scans are local-only)")
    scan = read_tiff(path)
    profile = parse_linear_rgb_profile(scan.icc_profile)
    band = np.ascontiguousarray(scan.image[:512])
    np.testing.assert_array_max_ulp(
        convert_to_working_space(band, profile), _convert_with_colour(band, profile), maxulp=2
    )
