"""`halide export` — convert a processed ACEScg TIFF into a delivery-ready sRGB PNG/JPEG."""

from __future__ import annotations

import argparse

import colour
import numpy as np

from halide.io.icc import UnsupportedICCProfileError, parse_linear_rgb_profile
from halide.io.raster import write_delivery_image
from halide.io.tiff import read_tiff

_D50_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
_ACESCG_MATRIX = colour.RGB_to_XYZ(
    np.eye(3),
    colourspace=colour.RGB_COLOURSPACES["ACEScg"],
    illuminant=_D50_XY,
    chromatic_adaptation_transform="Bradford",
    apply_cctf_decoding=False,
).T


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input ACEScg TIFF (output of `halide invert`/`halide batch`)")
    parser.add_argument("output", help="Output image path (.png, .jpg, or .jpeg)")
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality, 1-100 (default: 95; ignored for PNG)"
    )


def run(args: argparse.Namespace) -> int:
    scan = read_tiff(args.input)

    if scan.icc_profile is None:
        print(f"Warning: {args.input} has no embedded ICC profile; assuming it is ACEScg.")
    else:
        try:
            profile = parse_linear_rgb_profile(scan.icc_profile)
            if not np.allclose(profile.rgb_to_pcs_xyz, _ACESCG_MATRIX, atol=1e-3):
                print(
                    f"Warning: {args.input}'s embedded profile does not look like ACEScg — "
                    f"`halide export` expects the output of `halide invert`/`halide batch`. "
                    f"Proceeding anyway, but colors may be wrong."
                )
        except UnsupportedICCProfileError as exc:
            print(f"Warning: {args.input}'s embedded profile is unusable ({exc}); assuming ACEScg anyway.")

    write_delivery_image(args.output, scan.image, quality=args.quality)
    return 0
