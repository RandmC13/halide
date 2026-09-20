"""`halide export` — convert a processed ACEScg TIFF (or a directory of them, e.g. the output of
`halide batch`) into delivery-ready sRGB PNG/JPEG."""

from __future__ import annotations

import argparse
from pathlib import Path

import colour
import numpy as np

from halide.batch.orchestrator import TIFF_SUFFIXES, BatchJob, BatchResult
from halide.batch.progress import GridProgressRenderer
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

_FORMATS = ("png", "jpg", "jpeg")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", help="Input ACEScg TIFF, or a directory of them (output of `halide invert`/`halide batch`)"
    )
    parser.add_argument(
        "output",
        help="Output image path (.png, .jpg, or .jpeg) for a single file, or an output directory "
        "when `input` is a directory",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality, 1-100 (default: 95; ignored for PNG)"
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=_FORMATS,
        help="Output format when `input` is a directory (default: png). Ignored for a single "
        "file, where the output path's own extension is used instead.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix to append to output filenames when `input` is a directory (default: none)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the progress display when `input` is a directory"
    )


def _check_profile(input_path: Path, icc_profile: bytes | None) -> str | None:
    """Returns a warning message if `icc_profile` doesn't look like ACEScg (or is unusable/missing),
    or None if it looks fine."""
    if icc_profile is None:
        return f"{input_path} has no embedded ICC profile; assuming it is ACEScg."
    try:
        profile = parse_linear_rgb_profile(icc_profile)
        if not np.allclose(profile.rgb_to_pcs_xyz, _ACESCG_MATRIX, atol=1e-3):
            return (
                f"{input_path}'s embedded profile does not look like ACEScg — `halide export` "
                f"expects the output of `halide invert`/`halide batch`. Proceeding anyway, but "
                f"colors may be wrong."
            )
    except UnsupportedICCProfileError as exc:
        return f"{input_path}'s embedded profile is unusable ({exc}); assuming ACEScg anyway."
    return None


def _export_one(input_path: Path, output_path: Path, quality: int) -> str | None:
    scan = read_tiff(input_path)
    warning = _check_profile(input_path, scan.icc_profile)
    write_delivery_image(output_path, scan.image, quality=quality)
    return warning


def _run_single(args: argparse.Namespace, input_path: Path) -> int:
    scan = read_tiff(input_path)
    warning = _check_profile(input_path, scan.icc_profile)
    if warning:
        print(f"Warning: {warning}")
    write_delivery_image(args.output, scan.image, quality=args.quality)
    return 0


def _run_bulk(args: argparse.Namespace, input_dir: Path) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in TIFF_SUFFIXES)
    if not files:
        print(f"No TIFF files found in {input_dir}")
        return 1

    jobs = [
        BatchJob(input_path=f, output_path=output_dir / f"{f.stem}{args.suffix}.{args.format}")
        for f in files
    ]

    renderer = None if args.quiet else GridProgressRenderer(total=len(jobs))
    results: list[BatchResult] = []
    warnings: list[tuple[str, str]] = []

    if renderer:
        renderer.start()

    for i, job in enumerate(jobs):
        if renderer:
            renderer.mark_processing(i)
        try:
            warning = _export_one(job.input_path, job.output_path, args.quality)
            if warning:
                warnings.append((job.input_path.name, warning))
            result = BatchResult(job=job, error=None)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not abort the bulk export
            result = BatchResult(job=job, error=str(exc))
        results.append(result)
        if renderer:
            renderer.report(i, result)

    if renderer:
        renderer.finish()

    for name, warning in warnings:
        print(f"Warning: {name}: {warning}")

    failures = [r for r in results if r.error]
    if failures and renderer is None:
        for r in failures:
            print(f"FAILED: {r.job.input_path.name}: {r.error}")
    return 1 if failures else 0


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if input_path.is_dir():
        return _run_bulk(args, input_path)
    return _run_single(args, input_path)
