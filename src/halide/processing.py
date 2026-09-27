"""Ties I/O, color management, calibration, and the core pipeline together into "process one
negative scan" — the single place that logic lives, so the single-file CLI command and the batch
orchestrator (which calls this once per file inside a process pool) don't duplicate it.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from PIL import Image

from halide import banding
from halide import device as _device  # to_device/to_host looked up at call time: tests inject a fake
from halide.banding import map_in_bands
from halide.calibration.auto import auto_density_balance, roll_auto_density_balance
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core._xp import array_namespace
from halide.core.pipeline import negative_to_positive
from halide.core.tone_render import ResolvedTone, apply_tone, resolve_tone
from halide.core.types import DensityProfile, Stage, ToneCurveParams  # noqa: F401 -- Stage re-exported
from halide.device import ComputeDevice
from halide.io.icc import (
    LinearRGBProfile,
    UnsupportedICCProfileError,
    convert_to_working_space,
    output_profile_bytes,
    parse_linear_rgb_profile,
)
from halide.io.contact_sheet import (
    DEFAULT_FRAME_WIDTH,
    save_thumbnail,
    thumbnail_from_display,
    thumbnail_from_linear,
)
from halide.io.raster import to_srgb_8bit, write_delivery_image, write_srgb_8bit_image
from halide.io.scan_metadata import DarktableState, ScanSettings, read_scan_metadata
from halide.io.tiff import copy_exif_metadata, read_tiff, read_tiff_description, set_description, write_tiff

IDENTITY_PROFILE = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))


@functools.cache
def _acescg_matrix() -> np.ndarray:
    # colour-science is imported where it's used, not at module level: importing it (and the
    # scipy it pulls in) is most of the CLI's startup time, which `halide --help` shouldn't pay.
    import colour

    d50_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    return colour.RGB_to_XYZ(
        np.eye(3),
        colourspace=colour.RGB_COLOURSPACES["ACEScg"],
        illuminant=d50_xy,
        chromatic_adaptation_transform="Bradford",
        apply_cctf_decoding=False,
    ).T


class ScanColorError(Exception):
    """Raised when a scan's embedded ICC profile is missing or unsupported."""


class PrintInputError(Exception):
    """Raised when `halide print` is given something that isn't a flat positive to print."""


_PROVENANCE_KEY = "halide"


def _halide_version() -> str:
    try:
        return version("halide")
    except PackageNotFoundError:
        return "unknown"


def provenance_json(
    resolved: ResolvedTone, profile: DensityProfile | None, scan_gain: float = 1.0, device: str = "cpu"
) -> str:
    """What was done to produce an output file, written into its TIFF ImageDescription: the
    printing decision (fitted or pinned) and, where known, the calibration. For reproducibility,
    and so `halide print` can exactly undo a flat output's exposure scale when the file comes back
    untouched. External editors (darktable) are not expected to preserve it — nothing *requires*
    it to be present.

    `device` ("cpu" / "gpu") is where the frame was actually developed: the two agree only to
    float32 rounding (docs/plans/gpu-acceleration.md, D2), not bit for bit, so a file says which
    made it. A GPU frame that fell back to the CPU records "cpu"."""
    record: dict = {
        "output": "flat" if resolved.mode == "linear" else "print",
        "version": _halide_version(),
        "device": device,
    }
    if resolved.mode == "linear":
        record["linear_scale"] = float(resolved.linear_scale)
    else:
        record["exposure"] = float(resolved.exposure)
        record["contrast"] = float(resolved.contrast)
    if profile is not None:
        record["white_balance"] = [float(v) for v in profile.white_balance]
        record["density_scale"] = [float(v) for v in profile.density_scale]
        if profile.film_stock:
            record["film_stock"] = profile.film_stock  # for contact sheets' edge print
    if scan_gain != 1.0:
        record["scan_gain"] = float(scan_gain)
    return json.dumps({_PROVENANCE_KEY: record})


def read_provenance(description: str | None) -> dict | None:
    if not description:
        return None
    try:
        data = json.loads(description)
    except ValueError:
        return None
    record = data.get(_PROVENANCE_KEY) if isinstance(data, dict) else None
    return record if isinstance(record, dict) else None


def load_working_space_image(path: str | Path) -> np.ndarray:
    """Read a TIFF and convert it into the internal ACEScg working space, validating its embedded
    ICC profile along the way. Raises ScanColorError with a specific, actionable message if the
    profile is missing or unsupported.

    The returned array is a fresh buffer the caller owns outright (nothing else references it), so
    callers may develop it in place. The conversion itself runs band by band into the decoded
    buffer (see halide.banding): converting the whole frame at once held ~5 frames of
    colour-science's float64 temporaries (+900 MiB on a real scan)."""
    image, source_profile = _read_scan(path)
    return _to_working_space(image, source_profile)


def _read_scan(path: str | Path) -> tuple[np.ndarray, LinearRGBProfile | None]:
    """load_working_space_image's first half: decode and validate, but don't convert yet. Returns
    the writable decoded buffer and the profile to convert it from — None when it's already in the
    working space. Split out for the device path, which uploads the buffer *unconverted* (the
    conversion runs on the GPU) and needs it untouched to start over on the CPU if the GPU fails."""
    scan = read_tiff(path)
    if scan.icc_profile is None:
        raise ScanColorError(f"{path}: no embedded ICC profile found; cannot verify color space")
    if scan.icc_profile == output_profile_bytes():
        # Tagged with halide's own ACEScg output profile (e.g. a flat positive coming back for
        # `halide print`): already in the working space by definition. Converting anyway isn't a
        # true identity — the profile's s15Fixed16 matrix round-trips ACEScg only to ~1e-4 per
        # channel — so skip it rather than add that drift to a file halide itself wrote.
        return np.require(scan.image, requirements="W"), None
    try:
        source_profile = parse_linear_rgb_profile(scan.icc_profile)
    except UnsupportedICCProfileError as exc:
        raise ScanColorError(f"{path}: unsupported color profile — {exc}") from exc
    # copies only if tifffile handed back read-only data
    return np.require(scan.image, requirements="W"), source_profile


def _to_working_space(image, source_profile: LinearRGBProfile | None, band_bytes: int | None = None):
    """Convert `image` (host or device array) in place, band by band; a no-op for None."""
    if source_profile is None:
        return image
    return map_in_bands(image, lambda band: convert_to_working_space(band, source_profile), band_bytes=band_bytes)


def process_scan(
    input_path: str | Path,
    output_path: str | Path | None,
    stage: Stage,
    density_profile: DensityProfile | None,
    tone_params: ToneCurveParams,
    scan_gain: float = 1.0,
    thumbnail_path: str | Path | None = None,
    thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH,
    device: ComputeDevice | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> ResolvedTone | None:
    """Process one negative scan end to end and write the result.

    `scan_gain` (from --match-scan-exposure, see calibration/scan_consistency.py) multiplies the
    linear scan before calibration, putting a frame digitized at a different camera exposure back at
    the exposure its profile was solved at. 1.0 = untouched.

    `density_profile=None` means "compute a per-frame automatic profile from this image" — not
    valid combined with `stage=Stage.INVERT_ONLY`, which always uses the identity profile.

    `thumbnail_path`, if given, also writes a contact-sheet thumbnail of the result (see
    io/contact_sheet.py); `output_path=None` then skips the full-size TIFF entirely — what
    `batch --contact-sheet` without an output directory does, so previewing a roll's settings doesn't
    fill a folder with TIFFs. The thumbnail is made from exactly the same full-resolution develop
    (including the per-frame print fit), so the preview matches what a real run would write.

    `device`: None or a CPU device runs on the CPU, exactly as before the GPU work. A GPU device
    develops the frame there (see _run_on_device); if the GPU fails for any reason — out of memory,
    a driver error — the frame is redone on the CPU and `on_warning` is told why (printed if no
    callback is given). A GPU problem never fails a frame the CPU could have developed.

    Returns the tone values actually used (None for Stage.DENSITY_ONLY, which has no tone stage).
    """
    # One full-frame host buffer for the whole run: decoded, developed (or downloaded into) and
    # written in place.
    image, source_profile = _read_scan(input_path)

    def develop(frame, band_bytes=None):
        return _develop_frame(frame, source_profile, scan_gain, density_profile, stage, tone_params, band_bytes)

    developed, image = _run_on_device(input_path, image, device, develop, on_warning)
    used_device = "gpu" if developed is not None else "cpu"
    if developed is None:
        developed = develop(image)
    resolved, profile = developed

    record = provenance_json(resolved, profile, scan_gain, used_device) if resolved is not None else None
    if output_path is not None:
        write_tiff(output_path, image, icc_profile=output_profile_bytes())
    if thumbnail_path is not None:
        save_thumbnail(thumbnail_path, thumbnail_from_linear(image, thumbnail_long_edge),
                       read_provenance(record))
    # Free the frame before exiftool (a separate process) rewrites the output, so a batch worker
    # never holds a developed frame and exiftool's own memory at once — see the per-worker memory
    # estimate in batch/orchestrator.py.
    del image
    if output_path is not None:
        # Output is always ACEScg, a different profile than the source — exiftool must not clobber
        # the ACEScg tag we just wrote with the source's own ICC bytes.
        copy_exif_metadata(str(input_path), str(output_path), drop_icc=True)
        if record is not None:
            set_description(output_path, record)
    return resolved


def _develop_frame(
    frame,
    source_profile: LinearRGBProfile | None,
    scan_gain: float,
    density_profile: DensityProfile | None,
    stage: Stage,
    tone_params: ToneCurveParams,
    band_bytes: int | None = None,
) -> tuple[ResolvedTone | None, DensityProfile]:
    """All of process_scan's arithmetic, in place on `frame` — the decoded, not yet converted scan,
    as a host (numpy) array or a device (CuPy) array. One implementation for both: core/ picks the
    array library from the array (core/_xp.py), so the GPU runs exactly the CPU's steps in the
    CPU's order, and the CPU path is the code it always was. Returns (tone used, profile used)."""
    _to_working_space(frame, source_profile, band_bytes)
    if scan_gain != 1.0:
        # A scalar of the frame's own dtype: the multiply stays in float32, and a scalar (unlike a
        # 0-d numpy array) is accepted as an operand by a device array too.
        frame *= frame.dtype.type(scan_gain)

    if stage is Stage.INVERT_ONLY:
        profile = IDENTITY_PROFILE
    elif density_profile is not None:
        profile = density_profile
    else:
        # Auto calibration isn't device-ready yet (plan Task 8): on a GPU it gets a host copy of
        # the frame — one extra frame of host memory, for the length of this call.
        profile = auto_density_balance(_on_host(frame))

    if stage is Stage.DENSITY_ONLY:
        map_in_bands(frame, lambda band: apply_density_balance(apply_white_balance(band, profile), profile),
                     band_bytes=band_bytes)
        return None, profile
    return _develop_in_place(frame, profile, tone_params, band_bytes), profile


def _develop_in_place(
    image, profile: DensityProfile, tone_params: ToneCurveParams, band_bytes: int | None = None
) -> ResolvedTone:
    """core.pipeline.develop, applied band by band into `image` (which the caller owns): the same
    per-pixel stages in the same order, with the one whole-frame step — the print fit — run on the
    full buffer between the two banded passes, exactly where develop() runs it. Bit-identical to
    develop() (pinned by tests/unit/test_banding.py), at ~1 frame of memory instead of ~9."""
    map_in_bands(image, lambda band: negative_to_positive(band, profile), band_bytes=band_bytes)
    resolved = _tone_on_host(resolve_tone(image, tone_params))
    map_in_bands(image, lambda band: apply_tone(band, resolved, tone_params.curve_path), band_bytes=band_bytes)
    return resolved


def _on_host(frame) -> np.ndarray:
    return frame if array_namespace(frame) is np else _device.to_host(frame)


def _tone_on_host(resolved: ResolvedTone) -> ResolvedTone:
    """resolve_tone's result with every value a host scalar. On a device frame, the flat output's
    linear_scale comes back as a 0-d device array (the fit's exposure/contrast are already floats).
    It is brought back as the CPU has it — a numpy scalar of the frame's dtype, via `[()]` — not as
    a Python float, which would make the flat output's scaling run in float64 and move its last bit
    (see estimate_linear_scale). On the CPU this returns `resolved` unchanged."""
    scale = resolved.linear_scale
    if scale is None or isinstance(scale, (float, int, np.generic)):
        return resolved
    return dataclasses.replace(resolved, linear_scale=_device.to_host(scale)[()])


def _run_on_device(input_path, host: np.ndarray, device: ComputeDevice | None, work, on_warning):
    """Run `work(frame, band_bytes)` on a GPU copy of `host` and download the result into `host`.

    Returns (work's result, host), or (None, host) when there's no GPU to use or it failed — then
    `host` is the decoded, unconverted scan, ready for the CPU to start over on. That is why the
    device path writes into `host` only once, at the very end (the plan's "out-of-memory never
    fails a frame", §3.3): up to then a failure leaves it as decoded. The one exception is the
    download itself — a GPU runs asynchronously, so an earlier kernel's error can surface there,
    after part of `host` may have been overwritten — and then the scan is read again from disk.

    Host memory stays ~1 frame: `host` is both the upload source and the download target."""
    if device is None or device.kind != "gpu":
        return None, host
    frame = None
    downloading = False
    try:
        frame = _device.to_device(host)
        result = work(frame, banding.DEVICE_BAND_BYTES)
        downloading = True
        _device.to_host(frame, out=host)
        return result, host
    except Exception as exc:  # noqa: BLE001 — any GPU problem: redo on the CPU, never fail the frame
        del frame
        _device.release_memory()
        _warn(on_warning, _gpu_fallback_message(input_path, exc))
        if downloading:
            host, _ = _read_scan(input_path)
        return None, host


def _gpu_fallback_message(input_path, exc: Exception) -> str:
    text = str(exc)
    # cupy.cuda.memory.OutOfMemoryError is a MemoryError; cuBLAS and the runtime report their own
    # allocation failures as status codes instead.
    if isinstance(exc, MemoryError) or "cudaErrorMemoryAllocation" in text or "ALLOC_FAILED" in text:
        return f"{input_path}: out of GPU memory — developed this frame on the CPU instead"
    detail = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return f"{input_path}: the GPU failed ({detail}) — developed this frame on the CPU instead"


def _warn(on_warning: Callable[[str], None] | None, message: str) -> None:
    if on_warning is not None:
        on_warning(message)
    else:
        print(f"Warning: {message}")


_DISPLAY_SUFFIXES = (".png", ".jpg", ".jpeg")


def thumbnail_existing_output(
    input_path: str | Path, thumbnail_path: str | Path, thumbnail_long_edge: int = DEFAULT_FRAME_WIDTH
) -> None:
    """A contact-sheet thumbnail of an already-processed file: halide's own TIFF output (or any
    linear, profile-embedded TIFF — colour-managed the same way as a scan), or a display-encoded
    PNG/JPEG such as `halide export` writes. The file's recorded printing decision, if any, comes
    along for the caption."""
    path = Path(input_path)
    if path.suffix.lower() in _DISPLAY_SUFFIXES:
        with Image.open(path) as image:
            save_thumbnail(thumbnail_path, thumbnail_from_display(image, thumbnail_long_edge), None)
        return
    record = read_provenance(read_tiff_description(path))
    image = load_working_space_image(path)
    save_thumbnail(thumbnail_path, thumbnail_from_linear(image, thumbnail_long_edge), record)


def print_scan(
    input_path: str | Path,
    output_path: str | Path,
    tone_params: ToneCurveParams,
    device: ComputeDevice | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> tuple[ResolvedTone, str | None]:
    """`halide print`: apply only the print stage (fitted exposure + grade, paper curve) to a flat
    linear positive — typically `halide invert --output flat`'s output after scene-level editing in
    darktable. The input goes through the same ICC validation/conversion as a negative scan, so a
    gamma-encoded or unprofiled file is rejected the same way.

    Returns (resolved tone, warning-or-None). If the file still carries halide's flat-output
    provenance, its exposure scale is undone first, so a pinned --exposure means exactly what it
    means on `invert` and an untouched flat file prints identically to `invert --output print`.
    Without that metadata (darktable won't normally keep it), a pinned exposure can't be reproduced
    — it's dropped in favour of the fit, with a warning. The fitted exposure itself doesn't need the
    metadata: it's invariant to a global multiply (see core.tone_render.fit_print).

    `device` / `on_warning`: as process_scan — a GPU failure redoes the print on the CPU and is
    reported through `on_warning`, separately from the returned (metadata) warning.
    """
    # Header only: decoding the pixels here just for the description held a second full frame
    # alongside the decoded image.
    provenance = read_provenance(read_tiff_description(input_path))
    if provenance is not None and provenance.get("output") == "print":
        raise PrintInputError(
            f"{input_path} is already a halide print (it has the tone curve applied) — `halide print` "
            f"expects a flat positive from `halide invert --output flat`"
        )
    image, source_profile = _read_scan(input_path)

    warning = None
    exposure = tone_params.exposure
    scale = provenance.get("linear_scale") if provenance is not None else None
    if not (isinstance(scale, (int, float)) and scale > 0):
        scale = None
        if exposure is not None:
            warning = (
                f"{input_path} has no halide flat-output metadata (normal after editing elsewhere), so a "
                f"pinned exposure of {exposure:+.3f} can't be reproduced on it — fitting exposure instead"
            )
            exposure = None
    print_params = ToneCurveParams(
        mode="paper", exposure=exposure, contrast=tone_params.contrast, curve_path=tone_params.curve_path
    )

    def print_frame(frame, band_bytes=None) -> ResolvedTone:
        # The whole print stage in place on `frame` (host or device array) — see _develop_frame.
        _to_working_space(frame, source_profile, band_bytes)
        if scale is not None:
            frame /= frame.dtype.type(scale)  # a scalar of the frame's dtype: see _develop_frame
        resolved = _tone_on_host(resolve_tone(frame, print_params))
        map_in_bands(frame, lambda band: apply_tone(band, resolved, print_params.curve_path), band_bytes=band_bytes)
        return resolved

    resolved, image = _run_on_device(input_path, image, device, print_frame, on_warning)
    used_device = "gpu" if resolved is not None else "cpu"
    if resolved is None:
        resolved = print_frame(image)
    write_tiff(output_path, image, icc_profile=output_profile_bytes())
    del image  # before exiftool runs — see process_scan
    copy_exif_metadata(str(input_path), str(output_path), drop_icc=True)
    set_description(output_path, provenance_json(resolved, None, device=used_device))
    return resolved, warning


def export_delivery_image(
    input_path: str | Path,
    output_path: str | Path,
    quality: int = 95,
    device: ComputeDevice | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> str | None:
    """Convert one processed ACEScg TIFF into a delivery-ready sRGB PNG/JPEG. The single place this
    logic lives, so `halide export`'s single-file and bulk-directory modes, and the export worker
    pool, don't duplicate it — same reasoning as process_scan above.

    Returns a warning message if the input's embedded ICC profile doesn't look like ACEScg (or is
    missing/unusable), or None if it looks fine. Unlike ScanColorError elsewhere in this module,
    this is not fatal — export can still proceed by assuming ACEScg, it just may be wrong.

    `device`: None or a CPU device runs the sRGB conversion on the CPU exactly as before. A GPU
    device converts there instead (see `_export_srgb_on_device`); on any GPU problem the conversion
    is redone on the CPU and `on_warning` is told why (printed if no callback is given) — the same
    "never fail a frame the CPU could have handled" contract as `process_scan`/`_run_on_device`.
    Nothing here is written into `scan.image` on the way, so a fallback needs no re-read.
    """
    scan = read_tiff(input_path)
    warning = None
    if scan.icc_profile is None:
        warning = f"{input_path} has no embedded ICC profile; assuming it is ACEScg."
    else:
        try:
            profile = parse_linear_rgb_profile(scan.icc_profile)
            if not np.allclose(profile.rgb_to_pcs_xyz, _acescg_matrix(), atol=1e-3):
                warning = (
                    f"{input_path}'s embedded profile does not look like ACEScg — `halide export` "
                    f"expects the output of `halide invert`/`halide batch`. Proceeding anyway, but "
                    f"colors may be wrong."
                )
        except UnsupportedICCProfileError as exc:
            warning = f"{input_path}'s embedded profile is unusable ({exc}); assuming ACEScg anyway."

    srgb_8bit = _export_srgb_on_device(input_path, scan.image, device, on_warning)
    if srgb_8bit is not None:
        write_srgb_8bit_image(output_path, srgb_8bit, quality=quality)
    else:
        write_delivery_image(output_path, scan.image, quality=quality)
    return warning


def _export_srgb_on_device(input_path, acescg_image: np.ndarray, device: ComputeDevice | None, on_warning):
    """`export_delivery_image`'s device path: upload `acescg_image` once, run `to_srgb_8bit`
    (namespace-generic, halide.io.raster) over it in DEVICE_BAND_BYTES bands, and download each
    band straight into a host uint8 buffer this function owns alone — never into `acescg_image`
    itself. That means a failure partway through never needs to re-read the scan the way
    `_run_on_device` does for the develop path: `acescg_image` was only ever read from, so a plain
    CPU conversion of it afterwards is exactly as if the GPU had never been tried.

    Returns the finished (H, W, 3) uint8 array, or None when there's no GPU to use or it failed
    (then `on_warning` is told why, as with `_run_on_device`)."""
    if device is None or device.kind != "gpu":
        return None
    frame = None
    try:
        frame = _device.to_device(acescg_image)
        srgb_8bit = np.empty(acescg_image.shape, dtype=np.uint8)
        row_bytes = int(np.prod(acescg_image.shape[1:], dtype=np.int64)) * acescg_image.dtype.itemsize
        for band in banding.band_slices(acescg_image.shape[0], row_bytes, banding.DEVICE_BAND_BYTES):
            _device.to_host(to_srgb_8bit(frame[band]), out=srgb_8bit[band])
        return srgb_8bit
    except Exception as exc:  # noqa: BLE001 — any GPU problem: redo on the CPU, never fail the export
        del frame
        _device.release_memory()
        _warn(on_warning, _gpu_fallback_message(input_path, exc))
        return None


def estimate_roll_density_profile(
    input_paths: list[str | Path],
    stride: int = 8,
    scan_gains: dict[str, float] | None = None,
    on_skip: Callable[[Path, Exception], None] | None = None,
) -> DensityProfile:
    """Estimate one shared density-balance profile from a whole roll's worth of input files, for
    --auto-density-roll batch mode. Reads every file (downsampled by `stride` for speed/memory —
    a roll's worth of full-resolution scans held in memory at once would be wasteful for what is
    just a statistical estimate) and combines them via calibration.auto.roll_auto_density_balance.

    Runs in the main process, ahead of (and outside) the per-worker try/except in
    batch.orchestrator — so a single unreadable/malformed file here must not abort the whole roll
    estimate any more than it aborts the rest of the batch. Unreadable files are skipped with a
    warning (printed, or passed to `on_skip` so a caller mid-display can show it its own way); the
    actual per-file error still surfaces normally when that file is processed for
    real in the worker pool.

    Catches broad `Exception`, not just `ScanColorError` — found via testing with a genuinely
    corrupt file (not just one missing/unsupported ICC data): `read_tiff` can raise straight from
    `tifffile` (e.g. `TiffFileError` on a truncated or non-TIFF file), which isn't a
    `ScanColorError` and was crashing the whole roll estimate before a single frame was even
    color-managed. `batch.orchestrator._worker` already catches broadly for the same reason — a
    corrupt file is exactly the kind of one-bad-frame case this function exists to tolerate.
    """
    images = []
    for path in input_paths:
        try:
            # .copy(): a strided slice is a *view* that keeps the whole full-resolution frame alive
            # (~180 MiB each for a real scan) — holding a roll's worth of those got the process
            # OOM-killed on a real 37-frame roll. The copy is ~2 MiB and frees the full frame.
            image = load_working_space_image(path)[::stride, ::stride, :].copy()
            gain = (scan_gains or {}).get(str(path), 1.0)
            images.append(image * np.asarray(gain, dtype=image.dtype) if gain != 1.0 else image)
        except Exception as exc:  # noqa: BLE001 — one corrupt frame must not abort the roll estimate
            if on_skip is not None:
                on_skip(Path(path), exc)
            else:
                print(f"Warning: skipping {path} while estimating roll density balance ({exc})")
    if not images:
        raise ScanColorError("no readable frames found to estimate a roll density-balance profile from")
    return roll_auto_density_balance(images)


def read_roll_scan_metadata(paths: list[str | Path]) -> dict[str, tuple[ScanSettings | None, DarktableState | None]]:
    """Scan settings + darktable export state for each file, from headers only (see
    io/scan_metadata.py) — cheap enough to run over a whole roll before processing starts."""
    return {str(path): read_scan_metadata(path) for path in paths}
