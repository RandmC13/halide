"""Ties I/O, color management, calibration, and the core pipeline together into "process one
negative scan" — the single place that logic lives, so the single-file CLI command and the batch
orchestrator (which calls this once per file inside a process pool) don't duplicate it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import json
import re
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from halide import banding
from halide import device as _device  # to_device/to_host looked up at call time: tests inject a fake
from halide.banding import map_in_bands
from halide.calibration.auto import auto_density_balance, roll_auto_density_balance
from halide.core._xp import array_namespace
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.pipeline import negative_to_positive
from halide.core.tone_render import ResolvedTone, apply_tone, resolve_tone
from halide.io.lut import load_paper_curve
from halide.core.types import DensityProfile, Stage, ToneCurveParams  # noqa: F401 -- Stage re-exported
from halide.device import ComputeDevice
from halide.io.atomic import atomic_output
from halide.io.exiftool import ExifToolError
from halide.io.roll import RAW_SUFFIXES
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
from halide.io.raster import srgb_8bit_from_acescg, to_srgb_8bit, write_delivery_image, write_srgb_8bit_image
from halide.io.scan_metadata import DarktableState, ScanSettings, read_scan_metadata
from halide.io.tiff import (
    copy_exif_metadata,
    read_tiff,
    read_tiff_description,
    read_tiff_shape,
    set_description,
    write_tiff,
)

# A failure exiftool reports for the file it was asked to tag (see io/tiff.py::copy_exif_metadata):
# ExifToolError from its kept-open session, or CalledProcessError if the command had to fall back
# to a one-shot call. Either way the pixel data is already good — see process_scan/print_scan.
_EXIF_FAILURE = (ExifToolError, subprocess.CalledProcessError)

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


class ScanInputError(ScanColorError):
    """Raised when an input file isn't a scan halide can develop: not a TIFF, unreadable, or not
    RGB (F25), or already a halide positive (F15). A ScanColorError subclass so it travels the same
    way — unwrapped through the GPU paths, reported per frame by a batch, a plain message from the
    CLI — because it too is a fact about the input, never a GPU failure to retry on the CPU."""


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
    # allow_nan=False (F05): a NaN/inf value here would otherwise be written as a bare, non-standard
    # `NaN`/`Infinity` token — valid to json.dumps by default, but not valid JSON for any other
    # reader — so this raises a clear ValueError instead of silently writing a broken file.
    return json.dumps({_PROVENANCE_KEY: record}, allow_nan=False)


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
    return _to_working_space(image, source_profile, name=str(path))


_EXPORT_HELP = ("halide develops linear TIFFs exported from darktable or RawTherapee - see the README's "
                "'Exporting your scans'.")


def _looks_like(path: Path) -> str:
    """What a file that isn't a readable TIFF appears to be, from its first bytes and its suffix."""
    try:
        with open(path, "rb") as handle:
            magic = handle.read(8)
    except OSError:
        magic = b""
    if magic[:2] == b"\xff\xd8":
        return "it looks like a JPEG"
    if magic[:4] == b"\x89PNG":
        return "it looks like a PNG"
    if path.suffix.lower() in RAW_SUFFIXES:
        return "it looks like a raw file"
    return "it may be damaged or cut short"


def _not_a_tiff(path: Path) -> ScanInputError:
    return ScanInputError(f"{path.name} isn't a TIFF halide can read ({_looks_like(path)}). "
                          f"{_EXPORT_HELP}")


def _decode_tiff(path: str | Path, out: np.ndarray | None = None):
    """read_tiff, with anything the decoder raises on a damaged or cut-short file (tifffile's own
    errors, a libdeflate/zlib failure, a short read) reported as the plain "isn't a TIFF halide can
    read" message instead of library text. Not a blanket catch: running out of memory is not a bad file."""
    try:
        return read_tiff(path, out=out)
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception:  # noqa: BLE001 -- see docstring: every decoder failure means "unreadable file"
        raise _not_a_tiff(Path(path)) from None


def _check_scan_layout(path: str | Path) -> None:
    """Fail with a plain message, from the header alone, for anything that isn't a readable RGB TIFF
    (F25) — before decoding, so a JPEG, a raw file, a greyscale or RGBA export never surfaces as a
    tifffile/numpy traceback."""
    path = Path(path)
    try:
        shape = read_tiff_shape(path)
    except FileNotFoundError:
        raise ScanInputError(f"input file not found: {path}") from None
    except Exception:  # noqa: BLE001 -- tifffile raises its own types for a non-TIFF; all mean the same
        raise _not_a_tiff(path) from None
    channels = shape[-1] if len(shape) >= 3 else 1
    if channels == 3:
        return
    if channels == 4:
        raise ScanInputError(f"{path.name} has an alpha (transparency) channel; export without it")
    if channels == 1:
        raise ScanInputError(f"{path.name} is greyscale; halide needs an RGB scan of a colour negative")
    raise ScanInputError(f"{path.name} has {channels} channels; halide needs RGB")


def _reject_positive(path: str | Path) -> None:
    """F15: a file halide itself already wrote (it carries halide's provenance) isn't a negative —
    developing it again would invert a positive. Header only; a file that can't be read is left for
    _read_scan to report properly."""
    path = Path(path)
    try:
        record = read_provenance(read_tiff_description(path))
    except Exception:  # noqa: BLE001 -- reported by the real read, just after
        return
    if record is None:
        return
    made = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    raise ScanInputError(f"{path.name} is already a halide positive (made on {made}). To re-print it use "
                         f"`halide print`; to develop again, point halide at the original scan")


def _read_scan(
    path: str | Path, out: np.ndarray | None = None
) -> tuple[np.ndarray, LinearRGBProfile | None]:
    """load_working_space_image's first half: decode and validate, but don't convert yet. Returns
    the writable decoded buffer and the profile to convert it from — None when it's already in the
    working space. Split out for the device path, which uploads the buffer *unconverted* (the
    conversion runs on the GPU) and needs it untouched to start over on the CPU if the GPU fails.

    `out`, if given, is decoded into directly (see halide.io.tiff.read_tiff) — used by the GPU
    service path (halide/gpu_service.py, halide/shared_frames.py) to decode straight into a
    shared-memory frame instead of this process's own heap. Callers that don't pass `out` see no
    change in behavior."""
    _check_scan_layout(path)
    scan = _decode_tiff(path, out)
    if scan.icc_profile is None:
        raise ScanColorError(f"{path}: no embedded ICC profile found; cannot verify colour space")
    if scan.icc_profile == output_profile_bytes():
        # Tagged with halide's own ACEScg output profile (e.g. a flat positive coming back for
        # `halide print`): already in the working space by definition. Converting anyway isn't a
        # true identity — the profile's s15Fixed16 matrix round-trips ACEScg only to ~1e-4 per
        # channel — so skip it rather than add that drift to a file halide itself wrote.
        return np.require(scan.image, requirements="W"), None
    try:
        source_profile = parse_linear_rgb_profile(scan.icc_profile)
    except UnsupportedICCProfileError as exc:
        # icc.py's damaged / not-D50 messages are whole sentences; the rest are terse facts.
        lead = "" if str(exc).startswith("the embedded colour profile") else "unusable colour profile — "
        raise ScanColorError(f"{path}: {lead}{exc}") from exc
    # copies only if tifffile handed back read-only data (never the case when `out` was given)
    return np.require(scan.image, requirements="W"), source_profile


_MAX_NONFINITE_FRACTION = 0.01  # F05: above this, the export itself looks broken, not just noisy


def _clean_nonfinite_band(band, xp) -> int:
    """Count *pixel locations* in `band` with a non-finite value in any channel, and zero every
    non-finite value in place — 0 if there are none, so clean input is never touched by
    `nan_to_num` (bit-identical, F05's hard constraint).

    Counts locations, not raw values: one corrupted input value spreads across all 3 channels once
    the ICC conversion's matrix multiply mixes them, but a photographer reading "N pixels" means N
    spots on the frame, not N individual R/G/B numbers — so a single bad input pixel is reported as
    1, not 3, even though all 3 of its output channels get zeroed."""
    nonfinite = ~xp.isfinite(band)
    count = int(xp.count_nonzero(xp.any(nonfinite, axis=-1)))
    if count:
        xp.nan_to_num(band, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return count


def _report_nonfinite(name: str, count: int, total_pixels: int, on_warning: Callable[[str], None] | None) -> None:
    """After `_clean_nonfinite_band` has zeroed `count` (of `total_pixels`) pixel locations: warn
    about a few (the reciprocal's MIN_TRANSMITTANCE floor then prints them as the brightest white,
    and the print fit's percentiles are robust to a handful) — or fail the frame outright above
    `_MAX_NONFINITE_FRACTION`, since that many means the export is broken, not the scan (F05)."""
    fraction = count / total_pixels
    if fraction > _MAX_NONFINITE_FRACTION:
        raise ScanColorError(
            f"{name}: {fraction * 100:.1f}% of its pixels aren't valid numbers - this export looks "
            f"broken; re-export it from the raw converter"
        )
    if count == 1:
        subject = "1 pixel wasn't a valid number (NaN/inf) and was treated as clear film"
    else:
        subject = f"{count} pixels weren't valid numbers (NaN/inf) and were treated as clear film"
    _warn(on_warning, f"{name}: {subject}. If this is more than a handful, check the raw "
                      f"converter's export")


def _to_working_space(image, source_profile: LinearRGBProfile | None, band_bytes: int | None = None,
                      *, name: str | None = None, on_warning: Callable[[str], None] | None = None):
    """Convert `image` (host or device array) in place, band by band; a no-op for None.

    Also where non-finite (NaN/inf) pixels are handled (F05): a raw converter's export can contain
    a few, e.g. from a highlight-recovery artifact. Counted and zeroed per band — only on a band
    that actually has any, so clean input costs nothing extra and stays bit-identical — before they
    can poison invert()'s reciprocal, the print fit's percentiles, or the JSON provenance with NaN.
    `name`/`on_warning` are None for callers with no live warning channel (thumbnails, previews,
    the roll density pre-pass): cleaning still happens, just without a message naming the file."""
    if source_profile is None:
        return image
    xp = array_namespace(image)
    counts: list[int] = []

    def convert(band):
        # A NaN/inf input propagates through the profile matmul (e.g. inf * -coef + inf = NaN) and
        # numpy warns about it ("invalid value encountered in matmul") every time — noise once this
        # is an intentionally handled case, not a bug, so it's suppressed here rather than left to
        # print on every real occurrence; np.errstate is thread-local/scoped, not a global change.
        with np.errstate(invalid="ignore"):
            converted = convert_to_working_space(band, source_profile)
        counts.append(_clean_nonfinite_band(converted, xp))
        return converted

    result = map_in_bands(image, convert, band_bytes=band_bytes)
    total = sum(counts)
    if total:
        total_pixels = image.size // image.shape[-1]  # locations, not raw values — see _clean_nonfinite_band
        _report_nonfinite(name if name is not None else str(image.shape), total, total_pixels, on_warning)
    return result


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
    service=None,
    shm_prefix: str | None = None,
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

    `service`: a batch worker's gpu_service.ServiceClient — the frame is decoded into shared memory
    (segment names starting with `shm_prefix`, see shared_frames.batch_prefix) and developed by the
    batch's one GPU service process instead of in this process; `device` is then not used. Any
    failure (the service's device, the service itself gone, no room in shared memory) develops the
    frame on the CPU here instead, with a warning, exactly as the in-process GPU path does.

    Returns the tone values actually used (None for Stage.DENSITY_ONLY, which has no tone stage).
    """
    _reject_positive(input_path)  # F15: before anything is written
    output_path = Path(output_path) if output_path is not None else None
    # Written under a hidden temp name beside the real one and only moved into place once every
    # step below (the TIFF write, then exiftool, then the provenance description) has run — a
    # killed or failed write never leaves a truncated file under the real output name (F04).
    atomic_ctx = atomic_output(output_path) if output_path is not None else contextlib.nullcontext(None)
    exif_warning = None
    with atomic_ctx as tmp_path:
        with contextlib.ExitStack() as shared:
            # One full-frame host buffer for the whole run: decoded, developed (or downloaded into)
            # and written in place — a shared-memory one when a GPU service develops it.
            frame = _shared_frame(input_path, service, shm_prefix, shared, on_warning)
            image, source_profile = _read_scan(input_path, out=None if frame is None else frame.array)
            request = DevelopRequest(source_profile=source_profile, scan_gain=scan_gain, density_profile=density_profile,
                                     stage=stage, tone_params=tone_params)

            if frame is not None:
                reply, image, used_device = _run_on_service(input_path, frame, service, "develop", request, on_warning)
                developed = None if reply is None else (reply.resolved, reply.profile)
            else:
                developed, image = _run_on_device(input_path, image, device, develop_request, request, on_warning)
                used_device = "gpu" if developed is not None else "cpu"
            if developed is None:
                developed = develop_request(image, request, name=str(input_path), on_warning=on_warning)
            resolved, profile = developed

            record = provenance_json(resolved, profile, scan_gain, used_device) if resolved is not None else None
            if tmp_path is not None:
                write_tiff(tmp_path, image, icc_profile=output_profile_bytes())
            if thumbnail_path is not None:
                save_thumbnail(thumbnail_path, thumbnail_from_linear(image, thumbnail_long_edge),
                               read_provenance(record))
            # Free the frame before exiftool (a separate process, kept running between frames on Linux —
            # halide.io.exiftool) rewrites the output, so its peak while writing never coincides with a
            # developed frame — see the per-worker memory estimate in batch/orchestrator.py. A shared
            # frame is unlinked on leaving this block.
            del image, frame
        if tmp_path is not None:
            # Output is always ACEScg, a different profile than the source — exiftool must not clobber
            # the ACEScg tag we just wrote with the source's own ICC bytes. Run on the temp path, like
            # the TIFF write itself, so the atomic replace below only ever exposes a complete file.
            try:
                copy_exif_metadata(str(input_path), str(tmp_path), drop_icc=True)
            except _EXIF_FAILURE as exc:
                # The pixels are already fully developed and written — a missing camera-metadata
                # copy is a warning, not a failed frame (2.3-4).
                exif_warning = f"{input_path}: developed, but its camera metadata couldn't be copied ({exc})"
            if record is not None:
                set_description(tmp_path, record)
    if exif_warning is not None:
        _warn(on_warning, exif_warning)
    return resolved


def _shared_frame(input_path, service, shm_prefix: str | None, stack: contextlib.ExitStack, on_warning,
                  extra_uint8: bool = False):
    """A shared-memory frame the size of `input_path`'s decoded scan (float32, as read_tiff always
    decodes), entered on `stack` so it's unlinked when the caller is done — or None when there's no
    `service` to hand it to, or no room for it. With `extra_uint8`, (frame, uint8 output frame) for
    an export, or None.

    No room (SharedMemoryUnavailable: /dev/shm full — more workers than it holds, or something
    else filled it) warns and returns None: the frame is then developed on the CPU in this worker,
    never lost. A header that can't be read returns None silently, so the decode that follows
    fails with the same error the CPU path gives."""
    if service is None:
        return None
    dead_reason = getattr(service, "dead_reason", None)
    if dead_reason is not None:
        # The service is gone for good: no point making a shared frame nobody will develop.
        action = "exported this file" if extra_uint8 else "developed this frame"
        _warn(on_warning, f"{input_path}: the GPU service is no longer available ({dead_reason}) — {action} "
                          f"on the CPU instead")
        return None
    from halide.shared_frames import SharedMemoryUnavailable, new_frame

    try:
        shape = read_tiff_shape(input_path)
    except Exception:  # noqa: BLE001 — reported by the real decode, just after
        return None
    try:
        frame = stack.enter_context(new_frame(shape, np.float32, prefix=shm_prefix))
        if not extra_uint8:
            return frame
        return frame, stack.enter_context(new_frame(shape, np.uint8, prefix=shm_prefix))
    except SharedMemoryUnavailable as exc:
        action = "exported this file" if extra_uint8 else "developed this frame"
        _warn(on_warning, f"{input_path}: no room in shared memory for the GPU service ({exc}) — {action} "
                          f"on the CPU instead")
        return None


def _run_on_service(input_path, frame, service, method: str, request, on_warning):
    """`service.<method>(frame, request, name=...)` — develop or print_ — on a shared frame the scan
    was decoded into. Returns (reply, buffer, device name): the service writes the result back into
    the frame, so the buffer is `frame.array`. On any failure — the request failed on the service's
    device (DeviceJobFailed), or the service is gone or stuck (ServiceUnavailable) — (None, the
    buffer the CPU should start over on, "cpu"): fall_back_to_cpu's rules, so a frame the service
    may have half-written, or may still write into, is re-read into a private buffer.

    A ScanColorError from the service (bad input data — F05, R8) is neither of those: it isn't
    caught here, so it propagates straight out, exactly as run_device_job's in-process counterpart
    does — no fallback, no "GPU failed" wording. Any warnings the service collected while running
    the job (it has nothing of its own to print them to) are re-emitted here through this worker's
    own `on_warning`, exactly as the in-process paths emit them directly."""
    from halide.gpu_service import ServiceUnavailable

    try:
        reply = getattr(service, method)(frame, request, name=str(input_path))
    except (DeviceJobFailed, ServiceUnavailable) as failed:
        return None, fall_back_to_cpu(input_path, frame.array, failed.failure, on_warning), "cpu"
    for message in reply.warnings:
        _warn(on_warning, message)
    return reply, frame.array, service.device_kind


@dataclasses.dataclass(frozen=True)
class DevelopRequest:
    """Everything `develop_request` needs besides the frame itself — process_scan's arithmetic as
    plain, picklable data, so the GPU service process (halide/gpu_service.py) can run exactly what
    the in-process device path runs. `source_profile` is the profile to convert the scan from
    (None when it's already in the working space); the rest are process_scan's own arguments."""

    source_profile: LinearRGBProfile | None
    scan_gain: float
    density_profile: DensityProfile | None
    stage: Stage
    tone_params: ToneCurveParams


@dataclasses.dataclass(frozen=True)
class PrintRequest:
    """print_scan's print stage as picklable data (see DevelopRequest). `scale` is the flat file's
    recorded linear_scale to undo first, or None; `print_params` are the tone parameters actually
    printed with (a pinned exposure already dropped when it can't be reproduced)."""

    source_profile: LinearRGBProfile | None
    scale: float | None
    print_params: ToneCurveParams


@dataclasses.dataclass(frozen=True)
class ExportRequest:
    """export_delivery_image's sRGB conversion takes no settings today (quality only matters when
    the file is written, in the worker). A dataclass anyway, so every service request has the same
    shape and a future export option has somewhere to go."""


def develop_request(frame, request: DevelopRequest, band_bytes: int | None = None, *,
                    name: str | None = None, on_warning: Callable[[str], None] | None = None
                    ) -> tuple[ResolvedTone | None, DensityProfile]:
    """All of process_scan's arithmetic, in place on `frame` — the decoded, not yet converted scan,
    as a host (numpy) array or a device (CuPy) array. One implementation for the CPU, the
    in-process GPU path and the GPU service: core/ picks the array library from the array
    (core/_xp.py), so the GPU runs exactly the CPU's steps in the CPU's order, and the CPU path is
    the code it always was. Returns (tone used, profile used).

    `name`/`on_warning`: see _to_working_space (F05's non-finite handling). The GPU service calls
    this with neither (it runs in its own process, with no live callback to report through), so a
    frame developed that way cleans non-finite pixels silently rather than warning about them — a
    known, narrow gap, no worse than the pre-existing one for auto-density's own warnings.warn."""
    _to_working_space(frame, request.source_profile, band_bytes, name=name, on_warning=on_warning)
    if request.scan_gain != 1.0:
        # A scalar of the frame's own dtype: the multiply stays in float32, and a scalar (unlike a
        # 0-d numpy array) is accepted as an operand by a device array too.
        frame *= frame.dtype.type(request.scan_gain)

    stage = request.stage
    if stage is Stage.INVERT_ONLY:
        profile = IDENTITY_PROFILE
    elif request.density_profile is not None:
        profile = request.density_profile
    else:
        # On a GPU this runs on the device frame itself; only the two 3-vector percentiles it
        # solves from come back to the host (calibration/auto.py).
        profile = auto_density_balance(frame)

    if stage is Stage.DENSITY_ONLY:
        map_in_bands(frame, lambda band: apply_density_balance(apply_white_balance(band, profile), profile),
                     band_bytes=band_bytes)
        return None, profile
    return _develop_in_place(frame, profile, request.tone_params, band_bytes), profile


def print_request(frame, request: PrintRequest, band_bytes: int | None = None, *,
                  name: str | None = None, on_warning: Callable[[str], None] | None = None) -> ResolvedTone:
    """print_scan's whole print stage in place on `frame` (host or device array) — see
    develop_request."""
    _to_working_space(frame, request.source_profile, band_bytes, name=name, on_warning=on_warning)
    if request.scale is not None:
        frame /= frame.dtype.type(request.scale)  # a scalar of the frame's dtype: see develop_request
    print_params = request.print_params
    curve = load_paper_curve(print_params.curve_path)
    resolved = _tone_on_host(resolve_tone(frame, print_params, curve))
    map_in_bands(frame, lambda band: apply_tone(band, resolved, curve), band_bytes=band_bytes)
    return resolved


def export_request(frame, out: np.ndarray, request: ExportRequest, band_bytes: int | None = None) -> None:
    """export_delivery_image's sRGB conversion of `frame` (host or device array, only ever read)
    into `out`, a host uint8 buffer of the same shape, band by band. On the host this is
    io/raster.py's srgb_8bit_from_acescg exactly; on a device each band comes straight back into
    its rows of `out`, so the device never holds a second full-size frame."""
    row_bytes = int(np.prod(frame.shape[1:], dtype=np.int64)) * frame.dtype.itemsize
    for band in banding.band_slices(frame.shape[0], row_bytes, band_bytes):
        srgb = to_srgb_8bit(frame[band])
        if isinstance(srgb, np.ndarray):
            out[band] = srgb
        else:
            _device.to_host(srgb, out=out[band])


def _develop_in_place(
    image, profile: DensityProfile, tone_params: ToneCurveParams, band_bytes: int | None = None
) -> ResolvedTone:
    """core.pipeline.develop, applied band by band into `image` (which the caller owns): the same
    per-pixel stages in the same order, with the one whole-frame step — the print fit — run on the
    full buffer between the two banded passes, exactly where develop() runs it. Bit-identical to
    develop() (pinned by tests/unit/test_banding.py), at ~1 frame of memory instead of ~9."""
    map_in_bands(image, lambda band: negative_to_positive(band, profile), band_bytes=band_bytes)
    curve = load_paper_curve(tone_params.curve_path)
    resolved = _tone_on_host(resolve_tone(image, tone_params, curve))
    map_in_bands(image, lambda band: apply_tone(band, resolved, curve), band_bytes=band_bytes)
    return resolved


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


@dataclasses.dataclass(frozen=True)
class DeviceFailure:
    """Why a device job failed, as plain data — what the in-process device path and the GPU
    service's reply both carry, so a worker applies the same fallback rules either way (see
    fall_back_to_cpu). Not the exception itself: a CuPy exception can't be unpickled in a worker
    that must never import CuPy.

    `host_touched`: the failure happened after the device began writing back into the host buffer
    (the download — a GPU runs asynchronously, so an earlier kernel's error can surface there), or
    it can't be ruled out (a service that died or stopped answering mid-request — and one that is
    merely stuck may still write into it later). The buffer may then be half-written, and the scan
    must be read again from disk, into a fresh buffer, before the CPU starts over
    (fall_back_to_cpu). An export never writes its input; its output buffer is never trusted after
    any failure, whatever this says (export_fallback)."""

    type_name: str
    message: str
    out_of_memory: bool
    host_touched: bool = False

    @classmethod
    def from_exception(cls, exc: BaseException, host_touched: bool = False) -> DeviceFailure:
        text = str(exc)
        # cupy.cuda.memory.OutOfMemoryError is a MemoryError; cuBLAS and the runtime report their
        # own allocation failures as status codes instead.
        out_of_memory = isinstance(exc, MemoryError) or "cudaErrorMemoryAllocation" in text or "ALLOC_FAILED" in text
        return cls(type_name=type(exc).__name__, message=text, out_of_memory=out_of_memory,
                   host_touched=host_touched)

    def warning(self, input_path, action: str = "developed this frame") -> str:
        """The fallback warning, worded as it always has been. `action` names what actually fell
        back to the CPU — "developed this frame" for the develop path (the default), "exported this
        file" for export: export doesn't develop anything, so the develop wording would misdescribe
        what happened."""
        if self.out_of_memory:
            return f"{input_path}: out of GPU memory — {action} on the CPU instead"
        if self.type_name == "ServiceUnavailable":
            # A batch's shared GPU service is gone (gpu_service.ServiceUnavailable). Worded so the
            # CLI can say it once for the whole run instead of once per frame (service_stopped).
            return f"{input_path}: the GPU service stopped ({_service_reason(self.message)}) — {action} on the CPU instead"
        detail = f"{self.type_name}: {self.message}" if self.message else self.type_name
        return f"{input_path}: the GPU failed ({detail}) — {action} on the CPU instead"


def _service_reason(message: str) -> str:
    """The reason a GPU service stopped, in words: gpu_service's own message minus its "the GPU
    service" lead-in ("stopped responding (EOFError)" -> "no reply: EOFError")."""
    text = message.removeprefix("the GPU service ")
    for lead, replacement in (("stopped responding (", "no reply: "), ("can't be reached (", "couldn't connect: ")):
        if text.startswith(lead) and text.endswith(")"):
            return replacement + text[len(lead):-1]
    return text


_SERVICE_STOPPED = re.compile(r"the GPU service stopped \((.*?)\) — ")


def service_stopped(warning: str) -> str | None:
    """The reason, if `warning` (a frame's fallback warning) says the GPU service stopped — so a
    batch reports that once ("The GPU service stopped (...); developing the remaining frames on
    the CPU.") instead of on every frame after it."""
    found = _SERVICE_STOPPED.search(warning)
    return found.group(1) if found else None


class DeviceJobFailed(Exception):
    """A device job failed; `failure` says how (see DeviceFailure). Raised by run_device_job /
    run_device_export and, from a reply, by the GPU service's client."""

    def __init__(self, failure: DeviceFailure):
        super().__init__(f"{failure.type_name}: {failure.message}" if failure.message else failure.type_name)
        self.failure = failure


def run_device_job(host: np.ndarray, job, request):
    """Upload `host`, run `job(frame, request, DEVICE_BAND_BYTES)` on the device copy (job is
    develop_request or print_request), and download the result back into `host`. Returns job's
    result, or raises DeviceJobFailed after handing CuPy's cached memory back.

    `host` is written only once, at the very end (the plan's "out-of-memory never fails a frame",
    §3.3): up to the download a failure leaves it exactly as decoded, ready for the CPU to start
    over on. Host memory stays ~1 frame: `host` is both the upload source and the download target.
    The one place this runs: the in-process device path (_run_on_device) and the GPU service.

    A ScanColorError (F05's "too many non-finite pixels" — or any future input-data check) is *not*
    a device problem (R8): it propagates unwrapped, not as DeviceJobFailed, so the caller neither
    prints "the GPU failed" nor retries on the CPU (which would just raise the identical error a
    second time) — the frame fails once, with the same plain message the CPU path itself gives."""
    frame = None
    downloading = False
    try:
        frame = _device.to_device(host)
        result = job(frame, request, banding.DEVICE_BAND_BYTES)
        downloading = True
        _device.to_host(frame, out=host)
        return result
    except ScanColorError:
        del frame
        _device.release_memory()
        raise
    except Exception as exc:  # noqa: BLE001 — any GPU problem: reported, then redone on the CPU
        del frame
        _device.release_memory()
        raise DeviceJobFailed(DeviceFailure.from_exception(exc, host_touched=downloading)) from exc


def run_device_export(host: np.ndarray, out: np.ndarray, request: ExportRequest) -> None:
    """export_request on a device copy of `host`, downloaded band by band into `out`. Raises
    DeviceJobFailed on any device problem; `host` is only ever read, so a failure never needs a
    re-read (host_touched stays False) — a plain CPU conversion of it afterwards is exactly as if
    the GPU had never been tried."""
    frame = None
    try:
        frame = _device.to_device(host)
        export_request(frame, out, request, banding.DEVICE_BAND_BYTES)
    except Exception as exc:  # noqa: BLE001 — any GPU problem: reported, then redone on the CPU
        del frame
        _device.release_memory()
        raise DeviceJobFailed(DeviceFailure.from_exception(exc)) from exc


def fall_back_to_cpu(input_path, host: np.ndarray, failure: DeviceFailure,
                     on_warning: Callable[[str], None] | None, action: str = "developed this frame") -> np.ndarray:
    """After a device job failed (in this process or in the GPU service): warn, and return the
    buffer the CPU should start over on — `host` itself when the device never wrote into it, else
    the scan decoded again from disk into a fresh buffer. Fresh rather than back into `host`: a
    GPU service that stopped answering may still be writing into a shared frame it was given."""
    _warn(on_warning, failure.warning(input_path, action))
    if failure.host_touched:
        host, _ = _read_scan(input_path)
    return host


def export_fallback(input_path, acescg_image: np.ndarray, failure: DeviceFailure,
                    on_warning: Callable[[str], None] | None) -> np.ndarray:
    """After a device export failed (in this process or in the GPU service): warn, and convert
    `acescg_image` on the CPU into a fresh uint8 buffer the caller owns. Never into the buffer the
    device was writing: it is partly written, and a GPU service that stopped answering may still be
    writing bands into it. `acescg_image` itself is only ever read by the device, so it is always
    safe to convert from, and never needs a re-read."""
    _warn(on_warning, failure.warning(input_path, "exported this file"))
    return srgb_8bit_from_acescg(acescg_image)


def _run_on_device(input_path, host: np.ndarray, device: ComputeDevice | None, job, request, on_warning):
    """Run `job(frame, request, band_bytes)` on a GPU copy of `host` and download the result into
    `host` (see run_device_job). `job` is bound with `name`/`on_warning` first (F05's non-finite
    handling) — this runs in-process (unlike the GPU service), so the caller's own warning channel
    is still reachable here.

    Returns (job's result, host), or (None, host) when there's no GPU to use or it failed — then
    `host` is the decoded, unconverted scan, ready for the CPU to start over on (read again from
    disk if the failure came during the download; see fall_back_to_cpu)."""
    if device is None or device.kind != "gpu":
        return None, host
    bound_job = functools.partial(job, name=str(input_path), on_warning=on_warning)
    try:
        return run_device_job(host, bound_job, request), host
    except DeviceJobFailed as failed:
        return None, fall_back_to_cpu(input_path, host, failed.failure, on_warning)


def _gpu_fallback_message(input_path, exc: Exception, action: str = "developed this frame") -> str:
    """The fallback warning for `exc` — see DeviceFailure.warning."""
    return DeviceFailure.from_exception(exc).warning(input_path, action)


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
        try:
            with Image.open(path) as image:
                save_thumbnail(thumbnail_path, thumbnail_from_display(image, thumbnail_long_edge), None)
        except OSError:
            raise ScanInputError(f"{path.name} isn't an image halide can read (it may be damaged or cut short). "
                                 "Export it again") from None
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
    service=None,
    shm_prefix: str | None = None,
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

    `device` / `on_warning` / `service` / `shm_prefix`: as process_scan — a GPU failure redoes the
    print on the CPU and is reported through `on_warning`, separately from the returned (metadata)
    warning.
    """
    # Header only: decoding the pixels here just for the description held a second full frame
    # alongside the decoded image.
    provenance = read_provenance(read_tiff_description(input_path))
    if provenance is not None and provenance.get("output") == "print":
        raise PrintInputError(
            f"{input_path} is already a halide print (it has the tone curve applied) — `halide print` "
            f"expects a flat positive from `halide invert --output flat`"
        )
    output_path = Path(output_path)
    exif_warning = None
    # Same atomic-write contract as process_scan (F04): everything below runs on a hidden temp
    # path, moved over the real name only once it's all done.
    with atomic_output(output_path) as tmp_path:
        with contextlib.ExitStack() as shared:
            frame = _shared_frame(input_path, service, shm_prefix, shared, on_warning)
            image, source_profile = _read_scan(input_path, out=None if frame is None else frame.array)

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

            request = PrintRequest(source_profile=source_profile, scale=scale, print_params=print_params)
            if frame is not None:
                reply, image, used_device = _run_on_service(input_path, frame, service, "print_", request, on_warning)
                resolved = None if reply is None else reply.resolved
            else:
                resolved, image = _run_on_device(input_path, image, device, print_request, request, on_warning)
                used_device = "gpu" if resolved is not None else "cpu"
            if resolved is None:
                resolved = print_request(image, request, name=str(input_path), on_warning=on_warning)
            write_tiff(tmp_path, image, icc_profile=output_profile_bytes())
            del image, frame  # before exiftool runs — see process_scan
        try:
            copy_exif_metadata(str(input_path), str(tmp_path), drop_icc=True)
        except _EXIF_FAILURE as exc:
            # The print itself is already fully written — see process_scan's own exif handling.
            exif_warning = f"{input_path}: printed, but its camera metadata couldn't be copied ({exc})"
        set_description(tmp_path, provenance_json(resolved, None, device=used_device))
    if exif_warning is not None:
        _warn(on_warning, exif_warning)
    return resolved, warning


def export_delivery_image(
    input_path: str | Path,
    output_path: str | Path,
    quality: int = 95,
    device: ComputeDevice | None = None,
    on_warning: Callable[[str], None] | None = None,
    service=None,
    shm_prefix: str | None = None,
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

    `service` / `shm_prefix`: as process_scan — the batch's GPU service converts the frame from one
    shared-memory buffer into another. After any failure its output buffer is never used (see
    export_fallback): the CPU converts into a buffer of this process's own.
    """
    output_path = Path(output_path)
    with contextlib.ExitStack() as shared:
        frames = _shared_frame(input_path, service, shm_prefix, shared, on_warning, extra_uint8=True)
        scan = _decode_tiff(input_path, None if frames is None else frames[0].array)
        warning = None
        if scan.icc_profile is None:
            warning = f"{input_path} has no embedded ICC profile; assuming it is ACEScg"
        else:
            try:
                profile = parse_linear_rgb_profile(scan.icc_profile)
                if not np.allclose(profile.rgb_to_pcs_xyz, _acescg_matrix(), atol=1e-3):
                    warning = (
                        f"{input_path}'s embedded profile does not look like ACEScg — `halide export` "
                        f"expects the output of `halide invert`/`halide batch`. Proceeding anyway, but "
                        f"colors may be wrong"
                    )
            except UnsupportedICCProfileError as exc:
                warning = f"{input_path}'s embedded profile is unusable ({exc}); assuming ACEScg anyway"

        if frames is not None:
            srgb_8bit = _export_on_service(input_path, frames, service, on_warning)
        else:
            srgb_8bit = _export_srgb_on_device(input_path, scan.image, device, on_warning)
        # Written under a hidden temp name beside the real one, moved into place only once the
        # PNG/JPEG is fully encoded (F04) — see process_scan's own atomic_output for why.
        with atomic_output(output_path) as tmp_path:
            if srgb_8bit is not None:
                write_srgb_8bit_image(tmp_path, srgb_8bit, quality=quality)
            else:
                write_delivery_image(tmp_path, scan.image, quality=quality)
        del scan, frames, srgb_8bit
    return warning


def _export_on_service(input_path, frames, service, on_warning) -> np.ndarray:
    """export_delivery_image through the GPU service: `frames` is (the decoded ACEScg frame, its
    uint8 output), both shared. Returns the output frame's array, or after any failure the CPU's
    conversion of the (only ever read) input into a fresh buffer — never the output frame, which a
    service that stopped answering may still be writing into (export_fallback)."""
    from halide.gpu_service import ServiceUnavailable

    frame, out = frames
    try:
        service.export(frame, out, ExportRequest())
    except (DeviceJobFailed, ServiceUnavailable) as failed:
        return export_fallback(input_path, frame.array, failed.failure, on_warning)
    return out.array


def _export_srgb_on_device(input_path, acescg_image: np.ndarray, device: ComputeDevice | None, on_warning):
    """`export_delivery_image`'s device path: `acescg_image` converted on the device into a host
    uint8 buffer this function owns alone — never into `acescg_image` itself (see
    run_device_export), so a failure never needs a re-read.

    Returns the finished (H, W, 3) uint8 array — converted on the CPU after all if the GPU failed
    (see export_fallback; `on_warning` is told why) — or None when there's no GPU to use."""
    if device is None or device.kind != "gpu":
        return None
    srgb_8bit = np.empty(acescg_image.shape, dtype=np.uint8)
    try:
        run_device_export(acescg_image, srgb_8bit, ExportRequest())
    except DeviceJobFailed as failed:
        return export_fallback(input_path, acescg_image, failed.failure, on_warning)
    return srgb_8bit


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
