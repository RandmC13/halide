"""Read how a scan was *made* — the digitizing camera's exposure settings (EXIF) and, for darktable
exports, the raw white balance and which processing modules were active (the XMP history darktable
embeds) — from TIFF headers only, never pixel data.

This exists because a density-balance profile is only exactly valid at the scan conditions it was
solved at. Density balance is a per-channel power function, so a frame digitized brighter by a
factor k comes out of it scaled by k**density_scale — a different factor per channel, i.e. a colour
shift, not just a brightness change (the reference blog: "the white balance value will vary
depending on exposure ... consistency is important"). A per-frame raw white-balance difference
shifts colour the same way. Found on a real roll: 37 frames digitized with the camera metering each
frame (1/25-1/60s) and "as shot" (camera auto) white balance varying ~5-7% per channel.

Everything here is best-effort: a file without EXIF or without darktable history just yields None.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import tifffile

_XMP_TAG = 700

# darktable modules that don't reshape tone or colour: raw decoding, the colour-profile matrices,
# geometry, and cleanup. `denoiseprofile` and `highlights` (reconstruction of *clipped* raw values
# only) aren't strictly linear, but only act on noise/clipped pixels — not on the tonal
# relationships the inversion depends on. Anything else enabled (tone curves, levels, filmic/
# sigmoid, shadows & highlights, color balance, negadoctor, exposure...) is reported.
_NON_TONAL_DARKTABLE_MODULES = frozenset({
    "rawprepare", "demosaic", "colorin", "colorout", "gamma", "flip", "crop", "clipping", "ashift",
    "lens", "denoiseprofile", "rawdenoise", "hotpixels", "cacorrect", "cacorrectrgb", "temperature",
    "highlights", "retouch", "spots", "finalscale", "dither", "overexposed", "rawoverexposed",
})


@dataclass(frozen=True)
class ScanSettings:
    """The digitizing camera's exposure for one scan."""

    exposure_time: float  # seconds
    f_number: float
    iso: float

    @property
    def relative_exposure(self) -> float:
        """Proportional to how much light reached the sensor per unit film transmittance."""
        return self.exposure_time * self.iso / self.f_number**2

    def describe(self) -> str:
        t = self.exposure_time
        shutter = f"1/{round(1 / t)}" if 0 < t < 1 else f"{t:g}s"
        return f"{shutter} f/{self.f_number:g} ISO{self.iso:g}"


@dataclass(frozen=True)
class DarktableState:
    """The final (history_end) state of a darktable export's processing history."""

    white_balance: tuple[float, float, float] | None  # raw R/G/B multipliers of the temperature module
    tonal_modules: tuple[str, ...]  # enabled modules that reshape tone/colour (should be empty)


def _rational(value) -> float | None:
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return value[0] / value[1]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def settings_from_exif(exif: dict) -> ScanSettings | None:
    exposure_time = _rational(exif.get("ExposureTime"))
    f_number = _rational(exif.get("FNumber"))
    iso = exif.get("ISOSpeedRatings", exif.get("PhotographicSensitivity"))
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    if not exposure_time or not f_number or not iso:
        return None
    return ScanSettings(exposure_time=exposure_time, f_number=f_number, iso=float(iso))


def darktable_state_from_xmp(xmp: str) -> DarktableState | None:
    end_match = re.search(r'darktable:history_end="(\d+)"', xmp)
    history = re.search(r"<darktable:history>(.*?)</darktable:history>", xmp, re.S)
    if end_match is None or history is None:
        return None
    items = [dict(re.findall(r'darktable:(\w+)="([^"]*)"', item))
             for item in re.findall(r"<rdf:li(.*?)/>", history.group(1), re.S)]
    # Later entries override earlier ones for the same module instance, up to history_end.
    final: dict[tuple[str, str], dict] = {}
    for item in items[: int(end_match.group(1))]:
        if "operation" in item:
            final[(item["operation"], item.get("multi_priority", "0"))] = item

    white_balance = None
    temperature = final.get(("temperature", "0"))
    if temperature and temperature.get("enabled") == "1":
        params = temperature.get("params", "")
        if re.fullmatch(r"[0-9a-f]{24,}", params):
            white_balance = struct.unpack("<3f", bytes.fromhex(params)[:12])
    tonal = sorted({op for (op, _), item in final.items()
                    if item.get("enabled") == "1" and op not in _NON_TONAL_DARKTABLE_MODULES})
    return DarktableState(white_balance=white_balance, tonal_modules=tuple(tonal))


def read_scan_metadata(path: str | Path) -> tuple[ScanSettings | None, DarktableState | None]:
    """(EXIF scan settings, darktable export state) — either may be None if absent/unreadable."""
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            exif = tags["ExifTag"].value if "ExifTag" in tags else {}
            xmp = tags[_XMP_TAG].value if _XMP_TAG in tags else b""
    except Exception:  # noqa: BLE001 — metadata is advisory; an unreadable header must not abort anything
        return None, None
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")
    settings = settings_from_exif(exif) if isinstance(exif, dict) else None
    return settings, darktable_state_from_xmp(xmp) if xmp else None
