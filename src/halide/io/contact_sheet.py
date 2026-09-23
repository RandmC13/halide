"""High-resolution contact sheets: a roll's frames laid out as strips of film on a sheet of paper,
for comparing the results of different settings at a glance — `halide contact` (from an existing
folder of processed output) and `halide batch --contact-sheet` (a preview that never writes the
full-size TIFFs at all).

Thumbnails are reduced in *linear* light (block averaging) before being encoded to sRGB, so fine
detail averages the way light does rather than darkening, as averaging gamma-encoded values would.
The sheet is sRGB with the profile embedded, the same delivery encoding as `halide export`.

Layout mirrors the terminal progress display (batch/progress.py): strips of up to six frames in job
order, each edged with sprocket holes, the last strip only as long as the frames left over. Each
frame's name — and, for halide's own output, the printing decision it records (grade, exposure,
scan gain) — is printed in the rebate below it, like a film's edge printing, so two sheets made with
different settings can be compared frame by frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from halide.io.raster import _SRGB_ICC_BYTES, to_srgb_8bit

FRAME_ASPECT = 3 / 2  # a 35mm frame's cell; other shapes are fitted inside it
DEFAULT_FRAME_WIDTH = 900  # px per frame cell — a 6-across sheet is ~6000 px wide
DEFAULT_COLUMNS = 6

_PAPER = (239, 236, 230)
_INK = (42, 42, 42)
_REBATE = (12, 12, 12)
_EDGE_PRINT = (224, 160, 96)  # the warm orange of a film's edge printing
_CAPTION = (150, 150, 150)
_FAILED = (70, 24, 24)
_THUMBNAIL_KEY = "halide"
_SHEET_MARKER = "halide contact sheet"


@dataclass(frozen=True)
class Tile:
    name: str
    image: np.ndarray | None  # uint8 sRGB, HxWx3; None for a frame that failed
    caption: str = ""


def downsample_linear(image: np.ndarray, target_long_edge: int) -> np.ndarray:
    """Integer block-mean reduction of a linear image to roughly (not below) `target_long_edge`."""
    factor = max(1, max(image.shape[:2]) // max(1, target_long_edge))
    if factor == 1:
        return image
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    blocks = image[:h, :w].reshape(h // factor, factor, w // factor, factor, image.shape[2])
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _fit_long_edge(image: Image.Image, target_long_edge: int) -> Image.Image:
    scale = target_long_edge / max(image.size)
    if scale >= 1:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def thumbnail_from_linear(acescg_image: np.ndarray, target_long_edge: int) -> np.ndarray:
    """A linear ACEScg image (halide's working space / TIFF output) -> uint8 sRGB thumbnail."""
    reduced = downsample_linear(acescg_image, target_long_edge)
    return np.asarray(_fit_long_edge(Image.fromarray(to_srgb_8bit(reduced)), target_long_edge))


def thumbnail_from_display(image: Image.Image, target_long_edge: int) -> np.ndarray:
    """An already display-encoded (sRGB) image, e.g. `halide export` output -> uint8 thumbnail."""
    return np.asarray(_fit_long_edge(image.convert("RGB"), target_long_edge))


def caption_from_provenance(record: dict | None) -> str:
    """The printing decision halide recorded in an output file (see processing.provenance_json).
    Plain "x", not "×": Pillow's built-in font has no multiplication sign (it drew an empty box)."""
    if not record:
        return ""
    bits = []
    if record.get("output") == "flat":
        bits.append(f"flat x{record['linear_scale']:.3g}" if "linear_scale" in record else "flat")
    elif "contrast" in record:
        bits.append(f"grade {record['contrast']:.2f} · exp {record['exposure']:+.2f}")
    if record.get("scan_gain", 1.0) != 1.0:
        bits.append(f"scan x{record['scan_gain']:.2f}")
    return " · ".join(bits)


def save_thumbnail(path: str | Path, thumbnail: np.ndarray, record: dict | None) -> None:
    """A small intermediate (a worker's output), carrying the frame's provenance for its caption."""
    info = PngInfo()
    if record:
        info.add_text(_THUMBNAIL_KEY, json.dumps(record))
    Image.fromarray(thumbnail).save(path, format="PNG", pnginfo=info)


def load_thumbnail(path: str | Path) -> tuple[np.ndarray, dict | None]:
    with Image.open(path) as image:
        record = image.info.get(_THUMBNAIL_KEY)
        return np.asarray(image.convert("RGB")), json.loads(record) if record else None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(8, size))


def render_sheet(
    tiles: list[Tile],
    title: str,
    subtitle: str = "",
    frame_width: int = DEFAULT_FRAME_WIDTH,
    columns: int = DEFAULT_COLUMNS,
) -> Image.Image:
    W = frame_width
    H = round(W / FRAME_ASPECT)
    gap = round(W * 0.05)  # rebate between frames in a strip
    rebate = round(H * 0.2)  # above and below the frames, holding the sprocket holes (and edge print below)
    strip_h = H + 2 * rebate
    margin = round(W * 0.2)
    strip_gap = round(H * 0.14)
    columns = max(1, columns)
    header_h = round(H * 0.34)

    widest = max(1, min(columns, len(tiles)))  # a sheet of 2 frames shouldn't be 6 frames wide
    strip_w_full = widest * W + (widest + 1) * gap
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)] or [[]]
    sheet_w = strip_w_full + 2 * margin
    sheet_h = margin + header_h + len(rows) * strip_h + (len(rows) - 1) * strip_gap + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), _PAPER)
    draw = ImageDraw.Draw(sheet)

    title_font, sub_font = _font(round(H * 0.11)), _font(round(H * 0.055))
    draw.text((margin, margin), title, fill=_INK, font=title_font)
    if subtitle:
        draw.text((margin, margin + round(H * 0.15)), subtitle, fill=_INK, font=sub_font)

    edge_font = _font(round(H * 0.055))
    hole_w, hole_h = round(W * 0.045), round(rebate * 0.34)
    holes_per_frame = 8  # a 35mm frame has eight perforations each side
    for r, row in enumerate(rows):
        x0 = margin
        y0 = margin + header_h + r * (strip_h + strip_gap)
        strip_w = len(row) * W + (len(row) + 1) * gap
        draw.rectangle([x0, y0, x0 + strip_w - 1, y0 + strip_h - 1], fill=_REBATE)

        pitch = (W + gap) / holes_per_frame
        n_holes = int((strip_w - gap) // pitch)
        for k in range(n_holes):
            hx = x0 + gap / 2 + k * pitch + (pitch - hole_w) / 2
            for hy in (y0 + round(rebate * 0.18), y0 + strip_h - round(rebate * 0.18) - hole_h):
                draw.rounded_rectangle([hx, hy, hx + hole_w, hy + hole_h], radius=max(1, hole_h // 4), fill=_PAPER)

        for c, tile in enumerate(row):
            cx = x0 + gap + c * (W + gap)
            cy = y0 + rebate
            if tile.image is None:
                draw.rectangle([cx, cy, cx + W - 1, cy + H - 1], fill=_FAILED)
                draw.text((cx + W // 2, cy + H // 2), "failed", fill=_EDGE_PRINT, font=title_font, anchor="mm")
            else:
                frame = Image.fromarray(tile.image)
                scale = min(W / frame.width, H / frame.height)
                size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
                if size != frame.size:
                    frame = frame.resize(size, Image.LANCZOS)
                sheet.paste(frame, (cx + (W - size[0]) // 2, cy + (H - size[1]) // 2))
            text_y = cy + H + round(rebate * 0.12)
            draw.text((cx, text_y), tile.name, fill=_EDGE_PRINT, font=edge_font)
            if tile.caption:
                name_w = draw.textlength(tile.name + "   ", font=edge_font)
                draw.text((cx + name_w, text_y), tile.caption, fill=_CAPTION, font=edge_font)
    return sheet


def write_sheet(path: str | Path, sheet: Image.Image, quality: int = 92) -> None:
    """JPEG or PNG by extension, with an sRGB profile embedded (as `halide export` does), and marked
    as a halide contact sheet (see is_contact_sheet)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        sheet.save(path, format="JPEG", quality=quality, icc_profile=_SRGB_ICC_BYTES, subsampling=0,
                   comment=_SHEET_MARKER.encode())
    elif suffix == ".png":
        info = PngInfo()
        info.add_text(_SHEET_MARKER, "1")
        sheet.save(path, format="PNG", icc_profile=_SRGB_ICC_BYTES, pnginfo=info)
    else:
        raise ValueError(f"{path}: unsupported contact sheet format {suffix!r} (use .jpg, .jpeg or .png)")


def is_contact_sheet(path: str | Path) -> bool:
    """Whether a PNG/JPEG is one of halide's own contact sheets (reads the header only). Found via
    testing: a sheet written into the folder it proofs was picked up as an extra "frame" the next
    time that folder was proofed."""
    try:
        with Image.open(path) as image:
            return image.info.get("comment") == _SHEET_MARKER.encode() or _SHEET_MARKER in image.info
    except Exception:  # noqa: BLE001 — unreadable here just means "not a sheet"; the real read reports it
        return False


def check_sheet_path(path: str | Path) -> None:
    """Fail fast on a bad extension, before any (possibly long) processing starts."""
    if Path(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"{path}: contact sheet must be .jpg, .jpeg or .png")

