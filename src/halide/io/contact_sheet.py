"""High-resolution contact sheets: a roll's frames laid out as a real contact print of the roll
looks, for comparing the results of different settings at a glance — `halide contact` (from an
existing folder of processed output), `halide batch --contact-sheet` (a preview that never writes
the full-size TIFFs at all), and the calibration picker's proof window (gui/proof_window.py).

Thumbnails are reduced in *linear* light (block averaging) before being encoded to sRGB, so fine
detail averages the way light does rather than darkening, as averaging gamma-encoded values would.
The sheet is sRGB with the profile embedded, the same delivery encoding as `halide export`.

The look follows a real printed contact sheet (the user's own reference, a Portra 400 roll): black
wherever the film is, frames butted in strips of up to six in job order (the last strip only as
long as the frames left over), and the film's orange edge printing - frame numbers and the film
stock above each frame, numbers and edge-code bars below. Under each frame a small dim line gives
its file name and, for halide's own output, the printing decision it records (grade, exposure, scan
gain), so two sheets made with different settings can be compared frame by frame. The film stock
comes from the profile (recorded in each output's provenance); unknown, the edge reads
"INVERTED BY HALIDE".
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from halide.io.raster import _SRGB_ICC_BYTES, to_srgb_8bit

FRAME_ASPECT = 3 / 2  # a 35mm frame's cell; other shapes are fitted inside it
DEFAULT_FRAME_WIDTH = 900  # px per frame cell — a 6-across sheet is ~6000 px wide
DEFAULT_COLUMNS = 6

_SHEET = (14, 13, 12)  # film prints black on a contact sheet: rebate, frame lines, the lot
_TITLE = (225, 215, 200)
_EDGE_PRINT = (224, 160, 96)  # the warm orange of a film's edge printing (theme.EDGE_PRINT in the GUI)
_CAPTION = (120, 110, 98)
_FAILED = (70, 24, 24)
_THUMBNAIL_KEY = "halide"
_SHEET_MARKER = "halide contact sheet"


@dataclass(frozen=True)
class Tile:
    name: str
    image: np.ndarray | None  # uint8 sRGB, HxWx3; None for a frame that failed
    caption: str = ""
    number: int | None = None  # frame number on the edge print; None = its position on the sheet


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


_EDGE_FONT_FILES = ("DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf", "LiberationSans-Bold.ttf")
_TEXT_FONT_FILES = ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A system font if there is one (bold for the edge print, like film's own lettering), else
    Pillow's built-in font, which has no bold but draws everything this sheet needs."""
    for name in _EDGE_FONT_FILES if bold else _TEXT_FONT_FILES:
        try:
            return ImageFont.truetype(name, max(8, size))
        except OSError:
            continue
    return ImageFont.load_default(size=max(8, size))


def _barcode(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, height: float, seed: int, fill) -> None:
    """Edge-code-style bars between x0 and x1 - decoration in the manner of film's DX edge
    barcode, fixed per frame number (the same frame always draws the same bars)."""
    rng = random.Random(seed)
    unit = max(1.0, (x1 - x0) / 110)
    x = x0
    while x < x1:
        width = unit * rng.choice((1, 1, 2, 3))
        tall = rng.random() < 0.6
        top = y if tall else y + height * 0.45
        right = min(x + width, x1) - 1
        if right < x:  # no room left for even one pixel (tiny frames)
            break
        draw.rectangle([x, top, right, y + height], fill=fill)
        x += width + unit * rng.choice((1, 1, 2))


@dataclass(frozen=True)
class SheetLayout:
    """Where everything sits on a sheet of `count` frames - the one place the geometry lives, so
    the renderer and anything that needs to know where a frame landed (the proof window's hover
    labels) can't disagree."""

    count: int
    frame_width: int = DEFAULT_FRAME_WIDTH
    columns: int = DEFAULT_COLUMNS

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", max(1, self.columns))

    @property
    def frame_w(self) -> int:
        return self.frame_width

    @property
    def frame_h(self) -> int:
        return round(self.frame_width / FRAME_ASPECT)

    @property
    def gap(self) -> int:  # frame line between frames on the strip
        return round(self.frame_width * 0.03)

    @property
    def top_edge(self) -> int:  # edge print above the frames
        return round(self.frame_h * 0.11)

    @property
    def bottom_edge(self) -> int:  # edge print and edge code below
        return round(self.frame_h * 0.12)

    @property
    def caption_h(self) -> int:
        return round(self.frame_h * 0.08)

    @property
    def strip_h(self) -> int:
        return self.top_edge + self.frame_h + self.bottom_edge + self.caption_h

    @property
    def strip_gap(self) -> int:
        return round(self.frame_h * 0.09)

    @property
    def margin(self) -> int:
        return round(self.frame_width * 0.12)

    @property
    def header_h(self) -> int:
        return round(self.frame_h * 0.32)

    @property
    def size(self) -> tuple[int, int]:
        widest = max(1, min(self.columns, self.count))
        rows = max(1, -(-self.count // self.columns))
        width = 2 * self.margin + widest * self.frame_w + (widest - 1) * self.gap
        height = 2 * self.margin + self.header_h + rows * self.strip_h + (rows - 1) * self.strip_gap
        return width, height

    def frame_box(self, index: int) -> tuple[int, int, int, int]:
        """(x, y, width, height) of frame `index`'s cell on the sheet."""
        row, column = divmod(index, self.columns)
        x = self.margin + column * (self.frame_w + self.gap)
        y = self.margin + self.header_h + row * (self.strip_h + self.strip_gap) + self.top_edge
        return x, y, self.frame_w, self.frame_h

    def frame_at(self, x: float, y: float) -> int | None:
        """The frame whose cell (with its edge print and caption) contains (x, y), if any."""
        for index in range(self.count):
            fx, fy, fw, fh = self.frame_box(index)
            if fx <= x < fx + fw and fy - self.top_edge <= y < fy + fh + self.bottom_edge + self.caption_h:
                return index
        return None


def render_sheet(
    tiles: list[Tile],
    title: str,
    subtitle: str = "",
    frame_width: int = DEFAULT_FRAME_WIDTH,
    columns: int = DEFAULT_COLUMNS,
    film_stock: str | None = None,
) -> Image.Image:
    """The sheet as a real contact print looks: black wherever the film is (its rebate prints black
    on paper), frames butted in strips, the film's orange edge printing above and below each frame -
    frame number and film stock on top ("INVERTED BY HALIDE" when the stock isn't known), the number
    with its half-frame "A" number and edge-code bars below - and, under that, a small dim line with
    the frame's file name and the printing decision halide recorded, so sheets from different
    settings stay self-describing."""
    layout = SheetLayout(len(tiles), frame_width, columns)
    W, H, margin, header_h = layout.frame_w, layout.frame_h, layout.margin, layout.header_h
    top_edge, bottom_edge, caption_h = layout.top_edge, layout.bottom_edge, layout.caption_h
    columns = layout.columns
    rows = [tiles[i:i + columns] for i in range(0, len(tiles), columns)] or [[]]
    sheet = Image.new("RGB", layout.size, _SHEET)
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, margin), title, fill=_TITLE, font=_font(round(H * 0.1)))
    if subtitle:
        draw.text((margin, margin + round(H * 0.14)), subtitle, fill=_CAPTION, font=_font(round(H * 0.05)))

    edge_font = _font(round(H * 0.046), bold=True)
    caption_font = _font(round(H * 0.045))
    stock = (film_stock or "Inverted by halide").upper()
    for r, row in enumerate(rows):
        frames_y = layout.frame_box(r * columns)[1]
        y0 = frames_y - top_edge
        for c, tile in enumerate(row):
            number = tile.number if tile.number is not None else r * columns + c + 1
            x = layout.frame_box(r * columns + c)[0]
            if tile.image is None:
                draw.rectangle([x, frames_y, x + W - 1, frames_y + H - 1], fill=_FAILED)
                draw.text((x + W // 2, frames_y + H // 2), "failed", fill=_EDGE_PRINT, font=edge_font, anchor="mm")
            else:
                frame = Image.fromarray(tile.image)
                scale = min(W / frame.width, H / frame.height)
                size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
                if size != frame.size:
                    frame = frame.resize(size, Image.LANCZOS)
                sheet.paste(frame, (x + (W - size[0]) // 2, frames_y + (H - size[1]) // 2))

            # top edge: frame number, then the stock
            top_mid = y0 + top_edge // 2
            draw.text((x + round(W * 0.03), top_mid), str(number), fill=_EDGE_PRINT, font=edge_font, anchor="lm")
            draw.text((x + round(W * 0.30), top_mid), stock, fill=_EDGE_PRINT, font=edge_font, anchor="lm")

            # bottom edge: number + code, half-frame number + code
            code_y = frames_y + H + round(bottom_edge * 0.3)
            code_h = round(bottom_edge * 0.38)
            code_mid = code_y + code_h // 2
            for label, start, end, seed in (
                (str(number), 0.03, 0.46, number * 2),
                (f"{number}A", 0.52, 0.97, number * 2 + 1),
            ):
                lx = x + round(W * start)
                draw.text((lx, code_mid), label, fill=_EDGE_PRINT, font=edge_font, anchor="lm")
                text_end = lx + draw.textlength(label, font=edge_font) + W * 0.015
                _barcode(draw, text_end, x + W * end, code_y, code_h, seed, _EDGE_PRINT)

            # the frame's file name and recorded printing decision, dim, below the film
            caption = f"{tile.name}   {tile.caption}" if tile.caption else tile.name
            draw.text((x + round(W * 0.03), frames_y + H + bottom_edge + caption_h // 2), caption,
                      fill=_CAPTION, font=caption_font, anchor="lm")
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

