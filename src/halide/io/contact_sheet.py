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

from halide.io.atomic import atomic_output
from halide.io.contact_sheet_defaults import DEFAULT_COLUMNS, DEFAULT_FRAME_WIDTH  # noqa: F401 -- re-exported
from halide.io.raster import _SRGB_ICC_BYTES, to_srgb_8bit

FRAME_ASPECT = 3 / 2  # a 35mm frame's cell; other shapes are fitted inside it

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
_ARROW_SPACE = 0.2  # the thin space between the half-frame arrow and its number, in em


def _packaged_font_path(name: str) -> Path | None:
    from importlib.resources import files

    path = Path(str(files("halide.assets") / "fonts" / name))
    return path if path.is_file() else None


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The vendored DejaVu (bold for the edge print, like film's own lettering); a system font only
    if that file is missing, then Pillow's built-in font, which has no bold but draws everything
    this sheet needs."""
    for name in _EDGE_FONT_FILES if bold else _TEXT_FONT_FILES:
        packaged = _packaged_font_path(name)
        try:
            return ImageFont.truetype(str(packaged) if packaged else name, max(8, size))
        except OSError:
            continue
    return ImageFont.load_default(size=max(8, size))


def edge_code_unit(height: float) -> int:
    """The width u of one clock bar (and of the gap after it) for a code `height` px tall."""
    return max(1, round(height / 5))


def edge_code_cells(seed: int, count: int) -> list[bool]:
    """The data track's cells (one per clock bar and per gap, so twice the clock's resolution).
    A start mark spans the first 3 clock bars, an end mark joins the last 2; between them roughly
    40% of the cells are set, never more than 3 in a row, fixed per `seed`."""
    rng = random.Random(seed)
    cells = [False] * count
    cells[:5] = [True] * 5  # the wide start mark: bars 0-2 and the gaps between them
    cells[count - 3:] = [True] * 3  # the end mark: the last two bars and the gap between
    run = 0
    for i in range(6, count - 4):  # cell 5 and cell count-4 stay open, so the marks stand apart
        if run < 3 and rng.random() < 0.4:
            cells[i] = True
            run += 1
        else:
            run = 0
    return cells


def _edge_code(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, height: float, seed: int, fill) -> None:
    """A DX-style two-track edge code between x0 and x1, in the manner of film's own: a clock track
    (the lower ~55%) of narrow evenly spaced bars, and above it a data track of blocks that join
    the bars they touch into tall ones. Fixed per frame (the same seed draws the same code)."""
    u = edge_code_unit(height)
    count = int((x1 - x0) // u)
    count -= 1 - count % 2  # end on a clock bar
    if count < 9:  # no room for the marks (tiny frames)
        return
    x0, y = round(x0), round(y)
    bottom = y + round(height) - 1
    data_bottom = y + round(height * 0.45) - 1
    clock_top = data_bottom + 1
    for i in range(0, count, 2):  # the clock track
        draw.rectangle([x0 + i * u, clock_top, x0 + (i + 1) * u - 1, bottom], fill=fill)
    for i, is_set in enumerate(edge_code_cells(seed, count)):  # the data track
        if is_set:
            draw.rectangle([x0 + i * u, y, x0 + (i + 1) * u - 1, data_bottom], fill=fill)


def _draw_arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, mid: int, height: int, fill) -> None:
    """A right-pointing arrow (shaft and filled head) drawn as shapes, so it needs no font glyph."""
    half = max(1, height // 2)
    head = max(2, min(x1 - x0, round(height * 0.9)))
    shaft = max(1, round(height / 6))
    draw.rectangle([x0, mid - shaft // 2, x1 - head, mid - shaft // 2 + shaft - 1], fill=fill)
    draw.polygon([(x1 - head, mid - half), (x1 - 1, mid), (x1 - head, mid + half)], fill=fill)


@dataclass(frozen=True)
class LowerEdge:
    """Where one frame's lower edge print sits: `N  [code]  ->NA  [code]`, all pixel boxes
    (x0, x1) on the sheet, with the code strip's top `code_y` and height `code_h`."""

    number_x: int
    code_a: tuple[float, float]
    arrow: tuple[int, int]
    half_label_x: int
    half_label_end: float
    code_b: tuple[float, float]
    code_y: int
    code_h: int


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

    def lower_edge(self, index: int, number: int) -> LowerEdge:
        """The lower edge print of frame `index`, whose frame number is `number`."""
        x, y, W, H = self.frame_box(index)
        font = _font(round(H * 0.046), bold=True)
        size = font.size if hasattr(font, "size") else round(H * 0.046)
        gap = W * 0.015
        number_x = x + round(W * 0.03)
        number_end = number_x + font.getlength(str(number))
        arrow_x = x + round(W * 0.50)
        arrow_w = round(size * 1.0)
        label_x = arrow_x + arrow_w + round(size * _ARROW_SPACE)
        label_end = label_x + font.getlength(f"{number}A")
        return LowerEdge(
            number_x=number_x,
            code_a=(number_end + gap, arrow_x - gap),
            arrow=(arrow_x, arrow_x + arrow_w),
            half_label_x=label_x,
            half_label_end=label_end,
            code_b=(label_end + gap, x + W * 0.97),
            code_y=y + H + round(self.bottom_edge * 0.27),
            code_h=round(self.bottom_edge * 0.42),
        )

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
    with its half-frame "A" number (arrowed, lower edge only) and DX-style edge code below - and, under that, a small dim line with
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

            # bottom edge: number, code, arrow + half-frame number, code
            low = layout.lower_edge(r * columns + c, number)
            code_mid = low.code_y + low.code_h // 2
            draw.text((low.number_x, code_mid), str(number), fill=_EDGE_PRINT, font=edge_font, anchor="lm")
            _draw_arrow(draw, low.arrow[0], low.arrow[1], code_mid, round(low.code_h * 0.5), _EDGE_PRINT)
            draw.text((low.half_label_x, code_mid), f"{number}A", fill=_EDGE_PRINT, font=edge_font, anchor="lm")
            _edge_code(draw, *low.code_a, low.code_y, low.code_h, number * 2, _EDGE_PRINT)
            _edge_code(draw, *low.code_b, low.code_y, low.code_h, number * 2 + 1, _EDGE_PRINT)

            # the frame's file name and recorded printing decision, dim, below the film
            caption = f"{tile.name}   {tile.caption}" if tile.caption else tile.name
            draw.text((x + round(W * 0.03), frames_y + H + bottom_edge + caption_h // 2), caption,
                      fill=_CAPTION, font=caption_font, anchor="lm")
    return sheet


def write_sheet(path: str | Path, sheet: Image.Image, quality: int = 92) -> None:
    """JPEG or PNG by extension, with an sRGB profile embedded (as `halide export` does), and marked
    as a halide contact sheet (see is_contact_sheet). Written atomically (halide.io.atomic): a
    sheet can take tens of seconds to render for a full roll, and a kill or crash partway through
    the save must not leave a truncated file under the real name."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"{path}: unsupported contact sheet format {suffix!r} (use .jpg, .jpeg or .png)")
    with atomic_output(path) as tmp:
        if suffix in (".jpg", ".jpeg"):
            sheet.save(tmp, format="JPEG", quality=quality, icc_profile=_SRGB_ICC_BYTES, subsampling=0,
                       comment=_SHEET_MARKER.encode())
        else:
            info = PngInfo()
            info.add_text(_SHEET_MARKER, "1")
            sheet.save(tmp, format="PNG", icc_profile=_SRGB_ICC_BYTES, pnginfo=info)


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

