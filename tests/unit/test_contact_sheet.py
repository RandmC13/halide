import numpy as np
import pytest
from PIL import Image

from halide.io.contact_sheet import (
    Tile,
    caption_from_provenance,
    check_sheet_path,
    is_contact_sheet,
    downsample_linear,
    load_thumbnail,
    render_sheet,
    save_thumbnail,
    thumbnail_from_linear,
    write_sheet,
)


def test_downsample_averages_in_linear_light():
    # A fine checkerboard of 0 and 1 (linear) must average to 0.5 in *linear* light — averaging the
    # gamma-encoded values instead would come out visibly darker.
    board = np.indices((64, 64)).sum(axis=0) % 2
    image = np.repeat(board[..., None], 3, axis=2).astype(np.float32)
    reduced = downsample_linear(image, 8)
    assert reduced.shape == (8, 8, 3)
    assert reduced == pytest.approx(np.full_like(reduced, 0.5))


def test_thumbnail_fits_the_requested_long_edge_keeping_aspect():
    image = np.full((300, 450, 3), 0.18, dtype=np.float32)
    thumb = thumbnail_from_linear(image, 120)
    assert thumb.dtype == np.uint8 and max(thumb.shape[:2]) == 120
    assert thumb.shape[1] / thumb.shape[0] == pytest.approx(1.5, rel=0.02)


def test_captions_come_from_the_recorded_printing_decision():
    assert caption_from_provenance({"output": "print", "contrast": 0.874, "exposure": 0.152}) == "grade 0.87 · exp +0.15"
    assert caption_from_provenance({"output": "print", "contrast": 1.0, "exposure": -0.2, "scan_gain": 1.333}) == (
        "grade 1.00 · exp -0.20 · scan x1.33"
    )
    assert caption_from_provenance({"output": "flat", "linear_scale": 0.0161}) == "flat x0.0161"
    assert caption_from_provenance(None) == ""


def test_thumbnail_roundtrip_keeps_its_provenance(tmp_path):
    thumb = np.zeros((10, 15, 3), dtype=np.uint8)
    save_thumbnail(tmp_path / "t.png", thumb, {"output": "print", "contrast": 0.9, "exposure": 0.1})
    loaded, record = load_thumbnail(tmp_path / "t.png")
    assert loaded.shape == thumb.shape and record["contrast"] == 0.9


def test_sheet_lays_frames_out_in_strips_with_a_short_last_strip(tmp_path):
    frame = np.full((60, 90, 3), 128, dtype=np.uint8)
    portrait = np.full((90, 60, 3), 200, dtype=np.uint8)
    tiles = [Tile(f"IMG_{i}", portrait if i == 2 else frame, "grade 0.90 · exp +0.10") for i in range(7)]
    tiles.append(Tile("IMG_broken", None))
    one_strip = render_sheet(tiles[:3], "Roll", frame_width=90, columns=3)
    two_strips = render_sheet(tiles, "Roll", "8 frames", frame_width=90, columns=6)
    assert two_strips.width > one_strip.width  # six across vs three across
    assert two_strips.height > one_strip.height  # a second (short) strip
    path = tmp_path / "sheet.jpg"
    write_sheet(path, two_strips)
    with Image.open(path) as written:
        assert written.size == two_strips.size and written.info.get("icc_profile")  # sRGB tagged
    assert is_contact_sheet(path)  # marked, so a later `halide contact` of this folder skips it
    Image.new("RGB", (4, 4)).save(tmp_path / "export.png")
    assert not is_contact_sheet(tmp_path / "export.png")


def test_sheet_is_black_like_a_real_contact_print():
    frame = np.full((60, 90, 3), 128, dtype=np.uint8)
    tiles = [Tile("IMG_1", frame, "grade 0.90", number=7), Tile("IMG_2", frame)]
    for stock in ("Kodak Portra 400", None):  # named stock, and the "INVERTED BY HALIDE" fallback
        sheet = np.asarray(render_sheet(tiles, "Roll", frame_width=300, columns=6, film_stock=stock))
        assert tuple(sheet[2, 2]) == tuple(sheet[-2, -2]) and max(sheet[2, 2]) < 20  # black margins
        orange = (sheet[..., 0] > 180) & (sheet[..., 1] > 120) & (sheet[..., 1] < 190) & (sheet[..., 2] < 130)
        assert orange.sum() > 200  # the edge print is there


def test_film_stock_travels_in_provenance_to_the_sheet():
    from halide.cli._contact_sheet import common_film_stock
    from halide.core.tone_render import ResolvedTone
    from halide.core.types import DensityProfile
    from halide.processing import provenance_json, read_provenance

    profile = DensityProfile((0.7, 1.0, 1.0), (1.1, 1.0, 0.8), film_stock="Kodak Portra 400")
    record = read_provenance(provenance_json(ResolvedTone(mode="paper", exposure=0.1, contrast=0.9), profile))
    assert record["film_stock"] == "Kodak Portra 400"
    assert common_film_stock([record, record]) == "Kodak Portra 400"
    assert common_film_stock([record, {**record, "film_stock": "Ilford XP2"}]) is None  # mixed rolls
    assert common_film_stock([record, None]) is None  # a frame without provenance: don't guess
    unnamed = read_provenance(provenance_json(ResolvedTone(mode="paper", exposure=0.1, contrast=0.9), DensityProfile((1, 1, 1), (1, 1, 1))))
    assert "film_stock" not in unnamed


def test_layout_locates_frames_where_the_renderer_draws_them():
    from halide.io.contact_sheet import SheetLayout

    colours = [np.full((60, 90, 3), 40 + 20 * i, dtype=np.uint8) for i in range(8)]
    tiles = [Tile(f"IMG_{i}", c) for i, c in enumerate(colours)]
    sheet = np.asarray(render_sheet(tiles, "Roll", frame_width=300, columns=6))
    layout = SheetLayout(len(tiles), 300, 6)
    assert sheet.shape[1::-1] == layout.size
    for i, colour in enumerate(colours):
        x, y, w, h = layout.frame_box(i)
        assert tuple(sheet[y + h // 2, x + w // 2]) == tuple(colour[0, 0])  # the frame's own pixels
        assert layout.frame_at(x + w // 2, y + h // 2) == i
    assert layout.frame_at(2, 2) is None  # the margin belongs to no frame


def test_sheet_format_is_checked_up_front():
    check_sheet_path("sheet.png")
    with pytest.raises(ValueError, match=".jpg, .jpeg or .png"):
        check_sheet_path("sheet.tif")
