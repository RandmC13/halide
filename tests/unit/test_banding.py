"""Band-by-band processing (halide.banding) must be invisible in the output: every full-resolution
path that develops, prints or exports in bands is pinned bit-for-bit (raw float bits, not a
tolerance) to the plain whole-array pipeline, at deliberately awkward band sizes — 1 row, and 7 rows
on an image whose height isn't a multiple of 7 (so the last band is short).

Why bit-identical is the bar: banding exists purely to cut peak memory (see halide.banding); halide
prints what the film recorded, so a memory optimisation that moved any pixel at all would be the
wrong trade.
"""

import tracemalloc

import numpy as np
import pytest
from PIL import Image

import halide.banding
import halide.io.tiff
from halide.banding import band_slices, map_in_bands
from halide.calibration.auto import auto_density_balance
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.pipeline import develop
from halide.core.tone_render import apply_tone, resolve_tone
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.icc import convert_to_working_space, parse_linear_rgb_profile
from halide.io.raster import to_srgb_8bit, write_delivery_image
from halide.io.tiff import read_tiff, write_tiff
from halide.processing import (
    IDENTITY_PROFILE,
    Stage,
    load_working_space_image,
    print_scan,
    process_scan,
    read_provenance,
)
from tests.unit.test_icc import LINEAR_TAGS, build_icc

PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))
SHAPE = (37, 53, 3)  # 37 rows: not a multiple of 7, so a 7-row band leaves a short last band


def _bits(array):
    return np.ascontiguousarray(array).view(np.uint8)


def _assert_identical(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(_bits(actual), _bits(expected))


@pytest.fixture(params=[1, 7], ids=["1-row bands", "7-row bands"])
def band_rows(request, monkeypatch):
    row_bytes = SHAPE[1] * SHAPE[2] * 4
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", row_bytes * request.param)
    return request.param


@pytest.fixture
def negative_path(tmp_path):
    """A noisy synthetic linear negative, including exact zeros and slight negatives (sensor-floor
    pixels) so the MIN_TRANSMITTANCE clamps in density balance and invert are exercised too."""
    rng = np.random.default_rng(7)
    image = rng.uniform(0.01, 0.3, size=SHAPE).astype(np.float32)
    image[0, :5] = 0.0
    image[5, 10:14] = -0.001
    path = tmp_path / "negative.tif"
    write_tiff(path, image, icc_profile=build_icc(LINEAR_TAGS))
    return path


def _whole_working_image(path):
    """The pre-banding load: whole-frame colour conversion."""
    scan = read_tiff(path)
    return convert_to_working_space(scan.image, parse_linear_rgb_profile(scan.icc_profile))


def test_band_slices_cover_every_row_exactly_once_with_a_short_last_band(monkeypatch):
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", 70)
    slices = list(band_slices(37, row_bytes=10))  # 7 rows per band
    assert [s.stop - s.start for s in slices] == [7, 7, 7, 7, 7, 2]
    assert np.array_equal(np.concatenate([np.arange(37)[s] for s in slices]), np.arange(37))


def test_band_slices_take_at_least_one_row_even_when_a_row_exceeds_the_budget(monkeypatch):
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", 5)
    assert [s.stop - s.start for s in band_slices(3, row_bytes=10)] == [1, 1, 1]


def test_band_slices_one_band_when_the_image_fits(monkeypatch):
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", 1024**2)
    assert list(band_slices(4, row_bytes=10)) == [slice(0, 4)]


def test_map_in_bands_writes_in_place_or_into_out(band_rows):
    src = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    out = map_in_bands(src.copy(), lambda b: b * 2)
    _assert_identical(out, src * 2)
    separate = np.empty(SHAPE, dtype=np.float64)
    assert map_in_bands(src, lambda b: b + 1, out=separate) is separate
    _assert_identical(separate, (src + 1).astype(np.float64))


def test_load_working_space_image_converts_in_bands_identically(negative_path, band_rows):
    image = load_working_space_image(negative_path)
    _assert_identical(image, _whole_working_image(negative_path))
    assert image.flags.writeable  # callers develop it in place


@pytest.mark.parametrize(
    "stage, tone, profile",
    [
        (Stage.FULL, ToneCurveParams(), PROFILE),
        (Stage.FULL, ToneCurveParams(mode="linear"), PROFILE),
        (Stage.FULL, ToneCurveParams(exposure=0.3, contrast=0.8), PROFILE),
        (Stage.FULL, ToneCurveParams(), None),  # per-frame auto calibration
        (Stage.INVERT_ONLY, ToneCurveParams(), None),
        (Stage.DENSITY_ONLY, ToneCurveParams(), PROFILE),
    ],
    ids=["print", "flat", "pinned", "auto", "invert-only", "density-only"],
)
@pytest.mark.parametrize("scan_gain", [1.0, 1.37])
def test_process_scan_matches_the_whole_array_pipeline(tmp_path, negative_path, band_rows, stage, tone, profile, scan_gain):
    out_path = tmp_path / "out.tif"
    resolved = process_scan(negative_path, out_path, stage, profile, tone, scan_gain=scan_gain)

    working = _whole_working_image(negative_path)
    if scan_gain != 1.0:
        working *= np.asarray(scan_gain, dtype=working.dtype)
    if stage is Stage.INVERT_ONLY:
        used = IDENTITY_PROFILE
    else:
        used = profile if profile is not None else auto_density_balance(working)
    if stage is Stage.DENSITY_ONLY:
        expected, expected_tone = apply_density_balance(apply_white_balance(working, used), used), None
    else:
        expected, expected_tone = develop(working, used, tone)

    _assert_identical(read_tiff(out_path).image, expected.astype(np.float32))
    assert resolved == expected_tone


def test_process_scan_thumbnail_is_made_from_the_same_developed_frame(tmp_path, negative_path, band_rows, monkeypatch):
    thumb = tmp_path / "thumb.png"
    process_scan(negative_path, None, Stage.FULL, PROFILE, ToneCurveParams(), thumbnail_path=thumb, thumbnail_long_edge=20)
    reference = tmp_path / "reference.png"
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", 1024**3)  # one band == the old whole-array path
    process_scan(negative_path, None, Stage.FULL, PROFILE, ToneCurveParams(), thumbnail_path=reference, thumbnail_long_edge=20)
    assert np.array_equal(np.asarray(Image.open(thumb)), np.asarray(Image.open(reference)))
    assert Image.open(thumb).info == Image.open(reference).info


def test_print_scan_matches_the_whole_array_print(tmp_path, negative_path, band_rows):
    flat = tmp_path / "flat.tif"
    process_scan(negative_path, flat, Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    printed = tmp_path / "printed.tif"
    resolved, _ = print_scan(flat, printed, ToneCurveParams())

    scan = read_tiff(flat)
    working = scan.image / np.asarray(read_provenance(scan.description)["linear_scale"], dtype=scan.image.dtype)
    expected_tone = resolve_tone(working, ToneCurveParams(mode="paper"))
    expected = apply_tone(working, expected_tone)

    _assert_identical(read_tiff(printed).image, expected.astype(np.float32))
    assert resolved == expected_tone


def test_write_delivery_image_matches_whole_frame_conversion(tmp_path, negative_path, band_rows):
    positive = develop(_whole_working_image(negative_path), PROFILE)[0]
    path = tmp_path / "delivery.png"
    write_delivery_image(path, positive)
    assert np.array_equal(np.asarray(Image.open(path)), to_srgb_8bit(positive))


@pytest.mark.parametrize(
    "tone, profile, bound",
    [(ToneCurveParams(), PROFILE, 2.0), (ToneCurveParams(mode="linear"), PROFILE, 2.5), (ToneCurveParams(), None, 3.0)],
    ids=["print", "flat", "auto"],
)
def test_process_scan_peak_memory_stays_near_one_frame(tmp_path, monkeypatch, tone, profile, bound):
    """Regression guard for the reason banding exists. Before it, one frame's develop peaked at
    ~9 frames (colour-science's float64 conversion, the paper-curve lookup's temporaries, a new
    array per stage — measured 9.0x on this exact test); banded, it's the frame itself, the print
    fit's luminance (1/3 frame) and small fixed buffers — measured 1.42x print, 2.0x flat (its
    highlight percentile copies the frame), 2.19x auto (the neutral-candidate statistics). Measured
    with tracemalloc (numpy reports its buffers to it). The band and TIFF-read budgets are scaled
    down to the proportions they have on a real ~181 MiB scan (~2% and ~9% of the frame), so this
    small frame measures the design rather than those fixed buffers."""
    rng = np.random.default_rng(3)
    image = rng.uniform(0.01, 0.3, size=(1024, 1536, 3)).astype(np.float32)
    source = tmp_path / "big.tif"
    write_tiff(source, image, icc_profile=build_icc(LINEAR_TAGS))
    frame_bytes = image.nbytes
    del image
    monkeypatch.setattr(halide.banding, "_BAND_BYTES", frame_bytes // 50)
    monkeypatch.setattr(halide.io.tiff, "_READ_BUFFER_BYTES", frame_bytes // 11)

    tracemalloc.start()
    try:
        process_scan(source, tmp_path / "out.tif", Stage.FULL, profile, tone)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < bound * frame_bytes, f"peak {peak / frame_bytes:.2f}x the frame"
