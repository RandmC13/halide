import argparse
import json
import struct

import numpy as np
import pytest

from halide.calibration.profile_store import (
    load_scan_reference,
    load_tone_override,
    rename_profile,
    save_named_profile,
)
from halide.calibration.scan_consistency import assess_roll, most_common_settings, scan_gain
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.scan_metadata import ScanSettings, darktable_state_from_xmp, settings_from_exif
from halide.io.tiff import read_tiff, write_tiff
from halide.processing import Stage, process_scan
from tests.unit.test_icc import LINEAR_TAGS, build_icc

S30 = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
S60 = ScanSettings(exposure_time=1 / 60, f_number=8.0, iso=100.0)
PROFILE = DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78))


def _wb_params(r, g, b):
    return struct.pack("<4f", r, g, b, 0.0).hex()


def _xmp(entries, history_end=None):
    lis = "".join(
        f'<rdf:li darktable:operation="{op}" darktable:enabled="{en}" darktable:multi_priority="0" '
        f'darktable:params="{params}"/>'
        for op, en, params in entries
    )
    end = len(entries) if history_end is None else history_end
    return f'<x darktable:history_end="{end}"><darktable:history><rdf:Seq>{lis}</rdf:Seq></darktable:history></x>'


def test_settings_from_exif_reads_rationals_and_describes():
    s = settings_from_exif({"ExposureTime": (1, 30), "FNumber": (8, 1), "ISOSpeedRatings": 100})
    assert s == S30
    assert s.describe() == "1/30 f/8 ISO100"
    assert settings_from_exif({"FNumber": (8, 1)}) is None


def test_darktable_final_state_honours_later_disables_and_history_end():
    xmp = _xmp([
        ("negadoctor", "1", "00"),
        ("temperature", "1", _wb_params(1.9, 1.0, 1.8)),
        ("negadoctor", "0", "00"),  # later disabled -> not active
        ("shadhi", "1", "00"),
        ("exposure", "1", "00"),  # beyond history_end -> ignored
    ], history_end=4)
    state = darktable_state_from_xmp(xmp)
    assert state.tonal_modules == ("shadhi",)
    assert state.white_balance == pytest.approx((1.9, 1.0, 1.8))


def test_scan_gain_and_most_common_reference():
    assert scan_gain(S60, S30) == pytest.approx(2.0)  # a half-exposed scan needs doubling
    assert most_common_settings([S30, S60, S30]) == S30


def test_assess_roll_reports_each_kind_of_inconsistency():
    clean = darktable_state_from_xmp(_xmp([("temperature", "1", _wb_params(1.9, 1.0, 1.8))]))
    other_wb = darktable_state_from_xmp(_xmp([("temperature", "1", _wb_params(1.95, 1.0, 1.7))]))
    edited = darktable_state_from_xmp(_xmp([("temperature", "1", _wb_params(1.9, 1.0, 1.8)), ("rgbcurve", "1", "00")]))

    consistent = assess_roll({"a": (S30, clean), "b": (S30, clean)})
    assert not consistent.has_issues and consistent.summary_lines() == []

    report = assess_roll({"a": (S30, clean), "b": (S60, other_wb), "c": (S30, edited)})
    assert report.exposure_inconsistent and report.exposure_spread_stops == pytest.approx(1.0)
    assert report.white_balance_inconsistent
    assert report.tonal_modules == {"c": ("rgbcurve",)}
    assert len(report.summary_lines()) == 3


def _negative(tmp_path, name, scale):
    rng = np.random.default_rng(0)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    img = base * 10.0 ** (-rng.uniform(0.0, 1.0, size=(32, 32, 1)).astype(np.float32)) * np.float32(scale)
    path = tmp_path / name
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def test_matching_scan_exposure_exactly_undoes_a_brighter_scan(tmp_path):
    # The same negative digitized one stop brighter. Through a density-balance profile that's a
    # per-channel shift (k ** density_scale) — a colour change, not just brightness — and the
    # matching gain must undo it exactly, through the real pipeline.
    tone = ToneCurveParams(exposure=0.2, contrast=0.8)  # pinned, so any difference is visible, not re-fitted away
    reference = _negative(tmp_path, "ref.tif", 1.0)
    brighter = _negative(tmp_path, "bright.tif", 2.0)
    out = {k: tmp_path / f"{k}.tif" for k in ("ref", "raw", "matched")}
    process_scan(reference, out["ref"], Stage.FULL, PROFILE, tone)
    process_scan(brighter, out["raw"], Stage.FULL, PROFILE, tone)
    process_scan(brighter, out["matched"], Stage.FULL, PROFILE, tone, scan_gain=scan_gain(S30, S60))
    ref, raw, matched = (read_tiff(p).image for p in out.values())

    assert matched == pytest.approx(ref, rel=1e-5, abs=1e-7)
    ratio = (raw / ref).reshape(-1, 3).mean(axis=0)
    assert abs(ratio[0] / ratio[1] - 1) > 0.01 or abs(ratio[2] / ratio[1] - 1) > 0.01  # the uncorrected colour shift is real


def test_profile_scan_reference_roundtrip_and_rename_keeps_sidecars(tmp_path):
    save_named_profile(PROFILE, "roll", profiles_dir=tmp_path, tone=ToneCurveParams(exposure=0.1, contrast=0.9), scan=S30)
    assert load_scan_reference(tmp_path / "roll.json") == S30

    new_path = rename_profile("roll", "roll16", profiles_dir=tmp_path)
    assert load_scan_reference(new_path) == S30
    assert load_tone_override(new_path).contrast == 0.9  # used to be silently dropped by rename
    assert json.loads(new_path.read_text())["name"] == "roll16"


def test_invert_match_without_a_known_reference_explains_what_to_do(tmp_path, monkeypatch):
    from halide.cli.commands import invert_cmd

    monkeypatch.setattr(invert_cmd, "read_scan_metadata", lambda path: (S30, None))
    args = argparse.Namespace(input="x.tif", match_scan_exposure=True, scan_reference=None, profile=None)
    with pytest.raises(SystemExit, match="--scan-reference FRAME"):
        invert_cmd._resolve_scan_gain(args, calibrated_here=False)
    # Calibrated from this very frame: nothing to match, and the frame's own settings get recorded.
    assert invert_cmd._resolve_scan_gain(args, calibrated_here=True) == (1.0, S30)


def test_roll_estimate_keeps_small_copies_not_views_of_full_frames(tmp_path, monkeypatch):
    # Regression: the downsampled frames used to be strided *views*, each pinning its whole
    # full-resolution parent (~180 MiB per real scan) — a real 37-frame roll got OOM-killed.
    import halide.processing as processing

    captured = []
    monkeypatch.setattr(processing, "roll_auto_density_balance", lambda images: captured.extend(images) or PROFILE)
    paths = [_negative(tmp_path, f"f{i}.tif", 1.0) for i in range(2)]
    processing.estimate_roll_density_profile(paths, stride=4)
    assert len(captured) == 2
    assert all(image.base is None for image in captured)  # owns its (small) data
