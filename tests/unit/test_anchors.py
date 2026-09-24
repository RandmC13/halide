from pathlib import Path

import numpy as np
import pytest

from halide.calibration import anchors
from halide.calibration.anchors import NeutralPoint
from halide.calibration.profile_store import load_anchors, rename_profile, save_named_profile, update_profile
from halide.core.density import fit_density_balance
from halide.core.types import DensityProfile
from halide.io.scan_metadata import ScanSettings

# A neutral axis in fit_density_balance's form: D_c = log10(wb_c) + D_G / s_c.
SCALE = (1.12, 1.0, 0.78)
WB = (0.73, 1.0, 1.05)
AT_1_30 = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
AT_1_60 = ScanSettings(exposure_time=1 / 60, f_number=8.0, iso=100.0)
AT_1_25 = ScanSettings(exposure_time=1 / 25, f_number=8.0, iso=100.0)


def _rgb(green_density, deviation=(0.0, 0.0, 0.0)):
    d = np.array([np.log10(WB[c]) + green_density / SCALE[c] for c in range(3)]) + np.asarray(deviation)
    return tuple(10.0 ** -d)


def _point(green_density, scan=AT_1_30, deviation=(0.0, 0.0, 0.0), frame="IMG_0001.tif"):
    """A neutral object as it would be sampled from a frame digitized at `scan`: a scan exposure k
    times the reference's multiplies the recorded transmittance by k (scan_consistency.py)."""
    k = scan.relative_exposure / AT_1_30.relative_exposure
    rgb = tuple(v * k for v in _rgb(green_density, deviation))
    return NeutralPoint(frame=Path(frame), x=10, y=20, rgb=rgb, scan=scan)


def test_points_from_differently_exposed_scans_fit_like_matched_ones():
    mixed = [_point(0.8, AT_1_25), _point(1.1, AT_1_60), _point(1.4, AT_1_30), _point(1.6, AT_1_60)]
    matched = [_point(0.8), _point(1.1), _point(1.4), _point(1.6)]
    reference = AT_1_30
    fitted_mixed = anchors.fit(mixed, reference)
    fitted_matched = anchors.fit(matched, reference)
    assert fitted_mixed.density_scale == pytest.approx(fitted_matched.density_scale, rel=1e-9)
    assert fitted_mixed.white_balance == pytest.approx(fitted_matched.white_balance, rel=1e-9)
    assert fitted_matched.density_scale == pytest.approx(SCALE, rel=1e-9)

    # Without normalising, the same picks give a different (wrong) calibration.
    raw = fit_density_balance([p.rgb for p in mixed])
    assert raw.density_scale != pytest.approx(SCALE, rel=1e-3)


def test_reference_scan_is_the_rolls_most_common_setting():
    assert anchors.reference_scan([AT_1_30, AT_1_60, AT_1_30, None]) == AT_1_30
    assert anchors.reference_scan([None, None]) is None


def test_can_fit_needs_two_points_spanning_the_minimum_density():
    assert not anchors.can_fit([_point(1.0)], AT_1_30)
    assert not anchors.can_fit([_point(1.0), _point(1.05)], AT_1_30)
    assert anchors.can_fit([_point(1.0), _point(1.2)], AT_1_30)


def test_agreement_needs_three_points_then_flags_the_odd_one():
    two = anchors.agreement([_point(0.8), _point(1.6)], AT_1_30)
    assert two == [None, None]

    points = [_point(g) for g in (0.8, 1.0, 1.2, 1.4, 1.6)] + [_point(1.15, deviation=(0.06, 0.0, -0.06))]
    result = anchors.agreement(points, AT_1_30)
    wall = result[-1]
    # judged against the other five - exact neutrals here - it sits 0.06 * (1.12 + 0.78) = CC 11.4 off
    assert wall.cc == pytest.approx(11.4, rel=1e-6)
    assert wall.direction == "R"
    assert wall.band == "red"
    assert anchors.worst(points, AT_1_30) == len(points) - 1

    # The outlier tilts every other point's leave-one-out fit, so good points read a few CC off in
    # the opposite direction (C) - the lowest, with the most leverage, even reaches amber. That's
    # why only the worst point gets the hint: remove it and the rest settle.
    assert {a.direction for a in result[:-1]} == {"C"}
    assert all(a.cc < wall.cc / 2 for a in result[:-1])
    settled = anchors.agreement(points[:-1], AT_1_30)
    assert all(a.band == "calm" and a.cc < 0.5 for a in settled)
    assert anchors.worst(points[:-1], AT_1_30) is None


def test_no_agreement_when_the_others_cant_support_a_fit():
    # Judging the D 1.2 point against two points 0.05 apart would extrapolate a nearly
    # unconstrained line - no reading rather than a false accusation.
    points = [_point(1.2), _point(1.45), _point(1.5, deviation=(0.01, 0, 0))]
    result = anchors.agreement(points, AT_1_30)
    assert result[0] is None
    assert result[1] is not None and result[2] is not None


@pytest.mark.parametrize("cc, expected", [(0.0, "calm"), (5.0, "calm"), (5.1, "amber"), (10.0, "amber"), (10.1, "red")])
def test_bands(cc, expected):
    assert anchors.band(cc) == expected


def test_agreement_label():
    assert anchors.Agreement(cc=7.9, direction="R", band="amber").label() == "CC 8 R"
    assert anchors.Agreement(cc=0.2, direction="B", band="calm").label() == "CC 0"


def test_near_duplicate_needs_similar_tone_and_colour():
    existing = [_point(1.0), _point(1.5)]
    assert anchors.near_duplicate(_point(1.52), existing, AT_1_30) == 1
    assert anchors.near_duplicate(_point(1.3), existing, AT_1_30) is None  # different tone
    assert anchors.near_duplicate(_point(1.52, deviation=(0.05, 0, 0)), existing, AT_1_30) is None  # different colour


def test_near_duplicate_compares_at_the_reference_exposure():
    # The same object sampled from a frame scanned twice as bright is the same object.
    assert anchors.near_duplicate(_point(1.5, AT_1_60), [_point(1.5, AT_1_30)], AT_1_30) == 0


def test_wedge_range_spans_the_rolls_green_density():
    frames = [np.linspace(0.6, 1.0, 1000), np.linspace(0.9, 1.7, 1000)]
    low, high = anchors.wedge_range(frames)
    assert low == pytest.approx(0.6, abs=0.01)
    assert high == pytest.approx(1.7, abs=0.01)
    assert anchors.wedge_range([]) is None


def test_point_dict_roundtrip():
    point = _point(1.2, AT_1_60)
    assert anchors.point_from_dict(anchors.point_to_dict(point)) == point
    no_exif = NeutralPoint(frame=Path("a.tif"), x=1, y=2, rgb=(0.1, 0.2, 0.3), scan=None)
    assert anchors.point_from_dict(anchors.point_to_dict(no_exif)) == no_exif


def test_profile_anchors_sidecar_roundtrips_and_survives_edit_and_rename(tmp_path):
    points = [_point(0.9), _point(1.5, AT_1_60)]
    profile = DensityProfile(white_balance=WB, density_scale=SCALE)
    records = [anchors.point_to_dict(p) for p in points]
    save_named_profile(profile, "roll16", profiles_dir=tmp_path, anchors=records, roll="/films/Roll16")

    update_profile("roll16", profiles_dir=tmp_path, film_stock="Kodak Portra 400")
    path = rename_profile("roll16", "roll16-portra", profiles_dir=tmp_path)
    saved, roll = load_anchors(path)
    assert roll == "/films/Roll16"
    assert [anchors.point_from_dict(r) for r in saved] == points


def test_profiles_without_anchors_load_empty(tmp_path):
    path = save_named_profile(DensityProfile(white_balance=WB, density_scale=SCALE), "old", profiles_dir=tmp_path)
    assert load_anchors(path) == ([], None)
