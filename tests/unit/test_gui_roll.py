"""gui/roll.py is the picker's Qt-free session model - these run without a display."""

from pathlib import Path

import numpy as np
import pytest

from halide.calibration.anchors import NeutralPoint
from halide.calibration.profile_store import save_named_profile
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui.roll import CalibrationSession
from halide.io.scan_metadata import ScanSettings

SCALE = (1.12, 1.0, 0.78)
WB = (0.73, 1.0, 1.05)
AT_1_30 = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100.0)
AT_1_60 = ScanSettings(exposure_time=1 / 60, f_number=8.0, iso=100.0)


def _rgb(green_density, deviation=(0.0, 0.0, 0.0)):
    d = np.array([np.log10(WB[c]) + green_density / SCALE[c] for c in range(3)]) + np.asarray(deviation)
    return tuple(10.0 ** -d)


def _session(tmp_path, names=("a.tif", "b.tif", "c.tif"), scans=None):
    session = CalibrationSession()
    paths = [tmp_path / n for n in names]
    session.set_roll(paths, scans or [AT_1_30] * len(paths), tmp_path)
    return session, paths


def _pt(frame, green_density, x=100, y=100, deviation=(0.0, 0.0, 0.0), scan=AT_1_30):
    return NeutralPoint(frame=frame, x=x, y=y, rgb=_rgb(green_density, deviation), scan=scan)


def test_frames_are_numbered_through_the_roll(tmp_path):
    session, paths = _session(tmp_path)
    assert [f.number for f in session.frames] == [1, 2, 3]
    assert session.reference == AT_1_30


def test_add_selects_and_nudges_on_near_duplicates(tmp_path):
    session, paths = _session(tmp_path)
    assert session.add_point(_pt(paths[0], 1.0)) is None
    assert session.selected == 0
    nudge = session.add_point(_pt(paths[1], 1.01, x=5))
    assert nudge is not None and "point 1" in nudge
    assert len(session.points) == 2  # nudged, never refused
    assert session.selected == 1


def test_remove_keeps_selection_consistent(tmp_path):
    session, paths = _session(tmp_path)
    for g in (0.8, 1.2, 1.6):
        session.add_point(_pt(paths[0], g, x=int(g * 100)))
    session.selected = 2
    session.remove_point(0)
    assert session.selected == 1
    session.remove_point(1)
    assert session.selected is None


def test_point_at_finds_the_nearest_point_on_that_frame_only(tmp_path):
    session, paths = _session(tmp_path)
    session.add_point(_pt(paths[0], 1.0, x=100, y=100))
    session.add_point(_pt(paths[0], 1.3, x=108, y=100))
    session.add_point(_pt(paths[1], 1.5, x=100, y=100))
    assert session.point_at(paths[0], 106, 100, radius=10) == 1
    assert session.point_at(paths[0], 300, 300, radius=10) is None
    assert session.point_at(paths[2], 100, 100, radius=10) is None


def test_positive_switches_from_estimate_to_fit_at_two_separated_points(tmp_path):
    session, paths = _session(tmp_path, scans=[AT_1_30, AT_1_60, AT_1_30])
    estimate = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0))
    session.frames[1].estimate = estimate
    assert session.positive_source(session.frames[1]) == (estimate, False, 1.0)

    session.add_point(_pt(paths[0], 0.9))
    session.add_point(_pt(paths[2], 1.5))
    profile, is_fit, gain = session.positive_source(session.frames[1])
    assert is_fit
    assert profile.density_scale == pytest.approx(SCALE)
    assert gain == pytest.approx(2.0)  # the 1/60 frame is printed at the 1/30 reference, as batch would


def test_views_carry_frame_numbers_density_and_agreement(tmp_path):
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(paths[i], g))
    session.add_point(_pt(paths[1], 1.1, deviation=(0.06, 0.0, -0.06), x=400))
    views = session.views()
    assert [v.frame_number for v in views] == [1, 2, 3, 2]
    assert views[1].green_density == pytest.approx(1.2)
    assert views[3].agreement.direction == "R"
    assert session.worst() == 3
    assert session.point_counts() == {paths[0]: 1, paths[1]: 2, paths[2]: 1}


def test_save_and_restore_round_trip(tmp_path):
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(paths[i], g))
    session.details = {"film_stock": "Kodak Portra 400", "process": "", "scanner": "", "notes": "trike + shirt"}
    session.tone_override = ToneCurveParams(mode="paper", exposure=0.2, contrast=0.8)
    profile = session.profile_to_save()
    assert profile.film_stock == "Kodak Portra 400" and profile.process is None
    path = save_named_profile(profile, "roll16", profiles_dir=tmp_path / "profiles", **session.sidecars())

    restored = CalibrationSession()
    folder = restored.restore(path)
    assert folder == tmp_path
    restored.set_roll(paths, [AT_1_30] * 3, folder)
    assert restored.points == session.points
    assert restored.details["notes"] == "trike + shirt"
    assert restored.tone_override.exposure == pytest.approx(0.2)
    assert restored.profile().density_scale == pytest.approx(session.profile().density_scale)


def test_restore_reattaches_points_to_a_moved_roll_by_file_name(tmp_path):
    old, old_paths = _session(tmp_path / "old")
    for i, g in enumerate((0.8, 1.2, 1.6)):
        old.add_point(_pt(old_paths[i], g))
    path = save_named_profile(old.profile_to_save(), "moved", profiles_dir=tmp_path / "profiles", **old.sidecars())

    session = CalibrationSession()
    assert session.restore(path) is None  # the recorded roll folder doesn't exist any more
    moved = [tmp_path / "new" / n for n in ("a.tif", "b.tif", "c.tif")]
    session.set_roll(moved, [AT_1_30] * 3, tmp_path / "new")
    assert [p.frame for p in session.points] == moved
    assert [v.frame_number for v in session.views()] == [1, 2, 3]


def test_reopened_profile_keeps_its_own_scan_reference(tmp_path):
    session, paths = _session(tmp_path, scans=[AT_1_60, AT_1_60, AT_1_30])
    session.add_point(_pt(paths[2], 0.9))
    session.add_point(_pt(paths[2], 1.5, x=300))
    path = save_named_profile(session.profile_to_save(), "ref", profiles_dir=tmp_path / "profiles", **session.sidecars())

    reopened = CalibrationSession()
    reopened.restore(path)
    reopened.set_roll(paths, [AT_1_30, AT_1_30, AT_1_30], tmp_path)  # a roll whose most common setting differs
    assert reopened.reference == AT_1_60  # the profile's recorded reference, not the new roll's


def test_cant_save_without_a_fit(tmp_path):
    session, paths = _session(tmp_path)
    session.add_point(_pt(paths[0], 1.0))
    with pytest.raises(ValueError):
        session.profile_to_save()


def test_reset_starts_over(tmp_path):
    session, paths = _session(tmp_path)
    session.add_point(_pt(paths[0], 1.0))
    session.details["film_stock"] = "x"
    session.reset()
    assert session.points == [] and session.selected is None and session.details["film_stock"] == ""
