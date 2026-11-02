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


def test_clear_points_on_one_frame_or_all(tmp_path):
    session, paths = _session(tmp_path)
    session.add_point(_pt(paths[0], 0.9))
    session.add_point(_pt(paths[1], 1.2))
    session.add_point(_pt(paths[0], 1.5, x=300))
    session.selected = 1
    assert session.clear_points(paths[0]) == 2
    assert [p.frame for p in session.points] == [paths[1]]
    assert session.selected == 0  # the selected point survived and moved up
    assert session.clear_points(paths[2]) == 0
    assert session.clear_points() == 1
    assert session.points == [] and session.selected is None


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
    # The roll's estimate (here, its one vote) is at the reference exposure: the frame gets its gain.
    assert session.positive_source(session.frames[1]) == (session.roll_estimate(tmp_path), False, 2.0)

    session.add_point(_pt(paths[0], 0.9))
    session.add_point(_pt(paths[2], 1.5))
    profile, is_fit, gain = session.positive_source(session.frames[1])
    assert is_fit
    assert profile.density_scale == pytest.approx(SCALE)
    assert gain == pytest.approx(2.0)  # the 1/60 frame is printed at the 1/30 reference, as batch would


def _estimate(red_wb, red_scale=1.0):
    return DensityProfile(white_balance=(red_wb, 1.0, 1.0), density_scale=(red_scale, 1.0, 1.0), source="auto")


def test_positive_before_points_is_the_rolls_median_at_the_reference(tmp_path):
    from halide.calibration.auto import at_scan_gain, median_profile

    session, paths = _session(tmp_path, scans=[AT_1_30, AT_1_60, AT_1_30])
    estimates = [_estimate(0.7, 1.1), _estimate(0.8, 1.2), _estimate(5.0, 1.3)]
    for frame, estimate in zip(session.frames, estimates):
        frame.estimate = estimate  # each measured on its frame as scanned
    expected = median_profile([estimates[0], at_scan_gain(estimates[1], 2.0), estimates[2]])
    profile, is_fit, gain = session.positive_source(session.frames[1])
    assert (profile, is_fit, gain) == (expected, False, 2.0)
    assert session.roll_votes(tmp_path) == 3
    assert session.positive_source(session.frames[0])[2] == 1.0


def test_a_new_preview_moves_the_rolls_estimate(tmp_path):
    session, paths = _session(tmp_path)
    session.frames[0].estimate = _estimate(0.5)
    assert session.roll_estimate(tmp_path).white_balance[0] == pytest.approx(0.5)
    session.frames[1].estimate = _estimate(0.7)
    assert session.roll_estimate(tmp_path).white_balance[0] == pytest.approx(np.sqrt(0.5 * 0.7))
    assert session.roll_votes(tmp_path) == 2


def test_a_frame_falls_back_to_its_own_estimate_only_before_its_roll_has_one(tmp_path):
    session, paths = _session(tmp_path)
    own = _estimate(0.9)
    assert session.positive_source(session.frames[0], own) == (own, False, 1.0)


def test_each_roll_open_gets_its_own_rolls_estimate(tmp_path):
    session, paths = _session(tmp_path, names=("a.tif", "b.tif", "c.tif", "d.tif"))
    other = tmp_path / "other"
    for frame, estimate in zip(session.frames, (_estimate(0.5), _estimate(0.5), _estimate(2.0), _estimate(2.0))):
        frame.estimate = estimate
    session.frames[2].roll = session.frames[3].roll = other
    assert session.positive_source(session.frames[0])[0].white_balance[0] == pytest.approx(0.5)
    assert session.positive_source(session.frames[3])[0].white_balance[0] == pytest.approx(2.0)


def test_views_carry_frame_numbers_density_and_agreement(tmp_path):
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(paths[i], g))
    session.add_point(_pt(paths[1], 1.1, deviation=(0.06, 0.0, -0.06), x=400))
    views = session.views()
    assert [v.frame_number for v in views] == [1, 2, 3, 2]
    assert views[1].green_density == pytest.approx(1.2)
    assert views[3].agreement.filters.startswith("CC ") and views[3].agreement.filters.endswith("Y + 7M")  # warm
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
    restored.restore(path)
    restored.set_roll(paths, [AT_1_30] * 3, session.roll_folder)
    assert restored.points == session.points
    assert restored.details["notes"] == "trike + shirt"
    assert restored.tone_override is None  # the print pin belongs to the roll, never the profile
    assert restored.profile().density_scale == pytest.approx(session.profile().density_scale)


def test_a_profile_saved_with_a_print_pin_still_restores_it(tmp_path):
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(paths[i], g))
    pin = ToneCurveParams(mode="paper", exposure=0.2, contrast=0.8)
    path = save_named_profile(session.profile_to_save(), "old", profiles_dir=tmp_path / "profiles",
                              **{**session.sidecars(), "tone": pin})
    restored = CalibrationSession()
    restored.restore(path)
    assert restored.tone_override == pin


def test_restore_reattaches_points_to_a_moved_roll_by_file_name(tmp_path):
    old, old_paths = _session(tmp_path / "old")
    for i, g in enumerate((0.8, 1.2, 1.6)):
        old.add_point(_pt(old_paths[i], g))
    path = save_named_profile(old.profile_to_save(), "moved", profiles_dir=tmp_path / "profiles", **old.sidecars())

    session = CalibrationSession()
    session.restore(path)
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


def test_a_roll_loaded_by_relative_path_is_saved_as_absolute(tmp_path, monkeypatch):
    # `halide open Roll16` from the folder above the roll: the profile must reopen from anywhere.
    monkeypatch.chdir(tmp_path)
    session = CalibrationSession()
    relative = [Path("roll") / n for n in ("a.tif", "b.tif", "c.tif")]
    session.set_roll(relative, [AT_1_30] * 3, Path("roll"))
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(session.frames[i].path, g))
    sidecars = session.sidecars()
    assert "roll" not in sidecars
    assert [a["frame"] for a in sidecars["anchors"]] == [str(tmp_path / "roll" / n) for n in ("a.tif", "b.tif", "c.tif")]

    (tmp_path / "roll").mkdir()
    path = save_named_profile(session.profile_to_save(), "rel", profiles_dir=tmp_path / "profiles", **sidecars)
    monkeypatch.chdir(tmp_path / "profiles")  # somewhere else entirely
    reopened = CalibrationSession()
    reopened.restore(path)
    assert [p.frame for p in reopened.points] == [tmp_path / "roll" / n for n in ("a.tif", "b.tif", "c.tif")]


def test_missing_frames_are_the_picked_frames_that_have_gone(tmp_path):
    session, paths = _session(tmp_path)
    for p in paths:
        p.write_bytes(b"")
    for i, g in enumerate((0.8, 1.2, 1.6)):
        session.add_point(_pt(paths[i], g))
    path = save_named_profile(session.profile_to_save(), "gone", profiles_dir=tmp_path / "profiles", **session.sidecars())
    paths[1].unlink()  # deleted from the roll since

    reopened = CalibrationSession()
    reopened.restore(path)
    reopened.set_roll([paths[0], paths[2]], [AT_1_30] * 2, tmp_path)
    assert reopened.missing_frames() == [paths[1]]
    assert len(reopened.points) == 3  # its point still counts in the fit


def test_a_moved_roll_is_found_again_by_file_name(tmp_path):
    old, old_paths = _session(tmp_path / "old")
    for i, g in enumerate((0.8, 1.2, 1.6)):
        old.add_point(_pt(old_paths[i], g))
    path = save_named_profile(old.profile_to_save(), "moved", profiles_dir=tmp_path / "profiles", **old.sidecars())

    session = CalibrationSession()
    session.restore(path)
    assert session.missing_frames() == old_paths  # nothing loaded, nothing on disk
    session.set_roll([tmp_path / "new" / n for n in ("a.tif", "b.tif", "c.tif")], [AT_1_30] * 3, tmp_path / "new")
    assert session.missing_frames() == []  # all re-attached by file name


def test_a_fit_the_points_dont_pin_down_is_reported_not_raised(tmp_path):
    # Points spread widely in green but with red nearly flat: can_fit passes the 0.1 D gate, yet
    # the fit refuses (red scale ~50). The picker must treat that as "can't fit yet".
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.3, 0.7, 1.1)):
        rgb = (10 ** -(0.5 + 0.02 * g), 10 ** -g, 10 ** -(0.05 + g / 0.78))
        session.add_point(NeutralPoint(frame=paths[i], x=10 * i, y=50, rgb=rgb, scan=AT_1_30))
    assert session.can_fit()
    assert session.profile() is None
    assert "red layer" in session.fit_problem()
    assert "further apart in density" in session.fit_problem()
    session.views()  # agreement / worst() must not raise either
    assert session.worst() is None or isinstance(session.worst(), int)
    with pytest.raises(ValueError):
        session.profile_to_save()


def test_no_fit_problem_for_good_points(tmp_path):
    session, paths = _session(tmp_path)
    session.add_point(_pt(paths[0], 0.4))
    session.add_point(_pt(paths[1], 1.2, x=5))
    assert session.fit_problem() is None
    assert session.profile() is not None


def test_reliability_follows_the_fit_and_the_rolls_range(tmp_path):
    session, paths = _session(tmp_path)
    for i, g in enumerate((0.95, 1.0, 1.05)):  # agree perfectly, but only 0.1 D apart
        session.add_point(_pt(paths[i], g))
    assert session.can_fit()
    assert session.reliability() is None  # previews not measured yet: no roll range to judge against

    for frame in session.frames:  # a roll spanning D 0.4-1.6 in green
        green = np.linspace(0.4, 1.6, 400)
        frame.preview = np.stack([10.0 ** -green] * 3, axis=-1)[None, :, :].astype(np.float32)
    reliability = session.reliability()
    assert reliability.is_limited
    assert reliability.warning().startswith("Fit reliable over D 0.")

    session.add_point(_pt(paths[0], 0.5, x=300))
    session.add_point(_pt(paths[2], 1.55, x=300))
    assert session.reliability().warning() is None


def test_set_rolls_numbers_each_roll_from_one_and_keeps_them_apart(tmp_path):
    session = CalibrationSession()
    one, two = tmp_path / "one", tmp_path / "two"
    session.set_rolls([
        (one, [one / "a.tif", one / "b.tif"], [AT_1_30] * 2),
        (two, [two / "a.tif", two / "b.tif", two / "c.tif"], [AT_1_30] * 3),
    ])
    assert [f.number for f in session.frames] == [1, 2, 1, 2, 3]
    assert [f.roll for f in session.frames] == [one] * 2 + [two] * 3
    assert session.rolls == [one, two] and session.roll_folder is None
    session.set_rolls([(one, [one / "a.tif"], [AT_1_30])])
    assert session.roll_folder == one and session.rolls == [one]
    session.set_rolls([(None, [one / "a.tif"], [AT_1_30])])  # loose scans belong to their folder
    assert session.rolls == [one] and session.roll_folder is None
