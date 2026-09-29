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


def _point(green_density, scan=AT_1_30, deviation=(0.0, 0.0, 0.0), frame="/films/Roll16/IMG_0001.tif"):
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
    # prints warm: short of blue (Y) and with extra red (M + Y) - an orange pack, largest first
    assert wall.filters == "CC 11Y + 7M"
    assert wall.band == "red"
    assert anchors.worst(points, AT_1_30) == len(points) - 1

    # The outlier tilts every other point's leave-one-out fit, so good points read a few CC off in
    # the opposite direction (cyan, the wall's complement) - the lowest, with the most leverage, even
    # reaches amber. That's why only the worst point gets the hint: remove it and the rest settle.
    assert all(a.filters.startswith("CC ") and a.filters.split(" + ")[0].endswith("C") for a in result[:-1])
    assert not any("Y" in a.filters for a in result[:-1])
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
    assert anchors.Agreement(cc=7.9, filters="CC 8Y + 3M", band="amber").label() == "CC 8Y + 3M"
    assert anchors.Agreement(cc=0.2, filters="neutral", band="calm").label() == "neutral"


# --- fit reliability (D-2: the 0.1 D Save gate stays; the picker says where the fit holds) ----------

# Review 2.1-1's simulation: a film line (wb 2.2/1/1.4, s 1.3/1/0.78), three picked points centred
# on D 1.0, scored at the ends of a typical negative, D 0.4 and 1.6.
_SIM_WB = (2.2, 1.0, 1.4)
_SIM_SCALE = (1.3, 1.0, 0.78)
_SIM_WEDGE = (0.4, 1.6)


def _sim_rgb(green_density, noise=(0.0, 0.0, 0.0)):
    d = np.array([np.log10(_SIM_WB[c]) + green_density / _SIM_SCALE[c] for c in range(3)]) + np.asarray(noise)
    return tuple(10.0 ** -d)


def _sim_points(span, noise=None):
    greens = np.linspace(1.0 - span / 2, 1.0 + span / 2, 3)
    return [
        NeutralPoint(frame=Path("/r/a.tif"), x=0, y=0, rgb=_sim_rgb(g, (0, 0, 0) if noise is None else noise[i]), scan=None)
        for i, g in enumerate(greens)
    ]


def _simulated_worst_end_cc(span, trials=600, seed=7):
    """Median worst-end cast of fits through noisy picks (0.005 D per channel, green included)."""
    from halide.core.density import describe_cast, neutral_residuals

    rng = np.random.default_rng(seed)
    ends = [_sim_rgb(_SIM_WEDGE[0]), _sim_rgb(_SIM_WEDGE[1])]
    worst = []
    for _ in range(trials):
        points = _sim_points(span, rng.normal(0.0, anchors.ASSUMED_PICK_ERROR, (3, 3)))
        profile = fit_density_balance([p.rgb for p in points])
        worst.append(max(describe_cast(r)[0] for r in neutral_residuals(profile, ends)))
    return float(np.median(worst))


@pytest.mark.parametrize("span, table_median, over_amber", [(0.1, 7.7, True), (0.3, 2.9, False), (0.6, 1.8, False)])
def test_fit_reliability_reproduces_the_review_simulation(span, table_median, over_amber):
    reliability = anchors.fit_reliability(_sim_points(span), None, _SIM_WEDGE)
    assert reliability.is_limited is over_amber
    assert (reliability.worst_cc_at_ends > anchors.AGREEMENT_AMBER_CC) is over_amber
    # The covariance prediction tracks a Monte Carlo of real noisy fits (which also jitter green,
    # which the prediction leaves out) - and the review's own table.
    simulated = _simulated_worst_end_cc(span)
    assert reliability.worst_cc_at_ends == pytest.approx(simulated, rel=0.35)
    assert reliability.worst_cc_at_ends == pytest.approx(table_median, rel=0.35)


def test_a_narrow_fit_names_the_range_it_holds_over():
    reliability = anchors.fit_reliability(_sim_points(0.1), None, _SIM_WEDGE)
    low, high = reliability.reliable_range
    assert _SIM_WEDGE[0] < low < 0.95 and 1.05 < high < _SIM_WEDGE[1]  # wider than the points, short of the roll
    assert low + high == pytest.approx(2.0, abs=0.01)  # centred on the points
    assert reliability.warning() == (
        f"Fit reliable over D {low:.1f}-{high:.1f} only - add a point in the shadows or highlights "
        "for the ends of the roll"
    )


def test_a_well_spread_fit_holds_over_the_whole_roll_and_says_nothing():
    reliability = anchors.fit_reliability(_sim_points(0.6), None, _SIM_WEDGE)
    assert reliability.reliable_range == pytest.approx(_SIM_WEDGE)
    assert reliability.warning() is None


def test_fit_reliability_is_none_without_a_fit_or_a_wedge():
    assert anchors.fit_reliability(_sim_points(0.6), None, None) is None  # roll still measuring
    assert anchors.fit_reliability(_sim_points(0.6)[:1], None, _SIM_WEDGE) is None  # one point
    assert anchors.fit_reliability(_sim_points(0.05), None, _SIM_WEDGE) is None  # under the Save gate
    # A fit the fitter refuses (red layer nearly flat against green) - no reading, no exception.
    flat_red = [
        NeutralPoint(frame=Path("/r/a.tif"), x=0, y=0, rgb=(0.2, 10.0 ** -g, 10.0 ** -(g / 0.78)), scan=None)
        for g in (0.8, 1.0, 1.2)
    ]
    with pytest.raises(ValueError):
        fit_density_balance([p.rgb for p in flat_red])
    assert anchors.fit_reliability(flat_red, None, _SIM_WEDGE) is None


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


def test_point_dict_roundtrip(tmp_path):
    point = _point(1.2, AT_1_60)
    assert anchors.point_from_dict(anchors.point_to_dict(point)) == point
    no_exif = NeutralPoint(frame=tmp_path / "a.tif", x=1, y=2, rgb=(0.1, 0.2, 0.3), scan=None)
    assert anchors.point_from_dict(anchors.point_to_dict(no_exif)) == no_exif


def test_a_relative_frame_is_stored_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "roll").mkdir()
    (tmp_path / "roll" / "a.tif").write_bytes(b"")
    relative = NeutralPoint(frame=Path("roll/a.tif"), x=1, y=2, rgb=(0.1, 0.2, 0.3), scan=None)
    assert anchors.point_to_dict(relative)["frame"] == str(tmp_path / "roll" / "a.tif")


def test_an_unfound_relative_frame_is_kept_as_recorded_not_guessed(tmp_path, monkeypatch):
    # An older profile opened from a different folder than it was made in: which folder its relative
    # path meant is unknowable, so saving must not bake the current folder into it.
    monkeypatch.chdir(tmp_path)
    record = {"frame": "pre-processed/a.tif", "x": 1, "y": 2, "rgb": [0.1, 0.2, 0.3], "scan": None}
    point = anchors.point_from_dict(record)
    assert point.frame == Path("pre-processed/a.tif")
    assert anchors.point_to_dict(point)["frame"] == "pre-processed/a.tif"


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
