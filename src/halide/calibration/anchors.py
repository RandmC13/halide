"""The calibration picker's model: the neutral points a user has clicked, across any number of frames
of a roll, and everything the picker says about them - pure, no Qt (gui/ only displays this).

A point stores the frame it came from, where, its raw (working-space) RGB, and that frame's scan
exposure. The RGB is kept as sampled; each point is normalised to the roll's *reference* scan
exposure only when fitting (`normalised_rgb`), because frames of one roll are often digitized at
different camera exposures (Roll 16: 1/25-1/60) and density balance is a power function - a scan
brighter by k shifts a pixel's colour, not just its brightness (see calibration/scan_consistency.py).
That's the same one global multiply --match-scan-exposure makes, against the same reference the
saved profile records, so a profile fitted here develops the roll correctly with that flag.

Agreement is judged per point against the fit through the *other* points
(core.density.leave_one_out_residuals) and reported as a colour-printing filter pack ("CC 8M + 8Y").

How far the fit can be trusted is judged separately (fit_reliability): points that agree with each
other can still all sit in one narrow band of density, and the line through them is then a guess at
the ends of the roll.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from halide.calibration.auto import MIN_DENSITY_SEPARATION
from halide.calibration.scan_consistency import most_common_settings, scan_gain
from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.density import (
    describe_cast,
    fit_density_balance,
    leave_one_out_residuals,
    neutral_residuals,
    predicted_cast_error,
)
from halide.core.types import DensityProfile
from halide.io.scan_metadata import ScanSettings

# Agreement bands, in CC (density x 100) from what the other points agree neutral is. Calibrated on
# Roll 16: trusted whites (trike, car, T-shirt, clouds) read CC 0.8-4.8 against each other; the two
# objects the investigation had already found suspect - a warm-lit cloud edge (0147) and the cream
# Alhambra wall (0159) - read CC 9.6 Y and CC 7.9 R. So "amber" is where a point deserves a second
# look, and "red" is a point that almost certainly isn't neutral.
AGREEMENT_AMBER_CC = 5.0
AGREEMENT_RED_CC = 10.0
MIN_POINTS_FOR_AGREEMENT = 3  # with two points the line passes through both - nothing to disagree with

# Near-duplicate nudge: a new point this close in green density AND colour (raw density
# differences between channels, so it needs no fit) to an existing one adds little information.
DUPLICATE_DENSITY = 0.05
DUPLICATE_CHROMA = 0.02

# The step wedge spans the roll's own density range: the film-base end and the highlight end the
# print fit anchors to (core/tone_render.py::fit_print uses the 99.5th percentile).
WEDGE_LOW_PERCENTILE = 0.5
WEDGE_HIGH_PERCENTILE = 99.5

# The error assumed in each picked point's density, per channel, when predicting how far the fit
# can be trusted (fit_reliability): CC 0.5 - well below the CC 1-5 real "trusted" whites were
# measured off neutral on Roll 16, so the prediction is the best case, not a worst one.
ASSUMED_PICK_ERROR = 0.005
_RELIABILITY_STEPS = 1001  # D_G samples across the wedge when finding the reliable range


@dataclass(frozen=True)
class NeutralPoint:
    frame: Path
    x: int  # full-resolution pixel coordinates
    y: int
    rgb: tuple[float, float, float]  # as sampled, at the frame's own scan exposure
    scan: ScanSettings | None  # the frame's scan exposure, None if its EXIF didn't say


@dataclass(frozen=True)
class Agreement:
    cc: float  # the largest filter in the pack - what the bands are judged on
    filters: str  # "CC 20Y + 10M" or "neutral", see core.density.describe_cast
    band: str  # "calm" | "amber" | "red"

    def label(self) -> str:
        return self.filters


@dataclass(frozen=True)
class Reliability:
    """How far a fit can be trusted across the roll (fit_reliability)."""

    worst_cc_at_ends: float  # predicted cast at whichever end of the wedge is worse
    reliable_range: tuple[float, float]  # green density span where the prediction stays <= CC 5
    reliable_anywhere: bool = True  # False: nowhere on the roll (reliable_range is then just the best spot)

    @property
    def is_limited(self) -> bool:
        return self.worst_cc_at_ends > AGREEMENT_AMBER_CC

    def warning(self) -> str | None:
        """The picker's note under the step wedge (and the CLI's, on saving a picked profile)."""
        if not self.is_limited:
            return None
        if not self.reliable_anywhere:
            return (
                "Fit not reliable anywhere on the roll - add points spread from the shadows to the "
                "highlights"
            )
        low, high = self.reliable_range
        return (
            f"Fit reliable over D {low:.1f}-{high:.1f} only - add a point in the shadows or "
            "highlights for the ends of the roll"
        )


def reference_scan(settings: Iterable[ScanSettings | None]) -> ScanSettings | None:
    """The roll's reference scan exposure: its most common setting (ties -> brighter), the same
    fallback `batch --match-scan-exposure` uses, so the fitted profile and a later batch agree."""
    known = [s for s in settings if s is not None]
    return most_common_settings(known) if known else None


def normalised_rgb(point: NeutralPoint, reference: ScanSettings | None) -> tuple[float, float, float]:
    if point.scan is None or reference is None:
        return point.rgb
    gain = scan_gain(point.scan, reference)
    return tuple(float(v) * gain for v in point.rgb)


def green_density(rgb: Sequence[float]) -> float:
    return float(np.log10(1.0 / max(float(rgb[1]), MIN_TRANSMITTANCE)))


def can_fit(points: Sequence[NeutralPoint], reference: ScanSettings | None) -> bool:
    """At least two points spanning MIN_DENSITY_SEPARATION of green density - the same minimum the
    auto tier warns below (calibration/auto.py): closer than that and the slope is mostly noise."""
    if len(points) < 2:
        return False
    greens = [green_density(normalised_rgb(p, reference)) for p in points]
    return max(greens) - min(greens) >= MIN_DENSITY_SEPARATION


def fit(points: Sequence[NeutralPoint], reference: ScanSettings | None) -> DensityProfile:
    return fit_density_balance([normalised_rgb(p, reference) for p in points])


def band(cc: float) -> str:
    if cc > AGREEMENT_RED_CC:
        return "red"
    if cc > AGREEMENT_AMBER_CC:
        return "amber"
    return "calm"


def agreement(points: Sequence[NeutralPoint], reference: ScanSettings | None) -> list[Agreement | None]:
    """Per point, how far it is from what the other points agree neutral is. None for every point
    below MIN_POINTS_FOR_AGREEMENT, and for any point whose others can't support a fit of their own:
    fewer than two of them, or spanning less than MIN_DENSITY_SEPARATION (the Save gate's minimum).
    Without that, a line through two nearly-equal-density points extrapolated to judge a third far
    away - found on Roll 16, where the trike (D 1.22) read "CC 8 M" against a car and T-shirt only
    0.06 D apart, an artefact of the extrapolation, not a real disagreement."""
    if len(points) < MIN_POINTS_FOR_AGREEMENT:
        return [None] * len(points)
    residuals = leave_one_out_residuals([normalised_rgb(p, reference) for p in points])
    out: list[Agreement | None] = []
    for i, row in enumerate(residuals):
        others = [p for j, p in enumerate(points) if j != i]
        if np.isnan(row).any() or not can_fit(others, reference):
            out.append(None)
            continue
        cc, filters = describe_cast(row)
        out.append(Agreement(cc=cc, filters=filters, band=band(cc)))
    return out


def fit_reliability(
    points: Sequence[NeutralPoint], reference: ScanSettings | None, wedge: tuple[float, float] | None
) -> Reliability | None:
    """How far the fit through `points` can be trusted over the roll's density range `wedge`
    (wedge_range): the cast a true neutral is predicted to print with at the wedge's ends if every
    pick was off by ASSUMED_PICK_ERROR (core.density.predicted_cast_error), and the span of green
    density where that stays within the calm agreement band (CC 5).

    Why: the Save gate only asks for MIN_DENSITY_SEPARATION (0.1 D) between points, and three
    points that close, agreeing perfectly with each other, still leave the line's tilt so loosely
    pinned that the ends of a typical roll can print CC 8-15 off (codebase review 2.1-1's
    simulation). Saving stays allowed; this is what the warning reports.

    None when there's nothing to judge: no fit yet (too few points, or one the fit refuses) or no
    wedge (the roll's previews haven't loaded)."""
    if wedge is None or not can_fit(points, reference):
        return None
    rgbs = [normalised_rgb(p, reference) for p in points]
    try:
        profile = fit_density_balance(rgbs)
    except ValueError:
        return None
    low, high = wedge
    greens = np.linspace(low, high, _RELIABILITY_STEPS)
    cc = predicted_cast_error(rgbs, profile, greens, ASSUMED_PICK_ERROR)
    within = np.flatnonzero(cc <= AGREEMENT_AMBER_CC)
    if within.size:  # the prediction is convex in D_G, so this is one contiguous span
        reliable = (float(greens[within[0]]), float(greens[within[-1]]))
    else:  # nowhere on the roll - keep the best spot, but the note says "nowhere", not a 1-point range
        best = float(greens[int(np.argmin(cc))])
        reliable = (best, best)
    return Reliability(
        worst_cc_at_ends=float(max(cc[0], cc[-1])), reliable_range=reliable, reliable_anywhere=bool(within.size)
    )


def _spread_without(points: Sequence[NeutralPoint], index: int, reference: ScanSettings | None) -> float:
    """RMS disagreement (CC) of every point except `index` against the fit through them."""
    others = [p for j, p in enumerate(points) if j != index]
    if not can_fit(others, reference):
        return float("inf")
    rgbs = [normalised_rgb(p, reference) for p in others]
    try:
        residuals = neutral_residuals(fit_density_balance(rgbs), rgbs)
    except ValueError:
        return float("inf")
    return float(np.sqrt(np.mean([describe_cast(r)[0] ** 2 for r in residuals])))


def worst(points: Sequence[NeutralPoint], reference: ScanSettings | None) -> int | None:
    """Index of the point to question first - the only one that gets the picker's "was this really
    neutral?" hint - or None when every point is calm.

    Not simply the point furthest from the others: a point at the end of the density range is
    judged against a line extrapolated from the rest, so while an outlier sits among those rest the
    good end point can read further off than the outlier itself (tests/unit/test_gui_roll.py: with
    three good points and one cream-wall-like outlier, the lowest good point read worst). Instead:
    of the points outside the calm band, the one whose removal leaves the others agreeing best -
    remove the real outlier and the rest are consistent; remove a good point and the outlier is
    still among them. Ties go to the point further from the others."""
    agreements = agreement(points, reference)
    candidates = [i for i, a in enumerate(agreements) if a is not None and a.band != "calm"]
    if not candidates:
        return None
    return min(candidates, key=lambda i: (_spread_without(points, i, reference), -agreements[i].cc))


def near_duplicate(
    candidate: NeutralPoint, points: Sequence[NeutralPoint], reference: ScanSettings | None
) -> int | None:
    """Index of an existing point the candidate is nearly the same as (tone and colour), if any -
    for the picker's "a different object adds more" nudge. Never used to refuse a pick."""
    new = np.log10(1.0 / np.maximum(normalised_rgb(candidate, reference), MIN_TRANSMITTANCE))
    for i, point in enumerate(points):
        old = np.log10(1.0 / np.maximum(normalised_rgb(point, reference), MIN_TRANSMITTANCE))
        delta = new - old
        if abs(delta[1]) < DUPLICATE_DENSITY and max(abs(delta[0] - delta[1]), abs(delta[2] - delta[1])) < DUPLICATE_CHROMA:
            return i
    return None


def wedge_range(green_densities: Iterable[np.ndarray]) -> tuple[float, float] | None:
    """(film-base end, highlight end) of the roll's green density, pooled over the given frames
    (the filmstrip's downsampled frames, already normalised to the reference scan exposure)."""
    arrays = [np.asarray(a, dtype=np.float64).ravel() for a in green_densities]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return None
    pooled = np.concatenate(arrays)
    low, high = np.percentile(pooled, [WEDGE_LOW_PERCENTILE, WEDGE_HIGH_PERCENTILE])
    return float(low), float(high)


# --- persistence (the profile's "anchors" sidecar, see calibration/profile_store.py) ------------------


def absolute(path: str | Path) -> Path:
    """`path` made absolute against the current folder, as typed otherwise (symlinks kept, unlike
    Path.resolve). Profiles record frames and the roll this way so they reopen from any folder - a
    relative path only meant something in the folder `halide calibrate` was started from."""
    return Path(os.path.abspath(path))


def recorded_path(path: str | Path) -> Path:
    """A path as a profile should record it: absolute, unless it's a relative path from an older
    profile that doesn't exist from the current folder. That one is kept as it was recorded - which
    folder it was relative to is unknowable, and making it absolute here would bake a wrong guess
    (the current folder) into the profile when it's saved again."""
    path = Path(path)
    return absolute(path) if path.is_absolute() or path.exists() else path


def point_to_dict(point: NeutralPoint) -> dict:
    return {
        "frame": str(recorded_path(point.frame)),
        "x": point.x,
        "y": point.y,
        "rgb": [float(v) for v in point.rgb],
        "scan": asdict(point.scan) if point.scan is not None else None,
    }


def point_from_dict(data: dict) -> NeutralPoint:
    scan = data.get("scan")
    return NeutralPoint(
        frame=recorded_path(data["frame"]),
        x=int(data["x"]),
        y=int(data["y"]),
        rgb=tuple(float(v) for v in data["rgb"]),
        scan=ScanSettings(**scan) if scan else None,
    )
