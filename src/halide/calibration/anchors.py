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
(core.density.leave_one_out_residuals) and reported as a colour-printing filter value ("CC 7.9 R").
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from halide.calibration.auto import MIN_DENSITY_SEPARATION
from halide.calibration.scan_consistency import most_common_settings, scan_gain
from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.density import describe_cast, fit_density_balance, leave_one_out_residuals
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


@dataclass(frozen=True)
class NeutralPoint:
    frame: Path
    x: int  # full-resolution pixel coordinates
    y: int
    rgb: tuple[float, float, float]  # as sampled, at the frame's own scan exposure
    scan: ScanSettings | None  # the frame's scan exposure, None if its EXIF didn't say


@dataclass(frozen=True)
class Agreement:
    cc: float
    direction: str  # R/G/B excess or C/M/Y (a deficit of R/G/B), see core.density.describe_cast
    band: str  # "calm" | "amber" | "red"

    def label(self) -> str:
        return f"CC {self.cc:.0f} {self.direction}" if self.cc >= 0.5 else "CC 0"


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
    below MIN_POINTS_FOR_AGREEMENT, and for any point whose others can't be fitted."""
    if len(points) < MIN_POINTS_FOR_AGREEMENT:
        return [None] * len(points)
    residuals = leave_one_out_residuals([normalised_rgb(p, reference) for p in points])
    out: list[Agreement | None] = []
    for row in residuals:
        if np.isnan(row).any():
            out.append(None)
            continue
        cc, direction = describe_cast(row)
        out.append(Agreement(cc=cc, direction=direction, band=band(cc)))
    return out


def worst(agreements: Sequence[Agreement | None]) -> int | None:
    """Index of the point to question first: the one furthest from the others, if it's outside the
    calm band. Only this point gets the picker's "is this really neutral?" hint. One strong outlier
    tilts every other point's leave-one-out fit, so the good points read a few CC off in the
    opposite direction - and the one with the most leverage (the end of the density range) can
    itself reach amber (tests/unit/test_anchors.py). Deal with the worst first and the rest settle;
    hinting at all of them would send the user after good points."""
    scored = [(a.cc, i) for i, a in enumerate(agreements) if a is not None and a.band != "calm"]
    return max(scored)[1] if scored else None


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


def point_to_dict(point: NeutralPoint) -> dict:
    return {
        "frame": str(point.frame),
        "x": point.x,
        "y": point.y,
        "rgb": [float(v) for v in point.rgb],
        "scan": asdict(point.scan) if point.scan is not None else None,
    }


def point_from_dict(data: dict) -> NeutralPoint:
    scan = data.get("scan")
    return NeutralPoint(
        frame=Path(data["frame"]),
        x=int(data["x"]),
        y=int(data["y"]),
        rgb=tuple(float(v) for v in data["rgb"]),
        scan=ScanSettings(**scan) if scan else None,
    )
