"""The calibration picker's session state, with no Qt: the roll's frames, the neutral points picked on
them, the live fit and what the picker says about it, and what a saved profile records. The widgets
(gui/main_window.py and friends) only display this and forward clicks to it - the same split as
gui/sampling.py, so the picking logic is unit-testable without a display.

Frames load in the background (gui/loaders.py) as small working-space previews
(`load_frame_preview`, run in worker processes) - enough for the filmstrip, the step wedge's roll
density range and the proof sheet's draft. The frame being picked on is loaded at full resolution
separately, since picks sample full-resolution pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from halide.calibration import anchors
from halide.calibration.anchors import Agreement, NeutralPoint
from halide.calibration.auto import auto_density_balance
from halide.calibration.profile_store import load_anchors, load_profile, load_scan_reference, load_tone_override
from halide.calibration.scan_consistency import scan_gain
from halide.core._constants import MIN_TRANSMITTANCE
from halide.core.types import DensityProfile, ToneCurveParams
from halide.io.scan_metadata import ScanSettings

PREVIEW_LONG_EDGE = 320  # filmstrip / proof-draft previews: small, but enough for a readable draft
DETAIL_FIELDS = ("film_stock", "process", "scanner", "notes")


def load_frame_preview(path: str) -> tuple[np.ndarray | None, DensityProfile | None, str | None]:
    """Worker-process job: a frame's small linear working-space preview (block-averaged in linear
    light, as contact-sheet thumbnails are) and its own auto-density estimate for the Positive view
    before any points exist. Returns (preview, estimate, error)."""
    from halide.io.contact_sheet import downsample_linear  # worker-side import, keeps this module light
    from halide.processing import load_working_space_image

    try:
        preview = downsample_linear(load_working_space_image(path), PREVIEW_LONG_EDGE)
    except Exception as exc:  # noqa: BLE001 - one unreadable frame shouldn't stop the roll
        return None, None, str(exc)
    try:
        estimate = auto_density_balance(preview)
    except ValueError:
        estimate = None
    return preview, estimate, None


@dataclass
class Frame:
    path: Path
    number: int  # 1-based position in the roll (the contact sheet's frame number)
    scan: ScanSettings | None
    preview: np.ndarray | None = None  # at the frame's own scan exposure
    estimate: DensityProfile | None = None
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.preview is not None or self.error is not None


@dataclass
class PointView:
    """One row of the point list: everything the list, markers and step wedge show for a point."""

    index: int
    point: NeutralPoint
    frame_number: int | None  # None: the frame isn't in the loaded roll ("frame missing")
    green_density: float  # at the reference scan exposure
    agreement: Agreement | None


@dataclass
class CalibrationSession:
    frames: list[Frame] = field(default_factory=list)
    roll_folder: Path | None = None
    points: list[NeutralPoint] = field(default_factory=list)
    selected: int | None = None
    details: dict[str, str] = field(default_factory=lambda: {k: "" for k in DETAIL_FIELDS})
    tone_override: ToneCurveParams | None = None  # the Print drawer's pinned exposure/grade, if any
    _reference_override: ScanSettings | None = None  # a reopened profile's recorded reference

    # --- the roll -----------------------------------------------------------------------------

    def reset(self) -> None:
        """A fresh calibration: loading a different roll starts over rather than mixing one roll's
        points into another's (the caller asks first when there are points to lose)."""
        self.points = []
        self.selected = None
        self.details = {k: "" for k in DETAIL_FIELDS}
        self.tone_override = None
        self._reference_override = None

    def set_roll(self, paths: list[Path], scans: list[ScanSettings | None], folder: Path | None) -> None:
        """Load a roll's frames. Points already in the session (a reopened profile) whose frame
        isn't at its recorded path but has a namesake in this roll - the roll was moved or copied -
        are re-attached to that frame, so they're shown on it again."""
        self.frames = [Frame(path=p, number=i + 1, scan=s) for i, (p, s) in enumerate(zip(paths, scans))]
        self.roll_folder = folder
        by_name = {p.name: p for p in paths}
        known = set(paths)
        self.points = [
            replace(p, frame=by_name[p.frame.name]) if p.frame not in known and p.frame.name in by_name else p
            for p in self.points
        ]

    def frame_for(self, path: Path) -> Frame | None:
        for frame in self.frames:
            if frame.path == path:
                return frame
        return None

    @property
    def reference(self) -> ScanSettings | None:
        """The scan exposure every point and the fit are normalised to: a reopened profile's own
        recorded reference (so reopening and saving again doesn't silently move it), else the roll's
        most common setting (batch's fallback), else the points' own."""
        if self._reference_override is not None:
            return self._reference_override
        if self.frames:
            return anchors.reference_scan(f.scan for f in self.frames)
        return anchors.reference_scan(p.scan for p in self.points)

    def gain(self, scan: ScanSettings | None) -> float:
        reference = self.reference
        return scan_gain(scan, reference) if scan is not None and reference is not None else 1.0

    def wedge_range(self) -> tuple[float, float] | None:
        greens = [
            -np.log10(np.maximum(f.preview[..., 1] * self.gain(f.scan), MIN_TRANSMITTANCE))
            for f in self.frames
            if f.preview is not None
        ]
        return anchors.wedge_range(greens)

    # --- points -------------------------------------------------------------------------------

    def add_point(self, point: NeutralPoint) -> str | None:
        """Add a point (never refused) and select it. Returns the nudge to show, if it adds little."""
        duplicate = anchors.near_duplicate(point, self.points, self.reference)
        self.points.append(point)
        self.selected = len(self.points) - 1
        if duplicate is None:
            return None
        return (
            f"Close to point {duplicate + 1} in tone and colour — a different neutral object adds "
            "more information than another sample of the same one."
        )

    def remove_point(self, index: int) -> None:
        del self.points[index]
        if self.selected is not None:
            if self.selected == index:
                self.selected = None
            elif self.selected > index:
                self.selected -= 1

    def point_at(self, frame: Path, x: int, y: int, radius: int) -> int | None:
        """The point on `frame` within `radius` full-resolution pixels of (x, y) - a click there
        selects it rather than adding a new one. Nearest wins."""
        best, best_d2 = None, radius * radius
        for i, p in enumerate(self.points):
            if p.frame != frame:
                continue
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 <= best_d2:
                best, best_d2 = i, d2
        return best

    def can_fit(self) -> bool:
        return anchors.can_fit(self.points, self.reference)

    def profile(self) -> DensityProfile | None:
        """The live fit, once there are enough points - else None (the Positive view then shows
        the auto estimate)."""
        if not self.can_fit():
            return None
        try:
            return anchors.fit(self.points, self.reference)
        except ValueError:
            return None

    def views(self) -> list[PointView]:
        reference = self.reference
        agreements = anchors.agreement(self.points, reference)
        numbers = {f.path: f.number for f in self.frames}
        return [
            PointView(
                index=i,
                point=p,
                frame_number=numbers.get(p.frame),
                green_density=anchors.green_density(anchors.normalised_rgb(p, reference)),
                agreement=agreements[i],
            )
            for i, p in enumerate(self.points)
        ]

    def worst(self) -> int | None:
        return anchors.worst(self.points, self.reference)

    def point_counts(self) -> dict[Path, int]:
        counts: dict[Path, int] = {}
        for p in self.points:
            counts[p.frame] = counts.get(p.frame, 0) + 1
        return counts

    # --- Positive view ------------------------------------------------------------------------

    def positive_source(self, frame: Frame, full_estimate: DensityProfile | None = None) -> tuple[DensityProfile | None, bool, float]:
        """(profile, is_fit, scan gain) to print `frame` with: the live fit at the reference scan
        exposure once there is one, else the frame's own auto estimate (no gain - it was measured on
        the frame as scanned). `full_estimate` prefers an estimate measured on the frame at display
        resolution over its small preview's, when the caller has one."""
        fitted = self.profile()
        if fitted is not None:
            return fitted, True, self.gain(frame.scan)
        return (full_estimate or frame.estimate), False, 1.0

    def print_tone(self) -> ToneCurveParams:
        return self.tone_override or ToneCurveParams()

    # --- save / reopen ------------------------------------------------------------------------

    def profile_to_save(self) -> DensityProfile:
        fitted = self.profile()
        if fitted is None:
            raise ValueError("not enough neutral points to save a calibration yet")
        return replace(fitted, source="manual", **{k: (self.details.get(k) or None) for k in DETAIL_FIELDS})

    def sidecars(self) -> dict:
        """Keyword arguments for profile_store.save_named_profile besides the profile itself."""
        return {
            "tone": self.tone_override,
            "scan": self.reference,
            "anchors": [anchors.point_to_dict(p) for p in self.points],
            "roll": str(self.roll_folder) if self.roll_folder else None,
        }

    def restore(self, profile_path: Path) -> Path | None:
        """Load a saved profile's picks, roll details, print override and scan reference into this
        session. Returns the roll folder it recorded, if that folder still exists (the caller loads
        it); points whose frames are gone stay in the fit - their RGB is stored."""
        profile = load_profile(profile_path)
        records, roll = load_anchors(profile_path)
        self.points = [anchors.point_from_dict(r) for r in records]
        self.selected = None
        self.details = {k: getattr(profile, k) or "" for k in DETAIL_FIELDS}
        self.tone_override = load_tone_override(profile_path)
        self._reference_override = load_scan_reference(profile_path)
        folder = Path(roll) if roll else None
        return folder if folder is not None and folder.is_dir() else None
