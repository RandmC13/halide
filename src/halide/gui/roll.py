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

import copy
import warnings
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
    return preview, quiet_auto_estimate(preview), None


def quiet_auto_estimate(image: np.ndarray) -> DensityProfile | None:
    """calibration/auto.py's estimate for the Positive view before any points exist, or None. Its
    "candidates unusually close in density" warning is meant for `--auto-density` users on the
    command line; here the estimate is already labelled a rough estimate in the window, so the
    warning would only be terminal noise (it would print on launch for some frames)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return auto_density_balance(image)
        except ValueError:
            return None


@dataclass
class Frame:
    path: Path
    number: int  # 1-based position in its own roll (the contact sheet's frame number)
    scan: ScanSettings | None
    roll: Path = field(default_factory=Path)  # the folder it belongs to
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
    note: str | None = None  # e.g. the frame had no scan EXIF, so its point wasn't normalised
    roll_name: str | None = None  # the roll a point not on the open frames was picked on, if its frame still exists


@dataclass
class CalibrationSession:
    frames: list[Frame] = field(default_factory=list)
    roll_folder: Path | None = None
    points: list[NeutralPoint] = field(default_factory=list)
    selected: int | None = None
    details: dict[str, str] = field(default_factory=lambda: {k: "" for k in DETAIL_FIELDS})
    tone_override: ToneCurveParams | None = None  # the Print drawer's pinned exposure/grade, if any
    _reference_override: ScanSettings | None = None  # a reopened profile's recorded reference
    _undo: tuple[list[NeutralPoint], int | None] | None = None  # (points, selected) before the last change
    profile_in_use: Path | None = None  # a saved profile used on this roll ("Use for this roll")
    _profile_points: list[NeutralPoint] = field(default_factory=list)  # ... and the points it came with

    # --- the roll -----------------------------------------------------------------------------

    def reset(self) -> None:
        """A fresh calibration: loading a different roll starts over rather than mixing one roll's
        points into another's (the caller asks first when there are points to lose)."""
        self.points = []
        self.selected = None
        self.details = {k: "" for k in DETAIL_FIELDS}
        self.tone_override = None
        self._reference_override = None
        self._undo = None
        self.profile_in_use = None
        self._profile_points = []

    def use_profile(self, profile_path: Path) -> None:
        """Put a saved profile on the roll that's open, keeping its frames: the profile's points,
        extra information, print pin and scan reference come in, and points clicked from now on
        refine it. Raises as restore() does, leaving the session as it was."""
        before = copy.copy(self)
        self.reset()
        try:
            self.restore(profile_path)
        except Exception:
            self.__dict__.update(before.__dict__)
            raise
        self.profile_in_use = profile_path
        self._profile_points = list(self.points)

    def new_points(self) -> list[NeutralPoint]:
        """The points clicked on this roll on top of a profile in use (all of them without one)."""
        return [p for p in self.points if p not in self._profile_points]

    def changed_from_profile(self) -> bool:
        """The points differ from the profile in use: some added, or some of its own removed."""
        return self.profile_in_use is not None and self.points != self._profile_points

    def forget_profile(self) -> None:
        """The profile in use has gone (deleted): its points stay, as an unsaved calibration."""
        self.profile_in_use = None
        self._profile_points = []

    def profile_rolls(self) -> list[str]:
        """The names of the rolls the profile in use was picked on, in order of first appearance."""
        return list(dict.fromkeys(p.frame.parent.name for p in self._profile_points))

    def adopt_points(self, profile_path: Path) -> None:
        """After a save: every point now belongs to the profile at `profile_path`."""
        if self.profile_in_use is not None:
            self.profile_in_use = profile_path
            self._profile_points = list(self.points)

    def set_roll(self, paths: list[Path], scans: list[ScanSettings | None], folder: Path | None) -> None:
        self.set_rolls([(folder, paths, scans)])

    def set_rolls(self, rolls: list[tuple[Path | None, list[Path], list[ScanSettings | None]]]) -> None:
        """Load one or more rolls' frames, numbered from 1 within each. A roll given without a folder
        (loose scans) belongs to its first scan's folder. Points already in the session (a reopened
        profile) whose frame is neither loaded nor on disk but has a namesake in the rolls - the roll
        was moved or copied - are re-attached to that frame, so they're shown on it again."""
        self.frames = []
        folders = []
        for folder, paths, scans in rolls:
            paths = [anchors.absolute(p) for p in paths]
            folder = anchors.absolute(folder) if folder is not None else None
            folders.append(folder)
            roll = folder or paths[0].parent
            self.frames += [Frame(path=p, number=i + 1, scan=s, roll=roll) for i, (p, s) in enumerate(zip(paths, scans))]
        self.roll_folder = folders[0] if len(folders) == 1 else None
        # shortcut: two moved rolls with the same file names re-attach to the first; match per roll if that ever matters
        by_name: dict[str, Path] = {}
        for f in self.frames:
            by_name.setdefault(f.path.name, f.path)
        known = {f.path for f in self.frames}
        self.points = [
            replace(p, frame=by_name[p.frame.name])
            if p.frame not in known and not p.frame.is_file() and p.frame.name in by_name else p
            for p in self.points
        ]

    @property
    def rolls(self) -> list[Path]:
        """The folders of the open frames, in order."""
        return list(dict.fromkeys(f.roll for f in self.frames))

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
        self._remember()
        self.points.append(point)
        self.selected = len(self.points) - 1
        if duplicate is None:
            return None
        return (
            f"Close to point {duplicate + 1} in tone and colour — a different neutral object adds "
            "more information than another sample of the same one."
        )

    def remove_point(self, index: int) -> None:
        self._remember()
        del self.points[index]
        if self.selected is not None:
            if self.selected == index:
                self.selected = None
            elif self.selected > index:
                self.selected -= 1

    def clear_points(self, frame: Path | None = None) -> int:
        """Remove every point, or just those on `frame`. A profile in use keeps its own points.
        Returns how many were removed."""
        keep = [p for p in self.points if (frame is not None and p.frame != frame) or p in self._profile_points]
        removed = len(self.points) - len(keep)
        if removed:
            self._remember()
            selected = self.points[self.selected] if self.selected is not None else None
            self.points = keep
            self.selected = keep.index(selected) if selected in keep else None
        return removed

    def _remember(self) -> None:
        self._undo = (list(self.points), self.selected)

    def undo(self) -> int | None:
        """Put the points back as they were before the last add, remove or clear (one level only).
        Returns how many points that brought back (negative: an added point taken away), or None if
        there was nothing to undo."""
        if self._undo is None:
            return None
        before = len(self.points)
        (self.points, self.selected), self._undo = self._undo, None
        return len(self.points) - before

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

    def fit_problem(self) -> str | None:
        """Why the points can't be fitted although they span enough density (e.g. a colour layer
        the points don't pin down), in plain words for the status line - else None."""
        if not self.can_fit():
            return None
        try:
            anchors.fit(self.points, self.reference)
        except ValueError as exc:
            return str(exc)
        return None

    def reliability(self, wedge: tuple[float, float] | None = None) -> anchors.Reliability | None:
        """How far the live fit can be trusted over the roll's density range (`wedge`, if the
        caller already has wedge_range()) - None until there's a fit (fit_problem's refusals
        included) and the roll's previews have been measured."""
        wedge = self.wedge_range() if wedge is None else wedge
        return anchors.fit_reliability(self.points, self.reference, wedge)

    def views(self) -> list[PointView]:
        reference = self.reference
        agreements = anchors.agreement(self.points, reference)
        numbers = {f.path: f.number for f in self.frames}
        return [
            PointView(
                index=i,
                point=p,
                frame_number=numbers.get(p.frame),
                roll_name=p.frame.parent.name if p.frame not in numbers and p.frame.is_file() else None,
                green_density=anchors.green_density(anchors.normalised_rgb(p, reference)),
                agreement=agreements[i],
                # A point from a frame with no scan EXIF can't be matched to the roll's exposure,
                # so it counts as it was sampled. Only worth saying when the roll has one.
                note=(f"point {i + 1}: frame has no scan exposure data, used unnormalised"
                      if p.scan is None and reference is not None else None),
            )
            for i, p in enumerate(self.points)
        ]

    def density_at(self, frame: Frame, rgb) -> float:
        """The green density the point list would show for a point with this rgb on this frame."""
        point = anchors.NeutralPoint(frame=frame.path, x=0, y=0, rgb=tuple(float(v) for v in rgb), scan=frame.scan)
        return anchors.green_density(anchors.normalised_rgb(point, self.reference))

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
        }

    def restore(self, profile_path: Path) -> None:
        """Load a saved profile's picks, extra information, print override and scan reference into this
        session. Points whose frames are gone stay in the fit - their RGB is stored."""
        profile = load_profile(profile_path)
        self.points = [anchors.point_from_dict(r) for r in load_anchors(profile_path)]
        self.selected = None
        self._undo = None
        self.details = {k: getattr(profile, k) or "" for k in DETAIL_FIELDS}
        self.tone_override = load_tone_override(profile_path)
        self._reference_override = load_scan_reference(profile_path)

    def missing_frames(self) -> list[Path]:
        """Frames that points were picked on but that are neither in the loaded roll nor on disk
        where the profile recorded them - moved or deleted since. Their points still count in the
        fit (the RGB is stored); they just can't be shown on their frame."""
        loaded = {f.path for f in self.frames}
        missing = {p.frame for p in self.points if p.frame not in loaded and not p.frame.is_file()}
        return sorted(missing)
