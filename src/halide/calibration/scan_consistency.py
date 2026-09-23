"""Is a roll's set of scans consistent enough for one calibration profile to be exactly valid across
it — and, optionally, the one correction that *can* be made exactly when it isn't.

Why consistency matters (see io/scan_metadata.py for the full reasoning): density balance is a
per-channel power function, so a scan-brightness factor k becomes a per-channel factor
k**density_scale after it — a colour shift, not just brightness. The same goes for a per-frame raw
white-balance difference.

What can be corrected, and what can't:
  - A difference in the digitizing camera's exposure (shutter/aperture/ISO) is one global multiply
    on linear sensor data. It commutes with every colour matrix downstream, so scaling the frame by
    the exposure ratio (scan_gain) undoes it exactly (up to shutter-speed accuracy). That's the
    optional --match-scan-exposure correction.
  - A raw white-balance difference is a per-channel multiply in the *camera's* colour space, applied
    before the raw converter's camera matrix. Undoing it in the exported colour space would need
    that matrix, which the TIFF doesn't carry — so it's only reported, never "corrected" with an
    approximation. The fix is at export: one fixed white balance for the whole roll.
  - Tone/colour modules active in the raw converter (curves, levels, shadows & highlights,
    filmic/sigmoid, negadoctor...) make the export non-linear. Not correctable after the fact.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from halide.io.scan_metadata import DarktableState, ScanSettings

SCANNING_GUIDANCE = (
    "For one calibration to be exactly valid across a roll, digitize every frame the same way: "
    "the same *manual* camera exposure (shutter, aperture, ISO — set once per roll, as bright as "
    "possible without clipping the film base), a fixed white balance (not auto), and export every "
    "frame with the same raw white balance and no tone/colour modules (only crop/rotate, lens "
    "correction, denoise and dust removal), as a linear, profile-embedded TIFF."
)

# Differences below these are EXIF/float rounding, not a real difference in how a frame was made.
_EXPOSURE_TOLERANCE_STOPS = 0.05
_WHITE_BALANCE_TOLERANCE = 0.005  # relative, per channel


def scan_gain(frame: ScanSettings, reference: ScanSettings) -> float:
    """Multiply a frame's linear pixels by this to put it at the reference's scan exposure."""
    return reference.relative_exposure / frame.relative_exposure


def most_common_settings(settings: list[ScanSettings]) -> ScanSettings:
    """The roll's most common scan settings (ties -> the brighter one) — the reference that leaves
    the most frames untouched when no profile-recorded reference is available."""
    counts = Counter(settings)
    return max(counts, key=lambda s: (counts[s], s.relative_exposure))


def _stops(a: float, b: float) -> float:
    return abs(math.log2(a / b))


@dataclass
class RollScanReport:
    settings: dict[str, ScanSettings | None]
    darktable: dict[str, DarktableState | None]
    exposure_groups: dict[str, list[str]] = field(default_factory=dict)
    exposure_spread_stops: float = 0.0
    white_balance_groups: dict[tuple[float, float, float], list[str]] = field(default_factory=dict)
    white_balance_spread: tuple[float, float, float] = (0.0, 0.0, 0.0)  # max/min - 1, per channel
    tonal_modules: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing_exif: list[str] = field(default_factory=list)

    @property
    def exposure_inconsistent(self) -> bool:
        return self.exposure_spread_stops > _EXPOSURE_TOLERANCE_STOPS

    @property
    def white_balance_inconsistent(self) -> bool:
        return max(self.white_balance_spread) > _WHITE_BALANCE_TOLERANCE

    @property
    def has_issues(self) -> bool:
        return self.exposure_inconsistent or self.white_balance_inconsistent or bool(self.tonal_modules)

    def summary_lines(self) -> list[str]:
        """Short, one-line-per-problem warnings (what `batch` prints before starting)."""
        lines = []
        if self.exposure_inconsistent:
            lines.append(
                f"frames were digitized at {len(self.exposure_groups)} different camera exposures "
                f"({self.exposure_spread_stops:.1f} stops apart) — one profile will shift colour "
                "from frame to frame, visibly so for a stop or more (--match-scan-exposure corrects this)"
            )
        if self.white_balance_inconsistent:
            spread = ", ".join(f"{c} {v * 100:.0f}%" for c, v in zip("RGB", self.white_balance_spread) if v > 0)
            lines.append(
                f"frames were exported with {len(self.white_balance_groups)} different raw white "
                f"balances ({spread} apart) — this shifts colour frame to frame and can't be "
                "corrected here; re-export with one fixed white balance"
            )
        if self.tonal_modules:
            names = ", ".join(sorted(self.tonal_modules))
            lines.append(f"tone/colour modules were active in darktable for: {names} — those exports aren't linear")
        return lines


def assess_roll(metadata: dict[str, tuple[ScanSettings | None, DarktableState | None]]) -> RollScanReport:
    settings = {name: s for name, (s, _) in metadata.items()}
    darktable = {name: d for name, (_, d) in metadata.items()}
    report = RollScanReport(settings=settings, darktable=darktable)

    known = {name: s for name, s in settings.items() if s is not None}
    report.missing_exif = sorted(name for name, s in settings.items() if s is None)
    for name, s in sorted(known.items()):
        report.exposure_groups.setdefault(s.describe(), []).append(name)
    if known:
        exposures = [s.relative_exposure for s in known.values()]
        report.exposure_spread_stops = _stops(max(exposures), min(exposures))

    balances = {name: d.white_balance for name, d in darktable.items() if d is not None and d.white_balance}
    for name, wb in sorted(balances.items()):
        report.white_balance_groups.setdefault(tuple(round(v, 4) for v in wb), []).append(name)
    if balances:
        channels = list(zip(*balances.values()))
        report.white_balance_spread = tuple(max(c) / min(c) - 1.0 for c in channels)

    report.tonal_modules = {name: d.tonal_modules for name, d in darktable.items() if d is not None and d.tonal_modules}
    return report
