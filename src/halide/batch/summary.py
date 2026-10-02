"""What happened to a roll, from its frames' BatchResults: the figures and flags the end-of-run
summary prints and the run log records (docs/specs/batch-summary-and-log.md). Pure: it prints
nothing and touches no files, so the screen and the log can't disagree."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from halide.cli.console.runsheet import RunSheet

# The user's choice (2026-10-02, option A): only a fitted grade well under the rest of the roll is
# worth a look. Checked on Roll 16: healthy frames 0.86-1.00 (median 1.00); the old badly cropped
# frames (an opaque film-holder strip in the crop) fitted ~0.46.
LOW_GRADE_MARGIN = 0.25
MIN_FRAMES_FOR_CHECK = 5  # a median of two or three frames says little
GRADE_CAP = 1.0  # fit_print caps the grade at the real paper's own
CHECK_HINT = "long tonal range, or something bright left in the crop?"
MAX_NAMES = 4

Row = tuple[str, str, str]  # (style, label, text); style is "row", "note", "ok" or "warn"


@dataclass(frozen=True)
class Flag:
    name: str
    contrast: float


@dataclass(frozen=True)
class RollSummary:
    kind: str  # "develop", "print" or "export"
    grades: list[float]  # fitted grades of developed frames
    exposures: list[float]  # exposures of frames with a fitted grade
    scales: list[float]  # flat (--output flat) scales
    flags: list[Flag]
    warning_groups: list[tuple[str, list[str]]]  # (message, frame names), in first-seen order
    service_stopped: str | None  # the reason, said once

    @classmethod
    def from_results(
        cls, results, kind: str, stopped: Callable[[str], str | None] = lambda w: None
    ) -> "RollSummary":
        """`stopped` recognises a "the GPU service stopped" frame warning and returns its reason
        (processing.service_stopped; passed in so this module stays free of the pipeline)."""
        developed = [r for r in results if r.error is None]
        fitted = [r for r in developed if r.contrast is not None and r.grade_fitted]
        grades = [r.contrast for r in fitted]
        flags = []
        if len(grades) >= MIN_FRAMES_FOR_CHECK:
            median = statistics.median(grades)
            # Rounded so 1.00 - 0.75 counts as exactly 0.25, not 0.24999999.
            flags = [Flag(r.job.input_path.name, r.contrast) for r in fitted
                     if round(median - r.contrast, 6) >= LOW_GRADE_MARGIN]
        groups: dict[str, list[str]] = {}
        stopped_reason = None
        for r in results:
            for part in (r.warning or "").split("; "):
                # Not every source strips the frame's own name; grouping needs it gone.
                for own in (f"{r.job.input_path}: ", f"{r.job.input_path.name}: "):
                    if part.startswith(own):
                        part = part[len(own):]
                        break
                if not part:
                    continue
                reason = stopped(part)
                if reason is not None:
                    stopped_reason = stopped_reason or reason
                else:
                    groups.setdefault(part, []).append(r.job.input_path.name)
        return cls(
            kind=kind,
            grades=grades,
            exposures=[r.exposure for r in fitted if r.exposure is not None],
            scales=[r.linear_scale for r in developed if r.linear_scale is not None],
            flags=flags,
            warning_groups=list(groups.items()),
            service_stopped=stopped_reason,
        )


def _span(values: list[float], fmt: str, joiner: str) -> str:
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo}{joiner}{hi}"


def _prints_text(s: RollSummary) -> str:
    median = statistics.median(s.grades)
    capped = sum(1 for g in s.grades if g >= GRADE_CAP - 1e-6)
    detail = f"median {median:.2f}" + (f", {capped} at the 1.0 cap" if capped else "")
    text = f"grade {_span(s.grades, '{:.2f}', '–')} ({detail})"
    if s.exposures:
        text += f"{RunSheet.SEP}exposure {_span(s.exposures, '{:+.2f}', ' to ')}"
    return text


def _group_text(message: str, names: list[str]) -> str:
    if len(names) == 1:
        return f"{names[0]}: {message}"
    shown = ", ".join(names[:MAX_NAMES])
    more = f" and {len(names) - MAX_NAMES} more" if len(names) > MAX_NAMES else ""
    return f"{len(names)} frames: {message} — {shown}{more}"


def summary_rows(s: RollSummary) -> list[Row]:
    """The rows printed under the end line (and recorded in the log): nothing at all when there's
    nothing to say."""
    rows: list[Row] = []
    if s.kind != "export":
        if s.grades:
            rows.append(("row", "Prints", _prints_text(s)))
        elif s.scales:
            rows.append(("row", "Prints", f"flat{RunSheet.SEP}scale {_span(s.scales, 'x{:.3g}', '–')}"))
    for flag in s.flags:
        rows.append(("warn", "Check", f"{flag.name}  grade {flag.contrast:.2f} — {CHECK_HINT}"))
    for message, names in s.warning_groups:
        rows.append(("warn", "Warnings", _group_text(message, names)))
    if s.service_stopped is not None:
        rows.append(("warn", "Warnings", f"The GPU service stopped ({s.service_stopped}); "
                                         "developing the remaining frames on the CPU"))
    return rows
