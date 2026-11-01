"""What the window's Develop runs: the real `halide batch` (and `halide export` for sharing copies)
as a command line, so a print developed in the window is the one the terminal would make. No Qt."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from halide.io.roll import RUN_LOG_NAME

DEFAULT_NAME = "{stem}"


@dataclass(frozen=True)
class DevelopSettings:
    roll: Path
    into: Path
    profile: Path
    output: str = "print"  # print | flat
    name: str = DEFAULT_NAME
    existing: str = "ask"  # ask | replace | skip
    match_scan_exposure: bool = False
    sharing: str = "none"  # none | jpg | png
    device: str | None = None  # None: halide's own default (or $HALIDE_DEVICE)

    @property
    def sharing_dir(self) -> Path:
        return self.into / "sharing"


def _batch_args(s: DevelopSettings) -> list[str]:
    """batch's arguments, passing only what differs from its defaults."""
    args = ["batch", str(s.roll), str(s.into), "--profile", str(s.profile)]
    if s.output != "print":
        args += ["--output", s.output]
    if s.name != DEFAULT_NAME:
        args.append(f"--name={s.name}")  # one token: a template may start with "-"
    if s.existing == "replace":
        args.append("--overwrite")
    elif s.existing == "skip":
        args.append("--skip-existing")
    if s.match_scan_exposure:
        args.append("--match-scan-exposure")
    if s.device:
        args += ["--device", s.device]
    return args


def _cli() -> list[str]:
    return [sys.executable, "-m", "halide.cli.main"]


def batch_command(s: DevelopSettings) -> list[str]:
    return _cli() + _batch_args(s)


def export_command(s: DevelopSettings) -> list[str]:
    """Sharing copies of the developed folder; its record goes into the same run log."""
    args = ["export", str(s.into), str(s.sharing_dir), "--format", s.sharing,
            "--log", str(s.into / RUN_LOG_NAME)]
    if s.existing == "replace":
        args.append("--overwrite")
    elif s.existing == "skip":
        args.append("--skip-existing")
    if s.device:
        args += ["--device", s.device]
    return _cli() + args


def shown_command(s: DevelopSettings) -> str:
    """The batch command as one line a person can paste into a terminal."""
    return shlex.join(["halide", *_batch_args(s)])


def _plan(s: DevelopSettings):
    """batch's own plan for this roll (its parser and its output naming), so the window never
    works out output names a second way. A refusal batch would make is a ValueError."""
    from halide.cli.commands.batch_cmd import _find_plans
    from halide.cli.main import build_parser

    try:
        args = build_parser().parse_args(_batch_args(s))
        plans, _ = _find_plans(args, s.roll)
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    return plans[0]


def existing_outputs(s: DevelopSettings) -> list[Path]:
    return [job.output_path for job in _plan(s).jobs if job.output_path.exists()]


def frame_count(s: DevelopSettings) -> int:
    return len(_plan(s).jobs)


def example_name(s: DevelopSettings) -> str:
    """One example for the Names field ("IMG_0138.tif → Roll16_01.tif"); a template batch would
    refuse is a ValueError with batch's own message."""
    return preview(s)[1]


def preview(s: DevelopSettings) -> tuple[int, str]:
    """(frames, example name) from one plan, for the panel to refresh on every keystroke."""
    jobs = _plan(s).jobs
    example = f"{jobs[0].input_path.name} → {jobs[0].output_path.name}" if jobs else ""
    return len(jobs), example
