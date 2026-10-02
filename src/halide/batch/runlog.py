"""The run log: a darkroom notebook appended to `halide-log.txt` in the output folder (or
`--log FILE`), one block per run (docs/specs/batch-summary-and-log.md).

Written as the run goes, not at the end — the runs that die are the ones you most want a record
of: (1) heading, command and run sheet before any frame develops; (2) one line per frame as it
finishes, flushed; (3) end line, summary and a closing rule. A run killed outright leaves a block
with no closing rule, and the next run says so. Writing never raises: the first failure is kept
in `error` (shown on the summary's Log row) and later writes are skipped."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from halide.cli.console import ICON_FAIL, ICON_OK, ICON_WARN, human_time

RULE_WIDTH = 100
LABEL_WIDTH = 15  # the run sheet's label column (RunSheet.LABEL_WIDTH)
_TAIL_BYTES = 256 * 1024  # enough for the last block of any real roll
_HEAD = "━━ "
_FINISHED = "━━ finished "
_ICONS = {"warn": ICON_WARN, "ok": ICON_OK}
_RESULT_WORD = {"develop": "developed", "print": "printed", "export": "exported"}


def _rule(text: str) -> str:
    return f"{_HEAD}{text} ".ljust(RULE_WIDTH, "━")


def plain_row_lines(rows) -> list[str]:
    """Run-sheet rows as plain text, unwrapped: a label once per run of equal labels, an icon
    for warnings and confirmations."""
    lines, last = [], None
    for style, label, text in rows:
        shown = "" if label == last else label
        icon = f"{_ICONS[style]} " if style in _ICONS else ""
        first, *rest = str(text).splitlines() or [""]
        lines.append(f"  {shown:<{LABEL_WIDTH}}{icon}{first}")
        lines += [f"  {'':<{LABEL_WIDTH}}{' ' * len(icon)}{more}" for more in rest]
        last = label
    return lines


def _one_line(text: str) -> str:
    """A frame line is one line: fold line breaks in a message into " / "."""
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


def previous_unfinished(path: Path) -> str | None:
    """The start time ("2026-10-02 14:31") of the log's last block if it has no closing rule —
    the run was killed — else None. Reads only the file's tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith(_FINISHED):
            return None
        if line.startswith(_HEAD):
            return line[len(_HEAD):].split(" · ")[0]
    return None


def _reason(path: Path, exc: Exception) -> str:
    if isinstance(exc, OSError):
        return f"couldn't write {path}: {exc.strerror or exc}"
    return f"couldn't write {path}: {type(exc).__name__}: {exc}"


class RunLog:
    def __init__(self, path: Path, kind: str, now: Callable[[], datetime] = datetime.now):
        self.path = Path(path)
        self.kind = kind  # "develop", "print" or "export"
        self._now = now
        self._file = None
        self.error: str | None = None
        self._width = len("Frame")

    def _write(self, lines: list[str], sync: bool = False) -> None:
        if self._file is None:
            return
        try:
            self._file.write("".join(f"{line.rstrip()}\n" for line in lines))
            self._file.flush()
            if sync:  # a power cut should leave every finished frame on disk
                os.fsync(self._file.fileno())
        except (OSError, ValueError) as exc:
            self.error = _reason(self.path, exc)
            self._close()

    def _close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def _line(self, name: str, result: str, grade="—", exposure="—", gain="—", time="—", notes="") -> str:
        if self.kind == "export":
            cells = [f"{name:<{self._width}}", f"{result:<11}", f"{time:<8}", notes]
        else:
            cells = [f"{name:<{self._width}}", f"{result:<11}", f"{grade:<15}", f"{exposure:<10}",
                     f"{gain:<11}", f"{time:<8}", notes]
        return "".join(cells)

    def start(self, command: str, sheet_rows, names: list[str], kept: list[str], version: str) -> None:
        try:
            unfinished = previous_unfinished(self.path)
            try:
                self._file = open(self.path, "a", encoding="utf-8", errors="backslashreplace")
            except OSError as exc:
                self.error = _reason(self.path, exc)
                return
            self._width = max(len(n) for n in [*names, *kept, "Frame"]) + 3
            lines = [_rule(f"{self._now():%Y-%m-%d %H:%M} · halide {version}")]
            if unfinished is not None:
                lines.append(f"{ICON_WARN} The previous run in this log (started {unfinished}) didn't finish — "
                             "it stopped after the frames listed above.")
            lines += [command, *plain_row_lines(sheet_rows), "",
                      self._line("Frame", "Result", "Grade", "Exposure", "Scan gain", "Time", "Notes")]
            lines += [self._line(name, "kept", notes="already there, kept") for name in kept]
            self._write(lines)
        except Exception as exc:
            if self.error is None:
                self.error = _reason(self.path, exc)
            self._close()

    def frame(self, result) -> None:
        try:
            name = result.job.input_path.name
            time = human_time(result.seconds) if result.seconds is not None else "—"
            if result.error:
                notes = "; ".join(_one_line(t) for t in (result.error, result.warning) if t)
                self._write([self._line(name, "failed", time=time, notes=notes)], sync=True)
                return
            grade, exposure = "—", "—"
            if result.linear_scale is not None:
                grade = f"flat x{result.linear_scale:.3g}"
            elif result.contrast is not None:
                grade = f"{result.contrast:.2f}" + ("" if result.grade_fitted else " (pinned)")
                exposure = f"{result.exposure:+.2f}" if result.exposure is not None else "—"
            gain = f"{result.job.scan_gain:.2f}"
            self._write([self._line(name, _RESULT_WORD[self.kind], grade, exposure, gain, time, _one_line(result.warning or ""))], sync=True)
        except Exception as exc:
            if self.error is None:
                self.error = _reason(self.path, exc)
            self._close()

    def finish(self, end_text: str | None, summary_rows, error: str | None = None,
               interrupted: bool = False) -> None:
        try:
            lines = [""]
            if end_text:
                lines.append(end_text)
            lines += plain_row_lines(summary_rows)
            if interrupted:
                lines.append(f"{ICON_FAIL} Stopped — interrupted")
            elif error:
                lines.append(f"{ICON_FAIL} Stopped by an error: {error}")
            lines += [_rule(f"finished {self._now():%H:%M}"), ""]
            self._write(lines)
        except Exception as exc:
            if self.error is None:
                self.error = _reason(self.path, exc)
        finally:
            self._close()
