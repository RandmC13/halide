"""The end-of-run summary and the run log, shared by the multi-frame commands (batch, and print/
export given folders). Stdlib-only at import, like every
module the parser loads."""

from __future__ import annotations

import argparse
from pathlib import Path

from halide.cli import console
from halide.io.roll import RUN_LOG_NAME


def add_log_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--log", metavar="FILE",
                       help=f"Append this run's record to FILE instead of {RUN_LOG_NAME} in the output folder")
    group.add_argument("--no-log", action="store_true", help="Don't write a run log")


def log_path(args, output_dir: Path | None) -> Path | None:
    """Where this run's log goes: --log FILE, else the output folder's halide-log.txt, else (a
    preview with no output folder) nowhere."""
    if args.no_log:
        return None
    if args.log:
        return Path(args.log)
    return output_dir / RUN_LOG_NAME if output_dir is not None else None


_PICTURE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def check_log_path(log: str | Path | None, pairs: list[tuple[Path, Path]]) -> None:
    """Refuse a --log FILE that is a picture or is one of this run's own scans or outputs: the log
    is appended to, so it would corrupt the file. `pairs` are (input, output) paths."""
    if log is None:
        return
    from halide.cli._output_policy import _same_file

    log = Path(log)
    if log.suffix.lower() in _PICTURE_SUFFIXES or any(_same_file(log, p) for pair in pairs for p in pair):
        raise SystemExit(
            f"--log {log} is a picture, not a log — halide never writes over a scan or a print. "
            f"Choose a .txt file, or leave --log out to use {RUN_LOG_NAME} in the output folder"
        )


class RunRecord:
    """One run's summary and log: `start` when the run sheet closes, `add` each result as it
    arrives, then `finish` (or `abort` on an unexpected error)."""

    def __init__(self, args, kind: str, output_dir: Path | None):
        from halide.batch.runlog import RunLog

        path = log_path(args, output_dir)
        self.kind = kind
        self.output_dir = output_dir
        self.quiet = args.quiet
        self.command = getattr(args, "command_line", "halide")
        self.log = RunLog(path, kind) if path is not None else None
        self.results: list = []
        self._closed = False  # finish and abort each close the log block, once between them
        self._compute: list = []

    def start(self, sheet: console.RunSheet, jobs, kept: list[str]) -> None:
        # The window shows no run sheet, so its summary repeats which device did the work.
        self._compute = [r for r in sheet.plain_rows if r[1] == "Compute"]
        if self.log is not None:
            from halide.processing import _halide_version

            self.log.start(self.command, sheet.plain_rows, [j.input_path.name for j in jobs], kept,
                           _halide_version())

    def add(self, result) -> None:
        self.results.append(result)
        if self.log is not None:
            self.log.frame(result)

    def finish(self, renderer) -> None:
        from halide.batch.summary import RollSummary, summary_rows
        from halide.processing import service_stopped

        self._sweep()
        rows = summary_rows(RollSummary.from_results(self.results, self.kind, stopped=service_stopped))
        self._closed = True
        if self.log is not None:
            self.log.finish(renderer.end_text, rows)
            if self.log.error is not None:
                rows.append(("warn", "Log", f"not written — {self.log.error}"))
            else:
                rows.append(("row", "Log", str(self.log.path)))
        console.print_rows(rows, quiet=self.quiet)
        log_at = next((i for i, r in enumerate(rows) if r[1] == "Log"), len(rows))
        renderer.summary(rows[:log_at] + self._compute + rows[log_at:])  # the machine renderer only

    def _sweep(self) -> None:
        """The pool is done, so any partial file left in the output folder belongs to a dead worker."""
        if self.output_dir is not None:
            from halide.io.atomic import sweep_partials

            sweep_partials(self.output_dir)

    def abort(self, exc: BaseException) -> None:
        self._sweep()
        if self.log is not None and not self._closed:
            self._closed = True
            if isinstance(exc, KeyboardInterrupt):  # also what SIGTERM/SIGHUP become
                self.log.finish(None, [], interrupted=True)
            else:
                self.log.finish(None, [], error=str(exc) or type(exc).__name__)
