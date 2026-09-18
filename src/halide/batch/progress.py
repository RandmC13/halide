"""Terminal rendering for batch progress. This module only draws — it knows nothing about the
pipeline math or multiprocessing in halide.batch.orchestrator, and is driven purely by
BatchResult callbacks so it can be swapped out (or skipped, e.g. --quiet) without touching
processing logic at all."""

from __future__ import annotations

import sys

from halide.batch.orchestrator import BatchResult


class _T:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    @staticmethod
    def cursor_up(n: int) -> str:
        return f"\033[{n}A"


_BLOCK = "██"
_STATE_COLOR = {"pending": _T.DIM, "done": _T.GREEN, "error": _T.YELLOW}


class GridProgressRenderer:
    """A colored block grid, one block per job, redrawn in place as jobs complete — ported from
    the original script's terminal UI, decoupled from the processing it's reporting on."""

    def __init__(self, total: int, columns: int = 9):
        self.total = total
        self.columns = columns
        self.completed = 0
        self.failures: list[tuple[str, str]] = []  # (filename, error message)
        self._states = ["pending"] * total

    def _rows(self) -> int:
        return (self.total + self.columns - 1) // self.columns

    def _total_lines(self) -> int:
        return self._rows() * 2 + 2

    def start(self) -> None:
        sys.stdout.write("\n" * self._total_lines())

    def _redraw(self) -> None:
        rows = self._rows()
        sys.stdout.write(_T.cursor_up(self._total_lines()))
        for r in range(rows):
            row = ""
            for c in range(self.columns):
                idx = r * self.columns + c
                if idx < self.total:
                    color = _STATE_COLOR[self._states[idx]]
                    row += f"{color}{_BLOCK}{_T.RESET} "
            sys.stdout.write(f"{_T.CLEAR_LINE}    {row}\n")
            if r < rows - 1:
                sys.stdout.write(f"{_T.CLEAR_LINE}\n")
        sys.stdout.write(f"{_T.CLEAR_LINE}\n")
        sys.stdout.write(
            f"{_T.CLEAR_LINE}    [ Processed {self.completed}/{self.total}"
            f"{f' — {len(self.failures)} failed' if self.failures else ''} ]\n"
        )
        sys.stdout.flush()

    def report(self, index: int, result: BatchResult) -> None:
        self._states[index] = "error" if result.error else "done"
        self.completed += 1
        if result.error:
            self.failures.append((result.job.input_path.name, result.error))
        self._redraw()

    def finish(self) -> None:
        if self.failures:
            print(f"\n{_T.YELLOW}{_T.BOLD}Completed with {len(self.failures)} failure(s):{_T.RESET}")
            for name, error in self.failures:
                print(f"  {_T.YELLOW}✗{_T.RESET} {name}: {error}")
        else:
            print(f"\n{_T.GREEN}{_T.BOLD}All {self.total} frames processed successfully.{_T.RESET}")
