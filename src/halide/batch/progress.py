"""Terminal rendering for batch progress. This module only draws — it knows nothing about the
pipeline math or multiprocessing in halide.batch.orchestrator, and is driven purely by
BatchResult callbacks so it can be swapped out (or skipped, e.g. --quiet) without touching
processing logic at all."""

from __future__ import annotations

import random
import sys

from halide.batch.orchestrator import BatchResult


class _T:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    @staticmethod
    def cursor_up(n: int) -> str:
        return f"\033[{n}A"


_BLOCK = "██"
_STATE_COLOR = {"pending": _T.DIM, "processing": _T.ORANGE, "done": _T.GREEN, "error": _T.RED}


class GridProgressRenderer:
    """A colored block grid, one block per job, redrawn in place as jobs start and complete —
    ported from the original script's terminal UI, decoupled from the processing it's reporting
    on. Each job is assigned to a random grid cell (fixed at construction time) rather than its
    own index's position, so the grid fills in a scattered order rather than mechanically
    top-left-to-bottom-right regardless of what order jobs actually start/finish in."""

    def __init__(self, total: int, columns: int = 9):
        self.total = total
        self.columns = columns
        self.completed = 0
        self.failures: list[tuple[str, str]] = []  # (filename, error message)
        self._states = ["pending"] * total
        self._cell_job = list(range(total))  # cell_job[grid position] -> job index
        random.shuffle(self._cell_job)

    def _rows(self) -> int:
        return (self.total + self.columns - 1) // self.columns

    def _total_lines(self) -> int:
        rows = self._rows()
        # rows of blocks + a blank separator between each pair of rows + a blank line + the
        # "[ Processed x/y ]" line. Must exactly match what _redraw() actually writes below, or
        # its cursor_up() will drift by the difference on every subsequent redraw.
        return rows + max(rows - 1, 0) + 2

    def start(self) -> None:
        sys.stdout.write("\n" * self._total_lines())
        self._redraw()  # draw the initial all-pending grid immediately, not on the first report()

    def _redraw(self) -> None:
        rows = self._rows()
        sys.stdout.write(_T.cursor_up(self._total_lines()))
        for r in range(rows):
            row = ""
            for c in range(self.columns):
                pos = r * self.columns + c
                if pos < self.total:
                    job_index = self._cell_job[pos]
                    color = _STATE_COLOR[self._states[job_index]]
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

    def mark_processing(self, index: int) -> None:
        self._states[index] = "processing"
        self._redraw()

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
