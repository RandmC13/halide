"""The top-level exception boundary for the installed console script."""

from __future__ import annotations

import os
import sys
from typing import Callable
from halide.cli.console.style import _ANSI_ESCAPE, cancelled, dim, error


def _print_err(text: str) -> None:
    """Print to stderr, without colour codes when stderr itself isn't a terminal (2> file) — the
    message was styled for stdout's terminal."""
    force = os.environ.get("FORCE_COLOR")
    if not sys.stderr.isatty() and not (force and force != "0"):
        text = _ANSI_ESCAPE.sub("", text)
    print(text, file=sys.stderr)


def run_guarded(main_func: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    """The top-level exception boundary for the installed console script (and `python -m
    halide.cli.main`) — NOT used by main() itself. The integration test suite calls main() directly
    and relies on it raising SystemExit for domain errors rather than swallowing it into a return
    code (see tests/integration/test_*_cli.py's pytest.raises(SystemExit, ...) usage); only this
    wrapper, which nothing tests against directly, is meant to be user-facing.

    A simple `"--debug" in argv` scan (rather than parsing argv through argparse a second time)
    decides whether an unexpected error re-raises with its full traceback — deliberately independent
    of exactly where `--debug` appears, and of whether argparse itself would even accept it there.
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    debug = "--debug" in raw_args or os.environ.get("HALIDE_DEBUG") == "1"
    try:
        return main_func(argv)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        _print_err(cancelled())
        return 130
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            _print_err(error(code))
            return 1
        return code if isinstance(code, int) else 0
    except Exception as exc:  # noqa: BLE001 -- last-resort presentation layer, see docstring above
        if debug:
            raise
        _print_err(error(f"halide hit an unexpected error: {exc}"))
        _print_err(dim("(re-run with --debug, or set HALIDE_DEBUG=1, for the full traceback)"))
        return 1
