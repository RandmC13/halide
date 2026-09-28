"""What halide does when an output file already exists (the user's decision D-1, recorded in
CLAUDE.md): ask in a terminal, refuse in a script, --overwrite / --skip-existing to choose.
Writing over the scan being developed is refused in every mode, whatever the flags say.

Shared by every command that writes files (invert/batch/print/export/contact) so the policy reads
the same way everywhere, rather than each command inventing its own overwrite prompt (the gap this
replaces: `console.confirm_overwrite` silently proceeded whenever stdin wasn't a tty, and no bulk
command checked for pre-existing outputs at all — see docs/investigations/codebase-review-evidence
2.3-1, 2.3-2, 2.5-1, 4-3, 4-4)."""

from __future__ import annotations

import enum
import os
import sys
from pathlib import Path

from halide.cli import console


class OutputPolicy(enum.Enum):
    ASK = "ask"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip-existing"


def add_output_policy_arguments(parser) -> None:
    """A mutually exclusive `--overwrite` / `--skip-existing` group, shared by every command that
    writes files (and, later, by `--save-profile-as`'s own overwrite check)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output files that already exist (and a saved profile of the same name)",
    )
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="develop only the frames whose output doesn't exist yet - resumes an interrupted roll",
    )


def policy_from_args(args) -> OutputPolicy:
    if getattr(args, "overwrite", False):
        return OutputPolicy.OVERWRITE
    if getattr(args, "skip_existing", False):
        return OutputPolicy.SKIP_EXISTING
    return OutputPolicy.ASK


def is_interactive() -> bool:
    """True only when both stdin and stdout are real terminals — a script or a redirected/logged
    run must never be left blocked waiting on input it can never receive."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a.resolve() == b.resolve()


def check_not_input(pairs: list[tuple[Path, Path]]) -> None:
    """Refuse, in every mode and regardless of --overwrite, a run whose output path is the same
    file as its own input scan — this is never "overwriting an existing file", it's using a scan
    as both the input and the output of the same run, which would destroy it."""
    clashes = [out for src, out in pairs if _same_file(src, out)]
    if not clashes:
        return
    first = clashes[0]
    more = f" (and {len(clashes) - 1} more)" if len(clashes) > 1 else ""
    raise SystemExit(
        f"{first} is the scan itself{more} - halide never writes over a scan. "
        "Choose a different output folder, or add --suffix."
    )


def resolve_existing(
    pairs: list[tuple[Path, Path]], policy: OutputPolicy, *, interactive: bool
) -> list[tuple[Path, Path]]:
    """The pairs to actually process, after applying `policy` to whichever outputs already exist.
    Never prompts when nothing exists yet. --overwrite keeps every pair; --skip-existing drops the
    ones whose output is already there; the default (ASK) asks once for the whole batch in a
    terminal, and refuses — naming both flags — everywhere else (a script must never silently
    overwrite a previous run, and must never block on input it can't get)."""
    existing = [(src, out) for src, out in pairs if out.exists()]
    if not existing or policy is OutputPolicy.OVERWRITE:
        return list(pairs)
    if policy is OutputPolicy.SKIP_EXISTING:
        return [(src, out) for src, out in pairs if not out.exists()]

    n, total = len(existing), len(pairs)
    if total == 1:
        path = existing[0][1]
        question = f"{path} already exists - overwrite it?"
        refusal = (
            f"{path} already exists. Add --overwrite to replace it, "
            "or --skip-existing to leave it as is."
        )
        decline_hint = "(--skip-existing leaves it as is.)"
    else:
        where = existing[0][1] if n == 1 else existing[0][1].parent
        question = f"{n} of {total} outputs already exist in {where} - overwrite them?"
        refusal = (
            f"{n} of {total} outputs already exist in {where}. Add --overwrite to replace them, "
            "or --skip-existing to develop only the new frames."
        )
        decline_hint = "(--skip-existing develops only the new frames.)"

    if interactive:
        if console.confirm(question):
            return list(pairs)
        raise SystemExit(f"Nothing was written. {decline_hint}")
    raise SystemExit(refusal)
