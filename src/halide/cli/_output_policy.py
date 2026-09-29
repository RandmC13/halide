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
        help="Replace output files that already exist (and a saved profile of the same name)",
    )
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Develop only the frames whose output doesn't exist yet (resumes an interrupted roll)",
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


def check_not_input(pairs: list[tuple[Path, Path]], *, suggest_suffix: bool = True) -> None:
    """Refuse, in every mode and regardless of --overwrite, a run whose output path is the same
    file as its own input scan — this is never "overwriting an existing file", it's using a scan
    as both the input and the output of the same run, which would destroy it.

    `suggest_suffix` names --suffix in the fix-it hint — only for callers that actually have that
    flag (batch, bulk print/export); single-file commands without one (invert, single print/export)
    pass `suggest_suffix=False`, ending the hint at "Choose a different output path." instead."""
    clashes = [out for src, out in pairs if _same_file(src, out)]
    if not clashes:
        return
    first = clashes[0]
    more = f" (and {len(clashes) - 1} more)" if len(clashes) > 1 else ""
    fix = "Choose a different output folder, or add --suffix" if suggest_suffix else "Choose a different output path"
    raise SystemExit(f"{first} is the scan itself{more} - halide never writes over a scan. {fix}")


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
            "or --skip-existing to leave it as is"
        )
        decline_hint = "(--skip-existing leaves it as is.)"
    else:
        where = existing[0][1] if n == 1 else existing[0][1].parent
        question = f"{n} of {total} outputs already exist in {where} - overwrite them?"
        refusal = (
            f"{n} of {total} outputs already exist in {where}. Add --overwrite to replace them, "
            "or --skip-existing to develop only the new frames"
        )
        decline_hint = "(--skip-existing develops only the new frames.)"

    if interactive:
        if console.confirm(question):
            return list(pairs)
        raise SystemExit(f"Nothing was written. {decline_hint}")
    raise SystemExit(refusal)


def resolve_bulk_jobs(
    jobs: list,
    args,
    *,
    interactive: bool,
    extra_pairs: list[tuple[Path, Path]] | None = None,
) -> tuple[list, int, set[Path]]:
    """The shared `pairs -> check_not_input -> resolve_existing -> filter jobs` sequence every
    bulk command (batch, bulk print, bulk export) needs — each `job` must have `.input_path`/
    `.output_path` attributes (e.g. `BatchJob`). `extra_pairs` folds other outputs that should
    share the *same* single prompt/refusal (batch's `--contact-sheet` path) into the existence
    check, without being subject to job filtering themselves — the caller checks membership in the
    returned `kept_outputs` set for those.

    Returns (jobs whose output should still be written, how many jobs were dropped because their
    output already existed under --skip-existing, every output path that survived — job outputs
    and `extra_pairs` outputs alike)."""
    pairs = [(job.input_path, job.output_path) for job in jobs]
    check_not_input(pairs)
    all_pairs = pairs + list(extra_pairs or [])
    kept_outputs = {out for _, out in resolve_existing(all_pairs, policy_from_args(args), interactive=interactive)}
    kept_jobs = [job for job in jobs if job.output_path in kept_outputs]
    skipped = len(jobs) - len(kept_jobs)
    return kept_jobs, skipped, kept_outputs


# --- Pre-flight checks (F25): before any work starts, each a plain error naming the path ----------


def check_input_file(path: Path) -> None:
    """The input of a single-file command must exist and be a file."""
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")
    if path.is_dir():
        raise SystemExit(f"{path} is a folder, not a scan file. Name one TIFF, or use the folder form "
                         f"of the command (batch, or print/export with an output folder)")


def check_input_folder(path: Path) -> None:
    """The input of a bulk command must exist and be a folder."""
    if not path.exists():
        raise SystemExit(f"input directory not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"{path} is a file, not a folder of scans")


def _check_writable_folder(folder: Path) -> None:
    if not folder.is_dir():
        raise SystemExit(f"{folder} isn't a folder. Choose a different output location")
    if not os.access(folder, os.W_OK | os.X_OK):
        raise SystemExit(f"halide can't write to {folder} (permission denied). "
                         f"Choose a different output folder, or fix its permissions")


def prepare_output_folder(folder: Path) -> None:
    """A bulk command's output folder: created if missing (as before), then it must be writable —
    checked now, so a read-only folder fails before a single frame is developed rather than as one
    failure per frame."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # a file by that name: reported just below
    except OSError as exc:
        raise SystemExit(f"halide can't create the output folder {folder} ({exc.strerror or exc}). "
                         f"Choose a different output folder.") from exc
    _check_writable_folder(folder)


def check_output_parent(output_path: Path) -> None:
    """A single-file command's output: its folder must already exist and be writable."""
    parent = output_path.parent
    if not parent.exists():
        raise SystemExit(f"the output folder {parent} doesn't exist. Create it, or choose a different "
                         f"output path")
    _check_writable_folder(parent)
