"""What halide does when an output file already exists (the user's decision): ask in a terminal,
refuse in a script, --overwrite / --skip-existing to choose.
Writing over the scan being developed is refused in every mode, whatever the flags say.

Shared by every command that writes files (invert/batch/print/export/contact) so the policy reads
the same way everywhere, rather than each command inventing its own overwrite prompt (the gap this
replaces: `console.confirm_overwrite` silently proceeded whenever stdin wasn't a tty, and no bulk
command checked for pre-existing outputs at all)."""

from __future__ import annotations

import enum
import os
import sys
from pathlib import Path

from halide.cli import console
from halide.cli.console.prompts import is_interactive  # noqa: F401 - re-exported for the commands


class OutputPolicy(enum.Enum):
    ASK = "ask"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip-existing"


def add_output_policy_arguments(parser, *, saves_profile: bool = False) -> None:
    """A mutually exclusive `--overwrite` / `--skip-existing` group, shared by every command that
    writes files (and, later, by `--save-profile-as`'s own overwrite check)."""
    overwrite_help = "Replace output files that already exist"
    if saves_profile:
        overwrite_help += " (and a saved profile of the same name)"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite",
        action="store_true",
        help=overwrite_help,
    )
    group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Develop only the frames whose output doesn't exist yet (resumes an interrupted roll)",
    )


DRY_RUN_HELP = (
    "Show what this run would do — frames, outputs, calibration, workers — without developing "
    "anything or writing any file"
)


def add_dry_run_argument(parser) -> None:
    """`--dry-run`, shared by batch and the folder form of print/export."""
    parser.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)


def refuse_dry_run_for_single_file(args) -> None:
    """print/export take a file or a folder; only a folder run has anything to plan."""
    if getattr(args, "dry_run", False):
        raise SystemExit("--dry-run plans a folder run; a single file has nothing to plan")


def policy_from_args(args) -> OutputPolicy:
    if getattr(args, "overwrite", False):
        return OutputPolicy.OVERWRITE
    if getattr(args, "skip_existing", False):
        return OutputPolicy.SKIP_EXISTING
    return OutputPolicy.ASK


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
    raise SystemExit(f"{first} is the scan itself{more} — halide never writes over a scan. {fix}")


def check_unique_outputs(pairs: list[tuple[Path, Path]], fix: str = "rename one of them") -> None:
    """Refuse two scans that would be written to one output (a.tif and a.tiff both become a.png).
    Compared case-insensitively, because macOS disks are."""
    seen: dict[str, tuple[Path, Path]] = {}
    clashes = []
    for src, out in pairs:
        first = seen.setdefault(str(out).casefold(), (src, out))
        if first[0] is not src:
            clashes.append((first[0], src, first[1]))
    if not clashes:
        return
    a, b, out = clashes[0]
    more = f" (and {len(clashes) - 1} more)" if len(clashes) > 1 else ""
    raise SystemExit(f"{a.name} and {b.name} would both be written to {out}{more} — {fix}")


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
        question = f"{path} already exists — overwrite it?"
        refusal = (
            f"{path} already exists. Add --overwrite to replace it, "
            "or --skip-existing to leave it as is"
        )
        decline_hint = "(--skip-existing leaves it as is.)"
    else:
        where = existing[0][1] if n == 1 else existing[0][1].parent
        question = f"{n} of {total} outputs already exist in {where} — overwrite them?"
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


def _check_log(args, pairs: list[tuple[Path, Path]]) -> None:
    from halide.cli._run_log import check_log_path

    check_log_path(getattr(args, "log", None), pairs)


def resolve_bulk_jobs(
    jobs: list,
    args,
    *,
    interactive: bool,
    extra_pairs: list[tuple[Path, Path]] | None = None,
    fix: str = "rename one of them",
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
    check_unique_outputs(pairs, fix)
    _check_log(args, pairs + list(extra_pairs or []))
    all_pairs = pairs + list(extra_pairs or [])
    kept_outputs = {out for _, out in resolve_existing(all_pairs, policy_from_args(args), interactive=interactive)}
    kept_jobs = [job for job in jobs if job.output_path in kept_outputs]
    skipped = len(jobs) - len(kept_jobs)
    return kept_jobs, skipped, kept_outputs


def plan_bulk_jobs(
    jobs: list,
    args,
    *,
    extra_pairs: list[tuple[Path, Path]] | None = None,
    fix: str = "rename one of them",
) -> tuple[list, int, set[Path], int]:
    """resolve_bulk_jobs for a dry run: the same refusal to write over a scan and the same
    --overwrite / --skip-existing filtering, but the default (ASK) never prompts and never refuses
    for want of a terminal — every frame stays in the plan and the caller reports how many a real
    run would ask about. Returns (jobs, skipped under --skip-existing, surviving output paths,
    how many of the *jobs'* outputs already exist)."""
    pairs = [(job.input_path, job.output_path) for job in jobs]
    check_not_input(pairs)
    check_unique_outputs(pairs, fix)
    _check_log(args, pairs + list(extra_pairs or []))
    existing = sum(1 for _, out in pairs if out.exists())
    all_pairs = pairs + list(extra_pairs or [])
    policy = policy_from_args(args)
    if policy is OutputPolicy.SKIP_EXISTING:
        kept_outputs = {out for _, out in all_pairs if not out.exists()}
    else:
        kept_outputs = {out for _, out in all_pairs}
    kept_jobs = [job for job in jobs if job.output_path in kept_outputs]
    return kept_jobs, len(jobs) - len(kept_jobs), kept_outputs, existing


# --- Pre-flight checks: before any work starts, each a plain error naming the path ----------


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


def check_output_folder(folder: Path) -> None:
    """A dry run's version of prepare_output_folder: nothing is created. A folder that exists must
    be writable, exactly as in a real run; one that doesn't is left for the run sheet to say
    "(would be created)" — its parent isn't checked, as a real run would just make it."""
    if folder.exists() or folder.is_symlink():
        _check_writable_folder(folder)


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
                         f"Choose a different output folder") from exc
    _check_writable_folder(folder)


def check_output_parent(output_path: Path) -> None:
    """A single-file command's output: its folder must already exist and be writable."""
    parent = output_path.parent
    if not parent.exists():
        raise SystemExit(f"the output folder {parent} doesn't exist. Create it, or choose a different "
                         f"output path")
    _check_writable_folder(parent)
