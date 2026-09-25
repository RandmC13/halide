"""zsh tab completion that installs itself — no `halide completion` command to remember.

The completion script is generated from halide's own argparse parser by `shtab`, so it can't drift
from the real flags: every run of halide from a zsh terminal regenerates it and rewrites the file
only if it changed. It is a static script — pressing Tab never starts Python (importing halide
takes ~1 s, mostly colour-science, which would make every Tab lag).

zsh only finds completion functions in folders on its `fpath`, and there is no standard per-user
folder there, so the first run also appends one marked block to `~/.zshrc` pointing at halide's
folder (the user chose this over printing the line for them to paste). That happens once: a stamp
file records it, so a user who deletes the block doesn't get it back. `HALIDE_NO_COMPLETION=1`
turns all of this off. Nothing here may ever break a real run — every failure is swallowed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ZSHRC_BEGIN = "# >>> halide completion >>>"
ZSHRC_END = "# <<< halide completion <<<"

# zsh completion actions (see `man zshcompsys`, _files). `(#i)` = case-insensitive, so IMG_0156.TIF
# matches too; _files still offers folders to descend into, so directory inputs work as well.
_TIFF = {"zsh": "_files -g '*.(#i)tif(|f)'"}
_FILE = {"zsh": "_files"}
_DIR = {"zsh": "_files -/"}
_PROFILE = {"zsh": "_halide_profiles"}

# By argument dest; shtab completes nothing for an argument's value unless told how.
_COMPLETERS = {
    "input": _TIFF,
    "inputs": _TIFF,
    "scan_reference": _TIFF,
    "output": _FILE,
    "contact_sheet": _FILE,
    "input_dir": _DIR,
    "output_dir": _DIR,
    "profile": _PROFILE,
    "name": _PROFILE,  # profile show/edit/delete
    "old_name": _PROFILE,  # profile rename
}
# (subcommand, dest) exceptions to the table above.
_OVERRIDES = {
    ("contact", "inputs"): _FILE,  # processed TIFFs or exported PNG/JPEGs, then the sheet to write
}

# Saved profile names are the .json files in profile_store.default_profiles_dir() — listed by zsh
# itself so completing them doesn't start Python either.
_ZSH_PREAMBLE = """
_halide_profiles() {
  local -a names
  names=(${XDG_CONFIG_HOME:-$HOME/.config}/halide/profiles/*.json(N:t:r))
  _wanted profiles expl 'saved profile' compadd -a names
}
"""


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            yield from action.choices.items()


def _attach_completers(parser: argparse.ArgumentParser, command: str | None = None) -> None:
    for action in parser._actions:
        completer = _OVERRIDES.get((command, action.dest), _COMPLETERS.get(action.dest))
        if completer is not None and not action.choices:
            action.complete = completer
    for name, subparser in _subparsers(parser):
        _attach_completers(subparser, command or name)


def zsh_script(parser: argparse.ArgumentParser) -> str:
    import shtab

    _attach_completers(parser)
    script = shtab.complete(parser, "zsh", preamble={"zsh": _ZSH_PREAMBLE})
    # shtab's header tells the reader to copy the file into place by hand; here halide does that.
    first, *body = script.splitlines(keepends=True)
    while body and (body[0].startswith("#") or not body[0].strip()):
        body.pop(0)
    header = (
        "# Written by halide (src/halide/cli/completion.py) and regenerated whenever its flags\n"
        "# change — don't edit. Set HALIDE_NO_COMPLETION=1 to stop halide managing it.\n\n"
    )
    return first + header + "".join(body)


def completion_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    return (Path(data_home) if data_home else Path.home() / ".local" / "share") / "halide" / "zsh"


def zshrc_path() -> Path:
    return Path(os.environ.get("ZDOTDIR") or Path.home()) / ".zshrc"


def zshrc_block(directory: Path) -> str:
    # Appended at the end of .zshrc, i.e. usually after compinit (or oh-my-zsh) has already run,
    # so adding to fpath alone isn't enough: compdef registers _halide there and then. A zshrc
    # without the completion system at all gets it switched on, or Tab couldn't complete anything.
    return (
        f"\n{ZSHRC_BEGIN}\n"
        "# Added by halide for tab completion. Delete this block to remove it (it won't come back).\n"
        f'fpath=("{directory}" $fpath)\n'
        "(( $+functions[compdef] )) || { autoload -Uz compinit && compinit }\n"
        "autoload -Uz _halide && compdef _halide halide\n"
        f"{ZSHRC_END}\n"
    )


def ensure_zsh_completion(parser: argparse.ArgumentParser, *, interactive: bool | None = None) -> bool:
    """Write/refresh the completion file and, the first time, hook it into ~/.zshrc. Only for a
    person at a zsh terminal (not scripts, pipes or tests). Returns True if ~/.zshrc was edited,
    so the caller can say so."""
    if os.environ.get("HALIDE_NO_COMPLETION"):
        return False
    if Path(os.environ.get("SHELL", "")).name != "zsh":
        return False
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        return False

    directory = completion_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "_halide"
    script = zsh_script(parser)
    try:
        current = target.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != script:
        partial = target.with_name("_halide.tmp")
        partial.write_text(script, encoding="utf-8")
        partial.replace(target)

    stamp = directory / ".zshrc-added"
    if stamp.exists():
        return False
    zshrc = zshrc_path()
    try:
        existing = zshrc.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    edited = ZSHRC_BEGIN not in existing
    if edited:
        with zshrc.open("a", encoding="utf-8") as handle:
            handle.write(zshrc_block(directory))
    stamp.touch()
    return edited


def maybe_install_completion(parser: argparse.ArgumentParser) -> None:
    """ensure_zsh_completion plus the one-time note, never raising: completion is a convenience,
    and a read-only home or odd .zshrc must not turn a finished print into an error."""
    from halide.cli import console

    try:
        edited = ensure_zsh_completion(parser)
    except Exception:  # noqa: BLE001 -- see docstring
        return
    if edited:
        lines = len(zshrc_block(completion_dir()).strip().splitlines())
        print(console.success("Tab completion installed for zsh"), file=sys.stderr)
        print(
            console.dim(f"  (added {lines} lines to {zshrc_path()} — open a new terminal to use it)"),
            file=sys.stderr,
        )
