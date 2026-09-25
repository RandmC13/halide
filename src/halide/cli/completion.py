"""Tab completion for zsh, bash and fish that installs itself — no `halide completion` command to
remember.

Each shell's script is generated from halide's own argparse parser (`shtab` for zsh and bash,
which shtab supports; `fish_script` below for fish, which it doesn't), so it can't drift from the
real flags: every run of halide from an interactive terminal regenerates the script for the shell
it was run from and rewrites the file only if it changed. The scripts are static — pressing Tab
never starts Python (importing halide takes ~1 s, mostly colour-science, which would make every
Tab lag).

Getting the shell to load the script: fish autoloads `~/.config/fish/completions/halide.fish`, so
nothing else is needed. zsh and bash only read their rc file, so the first run appends one marked
block to `~/.zshrc` / `~/.bashrc` (`~/.bash_profile` on macOS, whose terminals start login
shells) — chosen over printing a line for the user to paste. A stamp file per shell makes that a
one-time step: a user who deletes the block (or the fish file) doesn't get it back.
`HALIDE_NO_COMPLETION=1` turns all of this off. Nothing here may ever break a real run — every
failure is swallowed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SHELLS = ("zsh", "bash", "fish")
RC_BEGIN = "# >>> halide completion >>>"
RC_END = "# <<< halide completion <<<"
_HEADER = (
    "# Written by halide (src/halide/cli/completion.py) and regenerated whenever its flags\n"
    "# change — don't edit. Set HALIDE_NO_COMPLETION=1 to stop halide managing it.\n"
)

# What an argument's value is, by argument dest — shells complete nothing for a value unless told.
_KINDS = {
    "input": "tiff",
    "inputs": "tiff",
    "scan_reference": "tiff",
    "output": "file",
    "contact_sheet": "file",
    "input_dir": "dir",
    "output_dir": "dir",
    "profile": "profile",
    "name": "profile",  # profile show/edit/delete
    "old_name": "profile",  # profile rename
}
# (subcommand, dest) exceptions to the table above.
_KIND_OVERRIDES = {
    ("contact", "inputs"): "file",  # processed TIFFs or exported PNG/JPEGs, then the sheet to write
}

# shtab completers per kind. zsh: an action for _arguments (`man zshcompsys`); `(#i)` =
# case-insensitive, so IMG_0156.TIF matches, and _files still offers folders to descend into.
# bash: a function printing candidates for the word in $1; a name containing `_file`/`_dir` makes
# shtab turn on bash's filename handling (trailing `/` on folders, escaped spaces).
_SHTAB = {
    "tiff": {"zsh": "_files -g '*.(#i)tif(|f)'", "bash": "_halide_tiff_files"},
    "file": {"zsh": "_files", "bash": "_shtab_compgen_files"},
    "dir": {"zsh": "_files -/", "bash": "_shtab_compgen_dirs"},
    "profile": {"zsh": "_halide_profiles", "bash": "_halide_profiles"},
}
# Saved profile names are the .json files in profile_store.default_profiles_dir(), listed by the
# shell itself so completing them doesn't start Python either.
_SHTAB_PREAMBLE = {
    "zsh": """
_halide_profiles() {
  local -a names
  names=(${XDG_CONFIG_HOME:-$HOME/.config}/halide/profiles/*.json(N:t:r))
  _wanted profiles expl 'saved profile' compadd -a names
}
""",
    "bash": """
_halide_tiff_files() {
  compgen -d -- "$1"
  compgen -f -- "$1" | grep -iE '\\.tiff?$'
}

_halide_profiles() {
  local f names=()
  for f in "${XDG_CONFIG_HOME:-$HOME/.config}"/halide/profiles/*.json; do
    [ -e "$f" ] && f=${f##*/} && names+=("${f%.json}")
  done
  compgen -W "${names[*]}" -- "$1"
}
""",
}


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            yield from action.choices.items()


def _kind(command: str | None, action: argparse.Action) -> str | None:
    if action.choices:
        return None
    return _KIND_OVERRIDES.get((command, action.dest), _KINDS.get(action.dest))


def _attach_completers(parser: argparse.ArgumentParser, command: str | None = None) -> None:
    for action in parser._actions:
        kind = _kind(command, action)
        if kind is not None:
            action.complete = _SHTAB[kind]
    for name, subparser in _subparsers(parser):
        _attach_completers(subparser, command or name)


def _shtab_script(parser: argparse.ArgumentParser, shell: str) -> str:
    import shtab

    _attach_completers(parser)
    script = shtab.complete(parser, shell, preamble=_SHTAB_PREAMBLE)
    # shtab's header tells the reader to copy the file into place by hand; here halide does that.
    # zsh's first line (`#compdef halide`) is what makes it a completion function, so it stays.
    lines = script.splitlines(keepends=True)
    keep = [lines.pop(0)] if lines[0].startswith("#compdef") else []
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return "".join(keep) + _HEADER + "\n" + "".join(lines)


def zsh_script(parser: argparse.ArgumentParser) -> str:
    return _shtab_script(parser, "zsh")


def bash_script(parser: argparse.ArgumentParser) -> str:
    return _shtab_script(parser, "bash")


# --- fish --------------------------------------------------------------------------------------

_FISH_VALUE = {
    "tiff": "-a '(__halide_tiffs)'",
    "file": "-F",
    "dir": "-a '(__fish_complete_directories (commandline -ct))'",
    "profile": "-a '(__halide_profiles)'",
}

# fish has no argparse-style completion engine, so the script carries a tiny parser of its own:
# walk the words typed so far, following subcommands and skipping options (and the value after
# an option that takes one), to know which subcommand and which positional the cursor is at.
_FISH_HELPERS = r"""
function __halide_state --description 'halide: subcommand and positional count at the cursor'
    set -l words (commandline -opc)
    set -e words[1]
    set -l path ''
    set -l npos 0
    set -l skip 0
    for word in $words
        if test $skip = 1
            set skip 0
            continue
        end
        switch $word
            case '--*=*'
                continue
            case '-*'
                contains -- (string trim -- "$path $word") $__halide_value_options; and set skip 1
                continue
        end
        set -l next (string trim -- "$path $word")
        if test $npos = 0; and contains -- $next $__halide_commands
            set path $next
        else
            set npos (math $npos + 1)
        end
    end
    echo "p:$path"
    echo $npos
end

function __halide_in --description 'halide: is the cursor inside this subcommand?'
    set -l state (__halide_state)
    test "$state[1]" = "p:$argv[1]"
end

function __halide_pos --description 'halide: is the cursor at this positional (or, with a 3rd argument, this one or later)?'
    set -l state (__halide_state)
    test "$state[1]" = "p:$argv[1]"; or return 1
    if set -q argv[3]
        test $state[2] -ge $argv[2]
    else
        test $state[2] -eq $argv[2]
    end
end

# Not __fish_complete_suffix: since fish 3.6 that only sorts matching files first, it still lists
# the rest.
function __halide_tiffs --description 'halide: folders and TIFF scans'
    for path in (__fish_complete_path (commandline -ct))
        set path (string split -f1 \t -- $path)
        if string match -qr '/$' -- $path; or string match -qir '\.tiff?$' -- $path
            echo $path
        end
    end
end

function __halide_profiles --description 'halide: saved profile names'
    set -l dir ~/.config
    set -q XDG_CONFIG_HOME; and set dir $XDG_CONFIG_HOME
    set -l files $dir/halide/profiles/*.json
    for file in $files
        string replace -r '^.*/(.*)\.json$' '$1\tsaved profile' -- $file
    end
end
"""


def _fish_quote(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _fish_description(help_text: str | None) -> str:
    """fish shows one line per candidate: the help's first sentence, trimmed."""
    if not help_text or help_text == argparse.SUPPRESS:
        return ""
    text = " ".join(help_text.split())
    text = re.split(r"\.\s+(?=[A-Z`])| — |; ", text)[0]  # a real sentence end, not "e.g. how"
    if text.count("(") > text.count(")"):  # cut inside a bracket: drop the unclosed aside
        text = text[: text.rfind("(")].rstrip()
    if len(text) > 80:
        text = text[:79].rstrip() + "…"
    return f" -d {_fish_quote(text)}"


def _fish_walk(parser, path, command, lines, commands, value_options):
    here = _fish_quote(path)
    positional = 0
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {choice.dest: choice.help for choice in action._choices_actions}
            for name, subparser in action.choices.items():
                sub_path = f"{path} {name}".strip()
                commands.append(sub_path)
                lines.append(
                    f"complete -c halide -n \"__halide_pos {here} 0\" -a {_fish_quote(name)}"
                    f"{_fish_description(helps.get(name))}"
                )
                _fish_walk(subparser, sub_path, command or name, lines, commands, value_options)
            continue
        kind = _kind(command, action)
        if action.option_strings:
            flags = " ".join(
                f"-l {flag[2:]}" if flag.startswith("--") else f"-s {flag[1:]}"
                for flag in action.option_strings
            )
            requires = ""
            if action.nargs != 0:
                value_options.extend(f"{path} {flag}".strip() for flag in action.option_strings)
                requires = "-r" if kind == "file" else "-x"
                if action.choices:
                    values = "-a " + _fish_quote(" ".join(str(c) for c in action.choices))
                else:
                    values = _FISH_VALUE.get(kind)
                if values:
                    # On a line of their own: fish gives -d to an option's values too, which would
                    # label `print` and `flat` (or every TIFF) with the whole flag's help.
                    lines.append(
                        f"complete -c halide -n \"__halide_in {here}\" {flags} {requires} {values}"
                    )
            lines.append(
                f"complete -c halide -n \"__halide_in {here}\" {flags} {requires}".rstrip()
                + _fish_description(action.help)
            )
        else:
            if kind is not None:
                more = " more" if action.nargs in ("*", "+") else ""
                lines.append(
                    f"complete -c halide -n \"__halide_pos {here} {positional}{more}\" "
                    f"{_FISH_VALUE[kind]}"
                )
            positional += 1


def fish_script(parser: argparse.ArgumentParser) -> str:
    lines: list[str] = []
    commands: list[str] = []
    value_options: list[str] = []
    _fish_walk(parser, "", None, lines, commands, value_options)
    return (
        _HEADER
        + "\ncomplete -c halide -e\n"
        + "complete -c halide -f  # only offer files where an argument actually takes one\n\n"
        + "set -g __halide_commands " + " ".join(_fish_quote(c) for c in commands) + "\n"
        + "set -g __halide_value_options " + " ".join(_fish_quote(o) for o in value_options) + "\n"
        + _FISH_HELPERS
        + "\n"
        + "\n".join(lines)
        + "\n"
    )


# --- installing --------------------------------------------------------------------------------


def _data_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    return (Path(data_home) if data_home else Path.home() / ".local" / "share") / "halide"


def _config_home() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    return Path(config_home) if config_home else Path.home() / ".config"


@dataclass(frozen=True)
class _Shell:
    name: str
    script: Callable[[argparse.ArgumentParser], str]
    target: Path  # where the completion script goes
    rc: Path | None  # the file to hook it into, or None if the shell finds it by itself

    @property
    def stamp(self) -> Path:
        return _data_dir() / "completion" / f"{self.name}.installed"

    def rc_block(self) -> str:
        if self.name == "zsh":
            # Appended at the end of .zshrc, i.e. usually after compinit (or oh-my-zsh) has run,
            # so adding to fpath alone isn't enough: compdef registers _halide there and then. A
            # zshrc without the completion system at all gets it switched on, or Tab couldn't
            # complete anything.
            body = (
                f'fpath=("{self.target.parent}" $fpath)\n'
                "(( $+functions[compdef] )) || { autoload -Uz compinit && compinit }\n"
                "autoload -Uz _halide && compdef _halide halide\n"
            )
        else:
            body = f'[ -f "{self.target}" ] && . "{self.target}"\n'
        return (
            f"\n{RC_BEGIN}\n"
            "# Added by halide for tab completion. Delete this block to remove it (it won't come back).\n"
            f"{body}{RC_END}\n"
        )


def shell_setup(shell: str) -> _Shell:
    if shell == "zsh":
        rc = Path(os.environ.get("ZDOTDIR") or Path.home()) / ".zshrc"
        return _Shell("zsh", zsh_script, _data_dir() / "completion" / "zsh" / "_halide", rc)
    if shell == "bash":
        rc = Path.home() / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
        return _Shell("bash", bash_script, _data_dir() / "completion" / "halide.bash", rc)
    if shell == "fish":
        return _Shell("fish", fish_script, _config_home() / "fish" / "completions" / "halide.fish", None)
    raise ValueError(f"no tab completion for {shell!r}")


def detect_shell() -> str | None:
    """The shell halide was started from: its parent process when that's a shell we support (a
    zsh user can still be running halide from bash), else the login shell in $SHELL."""
    try:
        import psutil

        name = psutil.Process().parent().name().lstrip("-")
        if name in SHELLS:
            return name
    except Exception:  # noqa: BLE001 -- best effort; $SHELL below is the fallback
        pass
    name = Path(os.environ.get("SHELL", "")).name
    return name if name in SHELLS else None


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def ensure_completion(
    parser: argparse.ArgumentParser, *, shell: str | None = None, interactive: bool | None = None
) -> str | None:
    """Write/refresh this shell's completion script and, the first time, hook it in. Only for a
    person at a terminal (not scripts, pipes or tests). Returns the one-time note to show when it
    has just been installed, else None."""
    if os.environ.get("HALIDE_NO_COMPLETION"):
        return None
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        return None
    shell = shell or detect_shell()
    if shell not in SHELLS:
        return None
    setup = shell_setup(shell)
    first_time = not setup.stamp.exists()

    if setup.rc is None:
        # The script lives in the shell's own folder, next to the user's files: never replace a
        # halide.fish that isn't ours, and don't recreate one the user deleted.
        try:
            existing = setup.target.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        if (existing is None and not first_time) or (existing is not None and _HEADER not in existing):
            return None
    _write_if_changed(setup.target, setup.script(parser))
    if not first_time:
        return None

    setup.stamp.parent.mkdir(parents=True, exist_ok=True)
    setup.stamp.touch()
    if setup.rc is None:
        # Verified: an already-open fish has cached "no completions for halide" and won't look
        # again, so this is for new terminals too.
        return f"added {_tilde(setup.target)} — open a new terminal to use it"
    try:
        rc_text = setup.rc.read_text(encoding="utf-8")
    except FileNotFoundError:
        rc_text = ""
    if RC_BEGIN in rc_text:
        return None
    block = setup.rc_block()
    with setup.rc.open("a", encoding="utf-8") as handle:
        handle.write(block)
    lines = len(block.strip().splitlines())
    return f"added {lines} lines to {_tilde(setup.rc)} — open a new terminal to use it"


def _tilde(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def maybe_install_completion(parser: argparse.ArgumentParser) -> None:
    """ensure_completion plus the one-time note, never raising: completion is a convenience, and a
    read-only home or odd rc file must not turn a finished print into an error."""
    from halide.cli import console

    try:
        shell = detect_shell()
        note = ensure_completion(parser, shell=shell)
    except Exception:  # noqa: BLE001 -- see docstring
        return
    if note:
        print(console.success(f"Tab completion installed for {shell}"), file=sys.stderr)
        print(console.dim(f"  ({note})"), file=sys.stderr)
