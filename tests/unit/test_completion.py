import shutil
import subprocess

import pytest

from halide.cli import completion
from halide.cli.main import build_parser


@pytest.fixture
def zsh_home(tmp_path, monkeypatch):
    """A fake home for a zsh user: halide's data dir and .zshrc both under tmp_path."""
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("ZDOTDIR", str(tmp_path))
    monkeypatch.delenv("HALIDE_NO_COMPLETION", raising=False)
    return tmp_path


def test_script_covers_every_subcommand_and_flag():
    script = completion.zsh_script(build_parser())
    parser = build_parser()
    for name, subparser in completion._subparsers(parser):
        assert f"{name}:" in script
        for action in subparser._actions:
            for flag in action.option_strings:
                assert flag in script, f"{name} {flag} missing from the completion script"


def _action(parser, command, dest):
    subparser = dict(completion._subparsers(parser))[command]
    return next(a for a in subparser._actions if a.dest == dest)


def test_script_completes_values_not_just_flag_names():
    parser = build_parser()
    script = completion.zsh_script(parser)
    assert _action(parser, "invert", "input").complete == completion._TIFF
    assert _action(parser, "batch", "input_dir").complete == completion._DIR
    assert _action(parser, "invert", "profile").complete == completion._PROFILE
    assert _action(parser, "contact", "inputs").complete == completion._FILE  # the override
    assert ":Input linear TIFF scan of a negative:_files -g '*.(#i)tif(|f)'" in script
    assert ":Directory of input linear TIFF scans:_files -/" in script
    assert "_halide_profiles() {" in script  # the preamble is included...
    assert ":profile:_halide_profiles" in script  # ...and used for --profile
    assert ":output_mode:(print flat)" in script  # choices still come from argparse
    assert "copy this to a file" not in script.lower()  # shtab's manual-install header is gone


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not installed")
def test_script_is_valid_zsh(tmp_path):
    path = tmp_path / "_halide"
    path.write_text(completion.zsh_script(build_parser()))
    result = subprocess.run(["zsh", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_first_run_writes_script_and_hooks_zshrc_once(zsh_home):
    (zsh_home / ".zshrc").write_text("autoload -Uz compinit && compinit\n")

    assert completion.ensure_zsh_completion(build_parser(), interactive=True) is True
    script = zsh_home / "data" / "halide" / "zsh" / "_halide"
    assert script.read_text().startswith("#compdef halide")
    zshrc = (zsh_home / ".zshrc").read_text()
    assert zshrc.startswith("autoload -Uz compinit && compinit\n")  # user's own lines untouched
    assert zshrc.count(completion.ZSHRC_BEGIN) == 1
    assert f'fpath=("{script.parent}" $fpath)' in zshrc

    assert completion.ensure_zsh_completion(build_parser(), interactive=True) is False
    assert (zsh_home / ".zshrc").read_text() == zshrc


def test_removed_block_is_not_added_back(zsh_home):
    completion.ensure_zsh_completion(build_parser(), interactive=True)
    (zsh_home / ".zshrc").write_text("# the user took it out\n")

    assert completion.ensure_zsh_completion(build_parser(), interactive=True) is False
    assert completion.ZSHRC_BEGIN not in (zsh_home / ".zshrc").read_text()


def test_existing_block_is_not_duplicated(zsh_home):
    """e.g. the stamp was lost with a cleared ~/.local/share, but .zshrc still has the block."""
    (zsh_home / ".zshrc").write_text(completion.zshrc_block(completion.completion_dir()))
    assert completion.ensure_zsh_completion(build_parser(), interactive=True) is False
    assert (zsh_home / ".zshrc").read_text().count(completion.ZSHRC_BEGIN) == 1


def test_stale_script_is_regenerated(zsh_home):
    completion.ensure_zsh_completion(build_parser(), interactive=True)
    script = zsh_home / "data" / "halide" / "zsh" / "_halide"
    script.write_text("#compdef halide\n# from an older halide\n")

    completion.ensure_zsh_completion(build_parser(), interactive=True)
    assert script.read_text() == completion.zsh_script(build_parser())


@pytest.mark.parametrize(
    "setup",
    [
        lambda mp: mp.setenv("SHELL", "/bin/bash"),
        lambda mp: mp.setenv("HALIDE_NO_COMPLETION", "1"),
    ],
    ids=["not zsh", "opted out"],
)
def test_does_nothing_outside_its_remit(zsh_home, monkeypatch, setup):
    setup(monkeypatch)
    assert completion.ensure_zsh_completion(build_parser(), interactive=True) is False
    assert not (zsh_home / "data").exists()
    assert not (zsh_home / ".zshrc").exists()


def test_does_nothing_when_not_at_a_terminal(zsh_home):
    assert completion.ensure_zsh_completion(build_parser(), interactive=False) is False
    assert not (zsh_home / "data").exists()


def test_failures_never_break_a_run(zsh_home, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise PermissionError("read-only home")

    monkeypatch.setattr(completion, "ensure_zsh_completion", boom)
    completion.maybe_install_completion(build_parser())  # must not raise
    assert capsys.readouterr().err == ""
