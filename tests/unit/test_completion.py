import shutil
import subprocess

import pytest

from halide.cli import completion
from halide.cli.main import build_parser

SCRIPTS = {"zsh": completion.zsh_script, "bash": completion.bash_script, "fish": completion.fish_script}
SYNTAX_CHECK = {"zsh": ["zsh", "-n"], "bash": ["bash", "-n"], "fish": ["fish", "--no-execute"]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home: halide's data dir, fish's config dir and every rc file under tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.delenv("HALIDE_NO_COMPLETION", raising=False)
    return tmp_path


def install(shell):
    return completion.ensure_completion(build_parser(), shell=shell, interactive=True)


@pytest.mark.parametrize("shell", completion.SHELLS)
def test_script_covers_every_subcommand_and_flag(shell):
    script = SCRIPTS[shell](build_parser())
    parser = build_parser()
    for name, subparser in completion._subparsers(parser):
        assert name in script
        for action in subparser._actions:
            for flag in action.option_strings:
                word = flag.lstrip("-") if shell == "fish" else flag  # fish: `-l profile`
                assert word in script, f"{shell}: {name} {flag} missing"


@pytest.mark.parametrize("shell", completion.SHELLS)
def test_script_is_valid_shell_code(shell, tmp_path):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not installed")
    path = tmp_path / "script"
    path.write_text(SCRIPTS[shell](build_parser()))
    result = subprocess.run([*SYNTAX_CHECK[shell], str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_values_are_completed_not_just_flag_names():
    zsh = completion.zsh_script(build_parser())
    assert ":Input linear TIFF scan of a negative:_files -g '*.(#i)tif(|f)'" in zsh
    assert ":Directory of input linear TIFF scans:_files -/" in zsh
    assert ":profile:_halide_profiles" in zsh
    assert ":output_mode:(print flat)" in zsh

    bash = completion.bash_script(build_parser())
    assert "_shtab_halide_invert_pos_0_COMPGEN=_halide_tiff_files" in bash
    assert "_shtab_halide_batch_pos_0_COMPGEN=_shtab_compgen_dirs" in bash
    assert "_shtab_halide_invert___profile_COMPGEN=_halide_profiles" in bash

    fish = completion.fish_script(build_parser())
    assert "-n \"__halide_pos 'invert' 0\" -a '(__halide_tiffs)'" in fish
    assert "-n \"__halide_pos 'invert' 1\" -F" in fish
    assert "-n \"__halide_pos 'contact' 0 more\" -F" in fish  # nargs="+"
    assert "-l profile -x -a '(__halide_profiles)'" in fish
    assert "-l output -x -a 'print flat'" in fish
    assert "-n \"__halide_pos 'profile' 0\" -a 'show'" in fish  # nested subcommands

    for script in (zsh, bash, fish):
        assert "copy this" not in script.lower()  # shtab's manual-install header is gone


@pytest.mark.parametrize("shell", ["zsh", "bash"])
def test_first_run_writes_script_and_hooks_rc_once(home, shell):
    rc = home / f".{shell}rc"
    rc.write_text("# the user's own settings\n")

    note = install(shell)
    assert note and "open a new terminal" in note
    target = completion.shell_setup(shell).target
    assert target.read_text() == SCRIPTS[shell](build_parser())
    text = rc.read_text()
    assert text.startswith("# the user's own settings\n")  # untouched
    assert text.count(completion.RC_BEGIN) == 1
    assert str(target.parent if shell == "zsh" else target) in text

    assert install(shell) is None
    assert rc.read_text() == text


def test_fish_needs_no_rc_file(home):
    assert install("fish") == "added ~/config/fish/completions/halide.fish — open a new terminal to use it"
    target = home / "config" / "fish" / "completions" / "halide.fish"
    assert target.read_text() == completion.fish_script(build_parser())
    assert not (home / "config" / "fish" / "config.fish").exists()


@pytest.mark.parametrize("shell", ["zsh", "bash"])
def test_removed_block_is_not_added_back(home, shell):
    install(shell)
    rc = home / f".{shell}rc"
    rc.write_text("# the user took it out\n")
    assert install(shell) is None
    assert completion.RC_BEGIN not in rc.read_text()


def test_removed_fish_file_is_not_recreated(home):
    install("fish")
    target = completion.shell_setup("fish").target
    target.unlink()
    assert install("fish") is None
    assert not target.exists()


def test_someone_elses_fish_file_is_left_alone(home):
    target = completion.shell_setup("fish").target
    target.parent.mkdir(parents=True)
    target.write_text("# my own halide completions\n")
    assert install("fish") is None
    assert target.read_text() == "# my own halide completions\n"


def test_existing_block_is_not_duplicated(home):
    """e.g. the stamp was lost with a cleared ~/.local/share, but .zshrc still has the block."""
    (home / ".zshrc").write_text(completion.shell_setup("zsh").rc_block())
    assert install("zsh") is None
    assert (home / ".zshrc").read_text().count(completion.RC_BEGIN) == 1


def test_zdotdir_is_respected(home, monkeypatch):
    monkeypatch.setenv("ZDOTDIR", str(home / "zsh"))
    (home / "zsh").mkdir()
    install("zsh")
    assert completion.RC_BEGIN in (home / "zsh" / ".zshrc").read_text()
    assert not (home / ".zshrc").exists()


@pytest.mark.parametrize("shell", completion.SHELLS)
def test_stale_script_is_regenerated(home, shell):
    install(shell)
    target = completion.shell_setup(shell).target
    target.write_text(completion._HEADER + "# from an older halide\n")
    install(shell)
    assert target.read_text() == SCRIPTS[shell](build_parser())


@pytest.mark.parametrize(
    "shell, env",
    [("tcsh", {}), ("zsh", {"HALIDE_NO_COMPLETION": "1"})],
    ids=["unsupported shell", "opted out"],
)
def test_does_nothing_outside_its_remit(home, monkeypatch, shell, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert install(shell) is None
    assert not (home / "data").exists()
    assert not (home / ".zshrc").exists()


def test_does_nothing_when_not_at_a_terminal(home):
    assert completion.ensure_completion(build_parser(), shell="zsh", interactive=False) is None
    assert not (home / "data").exists()


def test_detect_shell_falls_back_to_login_shell(monkeypatch):
    import psutil

    class NotAShell:
        def name(self):
            return "python3"

    monkeypatch.setattr(psutil.Process, "parent", lambda self: NotAShell())
    monkeypatch.setenv("SHELL", "/usr/local/bin/fish")
    assert completion.detect_shell() == "fish"
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    assert completion.detect_shell() is None


def test_failures_never_break_a_run(home, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise PermissionError("read-only home")

    monkeypatch.setattr(completion, "ensure_completion", boom)
    completion.maybe_install_completion(build_parser())  # must not raise
    assert capsys.readouterr().err == ""
