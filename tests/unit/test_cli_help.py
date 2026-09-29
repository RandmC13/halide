"""--version, the examples every --help ends with, and the range checks on numbers."""

from __future__ import annotations

import argparse
import shlex

import pytest

from halide.cli import main as cli_main
from halide.cli._help import (
    contrast_grade, finite_float, positive_float, worker_count,
)

COMMANDS = ("invert", "batch", "print", "export", "contact", "check", "profile", "calibrate", "gpu")
PROFILE_COMMANDS = ("list", "show", "rename", "delete", "edit")


def _help(argv, capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        cli_main.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "halide 0.1.0"


@pytest.mark.parametrize("argv", [[c] for c in COMMANDS] + [["profile", c] for c in PROFILE_COMMANDS])
def test_every_subcommand_help_has_examples(argv, capsys):
    out = _help(argv, capsys)
    assert "examples:" in out
    assert "halide " + " ".join(argv) in out.split("examples:")[1]


@pytest.mark.parametrize("argv", [[c] for c in COMMANDS] + [["profile", c] for c in PROFILE_COMMANDS])
def test_every_example_parses_with_the_real_parser(argv, capsys):
    """Examples must use real, current flags: parse each one."""
    out = _help(argv, capsys)
    lines = [l.strip() for l in out.split("examples:")[1].splitlines() if l.strip()]
    assert 1 <= len(lines) <= 3
    parser = cli_main.build_parser()
    for line in lines:
        words = shlex.split(line)
        assert words[0] == "halide"
        parser.parse_args(words[1:])


def test_top_level_examples_parse(capsys):
    out = _help([], capsys)
    parser = cli_main.build_parser()
    for line in out.split("examples:")[1].splitlines():
        if line.strip():
            parser.parse_args(shlex.split(line)[1:])


@pytest.mark.parametrize("argv", [["profile", c] for c in PROFILE_COMMANDS])
def test_profile_subcommands_have_a_description(argv, capsys):
    out = _help(argv, capsys)
    body = out.split("\n\n")[1]
    assert body.strip() and not body.startswith("positional") and not body.startswith("options")


def test_calibrate_profile_help_is_a_whole_sentence(capsys):
    out = " ".join(_help(["calibrate"], capsys).split())
    assert "roll folder" in out.split(" --profile PROFILE ")[1].split("--device")[0]


def test_no_fine_tune_left_in_help(capsys):
    for c in COMMANDS:
        assert "fine-tune" not in _help([c], capsys).lower()


def _workers_help(command, capsys) -> str:
    out = _help([command], capsys)
    text = out.split("--workers")[1].split("--quiet")[0].split("--device")[0]
    return " ".join(text.split())


@pytest.mark.parametrize("command", ["batch", "print", "export", "contact"])
def test_workers_help_is_identical_on_every_command(command, capsys):
    assert _workers_help(command, capsys) == _workers_help("batch", capsys)


@pytest.mark.parametrize("command", ["invert", "batch"])
def test_manual_calibration_flags_state_their_default(command, capsys):
    out = " ".join(_help([command], capsys).split())
    for flag in ("--rm", "--bm", "--rs", "--bs"):
        assert "default: 1.0" in out.split(f" {flag} X ")[1].split(" --")[0]


@pytest.mark.parametrize("argv", [["invert", "--debug", "a.tif", "b.tif"], ["profile", "list", "--debug"]])
def test_debug_is_accepted_after_the_subcommand(argv):
    assert cli_main.build_parser().parse_args(argv).debug is True


def test_debug_default_stays_false_and_before_subcommand_still_works():
    parser = cli_main.build_parser()
    assert parser.parse_args(["invert", "a.tif", "b.tif"]).debug is False
    assert parser.parse_args(["--debug", "invert", "a.tif", "b.tif"]).debug is True


@pytest.mark.parametrize("flag, value, message", [
    ("--contrast", "-1", "paper grade"),
    ("--contrast", "0", "paper grade"),
    ("--contrast", "2.5", "at most 2"),
    ("--contrast", "nan", "finite"),
    ("--exposure", "inf", "finite"),
    ("--exposure", "nan", "finite"),
    ("--rm", "0", "above zero"),
    ("--bm", "-1", "above zero"),
    ("--rs", "-2", "above zero"),
    ("--bs", "inf", "finite"),
    ("--workers", "0", "1 or more"),
    ("--workers", "-3", "1 or more"),
])
def test_out_of_range_numbers_are_refused(flag, value, message, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main.build_parser().parse_args(["invert", "a.tif", "b.tif", f"{flag}={value}"])
    if flag == "--workers":  # invert has no --workers
        return
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3"])
def test_workers_below_one_refused_on_batch(value, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main.build_parser().parse_args(["batch", "in", "out", f"--workers={value}"])
    assert exc.value.code == 2
    assert "1 or more" in capsys.readouterr().err


def test_in_range_numbers_are_accepted():
    args = cli_main.build_parser().parse_args(
        ["batch", "in", "out", "--contrast", "1.5", "--exposure", "-0.4", "--rm", "0.9", "--workers", "2"]
    )
    assert (args.contrast, args.exposure, args.rm, args.workers) == (1.5, -0.4, 0.9, 2)
    assert contrast_grade("2") == 2.0 and finite_float("0") == 0.0
    assert positive_float("1e-3") == 1e-3 and worker_count("1") == 1


def test_non_numbers_get_a_plain_message():
    for fn in (contrast_grade, finite_float, positive_float, worker_count):
        with pytest.raises(argparse.ArgumentTypeError, match="number"):
            fn("abc")


def _overwrite_help(command, capsys) -> str:
    out = " ".join(_help([command], capsys).split())
    return out.split(" --overwrite Replace")[1].split("--skip-existing")[0]


@pytest.mark.parametrize("command", ["invert", "batch"])
def test_overwrite_help_mentions_the_profile_where_one_is_saved(command, capsys):
    assert "saved profile" in _overwrite_help(command, capsys)


@pytest.mark.parametrize("command", ["print", "export", "contact"])
def test_overwrite_help_says_nothing_of_profiles_where_none_is_saved(command, capsys):
    assert "profile" not in _overwrite_help(command, capsys)


@pytest.mark.parametrize("command", ["batch", "print", "export", "contact"])
def test_workers_help_doesnt_say_develop(command, capsys):
    assert "develop" not in _workers_help(command, capsys)
