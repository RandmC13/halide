"""Small CLI wording and behaviour fixes: help text, empty-folder errors, flags that would do
nothing, and a few state fixes."""

import re

import numpy as np
import pytest

from halide.cli.main import build_parser, main
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


def _tiff(path, value=0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tiff(path, np.full((8, 8, 3), value, dtype=np.float32), icc_profile=build_icc(LINEAR_TAGS))
    return path


def _help(capsys, *argv):
    with pytest.raises(SystemExit):
        main([*argv, "--help"])
    return " ".join(_ANSI.sub("", capsys.readouterr().out).split())


@pytest.mark.parametrize("command, phrase", [
    ("batch", "Develop only the frames whose output doesn't exist yet (resumes an interrupted roll)"),
    ("print", "Print only the frames whose output doesn't exist yet (resumes an interrupted roll)"),
    ("export", "Export only the files whose output doesn't exist yet (resumes an interrupted roll)"),
    ("invert", "Develop only if the output doesn't exist yet"),
    ("contact", "Keep the sheet if it already exists"),
])
def test_skip_existing_help_uses_the_commands_own_verb(capsys, command, phrase):
    assert phrase in _help(capsys, command)


def test_contact_usage_shows_frames_then_sheet(capsys):
    text = _help(capsys, "contact")
    assert "FRAMES... SHEET" in text and "inputs" not in text


def test_batch_save_profile_help_only_promises_what_batch_can_save(capsys):
    text = _help(capsys, "batch")
    assert "picked" not in text.split("--save-profile-as")[1].split("--notes")[0]


def test_batch_summary_matches_what_the_run_says(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "Develop a folder of negative scans" in " ".join(capsys.readouterr().out.split())


def test_help_examples_use_no_private_roll_names(capsys):
    for argv in (["--help"], ["profile", "--help"], ["profile", "rename", "--help"]):
        with pytest.raises(SystemExit):
            main(argv)
        assert "roll16" not in capsys.readouterr().out.lower().replace(" ", "")


@pytest.mark.parametrize("argv, match", [
    (["print", "{src}", "{out}"], r"No TIFF files found in .*right folder\?$"),
    (["check", "{src}"], r"No TIFF files found — is this the right folder\?$"),
    (["contact", "{src}", "{out}/s.jpg"], r"No processed frames .*right folder\?$"),
])
def test_empty_folder_is_an_error_with_a_hint(tmp_path, argv, match):
    src = tmp_path / "empty"
    src.mkdir()
    argv = [a.format(src=src, out=tmp_path / "out") for a in argv]
    with pytest.raises(SystemExit, match=match):
        main(argv)


def test_profile_list_with_none_says_how_to_make_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert main(["profile", "list"]) == 0
    assert "make one with `halide open ROLL`" in capsys.readouterr().out


@pytest.mark.parametrize("command, extra", [("print", []), ("export", [])])
def test_a_refused_log_leaves_no_output_folder(tmp_path, command, extra):
    src = tmp_path / "in"
    _tiff(src / "a.tiff")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="is a picture, not a log"):
        main([command, str(src), str(out), "--log", str(src / "a.tiff")])
    assert not out.exists()


def test_check_lists_a_file_given_twice_once_and_keeps_same_named_files_apart(tmp_path, capsys):
    a, b = _tiff(tmp_path / "x" / "f.tiff"), _tiff(tmp_path / "y" / "f.tiff")
    main(["check", str(a), str(a), str(b)])
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "Checked 2 frames" in out and "(no EXIF)  " in out and "x/f.tiff" in out and "y/f.tiff" in out


def test_quality_takes_1_to_100_and_not_with_png(tmp_path):
    src = _tiff(tmp_path / "p.tiff")
    for bad in ("0", "101", "x"):
        with pytest.raises(SystemExit) as caught:
            main(["export", str(src), str(tmp_path / "o.jpg"), "--quality", bad])
        assert caught.value.code == 2
    with pytest.raises(SystemExit, match="--quality only applies to JPEG"):
        main(["export", str(tmp_path), str(tmp_path / "o"), "--format", "png", "--quality", "90"])


@pytest.mark.parametrize("flag", ["--columns", "--frame-width"])
@pytest.mark.parametrize("value", ["0", "-3", "x"])
def test_sheet_layout_numbers_must_be_positive_whole_numbers(tmp_path, flag, value):
    with pytest.raises(SystemExit) as caught:
        main(["contact", str(tmp_path), str(tmp_path / "s.jpg"), flag, value])
    assert caught.value.code == 2  # argparse's own usage error


def test_gpu_yes_without_install_is_refused():
    with pytest.raises(SystemExit, match="--yes only goes with --install"):
        main(["gpu", "--yes"])


def test_invert_only_with_match_scan_exposure_warns_instead_of_failing(tmp_path, capsys):
    src = _tiff(tmp_path / "n.tiff")
    assert main(["invert", str(src), str(tmp_path / "o.tiff"), "--invert-only", "--match-scan-exposure"]) == 0
    assert "--match-scan-exposure has no effect here" in capsys.readouterr().out


def test_explicit_unit_scales_count_as_manual_calibration():
    from halide.cli._calibration_args import manual_calibration_given

    ns = build_parser().parse_args(["invert", "a.tif", "b.tif", "--rs", "1", "--bs", "1"])
    assert manual_calibration_given(ns)
    assert not manual_calibration_given(build_parser().parse_args(["invert", "a.tif", "b.tif"]))


def test_explicit_unit_scales_conflict_with_a_source(tmp_path):
    src = _tiff(tmp_path / "n.tiff")
    with pytest.raises(SystemExit, match="can't be combined"):
        main(["invert", str(src), str(tmp_path / "o.tiff"), "--rs", "1", "--auto-density"])


def test_open_unknown_profile_fails_before_any_window(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as caught:
        main(["open", "--profile", "nope"])
    assert "no saved profile named 'nope'" in str(caught.value) and "couldn't launch" not in str(caught.value)


def test_open_profile_with_a_roll_path_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--profile reopens"):
        main(["open", str(tmp_path), "--profile", "x"])


def test_calibrate_is_an_unknown_command(capsys):
    # Replaced by `halide open` outright: argparse's own error, with no "renamed" message.
    with pytest.raises(SystemExit) as caught:
        main(["calibrate"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'calibrate'" in err and "open" in err


def _fake_window(monkeypatch, saved):
    import sys
    import types

    fake_app = types.ModuleType("halide.gui.app")
    fake_app.main = lambda **kwargs: saved
    monkeypatch.setitem(sys.modules, "halide.gui.app", fake_app)
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("HALIDE_DEVICE", "cpu")


def test_open_prints_the_next_step_after_a_save(monkeypatch, capsys, tmp_path):
    roll = tmp_path / "portra roll 03"
    _fake_window(monkeypatch, ("Portra 400", roll))
    assert main(["open", str(roll)]) == 0
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "Saved profile 'Portra 400'" in out
    assert f"next: halide batch '{roll}' '{roll}-developed' --profile 'Portra 400'" in out


def test_open_after_a_save_on_loose_scans_names_placeholders(monkeypatch, capsys):
    _fake_window(monkeypatch, ("Portra 400", None))
    assert main(["open"]) == 0
    assert "next: halide batch ROLL_DIR OUT_DIR --profile 'Portra 400'" in capsys.readouterr().out


def test_open_prints_no_next_step_without_a_save(monkeypatch, capsys):
    _fake_window(monkeypatch, None)
    assert main(["open"]) == 0
    out = capsys.readouterr().out
    assert "Saved profile" not in out and "next:" not in out
