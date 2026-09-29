"""End-to-end tests for `halide profile` and the --save-profile-as / bare-name --profile flags,
isolated from the real default profiles directory via $XDG_CONFIG_HOME."""

import sys

import numpy as np
import pytest

from halide.calibration.profile_store import load_profile
from halide.cli.main import main
from halide.core.types import DensityProfile
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


@pytest.fixture(autouse=True)
def isolated_profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "halide" / "profiles"


@pytest.fixture
def negative_tiff(tmp_path):
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def test_invert_save_profile_as_then_reuse_by_name(negative_tiff, tmp_path, isolated_profiles_dir):
    output1 = tmp_path / "positive1.tiff"
    exit_code = main(
        [
            "invert", str(negative_tiff), str(output1),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "portra400-c41-v600",
        ]
    )
    assert exit_code == 0
    assert (isolated_profiles_dir / "portra400-c41-v600.json").exists()
    saved = load_profile(isolated_profiles_dir / "portra400-c41-v600.json")
    assert saved.name == "portra400-c41-v600"
    assert saved.white_balance == pytest.approx((2.28, 1.0, 1.47))

    # Reuse by bare name, no path needed.
    output2 = tmp_path / "positive2.tiff"
    exit_code = main(["invert", str(negative_tiff), str(output2), "--profile", "portra400-c41-v600"])
    assert exit_code == 0
    assert output2.exists()


def test_save_profile_as_errors_with_per_frame_auto_density(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="needs a single resolved profile"):
        main(
            [
                "invert", str(negative_tiff), str(output),
                "--auto-density", "--save-profile-as", "should-fail",
            ]
        )


def test_save_profile_as_errors_with_invert_only(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="needs a single resolved profile"):
        main(["invert", str(negative_tiff), str(output), "--invert-only", "--save-profile-as", "should-fail"])


def test_profile_list_show_rename_delete(negative_tiff, tmp_path, isolated_profiles_dir, capsys):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "ektar100",
        ]
    )
    capsys.readouterr()  # discard invert's own output

    assert main(["profile", "list"]) == 0
    assert "ektar100" in capsys.readouterr().out

    assert main(["profile", "show", "ektar100"]) == 0
    show_output = capsys.readouterr().out
    assert "2.28" in show_output

    assert main(["profile", "rename", "ektar100", "ektar100-renamed"]) == 0
    capsys.readouterr()
    assert not (isolated_profiles_dir / "ektar100.json").exists()
    assert (isolated_profiles_dir / "ektar100-renamed.json").exists()

    assert main(["profile", "delete", "ektar100-renamed", "--yes"]) == 0
    assert not (isolated_profiles_dir / "ektar100-renamed.json").exists()


def test_save_profile_as_with_notes(negative_tiff, tmp_path, isolated_profiles_dir):
    exit_code = main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "with-notes", "--notes", "shot under a light table",
        ]
    )
    assert exit_code == 0
    saved = load_profile(isolated_profiles_dir / "with-notes.json")
    assert saved.notes == "shot under a light table"


def test_profile_edit_with_flags(negative_tiff, tmp_path, isolated_profiles_dir, capsys):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "editable",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "profile", "edit", "editable",
            "--film-stock", "Kodak Ektar 100", "--notes", "anchored on frame 12",
        ]
    )
    assert exit_code == 0
    updated = load_profile(isolated_profiles_dir / "editable.json")
    assert updated.film_stock == "Kodak Ektar 100"
    assert updated.notes == "anchored on frame 12"
    # profile identity/calibration data untouched
    assert updated.name == "editable"
    assert updated.white_balance == pytest.approx((2.28, 1.0, 1.47))

    # clearing a field with an explicit empty string
    assert main(["profile", "edit", "editable", "--notes", ""]) == 0
    assert load_profile(isolated_profiles_dir / "editable.json").notes is None


def test_profile_edit_interactive_prompts(negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "interactive-edit",
        ]
    )

    # Simulate a real terminal: film_stock and notes get new values, process/scanner are left
    # as-is (empty input), matching the order fields are prompted in (_EDIT_FIELD_LABELS).
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    responses = iter(["Kodak Ektar 100", "", "", "developed at home, scanned on a V600"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    exit_code = main(["profile", "edit", "interactive-edit"])
    assert exit_code == 0

    updated = load_profile(isolated_profiles_dir / "interactive-edit.json")
    assert updated.film_stock == "Kodak Ektar 100"
    assert updated.notes == "developed at home, scanned on a V600"
    assert updated.process is None
    assert updated.scanner is None


def test_profile_edit_interactive_no_changes(negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch, capsys):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "untouched",
        ]
    )
    capsys.readouterr()

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    exit_code = main(["profile", "edit", "untouched"])
    assert exit_code == 0
    assert "No changes made" in capsys.readouterr().out


def test_profile_edit_nonexistent_errors():
    with pytest.raises(SystemExit, match="no saved profile named"):
        main(["profile", "edit", "nonexistent", "--notes", "x"])


def test_profile_edit_non_interactive_without_flags_errors(negative_tiff, tmp_path, isolated_profiles_dir):
    main(
        [
            "invert", str(negative_tiff), str(tmp_path / "out.tiff"),
            "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            "--save-profile-as", "editable",
        ]
    )
    with pytest.raises(SystemExit, match="interactive terminal"):
        main(["profile", "edit", "editable"])


def test_profile_list_empty(isolated_profiles_dir, capsys):
    assert main(["profile", "list"]) == 0
    assert "No saved profiles" in capsys.readouterr().out


def test_profile_show_nonexistent_errors():
    with pytest.raises(SystemExit, match="no such profile file"):
        main(["profile", "show", "nonexistent"])


def test_batch_save_profile_as_with_auto_density_roll(tmp_path, isolated_profiles_dir):
    rng = np.random.default_rng(0)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
        img[8:16, :] = HIGHLIGHT_RGB
        img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
        write_tiff(in_dir / f"f{i}.tiff", img, icc_profile=build_icc(LINEAR_TAGS))

    out_dir = tmp_path / "out"
    exit_code = main(
        ["batch", str(in_dir), str(out_dir), "--auto-density-roll", "--quiet", "--save-profile-as", "roll-cal"]
    )
    assert exit_code == 0
    assert (isolated_profiles_dir / "roll-cal.json").exists()


def _two_saved_profiles(negative_tiff, tmp_path):
    for name, rm in (("older-roll", 2.0), ("just-made", 2.28)):
        main(
            [
                "invert", str(negative_tiff), str(tmp_path / f"{name}.tiff"),
                "--rm", str(rm), "--bm", "1.47", "--rs", "1.32", "--bs", "0.78", "--save-profile-as", name,
            ]
        )


def _output_white_balance(path):
    from halide.io.tiff import read_tiff_description
    from halide.processing import read_provenance

    return read_provenance(read_tiff_description(path))["white_balance"]


def test_invert_without_a_source_offers_saved_profiles_newest_first(
    negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch, capsys
):
    _two_saved_profiles(negative_tiff, tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    assert main(["invert", str(negative_tiff), str(tmp_path / "out.tiff")]) == 0
    menu = capsys.readouterr().out
    assert menu.index("'just-made'") < menu.index("'older-roll'")  # the profile just made is first
    assert _output_white_balance(tmp_path / "out.tiff")[0] == pytest.approx(2.28)


def test_batch_without_a_source_offers_saved_profiles(negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch, capsys):
    _two_saved_profiles(negative_tiff, tmp_path)
    roll = tmp_path / "roll"
    roll.mkdir()
    (roll / "frame1.tiff").write_bytes(negative_tiff.read_bytes())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")  # the older profile

    assert main(["batch", str(roll), str(tmp_path / "out"), "--quiet"]) == 0
    assert "Profile 'older-roll'" in capsys.readouterr().out
    assert _output_white_balance(tmp_path / "out" / "frame1.tiff")[0] == pytest.approx(2.0)


def test_without_a_terminal_a_missing_source_is_still_an_actionable_error(negative_tiff, tmp_path, isolated_profiles_dir):
    with pytest.raises(SystemExit, match="halide calibrate ROLL_DIR"):
        main(["invert", str(negative_tiff), str(tmp_path / "out.tiff")])


def test_show_names_the_roll_and_flags_it_when_moved(tmp_path, capsys, isolated_profiles_dir):
    from halide.calibration.profile_store import save_named_profile
    from halide.core.types import DensityProfile

    roll = tmp_path / "Roll16"
    roll.mkdir()
    (roll / "a.tif").write_bytes(b"")
    anchors = [
        {"frame": str(roll / n), "x": 1, "y": 1, "rgb": [0.1, 0.1, 0.1], "scan": None} for n in ("a.tif", "b.tif")
    ]
    profile = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0), name="r16")
    save_named_profile(profile, "r16", anchors=anchors, roll=str(roll))

    assert main(["profile", "show", "r16"]) == 0
    out = capsys.readouterr().out
    assert f"Roll:           {roll}" in out and "not found" not in out
    assert "Points:         2 on 2 frames (1 missing)" in out

    roll.rename(tmp_path / "Roll16-moved")
    assert main(["profile", "show", "r16"]) == 0
    out = capsys.readouterr().out
    assert "not found - moved or deleted?" in out
    assert "(2 missing)" in out


def test_show_doesnt_invent_a_full_path_for_an_older_relative_roll(tmp_path, capsys, monkeypatch, isolated_profiles_dir):
    from halide.calibration.profile_store import save_named_profile
    from halide.core.types import DensityProfile

    anchors = [{"frame": "pre-processed/a.tif", "x": 1, "y": 1, "rgb": [0.1, 0.1, 0.1], "scan": None}]
    profile = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0), name="old")
    save_named_profile(profile, "old", anchors=anchors, roll="pre-processed")
    monkeypatch.chdir(tmp_path)

    assert main(["profile", "show", "old"]) == 0
    out = capsys.readouterr().out
    assert "Roll:           pre-processed\n" in out
    assert str(tmp_path / "pre-processed") not in out
    assert "recorded relative to the folder halide calibrate ran in" in out


# --- Task 3: profile store safety (F03, F24) ----------------------------------------------------


def _save(negative_tiff, output, name, *, rm=2.0, bm=1.0, extra=()):
    return main(
        [
            "invert", str(negative_tiff), str(output),
            "--rm", str(rm), "--bm", str(bm), "--save-profile-as", name, *extra,
        ]
    )


def test_save_profile_as_rejects_invalid_name_at_parse_time(negative_tiff, tmp_path, isolated_profiles_dir, capsys):
    output = tmp_path / "out.tiff"
    with pytest.raises(SystemExit):
        _save(negative_tiff, output, "bad/name")
    assert "can't contain / or \\" in capsys.readouterr().err
    assert not output.exists()  # rejected before anything was developed
    assert not isolated_profiles_dir.exists()


def test_save_profile_as_existing_non_interactive_needs_overwrite(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out1.tiff", "roll16", rm=2.0) == 0
    out2 = tmp_path / "out2.tiff"
    with pytest.raises(SystemExit, match=r"already exists.*--overwrite"):
        _save(negative_tiff, out2, "roll16", rm=3.0)
    assert not out2.exists()  # checked before any frame was developed
    assert load_profile(isolated_profiles_dir / "roll16.json").white_balance[0] == pytest.approx(2.0)


def test_save_profile_as_existing_with_overwrite_flag_replaces_it(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out1.tiff", "roll16", rm=2.0) == 0
    assert _save(negative_tiff, tmp_path / "out2.tiff", "roll16", rm=3.0, extra=("--overwrite",)) == 0
    assert load_profile(isolated_profiles_dir / "roll16.json").white_balance[0] == pytest.approx(3.0)


def test_save_profile_as_existing_interactive_confirms(negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch):
    assert _save(negative_tiff, tmp_path / "out1.tiff", "roll16", rm=2.0) == 0
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert _save(negative_tiff, tmp_path / "out2.tiff", "roll16", rm=3.0) == 0
    assert load_profile(isolated_profiles_dir / "roll16.json").white_balance[0] == pytest.approx(3.0)


def test_profile_rename_onto_existing_refused(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "a.tiff", "a") == 0
    assert _save(negative_tiff, tmp_path / "b.tiff", "b") == 0
    with pytest.raises(SystemExit, match="already exists"):
        main(["profile", "rename", "a", "b"])
    assert (isolated_profiles_dir / "a.json").exists()
    assert (isolated_profiles_dir / "b.json").exists()


def test_rename_with_a_path_explains_names_only(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "roll16") == 0
    path = str(isolated_profiles_dir / "roll16.json")
    with pytest.raises(SystemExit, match="`profile rename` takes a profile name"):
        main(["profile", "rename", path, "newname"])
    with pytest.raises(SystemExit, match="`profile delete` takes a profile name"):
        main(["profile", "delete", path])
    with pytest.raises(SystemExit, match="`profile edit` takes a profile name"):
        main(["profile", "edit", path, "--notes", "x"])


def test_profile_show_typo_suggests(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "Portra400") == 0
    with pytest.raises(SystemExit, match="did you mean 'Portra400'"):
        main(["profile", "show", "Portra40"])


def test_profile_rename_typo_suggests(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "Portra400") == 0
    with pytest.raises(SystemExit, match="did you mean 'Portra400'"):
        main(["profile", "rename", "Portra40", "newname"])


def test_profile_edit_typo_suggests(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "Portra400") == 0
    with pytest.raises(SystemExit, match="did you mean 'Portra400'"):
        main(["profile", "edit", "Portra40", "--notes", "x"])


def test_profile_delete_typo_suggests(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "Portra400") == 0
    with pytest.raises(SystemExit, match="did you mean 'Portra400'"):
        main(["profile", "delete", "Portra40"])


def test_profile_delete_non_interactive_requires_yes(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "roll16") == 0
    with pytest.raises(SystemExit, match="--yes"):
        main(["profile", "delete", "roll16"])
    assert (isolated_profiles_dir / "roll16.json").exists()


def test_profile_delete_interactive_confirms(tmp_path, isolated_profiles_dir, monkeypatch):
    from halide.calibration.profile_store import save_named_profile
    from halide.core.types import DensityProfile

    anchors = [
        {"frame": "a.tif", "x": 1, "y": 1, "rgb": [0.1, 0.1, 0.1], "scan": None},
        {"frame": "b.tif", "x": 2, "y": 2, "rgb": [0.2, 0.2, 0.2], "scan": None},
    ]
    save_named_profile(
        DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.0, 1.0, 1.0)), "roll16", anchors=anchors
    )
    saved = load_profile(isolated_profiles_dir / "roll16.json")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    captured = {}

    def fake_input(prompt=""):
        captured["prompt"] = prompt
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["profile", "delete", "roll16"]) == 0
    assert not (isolated_profiles_dir / "roll16.json").exists()
    assert "roll16" in captured["prompt"]
    assert "2 points" in captured["prompt"]
    assert saved.created_at in captured["prompt"]


def test_profile_delete_interactive_declining_keeps_the_profile(negative_tiff, tmp_path, isolated_profiles_dir, monkeypatch):
    assert _save(negative_tiff, tmp_path / "out.tiff", "roll16") == 0
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(SystemExit, match="Nothing deleted"):
        main(["profile", "delete", "roll16"])
    assert (isolated_profiles_dir / "roll16.json").exists()


def test_damaged_profile_json_gives_plain_message(isolated_profiles_dir):
    isolated_profiles_dir.mkdir(parents=True)
    (isolated_profiles_dir / "corruptme.json").write_text('{"white_balance": [1.0, 1.0, 1.0], "density_sca')

    with pytest.raises(SystemExit) as excinfo:
        main(["profile", "show", "corruptme"])
    message = str(excinfo.value)
    assert "is damaged" in message
    assert "halide calibrate --profile corruptme" in message
    assert "halide profile delete corruptme" in message


def test_profile_delete_with_slash_is_rejected_before_touching_anything(negative_tiff, tmp_path, isolated_profiles_dir):
    assert _save(negative_tiff, tmp_path / "out.tiff", "roll16") == 0
    with pytest.raises(SystemExit):
        main(["profile", "delete", "../roll16"])
    assert (isolated_profiles_dir / "roll16.json").exists()


def test_saving_narrow_picks_warns_where_the_fit_holds(negative_tiff, tmp_path, monkeypatch, capsys, isolated_profiles_dir):
    # D-2: picks that only pin the fit down over part of the frame are still saved, with the
    # picker's own "Fit reliable over …" note as a warning.
    picked = DensityProfile(white_balance=(1.5, 1.0, 0.7), density_scale=(1.1, 1.0, 0.9), source="manual")
    note = "Fit reliable over D 0.9-1.3 only - add a point in the shadows or highlights for the ends of the roll"
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: (picked, None, note))
    assert main(["invert", str(negative_tiff), str(tmp_path / "p.tiff"), "--pick", "--save-profile-as", "narrow"]) == 0
    assert (isolated_profiles_dir / "narrow.json").exists()
    out = capsys.readouterr().out
    assert "⚠ Warning: " + note in out

    # A fit that holds over the whole frame saves quietly; so does a pick without saving.
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: (picked, None, None))
    assert main(["invert", str(negative_tiff), str(tmp_path / "q.tiff"), "--pick", "--save-profile-as", "wide"]) == 0
    assert "Fit reliable" not in capsys.readouterr().out
    monkeypatch.setattr("halide.gui.quick_pick.run_quick_pick", lambda path: (picked, None, note))
    assert main(["invert", str(negative_tiff), str(tmp_path / "r.tiff"), "--pick"]) == 0
    assert "Fit reliable" not in capsys.readouterr().out
