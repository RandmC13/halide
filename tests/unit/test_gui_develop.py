"""The window's develop command, built and checked without Qt."""

from __future__ import annotations

import shlex
import sys
from dataclasses import replace

import pytest

from halide.calibration.profile_store import save_profile
from halide.cli.main import main
from halide.core.types import DensityProfile
from halide.gui.develop import (
    DevelopSettings,
    batch_command,
    example_name,
    existing_outputs,
    export_command,
    shown_command,
)
from tests.integration.test_run_log_cli import _negative


@pytest.fixture
def settings(tmp_path):
    roll = tmp_path / "Roll16"
    roll.mkdir()
    for i in (1, 2):
        _negative(roll / f"IMG_{i:04d}.tif", i)
    profile = tmp_path / "p.json"
    save_profile(DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78)), profile)
    return DevelopSettings(roll=roll, into=roll / "developed", profile=profile)


def test_defaults_pass_nothing_extra(settings):
    assert batch_command(settings) == [sys.executable, "-m", "halide.cli.main", "batch", str(settings.roll),
                                       str(settings.into), "--profile", str(settings.profile)]


@pytest.mark.parametrize("change,flags", [
    ({"output": "flat"}, ["--output", "flat"]),
    ({"name": "{roll}_{n:02}"}, ["--name={roll}_{n:02}"]),
    ({"existing": "replace"}, ["--overwrite"]),
    ({"existing": "skip"}, ["--skip-existing"]),
    ({"match_scan_exposure": True}, ["--match-scan-exposure"]),
    ({"device": "cpu"}, ["--device", "cpu"]),
])
def test_each_field_gives_its_flag(settings, change, flags):
    assert batch_command(replace(settings, **change))[8:] == flags


def test_the_shown_command_is_the_real_one(settings):
    s = replace(settings, name="{roll} {n}", existing="skip")
    assert shlex.split(shown_command(s)) == ["halide", *batch_command(s)[3:]]


def test_export_command(settings):
    s = replace(settings, sharing="jpg")
    assert export_command(s)[3:] == ["export", str(s.into), str(s.into / "sharing"), "--format", "jpg",
                                     "--log", str(s.into / "halide-log.txt")]


@pytest.mark.parametrize("name", ["{stem}", "{roll}_{n:02}"])
def test_existing_outputs_agree_with_what_batch_writes(settings, name):
    s = replace(settings, name=name)
    assert existing_outputs(s) == []
    assert main(batch_command(s)[3:] + ["--workers", "1"]) == 0
    assert sorted(existing_outputs(s)) == sorted(p for p in s.into.iterdir() if p.suffix == ".tif")
    assert len(existing_outputs(s)) == 2


def test_example_name_and_a_refused_template(settings):
    assert example_name(settings) == "IMG_0001.tif → IMG_0001.tif"
    assert example_name(replace(settings, name="{roll}_{n:02}")) == "IMG_0001.tif → Roll16_01.tif"
    with pytest.raises(ValueError, match="unknown field"):
        example_name(replace(settings, name="{date}"))
    with pytest.raises(ValueError, match="hidden file"):
        example_name(replace(settings, name=".x{n}"))


def test_a_template_starting_with_a_dash_is_a_name_not_a_flag(settings):
    assert example_name(replace(settings, name="-{stem}")) == "IMG_0001.tif → -IMG_0001.tif"
