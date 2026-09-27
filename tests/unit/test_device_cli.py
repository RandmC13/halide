"""`--device`/`$HALIDE_DEVICE` wired into the CLI: the shared flag, its resolution into the CLI's
usual clean error exit, the run-sheet/summary-line Compute row, and end-to-end behaviour on
`invert`/`print` (the two commands that actually pass the resolved device into processing in this
task — see docs/plans/gpu-acceleration.md Task 5 and its Ruling R7)."""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pytest

from halide.cli._device_args import DEFAULT_DEVICE, add_device_argument, device_row, resolve_device_arg
from halide.cli.main import build_parser, main
from halide.device import ComputeDevice, DeviceUnavailableError
from halide.io.tiff import read_tiff, write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# add_device_argument / resolve_device_arg (pure, no I/O)
# ---------------------------------------------------------------------------


def test_add_device_argument_offers_the_three_choices_and_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_device_argument(parser)
    args = parser.parse_args([])
    assert args.device is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "tpu"])


def test_default_device_constant_is_auto():
    # R3: one default, "auto", everywhere.
    assert DEFAULT_DEVICE == "auto"


def test_resolve_device_arg_uses_resolve_device(monkeypatch):
    import halide.cli._device_args as device_args

    seen = []
    monkeypatch.setattr(
        device_args, "resolve_device", lambda requested: seen.append(requested) or ComputeDevice(kind="cpu")
    )
    device = resolve_device_arg(argparse.Namespace(device="cpu"))
    assert device.kind == "cpu"
    assert seen == ["cpu"]


def test_resolve_device_arg_turns_bad_env_value_into_system_exit(monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "tpu")
    with pytest.raises(SystemExit, match="HALIDE_DEVICE"):
        resolve_device_arg(argparse.Namespace(device=None))


def test_resolve_device_arg_turns_unavailable_gpu_into_system_exit_naming_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # no cupy installed at all
    with pytest.raises(SystemExit, match="halide gpu --install"):
        resolve_device_arg(argparse.Namespace(device="gpu"))


# ---------------------------------------------------------------------------
# device_row
# ---------------------------------------------------------------------------


def test_device_row_cpu():
    assert device_row(ComputeDevice(kind="cpu")) == "CPU"


def test_device_row_gpu_names_the_card():
    device = ComputeDevice(kind="gpu", name="NVIDIA GeForce RTX 3070", memory_free=1, memory_total=2)
    assert device_row(device) == "GPU — NVIDIA GeForce RTX 3070"


# ---------------------------------------------------------------------------
# every command takes --device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["invert", "in.tif", "out.tif", "--device", "cpu"],
        ["batch", "in_dir", "out_dir", "--device", "cpu"],
        ["print", "in.tif", "out.tif", "--device", "cpu"],
        ["export", "in.tif", "out.png", "--device", "cpu"],
        ["contact", "in_dir", "sheet.jpg", "--device", "cpu"],
        ["calibrate", "--device", "cpu"],
    ],
)
def test_every_command_accepts_device_flag(argv):
    build_parser().parse_args(argv)  # must not raise


# ---------------------------------------------------------------------------
# end to end: invert
# ---------------------------------------------------------------------------

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)
MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


@pytest.fixture
def negative_tiff(tmp_path):
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def _provenance(path):
    description = read_tiff(path).description
    return json.loads(description)["halide"]


def test_invert_device_cpu_writes_device_cpu_in_provenance(negative_tiff, tmp_path):
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL, "--device", "cpu"]) == 0
    assert _provenance(output)["device"] == "cpu"


def test_invert_prints_the_device_on_its_summary_line(negative_tiff, tmp_path, capsys):
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL, "--device", "cpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "CPU" in out


def test_invert_device_gpu_without_cupy_errors_naming_install(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="halide gpu --install"):
        main(["invert", str(negative_tiff), str(output), *MANUAL, "--device", "gpu"])
    assert not output.exists()


def test_invert_device_absent_uses_halide_device_env(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "cpu")
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL]) == 0
    assert _provenance(output)["device"] == "cpu"


def test_invert_reports_gpu_fallback_reason_as_a_warning(negative_tiff, tmp_path, monkeypatch, capsys):
    import halide.cli._device_args as device_args

    monkeypatch.setattr(
        device_args,
        "resolve_device",
        lambda requested: ComputeDevice(kind="cpu", fallback_reason="cudaErrorInsufficientDriver"),
    )
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL]) == 0
    out = _strip(capsys.readouterr().out)
    assert "cudaErrorInsufficientDriver" in out


# ---------------------------------------------------------------------------
# end to end: print
# ---------------------------------------------------------------------------


def test_print_device_cpu_writes_device_cpu_in_provenance(negative_tiff, tmp_path):
    flat = tmp_path / "flat.tiff"
    assert main(["invert", str(negative_tiff), str(flat), *MANUAL, "--output", "flat"]) == 0
    printed = tmp_path / "printed.tiff"
    assert main(["print", str(flat), str(printed), "--device", "cpu"]) == 0
    assert _provenance(printed)["device"] == "cpu"


def test_print_prints_the_device_on_its_summary_line(negative_tiff, tmp_path, capsys):
    flat = tmp_path / "flat.tiff"
    assert main(["invert", str(negative_tiff), str(flat), *MANUAL, "--output", "flat"]) == 0
    capsys.readouterr()
    printed = tmp_path / "printed.tiff"
    assert main(["print", str(flat), str(printed), "--device", "cpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "CPU" in out


# ---------------------------------------------------------------------------
# run sheet: batch / print --directory / export --directory show a Compute row
# ---------------------------------------------------------------------------


def _write_negative(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll_dir(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(2):
        _write_negative(in_dir / f"frame_{i:02d}.tiff", seed=i)
    return in_dir


def test_batch_run_sheet_shows_compute_row(roll_dir, tmp_path, capsys):
    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--workers", "1", "--device", "cpu"]
    assert main(args) == 0
    out = _strip(capsys.readouterr().out)
    sheet = out[: out.index("Developing")]
    assert "Compute" in sheet and "CPU" in sheet


def test_batch_run_sheet_shows_gpu_fallback_reason(roll_dir, tmp_path, monkeypatch, capsys):
    import halide.cli._device_args as device_args

    monkeypatch.setattr(
        device_args,
        "resolve_device",
        lambda requested: ComputeDevice(kind="cpu", fallback_reason="the driver is too old"),
    )
    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--workers", "1"]
    assert main(args) == 0
    out = _strip(capsys.readouterr().out)
    assert "the driver is too old" in out
