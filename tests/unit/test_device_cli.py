"""`--device`/`$HALIDE_DEVICE` wired into the CLI: the shared flag, its resolution into the CLI's
usual clean error exit, the run-sheet/summary-line Compute row, and end-to-end behaviour on
`invert`/`print` (Task 5, Ruling R7) and `export`'s single-file mode (Task 6), and the multi-frame
commands handing the resolved device to their worker pools (Task 7)."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pytest

import halide.cli._device_args as _device_args
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


def test_device_flag_ignores_case_and_surrounding_space():
    parser = argparse.ArgumentParser()
    add_device_argument(parser)
    assert parser.parse_args(["--device", "GPU"]).device == "gpu"
    assert parser.parse_args(["--device", " Cpu "]).device == "cpu"


def test_default_device_constant_is_auto():
    # R3: one default, "auto", everywhere.
    assert DEFAULT_DEVICE == "auto"


def test_default_device_constant_is_reexported_from_halide_device():
    # cli/_device_args.DEFAULT_DEVICE must be an import of halide.device's own constant (checked by
    # source, not just value, since two equal string literals would pass a `==` check either way) —
    # see tests/unit/test_device.py::test_default_device_constant_actually_governs_resolve_device
    # for proof resolve_device itself is the thing that reads it.
    import ast

    source = ast.parse(pathlib.Path(_device_args.__file__).read_text())
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom) and node.module == "halide.device"
        for alias in node.names
    }
    assert "DEFAULT_DEVICE" in imported_names


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
# end to end: export (single file — Task 6; the bulk-directory mode's pool is further down)
# ---------------------------------------------------------------------------


def test_export_device_cpu_writes_a_delivery_image(negative_tiff, tmp_path):
    positive = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(positive), *MANUAL]) == 0
    delivery = tmp_path / "delivery.png"
    assert main(["export", str(positive), str(delivery), "--device", "cpu"]) == 0
    assert delivery.exists()


def test_export_prints_the_device_on_its_summary_line(negative_tiff, tmp_path, capsys):
    positive = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(positive), *MANUAL]) == 0
    capsys.readouterr()
    delivery = tmp_path / "delivery.png"
    assert main(["export", str(positive), str(delivery), "--device", "cpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "CPU" in out


def test_export_device_gpu_without_cupy_errors_naming_install(negative_tiff, tmp_path, monkeypatch):
    positive = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(positive), *MANUAL]) == 0
    monkeypatch.setitem(sys.modules, "cupy", None)
    delivery = tmp_path / "delivery.png"
    with pytest.raises(SystemExit, match="halide gpu --install"):
        main(["export", str(positive), str(delivery), "--device", "gpu"])
    assert not delivery.exists()


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


# ---------------------------------------------------------------------------
# Task 7: the multi-frame commands hand their resolved device to the worker pool
# ---------------------------------------------------------------------------

_FAKE_GPU = ComputeDevice(kind="gpu", name="Fake GPU", memory_free=6 * 2**30, memory_total=8 * 2**30)


@pytest.fixture
def gpu_resolves(monkeypatch):
    """--device auto finds a (fake) GPU; nothing runs on it — the pool runners are replaced."""
    import halide.cli._device_args as device_args

    monkeypatch.setattr(device_args, "resolve_device", lambda requested: _FAKE_GPU)


def _recording_runner(calls, warning="out of GPU memory — developed this frame on the CPU instead"):
    from halide.batch.orchestrator import BatchResult

    def runner(jobs, *args, **kwargs):
        calls.append(kwargs)
        results = [BatchResult(job=job, error=None, warning=warning) for job in jobs]
        for result in results:
            if kwargs.get("on_result"):
                kwargs["on_result"](result)
        return results

    return runner


def _recording_default_count(seen):
    def default_count(jobs, **kwargs):
        seen.append(kwargs.get("device"))
        return 1

    return default_count


@pytest.mark.parametrize("quiet", [False, True])
def test_batch_hands_the_device_to_its_workers_and_shows_their_warnings(roll_dir, tmp_path, monkeypatch, capsys,
                                                                     gpu_resolves, quiet):
    import halide.cli.commands.batch_cmd as batch_cmd

    calls, seen = [], []
    monkeypatch.setattr(batch_cmd, "run_batch", _recording_runner(calls))
    monkeypatch.setattr(batch_cmd, "default_worker_count", _recording_default_count(seen))
    args = ["batch", str(roll_dir), str(tmp_path / "out"), *MANUAL] + (["--quiet"] if quiet else [])
    assert main(args) == 0
    assert calls[0]["device"] is _FAKE_GPU
    assert seen == [_FAKE_GPU]  # the default worker count knows it's a GPU run (VRAM cap)
    out = _strip(capsys.readouterr().out)
    # A frame redone on the CPU is reported, even with --quiet (warnings always print).
    assert "frame_00.tiff: out of GPU memory" in out and "frame_01.tiff: out of GPU memory" in out


def test_export_directory_hands_the_device_to_its_workers(roll_dir, tmp_path, monkeypatch, capsys, gpu_resolves):
    import halide.cli.commands.export_cmd as export_cmd

    calls, seen = [], []
    monkeypatch.setattr(export_cmd, "run_export_batch", _recording_runner(calls))
    monkeypatch.setattr(export_cmd, "default_export_worker_count", _recording_default_count(seen))
    assert main(["export", str(roll_dir), str(tmp_path / "out"), "--quiet"]) == 0
    assert calls[0]["device"] is _FAKE_GPU and seen == [_FAKE_GPU]
    assert "out of GPU memory" in _strip(capsys.readouterr().out)


def test_print_directory_hands_the_device_to_its_workers(roll_dir, tmp_path, monkeypatch, capsys, gpu_resolves):
    import halide.cli.commands.print_cmd as print_cmd

    calls, seen = [], []
    monkeypatch.setattr(print_cmd, "run_print_batch", _recording_runner(calls))
    monkeypatch.setattr(print_cmd, "default_worker_count", _recording_default_count(seen))
    assert main(["print", str(roll_dir), str(tmp_path / "out"), "--quiet"]) == 0
    assert calls[0]["device"] is _FAKE_GPU and seen == [_FAKE_GPU]
    assert "out of GPU memory" in _strip(capsys.readouterr().out)


def test_batch_warns_when_explicit_workers_look_too_many_for_the_gpu(roll_dir, tmp_path, monkeypatch, capsys):
    # 1 GiB free and at least a CUDA context (256 MiB) per GPU worker: 8 workers won't all fit.
    # Frames that don't are redone on the CPU, so --workers 8 would mostly measure CPU fallbacks —
    # say so up front.
    import halide.cli._device_args as device_args
    import halide.cli.commands.batch_cmd as batch_cmd

    small = ComputeDevice(kind="gpu", name="Fake GPU", memory_free=1 * 2**30, memory_total=2 * 2**30)
    monkeypatch.setattr(device_args, "resolve_device", lambda requested: small)
    monkeypatch.setattr(batch_cmd, "run_batch", _recording_runner([], warning=None))
    assert main(["batch", str(roll_dir), str(tmp_path / "out"), *MANUAL, "--workers", "8"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "--workers 8 may not fit in free GPU memory" in out


def test_batch_names_gpu_memory_when_it_set_the_worker_count(roll_dir, tmp_path, monkeypatch, capsys):
    import halide.cli._device_args as device_args
    import halide.cli.commands.batch_cmd as batch_cmd

    # Room for one GPU worker's CUDA context and a small frame, not two: the card sets the count.
    small = ComputeDevice(kind="gpu", name="Fake GPU", memory_free=300 * 2**20, memory_total=2 * 2**30)
    monkeypatch.setattr(device_args, "resolve_device", lambda requested: small)
    monkeypatch.setattr(batch_cmd, "run_batch", _recording_runner([], warning=None))
    assert main(["batch", str(roll_dir), str(tmp_path / "out"), *MANUAL]) == 0
    workers = [line for line in _strip(capsys.readouterr().out).splitlines() if "Workers" in line]
    assert workers and "free GPU memory" in workers[0]


def test_batch_worker_row_on_the_cpu_is_unchanged(roll_dir, tmp_path, monkeypatch, capsys):
    import halide.cli.commands.batch_cmd as batch_cmd

    monkeypatch.setattr(batch_cmd, "run_batch", _recording_runner([], warning=None))
    assert main(["batch", str(roll_dir), str(tmp_path / "out"), *MANUAL, "--device", "cpu"]) == 0
    workers = [line for line in _strip(capsys.readouterr().out).splitlines() if "Workers" in line]
    assert workers and "free memory and CPU cores" in workers[0] and "GPU" not in workers[0]


def _recording_resolve(monkeypatch, device):
    import halide.cli._device_args as device_args

    calls = []
    monkeypatch.setattr(device_args, "resolve_device", lambda requested: calls.append(requested) or device)
    return calls


@pytest.mark.parametrize("env", [None, "auto", "cpu"])
def test_contact_never_probes_the_gpu_unless_asked_to(roll_dir, tmp_path, monkeypatch, capsys, env):
    # Its frames are already developed, so it never uses a GPU: resolving `auto` would only import
    # CuPy, make a CUDA context (~0.6 s, ~300 MB of GPU memory) and possibly warn about a GPU
    # the command was never going to use.
    if env is None:
        monkeypatch.delenv("HALIDE_DEVICE", raising=False)
    else:
        monkeypatch.setenv("HALIDE_DEVICE", env)
    calls = _recording_resolve(monkeypatch, ComputeDevice(kind="cpu", fallback_reason="cudaErrorInsufficientDriver"))
    assert main(["contact", str(roll_dir), str(tmp_path / "sheet.jpg"), "--workers", "1"]) == 0
    out = _strip(capsys.readouterr().out)
    assert calls == []
    assert "cudaErrorInsufficientDriver" not in out and "GPU not usable" not in out
    compute = [line for line in out.splitlines() if line.startswith("Compute")]
    assert compute and compute[0] == "Compute: CPU"


def test_contact_device_auto_flag_does_not_probe_either(roll_dir, tmp_path, monkeypatch, capsys):
    calls = _recording_resolve(monkeypatch, _FAKE_GPU)
    assert main(["contact", str(roll_dir), str(tmp_path / "sheet.jpg"), "--workers", "1", "--device", "auto"]) == 0
    assert calls == []


@pytest.mark.parametrize("how", ["flag", "env"])
def test_contact_explicit_gpu_still_fails_fast_without_one(roll_dir, tmp_path, monkeypatch, how):
    monkeypatch.setitem(sys.modules, "cupy", None)
    argv = ["contact", str(roll_dir), str(tmp_path / "sheet.jpg"), "--workers", "1"]
    if how == "flag":
        argv += ["--device", "gpu"]
    else:
        monkeypatch.setenv("HALIDE_DEVICE", " GPU ")
    with pytest.raises(SystemExit, match="halide gpu --install"):
        main(argv)
    assert not (tmp_path / "sheet.jpg").exists()


def test_contact_explicit_gpu_says_its_thumbnails_run_on_the_cpu(roll_dir, tmp_path, monkeypatch, capsys):
    calls = _recording_resolve(monkeypatch, _FAKE_GPU)
    assert main(["contact", str(roll_dir), str(tmp_path / "sheet.jpg"), "--workers", "1", "--device", "gpu"]) == 0
    assert calls == ["gpu"]
    compute = [line for line in _strip(capsys.readouterr().out).splitlines() if line.startswith("Compute")]
    assert compute and compute[0].startswith("Compute: CPU") and "need no GPU" in compute[0]


def test_contact_bad_env_value_is_still_a_clean_error(roll_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "tpu")
    with pytest.raises(SystemExit, match="HALIDE_DEVICE"):
        main(["contact", str(roll_dir), str(tmp_path / "sheet.jpg"), "--workers", "1"])


def test_empty_env_value_is_not_an_error_on_the_cli(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setenv("HALIDE_DEVICE", "")
    monkeypatch.setitem(sys.modules, "cupy", None)  # auto -> CPU, as with no variable at all
    assert main(["invert", str(negative_tiff), str(tmp_path / "out.tif"), *MANUAL]) == 0


def test_calibrate_hands_the_device_to_the_picker(monkeypatch, gpu_resolves):
    # The picker's contact-sheet window develops every frame through batch's own _worker.
    import types

    seen = {}
    fake_app = types.ModuleType("halide.gui.app")
    fake_app.main = lambda **kwargs: seen.update(kwargs)
    monkeypatch.setitem(sys.modules, "halide.gui.app", fake_app)
    monkeypatch.setenv("DISPLAY", ":99")
    assert main(["calibrate"]) == 0
    assert seen["device"] is _FAKE_GPU
