"""Shared run-sheet helpers of the multi-frame commands (cli/_run_sheet.py)."""

from __future__ import annotations

import re
from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.cli._run_sheet import print_frame_warnings
from halide.gpu_service import ServiceUnavailable
from halide.processing import DeviceFailure, service_stopped

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _result(name, warning):
    return BatchResult(job=BatchJob(Path(name), Path("out") / name), error=None, warning=warning)


def test_a_stopped_gpu_service_is_reported_once_not_on_every_frame(capsys):
    reason = "the GPU service stopped responding (EOFError)"
    failure = ServiceUnavailable(reason).failure
    frame_warning = failure.warning("a.tif").removeprefix("a.tif: ")
    results = [_result(f"{n}.tif", frame_warning) for n in "abc"] + [_result("d.tif", "something else")]
    print_frame_warnings(results)
    lines = _ANSI.sub("", capsys.readouterr().out).splitlines()
    assert lines == [
        "⚠ Warning: d.tif: something else",
        "⚠ Warning: The GPU service stopped (no reply: EOFError); developing the remaining frames on the CPU",
    ]


def test_a_frames_other_warnings_survive_next_to_the_service_one(capsys):
    stopped = ServiceUnavailable("the GPU service can't be reached (OSError)").failure.warning("a.tif")
    both = f"{stopped.removeprefix('a.tif: ')}; the exif copy failed"
    print_frame_warnings([_result("a.tif", both)])
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "a.tif: the exif copy failed" in out
    assert "couldn't connect: OSError" in out


def test_other_gpu_failures_are_not_mistaken_for_a_stopped_service():
    assert service_stopped(DeviceFailure("RuntimeError", "boom", False).warning("a.tif")) is None
    assert service_stopped(DeviceFailure("MemoryError", "", True).warning("a.tif")) is None
