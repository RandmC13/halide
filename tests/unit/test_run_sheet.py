"""Shared run-sheet helpers of the multi-frame commands (cli/_run_sheet.py)."""

from __future__ import annotations

import re
from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.gpu_service import ServiceUnavailable
from halide.processing import DeviceFailure, service_stopped

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _result(name, warning):
    return BatchResult(job=BatchJob(Path(name), Path("out") / name), error=None, warning=warning)


def test_other_gpu_failures_are_not_mistaken_for_a_stopped_service():
    assert service_stopped(DeviceFailure("RuntimeError", "boom", False).warning("a.tif")) is None
    assert service_stopped(DeviceFailure("MemoryError", "", True).warning("a.tif")) is None
