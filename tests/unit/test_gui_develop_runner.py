"""The develop runner against a fake halide child that prints protocol lines."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time

import pytest
from PySide6.QtWidgets import QApplication

from halide.gui.develop_runner import DevelopRunner

FAKE = r'''
import os, signal, sys, time
cancelled = False
def stop(*_):
    global cancelled
    cancelled = True
signal.signal(signal.SIGINT, stop)
mode = sys.argv[1]
assert os.environ["HALIDE_PROGRESS"] == "machine"
if mode == "refused":
    print("halide: error: some flag", file=sys.stderr)
    sys.exit(2)
print("Developing 2 frames")
print("not\ta protocol line at all")
print("start\t0\tframe_00.tif", flush=True)
time.sleep(0.05)
print("done\t0\tok\tgrade 0.88", flush=True)
if mode == "slow":
    for _ in range(200):
        if cancelled:
            break
        time.sleep(0.05)
    print("end\t1\t2\t1", flush=True)
    sys.exit(130)
print("start\t1\tframe_01.tif\ndone\t1\tfailed\tbad scan", flush=True)
print("summary\trow\tPrints\tgrade 0.88", flush=True)
print("summary\tPrints\tthe old three-field form", flush=True)
print("end\t1\t2\t0", flush=True)
sys.exit(1 if mode == "fail" else 0)
'''


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])  # a bare QCoreApplication would break later window tests


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "fake.py"
    path.write_text(FAKE)
    return lambda mode: [sys.executable, str(path), mode]


def _record(runner):
    events = []
    runner.frameStarted.connect(lambda i, n: events.append(("start", i, n)))
    runner.frameDone.connect(lambda i, ok, d: events.append(("done", i, ok, d)))
    runner.summaryRow.connect(lambda kind, k, v: events.append(("summary", kind, k, v)))
    runner.finished.connect(lambda a, b, c: events.append(("finished", a, b, c, runner.step)))
    runner.failed.connect(lambda t: events.append(("failed", t)))
    return events


def _pump(app, until, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert until()


def test_signals_in_order_and_other_lines_ignored(app, fake):
    runner = DevelopRunner()
    events = _record(runner)
    runner.run(fake("ok"))
    _pump(app, lambda: not runner.is_running() and events)
    assert events == [
        ("start", 0, "frame_00.tif"), ("done", 0, True, "grade 0.88"),
        ("start", 1, "frame_01.tif"), ("done", 1, False, "bad scan"),
        ("summary", "row", "Prints", "grade 0.88"), ("finished", 1, 2, False, 0),
    ]


def test_a_child_that_stops_without_an_end_line_fails_with_its_stderr(app, fake):
    runner = DevelopRunner()
    events = _record(runner)
    runner.run(fake("refused"))
    _pump(app, lambda: not runner.is_running() and events)
    assert events == [("failed", "halide: error: some flag")]


def test_cancel_sends_sigint_and_the_childs_own_end_line_arrives(app, fake):
    runner = DevelopRunner()
    events = _record(runner)
    runner.run(fake("slow"))
    _pump(app, lambda: any(e[0] == "done" for e in events))
    runner.cancel()
    _pump(app, lambda: not runner.is_running())
    assert events[-1] == ("finished", 1, 2, True, 0)


def test_run_after_chains_only_after_a_good_end(app, fake):
    runner = DevelopRunner()
    events = _record(runner)
    runner.run(fake("ok"))
    runner.run_after(fake("ok"))
    _pump(app, lambda: [e[0] for e in events].count("finished") == 2 and not runner.is_running())
    assert [e for e in events if e[0] == "finished"] == [("finished", 1, 2, False, 0), ("finished", 1, 2, False, 1)]

    events.clear()
    runner.run(fake("fail"))
    runner.run_after(fake("ok"))
    _pump(app, lambda: not runner.is_running() and events and events[-1][0] == "finished")
    time.sleep(0.2)
    app.processEvents()
    assert [e for e in events if e[0] == "finished"] == [("finished", 1, 2, False, 0)]


def test_a_child_with_nothing_to_do_ends_well_without_an_end_line(app):
    runner = DevelopRunner()
    events = _record(runner)
    runner.run([sys.executable, "-c", "print('Nothing to do — every output already exists')"])
    _pump(app, lambda: not runner.is_running() and events)
    assert events == [("finished", 0, 0, False, 0)]
