"""The end-of-run summary's figures and rows (batch/summary.py)."""

from __future__ import annotations

from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.batch.summary import CHECK_HINT, RollSummary, summary_rows
from halide.cli.console.runsheet import RunSheet

SEP = RunSheet.SEP


def _ok(name, grade=None, exposure=0.3, fitted=True, scale=None, warning=None):
    job = BatchJob(Path(name), Path("out") / name)
    if scale is not None:
        return BatchResult(job=job, error=None, warning=warning, linear_scale=scale)
    return BatchResult(job=job, error=None, warning=warning, contrast=grade, exposure=exposure,
                       grade_fitted=fitted)


def _failed(name, error="the scan is gamma-encoded", warning=None):
    return BatchResult(job=BatchJob(Path(name), Path("out") / name), error=error, warning=warning)


def _rows(results, kind="develop", **kw):
    return summary_rows(RollSummary.from_results(results, kind, **kw))


def test_prints_row_gives_range_median_and_cap_count():
    results = [_ok(f"{i}.tif", g, e) for i, (g, e) in enumerate(
        [(0.86, 0.12), (0.93, 0.40), (1.0, 0.61), (1.0, 0.3), (1.0, 0.2)])]
    assert _rows(results) == [
        ("row", "Prints", f"grade 0.86–1.00 (median 1.00, 3 at the 1.0 cap){SEP}exposure +0.12 to +0.61"),
    ]


def test_no_cap_part_when_nothing_is_at_the_cap_and_single_values_are_not_ranges():
    assert _rows([_ok("a.tif", 0.9, 0.2)]) == [("row", "Prints", f"grade 0.90 (median 0.90){SEP}exposure +0.20")]


def test_a_grade_0_25_under_the_median_is_flagged_and_0_24_is_not():
    results = [_ok(f"{i}.tif", 1.0) for i in range(4)] + [_ok("low.tif", 0.75), _ok("fine.tif", 0.76)]
    rows = _rows(results)
    checks = [r for r in rows if r[1] == "Check"]
    assert checks == [("warn", "Check", f"low.tif  grade 0.75 — {CHECK_HINT}")]


def test_no_check_with_fewer_than_five_fitted_frames():
    results = [_ok(f"{i}.tif", 1.0) for i in range(3)] + [_ok("low.tif", 0.4)]
    assert not [r for r in _rows(results) if r[1] == "Check"]


def test_pinned_grades_are_never_compared_and_give_no_prints_row():
    results = [_ok(f"{i}.tif", 0.7, fitted=False) for i in range(6)]
    assert _rows(results) == []


def test_pinned_frames_do_not_count_towards_the_five():
    results = [_ok(f"{i}.tif", 1.0) for i in range(4)] + [_ok("p.tif", 0.3, fitted=False)]
    assert not [r for r in _rows(results) if r[1] == "Check"]


def test_flat_runs_show_the_scale_range():
    rows = _rows([_ok("a.tif", scale=0.012), _ok("b.tif", scale=0.0193)])
    assert rows == [("row", "Prints", f"flat{SEP}scale x0.012–x0.0193")]


def test_export_has_no_prints_row():
    assert _rows([_ok("a.tif", 0.9)], kind="export") == []


def test_warnings_group_by_message_listing_up_to_four_names():
    msg = "metadata not copied (exiftool failed)"
    results = [_ok(f"{i}.tif", 1.0, warning=msg) for i in range(6)] + [_ok("x.tif", 1.0, warning="other")]
    rows = [r for r in _rows(results) if r[1] == "Warnings"]
    assert rows == [
        ("warn", "Warnings", f"6 frames: {msg} — 0.tif, 1.tif, 2.tif, 3.tif and 2 more"),
        ("warn", "Warnings", "x.tif: other"),
    ]


def test_a_frame_with_two_warnings_appears_in_both_groups():
    rows = _rows([_ok("a.tif", 1.0, warning="one; two")])
    assert [r[2] for r in rows if r[1] == "Warnings"] == ["a.tif: one", "a.tif: two"]


def test_a_stopped_gpu_service_is_said_once():
    stopped = lambda w: "no reply: EOFError" if w.startswith("GPU gone") else None  # noqa: E731
    results = [_ok(f"{n}.tif", 1.0, warning="GPU gone") for n in "abc"] + [_ok("d.tif", 1.0, warning="else")]
    rows = [r for r in _rows(results, stopped=stopped) if r[1] == "Warnings"]
    assert rows == [
        ("warn", "Warnings", "d.tif: else"),
        ("warn", "Warnings", "The GPU service stopped (no reply: EOFError); developing the remaining frames on the CPU"),
    ]


def test_every_frame_failed_gives_no_prints_or_check():
    results = [_failed(f"{i}.tif") for i in range(6)]
    assert _rows(results) == []


def test_failed_frames_warnings_are_still_grouped():
    rows = _rows([_failed("a.tif", warning="GPU out of memory")])
    assert rows == [("warn", "Warnings", "a.tif: GPU out of memory")]


def test_nothing_to_say_gives_no_rows():
    assert _rows([]) == []


def test_a_real_stopped_service_warning_is_said_once():
    from halide.gpu_service import ServiceUnavailable
    from halide.processing import service_stopped

    failure = ServiceUnavailable("the GPU service stopped responding (EOFError)").failure
    frame_warning = failure.warning("a.tif").removeprefix("a.tif: ")
    results = [_ok(f"{n}.tif", 1.0, warning=frame_warning) for n in "abc"] + [_ok("d.tif", 1.0, warning="something else")]
    rows = [r for r in _rows(results, stopped=service_stopped) if r[1] == "Warnings"]
    assert rows == [
        ("warn", "Warnings", "d.tif: something else"),
        ("warn", "Warnings", "The GPU service stopped (no reply: EOFError); developing the remaining frames on the CPU"),
    ]


def test_other_warnings_survive_next_to_the_service_one():
    from halide.gpu_service import ServiceUnavailable
    from halide.processing import service_stopped

    stopped = ServiceUnavailable("the GPU service can't be reached (OSError)").failure.warning("a.tif")
    both = f"{stopped.removeprefix('a.tif: ')}; the exif copy failed"
    texts = [r[2] for r in _rows([_ok("a.tif", 1.0, warning=both)], stopped=service_stopped)]
    assert "a.tif: the exif copy failed" in texts
    assert any("couldn't connect: OSError" in t for t in texts)
