"""The run log (batch/runlog.py): three flushed stages, readable as plain text."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from halide.batch.orchestrator import BatchJob, BatchResult
from halide.batch.runlog import RunLog, plain_row_lines, previous_unfinished

T0 = datetime(2026, 10, 2, 14, 31)
T1 = datetime(2026, 10, 2, 14, 32)
ROWS = [("row", "Roll", "Roll16 · 2 frames → out/"), ("warn", "Scans", "exposures differ"),
        ("note", "Scans", "details: halide check Roll16")]


def _clock(*times):
    it = iter(times)
    return lambda: next(it)


def _ok(name, grade=0.88, fitted=True, gain=1.0, warning=None, seconds=4.1):
    return BatchResult(job=BatchJob(Path(name), Path("out") / name, scan_gain=gain), error=None,
                       warning=warning, contrast=grade, exposure=0.39, grade_fitted=fitted, seconds=seconds)


def _run(path, results, kind="develop", kept=(), end="✓ Developed 2 frames in 9.0s", rows=(), error=None):
    log = RunLog(path, kind, now=_clock(T0, T1))
    log.start("halide batch in out", ROWS, [r.job.input_path.name for r in results], list(kept), "0.1.0")
    for r in results:
        log.frame(r)
    log.finish(end, list(rows), error=error)
    return log


def test_a_block_has_heading_command_sheet_table_end_and_closing_rule(tmp_path):
    path = tmp_path / "halide-log.txt"
    log = _run(path, [_ok("IMG_0138.tif"), _ok("IMG_0151.tif", 0.46)],
               rows=[("row", "Prints", "grade 0.46–0.88 (median 0.67)")])
    assert log.error is None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("━━ 2026-10-02 14:31 · halide 0.1.0 ━━")
    assert lines[1] == "halide batch in out"
    assert lines[2] == "  Roll           Roll16 · 2 frames → out/"
    assert lines[3] == "  Scans          ⚠ exposures differ"
    assert lines[4] == "                 details: halide check Roll16"
    assert lines[5] == ""
    assert lines[6].split() == ["Frame", "Result", "Grade", "Exposure", "Scan", "gain", "Time", "Notes"]
    assert lines[7].split() == ["IMG_0138.tif", "developed", "0.88", "+0.39", "1.00", "4.1s"]
    assert lines[8].split()[:3] == ["IMG_0151.tif", "developed", "0.46"]
    assert lines[9] == ""
    assert lines[10] == "✓ Developed 2 frames in 9.0s"
    assert lines[11] == "  Prints         grade 0.46–0.88 (median 0.67)"
    assert lines[12].startswith("━━ finished 14:32 ━━")
    assert lines[6].index("Grade") == lines[7].index("0.88")  # columns line up


def test_each_frame_is_on_disk_before_the_next(tmp_path):
    path = tmp_path / "halide-log.txt"
    log = RunLog(path, "develop", now=_clock(T0, T1))
    log.start("halide batch in out", ROWS, ["a.tif", "b.tif"], [], "0.1.0")
    log.frame(_ok("a.tif"))
    assert "a.tif" in path.read_text(encoding="utf-8")
    assert "finished" not in path.read_text(encoding="utf-8")


def test_two_runs_append_two_blocks_and_a_closed_block_gives_no_note(tmp_path):
    path = tmp_path / "halide-log.txt"
    _run(path, [_ok("a.tif")])
    _run(path, [_ok("a.tif")])
    text = path.read_text(encoding="utf-8")
    assert text.count("━━ finished") == 2
    assert "didn't finish" not in text


def test_an_unfinished_block_makes_the_next_run_say_so(tmp_path):
    path = tmp_path / "halide-log.txt"
    killed = RunLog(path, "develop", now=_clock(T0))
    killed.start("halide batch in out", ROWS, ["a.tif"], [], "0.1.0")
    killed.frame(_ok("a.tif"))  # then SIGKILL: no finish()
    killed._file.close()
    assert previous_unfinished(path) == "2026-10-02 14:31"
    _run(path, [_ok("a.tif")])
    assert ("⚠ The previous run in this log (started 2026-10-02 14:31) didn't finish — it stopped "
            "after the frames listed above.") in path.read_text(encoding="utf-8")


def test_the_unfinished_check_reads_only_the_tail_of_a_big_log(tmp_path):
    path = tmp_path / "halide-log.txt"
    path.write_text("x" * 1_000_000 + "\n", encoding="utf-8")
    killed = RunLog(path, "develop", now=_clock(T0))
    killed.start("halide batch in out", ROWS, ["a.tif"], [], "0.1.0")
    killed._file.close()
    assert previous_unfinished(path) == "2026-10-02 14:31"


def test_kept_failed_pinned_and_warned_frames(tmp_path):
    path = tmp_path / "halide-log.txt"
    failed = BatchResult(job=BatchJob(Path("bad.tif"), Path("out/bad.tif")), error="the scan is gamma-encoded")
    _run(path, [_ok("p.tif", 0.7, fitted=False, gain=1.26, warning="metadata not copied"), failed],
         kept=["old.tif"])
    text = path.read_text(encoding="utf-8")
    kept_line = next(line for line in text.splitlines() if line.startswith("old.tif"))
    assert kept_line.split() == ["old.tif", "kept", "—", "—", "—", "—", "already", "there,", "kept"]
    pinned = next(line for line in text.splitlines() if line.startswith("p.tif"))
    assert "0.70 (pinned)" in pinned and "1.26" in pinned and pinned.endswith("metadata not copied")
    bad = next(line for line in text.splitlines() if line.startswith("bad.tif"))
    assert bad.split()[:2] == ["bad.tif", "failed"] and bad.endswith("the scan is gamma-encoded")


def test_flat_and_export_tables(tmp_path):
    flat = tmp_path / "flat.txt"
    r = BatchResult(job=BatchJob(Path("a.tif"), Path("o/a.tif")), error=None, linear_scale=0.0153, seconds=2.0)
    _run(flat, [r])
    line = next(x for x in flat.read_text(encoding="utf-8").splitlines() if x.startswith("a.tif"))
    assert line.split()[:5] == ["a.tif", "developed", "flat", "x0.0153", "—"]
    exp = tmp_path / "export.txt"
    _run(exp, [BatchResult(job=BatchJob(Path("a.tif"), Path("o/a.png")), error=None, seconds=1.0)], kind="export")
    lines = exp.read_text(encoding="utf-8").splitlines()
    assert next(x for x in lines if x.startswith("Frame")).split() == ["Frame", "Result", "Time", "Notes"]
    assert next(x for x in lines if x.startswith("a.tif")).split() == ["a.tif", "exported", "1.0s"]


def test_print_kind_says_printed(tmp_path):
    path = tmp_path / "p.txt"
    _run(path, [_ok("a.tif")], kind="print")
    assert next(x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("a.tif")).split()[1] == "printed"


def test_non_ascii_names_stay_aligned_and_utf8(tmp_path):
    path = tmp_path / "halide-log.txt"
    _run(path, [_ok("Été 01.tif"), _ok("b.tif")])
    lines = path.read_bytes().decode("utf-8").splitlines()
    header = next(x for x in lines if x.startswith("Frame"))
    row = next(x for x in lines if x.startswith("Été 01.tif"))
    assert header.index("Result") == row.index("developed")


def test_cancel_before_any_frame_still_closes_the_block(tmp_path):
    path = tmp_path / "halide-log.txt"
    _run(path, [], end="✗ Cancelled - 0 of 37 frames developed; nothing half-written")
    text = path.read_text(encoding="utf-8")
    assert "Cancelled - 0 of 37" in text and "━━ finished" in text


def test_an_error_is_recorded_before_the_closing_rule(tmp_path):
    path = tmp_path / "halide-log.txt"
    _run(path, [_ok("a.tif")], end=None, error="MemoryError")
    lines = path.read_text(encoding="utf-8").splitlines()
    i = lines.index("✗ Stopped by an error: MemoryError")
    assert lines[i + 1].startswith("━━ finished")


def test_an_unwritable_log_reports_once_and_never_raises(tmp_path):
    log = _run(tmp_path / "missing-folder" / "halide-log.txt", [_ok("a.tif")])
    assert log.error is not None and "missing-folder" in log.error


def test_plain_row_lines_stack_labels():
    assert plain_row_lines([("warn", "Warnings", "a"), ("warn", "Warnings", "b"), ("ok", "Calibration", "saved")]) == [
        "  Warnings       ⚠ a",
        "                 ⚠ b",
        "  Calibration    ✓ saved",
    ]


def test_surrogate_escape_file_names_are_written_legibly(tmp_path):
    path = tmp_path / "halide-log.txt"
    # A filename with surrogate escapes (as seen on Linux with non-UTF-8 bytes in filenames)
    bad_name = "bad\udcff.tif"
    r = BatchResult(job=BatchJob(Path(bad_name), Path("o/bad.tif")), error=None, contrast=0.88,
                    exposure=0.39, grade_fitted=True, seconds=4.1)
    log = RunLog(path, "develop", now=_clock(T0, T1))
    log.start("halide batch in out", ROWS, [bad_name], [], "0.1.0")
    log.frame(r)
    log.finish("✓ Developed 1 frame in 4.1s", [], error=None)
    assert log.error is None
    text = path.read_text(encoding="utf-8")
    assert "bad\\udcff.tif" in text


def test_unknown_kind_does_not_raise_and_sets_error(tmp_path):
    path = tmp_path / "halide-log.txt"
    log = RunLog(path, "nonsense", now=_clock(T0, T1))
    log.start("halide batch in out", ROWS, ["a.tif"], [], "0.1.0")
    log.frame(_ok("a.tif"))
    log.finish("end", [])
    assert log.error is not None and "KeyError" in log.error


def test_multi_line_row_text_is_indented_under_the_text_column():
    lines = plain_row_lines([("warn", "Calibration", "first\nsecond\nthird"), ("row", "Log", "x")])
    pad = " " * (2 + 15 + 2)  # indent, label column, the icon and its space
    assert lines[0].endswith("first") and lines[1] == pad + "second" and lines[2] == pad + "third"
    assert lines[3].startswith("  Log")


def test_newlines_in_a_frame_note_are_folded_onto_one_line(tmp_path):
    path = tmp_path / "halide-log.txt"
    _run(path, [_ok("a.tif", warning="line one\nline two"), _ok("b.tif")])
    text = path.read_text(encoding="utf-8")
    line = next(x for x in text.splitlines() if x.startswith("a.tif"))
    assert line.endswith("line one / line two")
    assert "line two" not in [x.strip() for x in text.splitlines()]


def test_a_failed_frame_shows_its_error_then_its_warnings(tmp_path):
    path = tmp_path / "halide-log.txt"
    bad = BatchResult(job=BatchJob(Path("bad.tif"), Path("out/bad.tif")), error="boom", warning="w1; w2")
    _run(path, [bad])
    line = next(x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("bad.tif"))
    assert line.endswith("boom; w1; w2")


def test_an_interrupt_is_logged_as_interrupted_not_as_an_error(tmp_path):
    path = tmp_path / "halide-log.txt"
    log = RunLog(path, "develop", now=_clock(T0, T1))
    log.start("halide batch in out", ROWS, [], [], "0.1.0")
    log.finish(None, [], interrupted=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    i = lines.index("✗ Stopped — interrupted")
    assert lines[i + 1].startswith("━━ finished") and "error" not in "\n".join(lines[i:])


def test_each_frame_line_is_fsynced(tmp_path, monkeypatch):
    import os

    synced = []
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd))
    _run(tmp_path / "halide-log.txt", [_ok("a.tif"), _ok("b.tif")])
    assert len(synced) == 2


def test_a_failing_fsync_stays_inside_the_never_raise_contract(tmp_path, monkeypatch):
    import os

    def nope(fd):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", nope)
    log = _run(tmp_path / "halide-log.txt", [_ok("a.tif")])
    assert log.error is not None
