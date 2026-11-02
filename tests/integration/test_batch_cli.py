"""End-to-end tests for `halide batch` and the underlying orchestrator, including the specific
robustness property called out in the build plan: one frame's failure must not corrupt or block
the rest of the batch."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from halide.batch.orchestrator import discover_jobs, run_batch
from halide.cli.main import main
from halide.core.types import ToneCurveParams
import halide.processing as processing
from halide.io.tiff import read_tiff, read_tiff_description, write_tiff
from halide.processing import Stage, read_provenance
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


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
    for i in range(4):
        _write_negative(in_dir / f"frame_{i:02d}.tiff", seed=i)
    return in_dir


def test_discover_jobs_applies_suffix(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    jobs = discover_jobs(roll_dir, out_dir, suffix="_positive")
    assert len(jobs) == 4
    assert all(job.output_path.stem.endswith("_positive") for job in jobs)


def test_run_batch_processes_all_jobs(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams())
    assert len(results) == 4
    assert all(r.error is None for r in results)
    for job in jobs:
        assert job.output_path.exists()


def test_one_bad_frame_does_not_corrupt_the_rest(roll_dir, tmp_path):
    # Sabotage one input file: no embedded ICC profile at all -> guaranteed ScanColorError.
    bad_path = roll_dir / "frame_02.tiff"
    write_tiff(bad_path, np.zeros((16, 16, 3), dtype=np.float32))  # overwrite, no icc_profile

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams())

    assert len(results) == 4
    failed = [r for r in results if r.error is not None]
    succeeded = [r for r in results if r.error is None]
    assert len(failed) == 1
    assert failed[0].job.input_path.name == "frame_02.tiff"
    assert len(succeeded) == 3
    for r in succeeded:
        assert r.job.output_path.exists()
        result = read_tiff(r.job.output_path)
        assert np.all(np.isfinite(result.image))
    assert not failed[0].job.output_path.exists()


def test_run_batch_keyboard_interrupt_returns_partial_results_instead_of_raising(roll_dir, tmp_path, monkeypatch):
    # Regression test for the raw-traceback-on-Ctrl+C bug: _run_pool must catch KeyboardInterrupt
    # around its wait() loop and return whatever completed so far, not propagate it and discard
    # every already-finished result.
    import halide.batch.orchestrator as orchestrator

    def _raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "wait", _raise_interrupt)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = discover_jobs(roll_dir, out_dir)
    results = run_batch(jobs, Stage.FULL, density_profile=None, tone_params=ToneCurveParams(), max_workers=2)
    assert results == []


def test_batch_cli_end_to_end(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main(
        ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"]
    )
    assert exit_code == 0
    outputs = sorted(out_dir.glob("*.tiff"))
    assert len(outputs) == 4
    for path in outputs:
        result = read_tiff(path)
        assert np.all(np.isfinite(result.image))
        assert result.icc_profile is not None


def test_batch_cli_reports_failure_exit_code(roll_dir, tmp_path):
    write_tiff(roll_dir / "frame_02.tiff", np.zeros((16, 16, 3), dtype=np.float32))
    out_dir = tmp_path / "out"
    exit_code = main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"])
    assert exit_code == 1


def test_batch_cli_missing_input_directory_errors(tmp_path):
    # Regression test: a nonexistent input directory used to propagate a raw FileNotFoundError
    # traceback straight out of discover_jobs' iterdir() call.
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="input directory not found"):
        main(["batch", str(tmp_path / "does-not-exist"), str(out_dir)])


def test_batch_cli_empty_directory_errors(tmp_path):
    in_dir = tmp_path / "empty_in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match=r"No TIFF files found in .*right folder\? Subfolders are skipped \(use --recursive\)"):
        main(["batch", str(in_dir), str(out_dir)])


def test_auto_density_roll_conflicts_with_other_sources(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="can't be combined with --auto-density-roll"):
        main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--rm", "2.0", "--quiet"])


def test_batch_cli_prints_its_settings_as_one_run_sheet_before_developing(roll_dir, tmp_path, capsys):
    import re

    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--contrast", "0.8", "--workers", "1"]
    assert main(args) == 0
    out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out)
    sheet = out[: out.index("Developing")].splitlines()

    assert sheet[0] == sheet[-1] and set(sheet[0]) == {"▫", " "}  # framed by sprocket rules
    rows: dict[str, str] = {}
    for line in sheet[1:-1]:  # a continuation line (blank label) belongs to the row above it
        label = line[2:17].strip() or label
        rows[label] = f"{rows[label]} {line[17:].strip()}" if label in rows else line[17:]
    assert list(rows) == ["Roll", "Scans", "Calibration", "Output", "Compute", "Workers"]
    nbsp = "\u00a0"  # RunSheet.SEP's non-breaking space
    assert rows["Roll"].replace(" ", "") == f"in{nbsp}·4frames→{out_dir}"  # the long tmp path wraps
    # The automatic tiers point to the faithful one, on a dimmed line of their own
    assert rows["Calibration"] == (
        f"auto{nbsp}· one profile for the whole roll, from 4 frames automatic estimate from 4 frames — "
        "rough on a short roll; for faithful colour, pick neutral points with `halide open`"
    )
    assert rows["Output"] == f"print{nbsp}· grade 0.80{nbsp}· exposure fitted per frame"
    assert rows["Workers"] == "1 (--workers)"


def _plain(capsys):
    import re

    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out).split())


def _fake_roll_estimate(monkeypatch, warm_frame=None):
    """The roll's estimate with `warm_frame` (a file name) named as running warm."""
    from halide.core.types import DensityProfile

    def fake(paths, **kwargs):
        profile = DensityProfile(white_balance=(2.28, 1.0, 1.47), density_scale=(1.32, 1.0, 0.78))
        differing = [(Path(p), 19.0, "CC 19Y", "warm") for p in paths if Path(p).name == warm_frame]
        return processing.RollEstimate(profile, len(paths), differing)

    monkeypatch.setattr(processing, "estimate_roll_density_profile", fake)


def test_roll_estimate_names_a_frame_that_runs_warmer_than_the_rest(tmp_path, capsys, monkeypatch):
    in_dir = tmp_path / "roll"
    in_dir.mkdir()
    for i in range(7):
        _write_negative(in_dir / f"frame_{i + 1:02d}.tiff", seed=i)
    _fake_roll_estimate(monkeypatch, warm_frame="frame_07.tiff")
    assert main(["batch", str(in_dir), str(tmp_path / "out"), "--auto-density-roll", "--workers", "1"]) == 0
    out = _plain(capsys)
    assert "one profile for the whole roll, from 7 frames" in out
    assert "frame 7 runs warmer than the rest — they'll print that way, as the film recorded" in out
    assert "rough on a short roll" not in out
    assert "for the most faithful colour, pick neutral points" in out
    log = (tmp_path / "out" / "halide-log.txt").read_text()
    assert "frame 7 runs warmer than the rest" in " ".join(log.split())


def test_differing_note_wording():
    from halide.cli.commands.batch_cmd import _differing_note

    a, b, c, d, e, f = (Path(x) for x in "abcdef")
    numbers = {a: 4, b: 13, c: 9, d: 20, e: 2, f: 30}
    warm = [(b, 35.0, "CC 35Y", "warm"), (a, 11.0, "CC 11Y", "warm")]
    assert _differing_note(warm, numbers) == (
        "frames 4 and 13 run warmer than the rest — they'll print that way, as the film recorded")
    mixed = [(a, 20.0, "", "warm"), (c, 12.0, "", "cool")]
    assert _differing_note(mixed, numbers) == (
        "frames 4 (warmer) and 9 (cooler) differ in colour from the rest — they'll print that way, "
        "as the film recorded")
    many = [(b, 40.0, "", "magenta"), (a, 30.0, "", "magenta"), (c, 20.0, "", "magenta"),
            (d, 15.0, "", "magenta"), (e, 12.0, "", "magenta"), (f, 11.0, "", "magenta")]
    assert _differing_note(many, numbers).startswith("frames 4, 9, 13, 20 and 2 more run more magenta than")


def test_batch_auto_density_per_frame_points_to_the_roll_estimate(roll_dir, tmp_path, capsys):
    assert main(["batch", str(roll_dir), str(tmp_path / "out"), "--auto-density", "--workers", "1"]) == 0
    assert ("auto · each frame guessed on its own — --auto-density-roll makes one estimate for the whole roll"
            in _plain(capsys).replace("\u00a0", " "))


def test_batch_cli_run_sheet_names_the_roll_when_run_on_the_current_directory(roll_dir, tmp_path, monkeypatch, capsys):
    # Path(".").name is "", which printed a blank roll name for `halide batch . ...`.
    monkeypatch.chdir(roll_dir)
    assert main(["batch", ".", str(tmp_path / "out"), "--auto-density-roll", "--workers", "1"]) == 0
    import re

    out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out)
    roll_line = next(line for line in out.splitlines() if line.startswith("  Roll"))
    assert roll_line.split("Roll", 1)[1].split()[0].startswith("in")


def test_batch_into_its_own_folder_is_refused_before_any_work(roll_dir):
    import hashlib

    before = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in roll_dir.iterdir()}
    with pytest.raises(SystemExit) as exc:
        main(["batch", str(roll_dir), str(roll_dir), "--quiet"])
    message = str(exc.value.code)
    assert "is the scan itself" in message
    assert "--suffix" in message  # batch has --suffix, unlike invert
    after = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in roll_dir.iterdir()}
    assert after == before


def test_batch_rerun_non_interactive_refuses_then_skip_existing_develops_only_new(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    assert main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"]) == 0
    outputs = sorted(out_dir.glob("*.tiff"))
    assert len(outputs) == 4
    before = {p: p.read_bytes() for p in outputs}

    # A plain re-run into the same, now-populated folder must refuse, not silently overwrite.
    with pytest.raises(SystemExit) as exc:
        main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"])
    message = str(exc.value.code)
    assert "--overwrite" in message and "--skip-existing" in message
    assert {p: p.read_bytes() for p in outputs} == before

    # Add a new, not-yet-developed frame to the roll.
    _write_negative(roll_dir / "frame_04.tiff", seed=99)
    assert main(
        ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet", "--skip-existing"]
    ) == 0
    # The 4 original outputs are untouched; only the new frame was developed.
    assert {p: p.read_bytes() for p in outputs} == before
    assert (out_dir / "frame_04.tiff").exists()
    assert len(list(out_dir.glob("*.tiff"))) == 5


def test_batch_skip_existing_contact_sheet_covers_whole_roll_when_nothing_to_develop(roll_dir, tmp_path):
    # --skip-existing must never silently drop the requested contact sheet, and
    # must build it even when no frame needed developing.
    out_dir = tmp_path / "out"
    assert main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"]) == 0
    assert len(list(out_dir.glob("*.tiff"))) == 4

    reference = tmp_path / "reference.jpg"
    assert main(["contact", str(out_dir), str(reference), "--frame-width", "60", "--quiet", "--workers", "1"]) == 0

    sheet = tmp_path / "resumed.jpg"
    assert main(
        ["batch", str(roll_dir), str(out_dir), "--contact-sheet", str(sheet), "--auto-density-roll",
         "--quiet", "--workers", "1", "--frame-width", "60", "--skip-existing"]
    ) == 0
    assert sheet.exists()
    with Image.open(reference) as a, Image.open(sheet) as b:
        assert a.size == b.size  # same 4-frame layout as a direct `halide contact` over the folder


def test_batch_skip_existing_contact_sheet_covers_new_and_old_frames(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    assert main(["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--quiet"]) == 0
    assert len(list(out_dir.glob("*.tiff"))) == 4

    # Add a new, not-yet-developed frame.
    _write_negative(roll_dir / "frame_04.tiff", seed=99)

    sheet = tmp_path / "resumed.jpg"
    assert main(
        ["batch", str(roll_dir), str(out_dir), "--contact-sheet", str(sheet), "--auto-density-roll",
         "--quiet", "--workers", "1", "--frame-width", "60", "--skip-existing"]
    ) == 0
    assert len(list(out_dir.glob("*.tiff"))) == 5

    reference = tmp_path / "reference.jpg"
    assert main(["contact", str(out_dir), str(reference), "--frame-width", "60", "--quiet", "--workers", "1"]) == 0
    with Image.open(reference) as a, Image.open(sheet) as b:
        assert a.size == b.size  # covers all 5 frames, not just the newly-developed one


@pytest.mark.parametrize(
    "source, advised",
    [(["--auto-density-roll"], True), (["--auto-density"], True), (["--rm", "2.0", "--bm", "1.4"], False)],
)
def test_run_sheet_advises_picking_points_after_an_automatic_estimate(roll_dir, tmp_path, capsys, source, advised):
    assert main(["batch", str(roll_dir), str(tmp_path / "out"), *source, "--workers", "1"]) == 0
    out = " ".join(capsys.readouterr().out.split())  # a long row wraps with a hanging indent
    assert ("pick neutral points with `halide open`" in out) is advised  # a 4-frame roll: the short-roll wording


def test_auto_density_roll_with_another_source_is_argparses_refusal(roll_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["batch", str(roll_dir), str(tmp_path / "out"), "--auto-density-roll", "--auto-density", "--quiet"])
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_batch_has_no_pick_option(roll_dir, tmp_path, capsys):
    """Picking is `invert --pick` only; a roll is calibrated in `halide open`."""
    with pytest.raises(SystemExit) as refused:
        main(["batch", str(roll_dir), str(tmp_path / "out"), "--pick"])
    assert refused.value.code == 2
    assert "--pick" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["batch", "--help"])
    assert "--pick" not in capsys.readouterr().out


def test_results_carry_the_print_decision(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = run_batch(discover_jobs(roll_dir, out_dir), Stage.FULL, density_profile=None,
                        tone_params=ToneCurveParams())
    for r in results:
        assert r.error is None
        assert 0.0 < r.contrast <= 1.0
        assert r.exposure is not None
        assert r.grade_fitted is True
        assert r.linear_scale is None


def test_a_pinned_grade_is_marked_not_fitted(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = run_batch(discover_jobs(roll_dir, out_dir), Stage.FULL, density_profile=None,
                        tone_params=ToneCurveParams(contrast=0.7))
    assert all(r.contrast == pytest.approx(0.7) and r.grade_fitted is False for r in results)


def test_flat_results_carry_their_scale(roll_dir, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = run_batch(discover_jobs(roll_dir, out_dir), Stage.FULL, density_profile=None,
                        tone_params=ToneCurveParams(mode="linear"))
    assert all(r.linear_scale > 0 and r.contrast is None and r.grade_fitted is None for r in results)


def _resumed_roll(tmp_path, monkeypatch):
    """A 6-frame roll developed with --auto-density-roll, two outputs deleted, then resumed with
    --skip-existing. Returns (out_dir, paths the estimate saw on the resume, resumed names, kept mtimes)."""
    in_dir = tmp_path / "roll"
    in_dir.mkdir()
    for i in range(6):
        _write_negative(in_dir / f"f{i}.tiff", seed=i)
    out_dir = tmp_path / "out"
    args = ["batch", str(in_dir), str(out_dir), "--auto-density-roll", "--quiet"]
    assert main(args) == 0
    gone = ["f1.tiff", "f4.tiff"]
    for name in gone:
        (out_dir / name).unlink()
    kept = {p.name: os.stat(p).st_mtime_ns for p in out_dir.glob("*.tiff")}

    seen: list[list] = []
    real = processing.estimate_roll_density_profile

    def spy(paths, *a, **kw):
        seen.append([str(p) for p in paths])
        return real(paths, *a, **kw)

    monkeypatch.setattr(processing, "estimate_roll_density_profile", spy)
    assert main(args + ["--skip-existing"]) == 0
    return out_dir, in_dir, seen, gone, kept


def test_resuming_estimates_the_roll_profile_from_every_frame(tmp_path, monkeypatch):
    out_dir, in_dir, seen, gone, kept = _resumed_roll(tmp_path, monkeypatch)
    assert len(seen) == 1
    assert sorted(Path(p).name for p in seen[0]) == [f"f{i}.tiff" for i in range(6)]
    assert sorted(p.name for p in out_dir.glob("*.tiff")) == [f"f{i}.tiff" for i in range(6)]
    # Only the two missing outputs were developed again.
    assert {n: os.stat(out_dir / n).st_mtime_ns for n in kept} == kept


def test_resumed_frames_use_the_same_profile_as_the_rest_of_the_roll(tmp_path, monkeypatch):
    out_dir, _, _, gone, kept = _resumed_roll(tmp_path, monkeypatch)

    def profile(name):
        record = read_provenance(read_tiff_description(out_dir / name))
        return record["white_balance"], record["density_scale"]

    reference = profile(next(iter(kept)))
    for name in gone:
        assert profile(name) == reference


def test_resumed_minority_exposure_frame_is_matched_to_the_rolls_most_common_setting(tmp_path):
    from halide.calibration.scan_consistency import scan_gain
    from halide.io.scan_metadata import read_scan_metadata
    from tests.integration.test_check_cli import _frame

    in_dir = tmp_path / "roll"
    in_dir.mkdir()
    for i in range(5):
        _frame(in_dir / f"f{i}.tiff", exposure=(1, 50), seed=i)
    _frame(in_dir / "f5.tiff", exposure=(1, 25), seed=5)  # the minority exposure
    out_dir = tmp_path / "out"
    args = ["batch", str(in_dir), str(out_dir), "--auto-density-roll", "--match-scan-exposure", "--quiet"]
    assert main(args) == 0
    (out_dir / "f5.tiff").unlink()
    assert main(args + ["--skip-existing"]) == 0

    common, _ = read_scan_metadata(in_dir / "f0.tiff")
    own, _ = read_scan_metadata(in_dir / "f5.tiff")
    expected = scan_gain(own, common)
    assert expected != 1.0
    record = read_provenance(read_tiff_description(out_dir / "f5.tiff"))
    assert record["scan_gain"] == pytest.approx(expected)


def test_one_exposure_on_unevenly_digitized_scans_says_to_even_them_out(tmp_path, capsys):
    from tests.integration.test_check_cli import _frame

    in_dir = tmp_path / "roll"
    in_dir.mkdir()
    _frame(in_dir / "a.tiff", exposure=(1, 50), seed=1)
    _frame(in_dir / "b.tiff", exposure=(1, 25), seed=2)
    base = ["batch", str(in_dir), str(tmp_path / "out"), "--dry-run", "--rm", "0.2", "--bm", "0.3", "--rs", "1.3", "--bs", "0.8"]
    assert main(base + ["--exposure", "0.4"]) == 0
    assert "one --exposure across scans digitized at different exposures" in capsys.readouterr().out
    assert main(base + ["--exposure", "0.4", "--match-scan-exposure"]) == 0
    assert "one --exposure across" not in capsys.readouterr().out
    assert main(base) == 0
    assert "one --exposure across" not in capsys.readouterr().out
