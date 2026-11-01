"""`--dry-run` through the real commands: the run sheet a real run would show, and nothing else."""

from __future__ import annotations

import re

import numpy as np
import pytest

from halide.cli.main import main
from halide.core.types import DensityProfile
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\033\[[0-9;]*m")
MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


def _negative(path, seed):
    rng = np.random.default_rng(seed)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    density = rng.uniform(0.0, 1.0, size=(24, 24, 1)).astype(np.float32)
    write_tiff(path, base * 10.0 ** (-density), icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    for i in range(3):
        _negative(d / f"frame_{i:02d}.tif", i)
    return d


@pytest.fixture(autouse=True)
def profiles_home(tmp_path, monkeypatch):
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home / "halide" / "profiles"


def _snapshot(root):
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*"))


def _run(capsys, *argv):
    code = main(list(argv))
    return code, " ".join(_ANSI.sub("", capsys.readouterr().out).split())


def _develop(roll, out):
    assert main(["batch", str(roll), str(out), *MANUAL, "--workers", "1"]) == 0


def _save(name, created):
    from halide.calibration.profile_store import save_named_profile

    save_named_profile(DensityProfile(white_balance=(2.0, 1.0, 1.5), density_scale=(1.3, 1.0, 0.8),
                                      film_stock=None, created_at=None), name)
    import json
    from halide.calibration.profile_store import find_profile

    path = find_profile(name)
    data = json.loads(path.read_text())
    data["created_at"] = created
    path.write_text(json.dumps(data))


def test_batch_dry_run_writes_nothing_and_says_so(roll, tmp_path, capsys):
    before = _snapshot(tmp_path)
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run")
    assert code == 0
    assert not (tmp_path / "out").exists()
    assert _snapshot(tmp_path) == before
    assert "(would be created)" in out
    assert "Dry run — nothing developed. 3 frames would be developed →" in out
    assert "Workers" in out and "Output" in out


def test_existing_outputs_never_prompt_and_say_would_ask(roll, tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / "out"
    _develop(roll, out_dir)
    capsys.readouterr()
    monkeypatch.setattr("halide.cli.console.prompts.is_interactive", lambda: True)
    monkeypatch.setattr("halide.cli._output_policy.is_interactive", lambda: True)
    monkeypatch.setattr("halide.cli.console.confirm", lambda *a, **k: pytest.fail("a dry run must not ask"))
    before = _snapshot(tmp_path)
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run")
    assert code == 0 and _snapshot(tmp_path) == before
    assert "3 already exist — a real run would ask" in out
    assert "3 already exist (a real run would ask whether to replace them)" in out
    assert "no new frames would be developed" in out


def test_existing_outputs_non_interactive_does_not_refuse(roll, tmp_path, capsys, monkeypatch):
    _develop(roll, tmp_path / "out")
    capsys.readouterr()
    monkeypatch.setattr("halide.cli._output_policy.is_interactive", lambda: False)
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run")
    assert code == 0 and "would ask" in out


def test_overwrite_says_would_be_replaced(roll, tmp_path, capsys):
    _develop(roll, tmp_path / "out")
    capsys.readouterr()
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run", "--overwrite")
    assert code == 0 and "3 would be replaced" in out


def test_skip_existing_reports_kept_frames(roll, tmp_path, capsys):
    out_dir = tmp_path / "out"
    _develop(roll, out_dir)
    (out_dir / "frame_01.tif").unlink()
    capsys.readouterr()
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run", "--skip-existing")
    assert code == 0
    assert "Skipping 2 frames already developed" in out
    assert "1 frame would be developed" in out and "2 kept" in out
    assert not (out_dir / "frame_01.tif").exists()


@pytest.mark.parametrize("interactive", [True, False])
def test_no_calibration_lists_saved_profiles_without_asking(roll, tmp_path, capsys, monkeypatch, interactive):
    _save("older", "2026-01-01T00:00:00+00:00")
    _save("newer", "2026-06-01T00:00:00+00:00")
    monkeypatch.setattr("halide.cli.console.prompts.is_interactive", lambda: interactive)
    monkeypatch.setattr("halide.cli._calibration_args.is_interactive", lambda: interactive)
    monkeypatch.setattr("halide.cli.console.menu", lambda *a, **k: pytest.fail("a dry run must not ask"))
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), "--dry-run")
    assert code == 0
    assert "none given — a real run would ask you to choose (saved: newer, older)" in out


def test_no_calibration_and_no_saved_profiles(roll, tmp_path, capsys):
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), "--dry-run")
    assert code == 0
    assert "no saved profiles yet (make one with: halide calibrate" in out and "choose; no saved" in out


def test_auto_density_roll_reads_no_pixels(roll, tmp_path, capsys, monkeypatch):
    import halide.processing as processing

    monkeypatch.setattr(processing, "estimate_roll_density_profile",
                        lambda *a, **k: pytest.fail("a dry run must not estimate"))
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), "--auto-density-roll", "--dry-run")
    assert code == 0
    assert "would estimate one profile for the whole roll from all 3 frames" in out
    assert "automatic estimate" in out


def test_save_profile_as_saves_nothing(roll, tmp_path, capsys, profiles_home):
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run",
                     "--save-profile-as", "mine")
    assert code == 0 and "would save as 'mine'" in out
    assert not profiles_home.exists() or not list(profiles_home.glob("*.json"))


def test_the_gpu_service_is_never_started(roll, tmp_path, capsys, monkeypatch):
    import halide.batch.orchestrator as orchestrator
    import halide.cli._device_args as device_args
    from halide.device import ComputeDevice

    gpu = ComputeDevice(kind="gpu", name="Fake GPU", memory_free=6 * 2**30, memory_total=8 * 2**30)
    monkeypatch.setattr(device_args, "resolve_device", lambda requested, isolated=False: gpu)
    monkeypatch.setattr(orchestrator, "_start_service", lambda *a, **k: pytest.fail("service started"))
    monkeypatch.setattr(orchestrator, "_service_python_supported", lambda: True)
    import halide.shared_frames as shared_frames

    monkeypatch.setattr(shared_frames, "sweep_stale", lambda: pytest.fail("swept"))
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run")
    assert code == 0
    assert "GPU — Fake GPU" in out and "would share one GPU process between the workers" in out

    monkeypatch.setenv("HALIDE_GPU_SERVICE", "0")
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run")
    assert code == 0 and "one GPU context per worker" in out


def test_cpu_dry_run_never_calls_start_compute(roll, tmp_path, capsys, monkeypatch):
    import halide.cli.commands.batch_cmd as batch_cmd

    monkeypatch.setattr(batch_cmd, "start_compute", lambda *a, **k: pytest.fail("start_compute called"))
    monkeypatch.setattr(batch_cmd, "run_batch", lambda *a, **k: pytest.fail("developed"))
    assert _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run")[0] == 0


def test_preview_batch_names_the_sheet(roll, tmp_path, capsys):
    sheet = tmp_path / "s.jpg"
    code, out = _run(capsys, "batch", str(roll), "--contact-sheet", str(sheet), *MANUAL, "--dry-run")
    assert code == 0 and not sheet.exists()
    assert "would be developed for the contact sheet" in out and "s.jpg" in out


def test_quiet_still_prints_the_final_line(roll, tmp_path, capsys):
    code, out = _run(capsys, "batch", str(roll), str(tmp_path / "out"), *MANUAL, "--dry-run", "--quiet")
    assert code == 0 and "Dry run — nothing developed." in out and "Workers" not in out


@pytest.mark.parametrize("command, past", [("print", "printed"), ("export", "exported")])
def test_bulk_print_and_export_dry_run(tmp_path, capsys, command, past):
    src = tmp_path / "src"
    src.mkdir()
    _develop_flat(src)
    out_dir = tmp_path / "out"
    before = _snapshot(tmp_path)
    code, out = _run(capsys, command, str(src), str(out_dir), "--dry-run")
    assert code == 0 and not out_dir.exists() and _snapshot(tmp_path) == before
    assert f"Dry run — nothing {past}. 3 " in out and f"would be {past}" in out


def _develop_flat(src):
    raw = src.parent / "raw"
    raw.mkdir()
    for i in range(3):
        _negative(raw / f"f{i}.tif", i)
    assert main(["batch", str(raw), str(src), *MANUAL, "--workers", "1", "--no-log"]) == 0


@pytest.mark.parametrize("command", ["print", "export"])
def test_single_file_dry_run_is_an_error(tmp_path, command):
    f = tmp_path / "a.tif"
    _negative(f, 0)
    with pytest.raises(SystemExit) as exc:
        main([command, str(f), str(tmp_path / "o.tif"), "--dry-run"])
    assert "--dry-run plans a folder run; a single file has nothing to plan" in str(exc.value)
    assert not (tmp_path / "o.tif").exists()


def test_same_errors_as_a_real_run(roll, tmp_path):
    with pytest.raises(SystemExit, match="input directory not found"):
        main(["batch", str(tmp_path / "nope"), str(tmp_path / "out"), *MANUAL, "--dry-run"])
    with pytest.raises(SystemExit, match="is the scan itself"):
        main(["batch", str(roll), str(roll), *MANUAL, "--dry-run"])
    with pytest.raises(SystemExit, match="no such profile"):
        main(["batch", str(roll), str(tmp_path / "out"), "--profile", "nope", "--dry-run"])
    assert not (tmp_path / "out").exists()


def test_help_lists_dry_run(capsys):
    with pytest.raises(SystemExit):
        main(["batch", "--help"])
    assert "--dry-run" in capsys.readouterr().out


def test_existing_contact_sheet_is_reported_in_each_policy(roll, tmp_path, capsys):
    sheet = tmp_path / "s.jpg"
    sheet.write_bytes(b"x")
    out_dir = tmp_path / "out"
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run", "--contact-sheet", str(sheet))
    assert code == 0 and "the contact sheet" in out and "already exists — a real run would ask" in out
    assert "3 frames would be developed" in out and "whether to replace it" in out
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run", "--contact-sheet", str(sheet),
                     "--overwrite")
    assert "the contact sheet" in out and "would be replaced" in out
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run", "--contact-sheet", str(sheet),
                     "--skip-existing")
    assert "is kept (--skip-existing)" in out
    assert sheet.read_bytes() == b"x"


def test_existing_contact_sheet_in_a_preview(roll, tmp_path, capsys):
    sheet = tmp_path / "s.jpg"
    sheet.write_bytes(b"x")
    code, out = _run(capsys, "batch", str(roll), "--contact-sheet", str(sheet), *MANUAL, "--dry-run")
    assert code == 0 and "already exists — a real run would ask" in out
    code, out = _run(capsys, "batch", str(roll), "--contact-sheet", str(sheet), *MANUAL, "--dry-run", "--skip-existing")
    assert code == 0 and out.rstrip().endswith("already exists (--skip-existing).")
    assert "Dry run — nothing developed. Nothing to do" in out


def test_invert_only_with_save_profile_as_errors_like_a_real_run(roll, tmp_path):
    with pytest.raises(SystemExit, match="needs a single resolved profile"):
        main(["batch", str(roll), str(tmp_path / "out"), "--invert-only", "--save-profile-as", "x", "--dry-run"])


def test_everything_present_but_the_sheet_would_build_it_from_the_folder(roll, tmp_path, capsys):
    out_dir = tmp_path / "out"
    _develop(roll, out_dir)
    capsys.readouterr()
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run", "--skip-existing",
                     "--contact-sheet", str(tmp_path / "s.jpg"))
    assert code == 0 and "Dry run — nothing developed. Nothing to develop" in out
    assert "would be built from the folder" in out


def test_ask_mode_counts_new_frames_once(roll, tmp_path, capsys):
    out_dir = tmp_path / "out"
    _develop(roll, out_dir)
    (out_dir / "frame_00.tif").unlink()
    capsys.readouterr()
    code, out = _run(capsys, "batch", str(roll), str(out_dir), *MANUAL, "--dry-run")
    assert "1 new frame would be developed" in out and "2 already exist (a real run would ask" in out


def test_single_file_dry_run_refuses_before_resolving_the_device(tmp_path, monkeypatch):
    import halide.cli._device_args as device_args

    monkeypatch.setattr(device_args, "resolve_device", lambda *a, **k: pytest.fail("device resolved"))
    f = tmp_path / "a.tif"
    _negative(f, 0)
    for command in ("print", "export"):
        with pytest.raises(SystemExit, match="plans a folder run"):
            main([command, str(f), str(tmp_path / "o.tif"), "--dry-run"])


@pytest.mark.parametrize("extra, message", [
    (["--output", "flat", "--contrast", "0.8"], "--contrast does nothing with --output flat"),
    (["--output", "flat", "--exposure", "0.1"], "--exposure does nothing with --output flat"),
    (["--notes", "x"], "--notes is saved with a profile"),
])
@pytest.mark.parametrize("command, dry", [("invert", False), ("batch", False), ("batch", True)])
def test_flags_that_would_do_nothing_are_refused(tmp_path, roll, command, extra, message, dry):
    frame = roll / "frame_00.tif"
    if command == "invert":
        args = ["invert", str(frame), str(tmp_path / "out.tif")]
        outputs = [tmp_path / "out.tif"]
    else:
        args = ["batch", str(roll), str(tmp_path / "out")]
        outputs = [tmp_path / "out"]
    argv = args + MANUAL + extra + (["--dry-run"] if dry else [])
    with pytest.raises(SystemExit, match=message):
        main(argv)
    assert not any(p.exists() for p in outputs)


def test_flat_with_a_saved_profile_or_notes_with_save_as_is_fine(tmp_path, roll):
    out = tmp_path / "out.tif"
    assert main(["invert", str(roll / "frame_00.tif"), str(out), *MANUAL, "--output", "flat",
                 "--save-profile-as", "p", "--notes", "x"]) == 0
