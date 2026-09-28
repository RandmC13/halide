"""F09, F15, F16, F25: one folder listing everywhere, plain-language input errors, a positive is
never re-inverted, scans without camera EXIF are "not verifiable"."""

import os
import struct

import re

import numpy as np
import pytest
import tifffile

import halide.device
from halide.calibration.anchors import NeutralPoint
from halide.calibration.scan_consistency import assess_roll
from halide.cli.main import main
from halide.core._xp import register_namespace
from halide.core.types import DensityProfile, ToneCurveParams
from halide.device import ComputeDevice
from halide.gui.roll import CalibrationSession
from halide.io.roll import list_scans
from halide.io.scan_metadata import ScanSettings
from halide.io.tiff import write_tiff
from halide.processing import ScanColorError, ScanInputError, Stage, process_scan
from tests.unit import _fake_device
from tests.unit._fake_device import FakeDeviceArray, fake_xp
from tests.unit.test_icc import LINEAR_TAGS, build_icc

PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))


def _negative(path, shape=(16, 16, 3), icc=True):
    rng = np.random.default_rng(3)
    img = rng.uniform(0.02, 0.15, size=shape).astype(np.float32)
    if len(shape) == 3 and shape[-1] == 3:
        write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS) if icc else None)
    else:  # write_tiff always writes RGB; these are the wrong-shaped files it's there to refuse
        tifffile.imwrite(path, img, photometric="minisblack", planarconfig="contig")
    return path


# --- discovery (F09) ------------------------------------------------------------------------------


def test_list_scans_takes_every_case_of_the_suffix_sorted_by_name(tmp_path):
    for name in ("b.TIF", "a.tif", "c.Tiff", "d.TIFF"):
        (tmp_path / name).write_bytes(b"x")
    scans, skipped = list_scans(tmp_path)
    assert [p.name for p in scans] == ["a.tif", "b.TIF", "c.Tiff", "d.TIFF"]
    assert not skipped


def test_list_scans_reports_each_reason_for_skipping(tmp_path):
    from halide.io.contact_sheet import Tile, render_sheet, write_sheet

    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "._a.tif").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".b.halide-partial-1.tif").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "shot.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "raw.NEF").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    write_sheet(tmp_path / "sheet.jpg", render_sheet([Tile(name="a", image=None)], title="t", subtitle="s"))
    scans, skipped = list_scans(tmp_path)
    assert [p.name for p in scans] == ["a.tif"]
    assert skipped.hidden == 3 and skipped.macos == 2  # `._`, `.DS_Store`; the partial file is not macOS's
    assert skipped.contact_sheets == 1 and skipped.folders == 1
    assert skipped.other == {"JPEG": 1, "raw file": 1, "non-TIFF file": 1}
    assert skipped.describe() == "3 hidden files, 1 contact sheet, 1 JPEG, 1 non-TIFF file, 1 raw file, 1 folder"


def test_skipped_description_reads_like_the_run_sheet_example(tmp_path):
    for name in ("._a.tif", ".DS_Store", "x.jpg"):
        (tmp_path / name).write_bytes(b"\xff\xd8")
    (tmp_path / "sub").mkdir()
    _, skipped = list_scans(tmp_path)
    assert skipped.describe() == "2 hidden macOS files, 1 JPEG, 1 folder"


def test_the_contact_command_admits_pictures_but_not_its_own_sheets(tmp_path):
    from halide.io.contact_sheet import Tile, render_sheet, write_sheet

    (tmp_path / "a.tif").write_bytes(b"x")
    from PIL import Image

    Image.new("RGB", (4, 4)).save(tmp_path / "b.png")
    write_sheet(tmp_path / "sheet.png", render_sheet([Tile(name="a", image=None)], title="t", subtitle="s"))
    scans, skipped = list_scans(tmp_path, extra_suffixes=(".png", ".jpg"))
    assert [p.name for p in scans] == ["a.tif", "b.png"] and skipped.contact_sheets == 1


def test_no_frame_fails_on_mac_junk(tmp_path):
    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    (roll / "._a.tif").write_bytes(b"\x00\x05\x16\x07")
    (roll / ".DS_Store").write_bytes(b"x")
    out = tmp_path / "out"
    assert main(["batch", str(roll), str(out), "--rm", "0.9", "--bm", "1.1", "--rs", "1", "--bs", "1",
                 "--quiet", "--workers", "1"]) == 0
    assert [p.name for p in out.iterdir()] == ["a.tif"]


def test_batch_run_sheet_row_for_skipped_files(tmp_path, capsys):
    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    (roll / "._a.tif").write_bytes(b"x")
    (roll / "sub").mkdir()
    assert main(["batch", str(roll), str(tmp_path / "out"), "--rm", "0.9", "--bm", "1.1", "--rs", "1",
                 "--bs", "1", "--workers", "1"]) == 0
    out = " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", capsys.readouterr().out).split())
    assert "Skipped 1 hidden macOS file, 1 folder" in out


# --- input errors (F25) ---------------------------------------------------------------------------


def _fail(path):
    with pytest.raises(ScanInputError) as excinfo:
        process_scan(path, path.parent / "out.tif", Stage.FULL, PROFILE, ToneCurveParams())
    assert not (path.parent / "out.tif").exists()
    return str(excinfo.value)


def test_a_jpeg_is_named_as_one(tmp_path):
    jpeg = tmp_path / "scan.tif"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    message = _fail(jpeg)
    assert message.startswith("scan.tif isn't a TIFF halide can read (it looks like a JPEG).")
    assert "README's 'Exporting your scans'" in message


def test_a_raw_file_is_named_by_its_extension(tmp_path):
    raw = tmp_path / "IMG_1.NEF"
    raw.write_bytes(b"II*\x00garbage")  # tifffile may or may not parse it; either way it isn't RGB float
    message = _fail(raw)
    assert "IMG_1.NEF" in message


def test_a_raw_extension_that_is_not_a_tiff_says_raw_file(tmp_path):
    raw = tmp_path / "IMG_1.dng"
    raw.write_bytes(b"not a tiff at all")
    assert "it looks like a raw file" in _fail(raw)


def test_an_unreadable_file_says_damaged(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff at all")
    assert "(it may be damaged or cut short)" in _fail(bad)


def test_rgba_is_refused_plainly(tmp_path):
    assert _fail(_negative(tmp_path / "rgba.tif", shape=(8, 8, 4))) == \
        "rgba.tif has an alpha (transparency) channel; export without it"


def test_greyscale_is_refused_plainly(tmp_path):
    assert _fail(_negative(tmp_path / "grey.tif", shape=(8, 8))) == \
        "grey.tif is greyscale; halide needs an RGB scan of a colour negative"


def test_other_channel_counts_are_refused_plainly(tmp_path):
    assert _fail(_negative(tmp_path / "five.tif", shape=(8, 8, 5))) == \
        "five.tif has 5 channels; halide needs RGB"


def test_input_errors_are_scan_colour_errors_so_the_device_paths_pass_them_through():
    assert issubclass(ScanInputError, ScanColorError)


def test_input_error_on_a_gpu_is_not_a_gpu_failure(tmp_path, monkeypatch):
    register_namespace(FakeDeviceArray, fake_xp)
    monkeypatch.setattr(halide.device, "to_device", _fake_device.to_device)
    monkeypatch.setattr(halide.device, "to_host", _fake_device.to_host)
    device = ComputeDevice(kind="gpu", name="Fake GPU")
    scan = _negative(tmp_path / "a.tif")

    def refuse(*args, **kwargs):
        raise ScanInputError("a.tif has 5 channels; halide needs RGB.")

    monkeypatch.setattr("halide.processing.develop_request", refuse)
    warnings = []
    with pytest.raises(ScanInputError):
        process_scan(scan, tmp_path / "out.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=device,
                     on_warning=warnings.append)
    assert warnings == []  # no "the GPU failed" fallback, no retry


def test_a_damaged_profile_message_reads_as_a_sentence(tmp_path):
    from halide.processing import _read_scan

    scan = tmp_path / "x.tif"
    icc = bytearray(build_icc(LINEAR_TAGS))
    write_tiff(scan, np.full((4, 4, 3), 0.1, dtype=np.float32), icc_profile=bytes(icc[:200]))
    with pytest.raises(ScanColorError) as excinfo:
        _read_scan(scan)
    message = str(excinfo.value)
    assert "unsupported color profile" not in message and "color" not in message.replace("colour", "")


# --- re-inverting a positive (F15) ----------------------------------------------------------------


def test_reinverting_a_halide_output_is_refused(tmp_path):
    scan = _negative(tmp_path / "neg.tif")
    positive = tmp_path / "pos.tif"
    process_scan(scan, positive, Stage.FULL, PROFILE, ToneCurveParams())
    with pytest.raises(ScanInputError) as excinfo:
        process_scan(positive, tmp_path / "again.tif", Stage.FULL, PROFILE, ToneCurveParams())
    message = str(excinfo.value)
    assert message.startswith("pos.tif is already a halide positive (made on ")
    assert "To re-print it use `halide print`; to develop again, point halide at the original scan" in message
    assert not (tmp_path / "again.tif").exists()


def test_a_batch_reports_a_positive_per_frame_and_develops_the_rest(tmp_path):
    from halide.batch.orchestrator import discover_jobs, run_batch

    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    process_scan(_negative(tmp_path / "n.tif"), roll / "b.tif", Stage.FULL, PROFILE, ToneCurveParams())
    out = tmp_path / "out"
    out.mkdir()
    results = run_batch(discover_jobs(roll, out), Stage.FULL, density_profile=PROFILE,
                        tone_params=ToneCurveParams(), max_workers=1)
    by_name = {r.job.input_path.name: r for r in results}
    assert by_name["a.tif"].error is None
    assert "already a halide positive" in by_name["b.tif"].error


# --- EXIF (F16) -----------------------------------------------------------------------------------


def test_check_reports_no_exif_as_not_verifiable(tmp_path, capsys):
    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    _negative(roll / "b.tif")
    (roll / "._a.tif").write_bytes(b"x")
    main(["check", str(roll)])
    out = capsys.readouterr().out
    assert "not verifiable (no camera EXIF in 2 frames)" in out
    assert "consistent" not in out
    assert "Skipped 1 hidden macOS file" in out


def test_summary_lines_carry_the_not_verifiable_warning():
    report = assess_roll({"a.tif": (None, None), "b.tif": (ScanSettings(exposure_time=1 / 60, f_number=8.0, iso=100), None)})
    assert "not verifiable (no camera EXIF in 1 frame)" in report.summary_lines()


def test_a_point_from_a_frame_without_scan_exposure_data_is_noted(tmp_path):
    with_exif = ScanSettings(exposure_time=1 / 30, f_number=8.0, iso=100)
    session = CalibrationSession()
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    session.set_roll(paths, [with_exif, None], tmp_path)
    for i, (frame, scan) in enumerate(zip(paths, [with_exif, None])):
        session.add_point(NeutralPoint(frame=frame, x=1, y=1, rgb=(0.1, 0.1 + 0.05 * i, 0.1), scan=scan))
    notes = [v.note for v in session.views()]
    assert notes == [None, "point 2: frame has no scan exposure data, used unnormalised"]


# --- pre-flight (F25) -----------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_output_folder_fails_before_developing(tmp_path):
    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o500)
    try:
        with pytest.raises(SystemExit, match="can't write to"):
            main(["batch", str(roll), str(out), "--rm", "0.9", "--bm", "1.1", "--rs", "1", "--bs", "1"])
    finally:
        out.chmod(0o700)
    assert list(out.iterdir()) == []


def test_a_folder_given_as_a_single_input_is_named(tmp_path):
    with pytest.raises(SystemExit, match="is a folder, not a scan file"):
        main(["invert", str(tmp_path), str(tmp_path / "x.tif"), "--invert-only"])


def test_missing_output_folder_for_a_single_file_is_named(tmp_path):
    scan = _negative(tmp_path / "a.tif")
    with pytest.raises(SystemExit, match="output folder .* doesn't exist"):
        main(["invert", str(scan), str(tmp_path / "nope" / "x.tif"), "--invert-only"])
