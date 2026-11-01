"""One folder listing everywhere, plain-language input errors, a positive is
never re-inverted, scans without camera EXIF are "not verifiable"."""

import os
import struct

import re

import numpy as np
import pytest
import tifffile

import halide.device
from halide.io.roll import RUN_LOG_NAME
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


# --- discovery ------------------------------------------------------------------------------------


def test_list_scans_takes_every_case_of_the_suffix_sorted_by_name(tmp_path):
    for name in ("b.TIF", "a.tif", "c.Tiff", "d.TIFF"):
        (tmp_path / name).write_bytes(b"x")
    scans, skipped = list_scans(tmp_path)
    assert [p.name for p in scans] == ["a.tif", "b.TIF", "c.Tiff", "d.TIFF"]
    assert not skipped


def test_halide_run_log_is_skipped_silently(tmp_path):
    from halide.io.roll import RUN_LOG_NAME, list_scans

    (tmp_path / RUN_LOG_NAME).write_text("log\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")
    files, skipped = list_scans(tmp_path)
    assert files == []
    assert skipped.other == {"non-TIFF file": 1}  # notes.txt only


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
    assert [p.name for p in out.iterdir() if p.name != RUN_LOG_NAME] == ["a.tif"]


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


# --- input errors ---------------------------------------------------------------------------------


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


# --- re-inverting a positive ----------------------------------------------------------------------


def test_reinverting_a_halide_output_is_refused(tmp_path):
    scan = _negative(tmp_path / "neg.tif")
    positive = tmp_path / "pos.tif"
    process_scan(scan, positive, Stage.FULL, PROFILE, ToneCurveParams())
    with pytest.raises(ScanInputError) as excinfo:
        process_scan(positive, tmp_path / "again.tif", Stage.FULL, PROFILE, ToneCurveParams())
    message = str(excinfo.value)
    assert message.startswith("pos.tif is already a halide positive. To re-print")
    assert "made on" not in message
    assert "To re-print it use `halide print`; to develop again, point halide at the original scan" in message
    assert not (tmp_path / "again.tif").exists()


def test_a_batch_reports_a_positive_per_frame_and_develops_the_rest(tmp_path):
    from halide.batch.orchestrator import jobs_for_files, run_batch

    roll = tmp_path / "roll"
    roll.mkdir()
    _negative(roll / "a.tif")
    process_scan(_negative(tmp_path / "n.tif"), roll / "b.tif", Stage.FULL, PROFILE, ToneCurveParams())
    out = tmp_path / "out"
    out.mkdir()
    results = run_batch(jobs_for_files(list_scans(roll)[0], out), Stage.FULL, density_profile=PROFILE,
                        tone_params=ToneCurveParams(), max_workers=1)
    by_name = {r.job.input_path.name: r for r in results}
    assert by_name["a.tif"].error is None
    assert "already a halide positive" in by_name["b.tif"].error


# --- EXIF -----------------------------------------------------------------------------------------


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


# --- pre-flight -----------------------------------------------------------------------------------


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


def _positive(path):

    write_tiff(path, np.full((8, 8, 3), 0.2, np.float32), icc_profile=build_icc(LINEAR_TAGS))
    tifffile.tiffcomment(path, '{"halide": {"output": "print"}}')


def test_skip_positives_leaves_out_halide_output_and_counts_it(tmp_path):
    _negative(tmp_path / "a.tif")
    _positive(tmp_path / "a_d.tif")
    files, skipped = list_scans(tmp_path, skip_positives=True)
    assert [f.name for f in files] == ["a.tif"] and skipped.positives == 1 and skipped
    assert skipped.describe() == "1 halide positive"
    files, skipped = list_scans(tmp_path)
    assert len(files) == 2 and not skipped


def test_positives_wording_is_plural_and_merges():
    from halide.io.roll import Skipped

    total = Skipped(positives=2)
    total += Skipped(folders=1, positives=1)
    assert total.describe() == "1 folder, 3 halide positives"


def test_an_unreadable_header_is_still_listed_as_a_scan(tmp_path):
    (tmp_path / "broken.tif").write_bytes(b"not a tiff")
    files, skipped = list_scans(tmp_path, skip_positives=True)
    assert [f.name for f in files] == ["broken.tif"] and not skipped


# --- natural order --------------------------------------------------------------------------------


def test_natural_key_orders_numbers_as_numbers():
    from halide.io.roll import natural_key

    names = ["frame_10.tif", "frame_2.tif", "frame_1.tif", "IMG_0138.tif", "IMG_0139.tif"]
    assert sorted(names, key=natural_key) == [
        "IMG_0138.tif", "IMG_0139.tif", "frame_1.tif", "frame_2.tif", "frame_10.tif"]


def test_zero_padded_names_keep_their_old_order():
    from halide.io.roll import natural_key

    names = [f"IMG_{i:04d}.tif" for i in (175, 9, 138, 1000)]
    assert sorted(names, key=natural_key) == sorted(names)


def test_list_scans_uses_natural_order(tmp_path):
    from halide.io.roll import list_scans

    for n in (10, 2, 1):
        (tmp_path / f"f{n}.tif").write_bytes(b"")
    files, _ = list_scans(tmp_path)
    assert [f.name for f in files] == ["f1.tif", "f2.tif", "f10.tif"]


# --- finding rolls under a folder -----------------------------------------------------------------


def _scan(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return _negative(path)


def _output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return _positive(path)


def test_find_rolls_in_the_users_layout(tmp_path):
    from halide.io.roll import find_rolls

    top = tmp_path / "Holiday"
    for roll in ("Roll1", "Roll2"):
        _scan(top / roll / "pre-processed" / "IMG_0001.tif")
        (top / roll / "IMG_0001.CR2").write_bytes(b"raw")
    _output(top / "Roll1" / "inversion" / "IMG_0001.tif")
    rolls, skipped = find_rolls(top)
    assert [(r.name, r.folder.relative_to(top).as_posix()) for r in rolls] == [
        ("Roll1", "Roll1/pre-processed"), ("Roll2", "Roll2/pre-processed")]
    assert [f.name for f in rolls[0].files] == ["IMG_0001.tif"]
    assert skipped.halide_outputs == 1
    assert skipped.describe() == "1 folder of halide output"


def test_a_mixed_folder_is_a_roll_and_reports_its_positives(tmp_path):
    from halide.io.roll import find_rolls

    _scan(tmp_path / "T" / "a.tif")
    _output(tmp_path / "T" / "a_d.tif")
    rolls, skipped = find_rolls(tmp_path / "T")
    assert [f.name for f in rolls[0].files] == ["a.tif"] and rolls[0].left_out.positives == 1
    assert skipped.halide_outputs == 0


def test_destinations_are_excluded_even_when_new(tmp_path):
    from halide.io.roll import find_rolls

    top = tmp_path / "T"
    _scan(top / "A" / "x.tif")
    _scan(top / "A" / "out" / "x.tif")  # a copy, not a halide output
    rolls, skipped = find_rolls(top, exclude=lambda p: p == top / "A" / "out")
    assert [r.name for r in rolls] == ["A"] and skipped.destinations == 1
    assert skipped.describe() == "1 output folder of this run"


def test_flat_layout_and_the_input_itself(tmp_path):
    from halide.io.roll import find_rolls

    top = tmp_path / "Scans"
    _scan(top / "a.tif")
    _scan(top / "RollA" / "a.tif")
    rolls, _ = find_rolls(top)
    assert [r.name for r in rolls] == ["Scans", "RollA"]


def test_hidden_and_symlinked_folders_are_not_entered(tmp_path):
    from halide.io.roll import find_rolls

    top = tmp_path / "T"
    _scan(top / ".cache" / "a.tif")
    _scan(tmp_path / "elsewhere" / "a.tif")
    (top / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    rolls, _ = find_rolls(top)
    assert rolls == []


def test_two_scan_folders_in_one_roll_get_unique_names(tmp_path):
    from halide.io.roll import find_rolls

    top = tmp_path / "H"
    _scan(top / "Roll1" / "pre-processed" / "a.tif")
    _scan(top / "Roll1" / "rescans" / "a.tif")
    rolls, _ = find_rolls(top)
    assert [r.name for r in rolls] == ["Roll1 - pre-processed", "Roll1 - rescans"]


def test_rolls_come_in_natural_order(tmp_path):
    from halide.io.roll import find_rolls

    for name in ("Roll10", "Roll2"):
        _scan(tmp_path / "T" / name / "a.tif")
    rolls, _ = find_rolls(tmp_path / "T")
    assert [r.name for r in rolls] == ["Roll2", "Roll10"]
