"""`halide check` end to end on synthetic rolls whose headers carry real EXIF and darktable XMP
(written into the TIFF bytes, so `read_scan_metadata` parses them exactly as it does a real scan)."""

import re
import struct

import numpy as np
import tifffile

from halide.cli.main import main
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ICC_TAG, _XMP_TAG, _EXIF_TAG = 34675, 700, 34665
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _wb_params(r, g, b):
    return struct.pack("<4f", r, g, b, 0.0).hex()


def _xmp(white_balance):
    item = (
        f'<rdf:li darktable:operation="temperature" darktable:enabled="1" darktable:multi_priority="0" '
        f'darktable:params="{_wb_params(*white_balance)}"/>'
    )
    return f'<x darktable:history_end="1"><darktable:history><rdf:Seq>{item}</rdf:Seq></darktable:history></x>'


def _add_exif(path, exposure_time, f_number, iso):
    """Appends an Exif IFD to a little-endian TIFF and points a copy of IFD0 at it (tifffile can't
    write that tag itself). Every other offset in the file stays valid."""
    data = bytearray(path.read_bytes())
    assert data[:2] == b"II"
    ifd0 = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd0)[0]
    entries = [bytes(data[ifd0 + 2 + 12 * i: ifd0 + 14 + 12 * i]) for i in range(count)]
    next_ifd = data[ifd0 + 2 + 12 * count: ifd0 + 6 + 12 * count]

    def rational(value):
        offset = len(data)
        data.extend(struct.pack("<II", *value))
        return offset

    time_at, f_at = rational(exposure_time), rational(f_number)
    exif = [
        struct.pack("<HHII", 33434, 5, 1, time_at),
        struct.pack("<HHII", 33437, 5, 1, f_at),
        struct.pack("<HHIHH", 34855, 3, 1, iso, 0),
    ]
    exif_at = len(data)
    data.extend(struct.pack("<H", len(exif)) + b"".join(exif) + struct.pack("<I", 0))
    entries.append(struct.pack("<HHII", _EXIF_TAG, 4, 1, exif_at))
    entries.sort(key=lambda e: struct.unpack_from("<H", e)[0])
    new_ifd0 = len(data)
    data.extend(struct.pack("<H", len(entries)) + b"".join(entries) + bytes(next_ifd))
    struct.pack_into("<I", data, 4, new_ifd0)
    path.write_bytes(bytes(data))


def _frame(path, exposure=None, white_balance=None, seed=0):
    icc = build_icc(LINEAR_TAGS)
    extratags = [(_ICC_TAG, "B", len(icc), icc)]
    if white_balance is not None:
        xmp = _xmp(white_balance).encode()
        extratags.append((_XMP_TAG, "B", len(xmp), xmp))
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.02, 0.2, size=(8, 8, 3)).astype(np.float32)
    tifffile.imwrite(path, image, photometric="rgb", extratags=extratags, byteorder="<")
    if exposure is not None:
        _add_exif(path, exposure, (8, 1), 100)


def _check(roll, capsys):
    code = main(["check", str(roll)])
    return code, " ".join(_ANSI.sub("", capsys.readouterr().out).split())


def _roll(tmp_path, frames):
    roll = tmp_path / "roll"
    roll.mkdir()
    for i, kwargs in enumerate(frames):
        _frame(roll / f"frame_{i:02d}.tif", seed=i, **kwargs)
    return roll


WB = (1.9, 1.0, 1.8)


def test_a_consistent_roll_passes(tmp_path, capsys):
    roll = _roll(tmp_path, [dict(exposure=(1, 30), white_balance=WB)] * 3)
    code, out = _check(roll, capsys)
    assert code == 0
    assert "Checked 3 frames" in out and "1/30 f/8 ISO100 3 frames" in out
    assert out.count("consistent") == 2  # exposure and white balance
    assert "stops between" not in out and "not verifiable" not in out and "differs" not in out
    assert "none active" in out


def test_mixed_exposure_is_reported_in_stops_with_the_way_to_correct_it(tmp_path, capsys):
    roll = _roll(tmp_path, [dict(exposure=(1, 30), white_balance=WB), dict(exposure=(1, 30), white_balance=WB),
                            dict(exposure=(1, 60), white_balance=WB)])
    code, out = _check(roll, capsys)
    assert code == 1
    assert "1/30 f/8 ISO100 2 frames" in out and "1/60 f/8 ISO100 1 frames" in out
    assert "1.0 stops between the brightest and darkest scan" in out
    assert "--match-scan-exposure" in out


def test_mixed_white_balance_is_reported_and_not_called_correctable(tmp_path, capsys):
    roll = _roll(tmp_path, [dict(exposure=(1, 30), white_balance=WB), dict(exposure=(1, 30), white_balance=(1.95, 1.0, 1.7))])
    code, out = _check(roll, capsys)
    assert code == 1
    assert "R 1.900 G 1.000 B 1.800" in out and "R 1.950 G 1.000 B 1.700" in out
    assert "white balance differs between frames" in out and "not correctable after export" in out


def test_frames_without_exif_are_not_verifiable_not_consistent(tmp_path, capsys):
    roll = _roll(tmp_path, [dict(), dict()])
    code, out = _check(roll, capsys)
    assert "(no EXIF) 2 frames" in out
    assert "not verifiable (no camera EXIF in 2 frames)" in out
    assert "consistent" not in out.split("Raw white balance")[0]
    assert "no darktable history found" in out
