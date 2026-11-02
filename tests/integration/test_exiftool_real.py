"""copy_exif_metadata through the real exiftool, kept open (halide.io.exiftool), against the
one-shot call it replaced. Skipped where exiftool isn't installed. This keeps the
byte-for-byte comparison running on small synthetic files."""

import hashlib
import shutil
import subprocess

import numpy as np
import pytest
import tifffile

from halide.io import exiftool
from halide.io.icc import output_profile_bytes
from halide.io.tiff import copy_exif_metadata, write_tiff

pytestmark = pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool isn't installed")


@pytest.fixture(autouse=True)
def _fresh_session():
    exiftool._reset()
    yield
    exiftool._reset()


@pytest.fixture
def scan(tmp_path):
    # A source with metadata worth copying: camera make/model and a date, as a real scan has.
    path = tmp_path / "scan.tif"
    tifffile.imwrite(
        path,
        np.full((8, 8, 3), 0.25, dtype=np.float32),
        photometric="rgb",
        description="source description",
        software="darktable 5.0",
        datetime="2026:09:27 12:00:00",
        extratags=[(271, "s", 0, "Canon", True), (272, "s", 0, "Canon EOS R6", True)],
    )
    return path


def _output(path):
    write_tiff(path, np.full((8, 8, 3), 0.5, dtype=np.float32), icc_profile=output_profile_bytes())
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oneshot(source, dest, drop_icc):
    command = ["exiftool", "-TagsFromFile", str(source), "-all:all", "--ExifImageWidth", "--ExifImageHeight"]
    if drop_icc:
        command.append("--icc_profile")
    subprocess.run(command + ["-overwrite_original", str(dest)], check=True, capture_output=True)


@pytest.mark.parametrize("drop_icc", [True, False])
@pytest.mark.parametrize("name", ["out.tif", "Überbelichtung 日本 é.tif"])
def test_session_writes_the_same_bytes_as_the_oneshot_call(scan, tmp_path, drop_icc, name):
    # Includes a non-ASCII file name: the session passes file names as UTF-8 (-charset
    # filename=utf8), which must not change what is written.
    expected = _output(tmp_path / "expected.tif")
    _oneshot(scan, expected, drop_icc)
    actual = _output(tmp_path / name)
    assert copy_exif_metadata(scan, actual, drop_icc=drop_icc) is True
    if exiftool._KEPT_OPEN_SUPPORTED:
        assert exiftool.session() is not None  # it really went through the session
    assert _sha(actual) == _sha(expected)
    with tifffile.TiffFile(actual) as tif:
        assert tif.pages[0].tags[272].value == "Canon EOS R6"


@pytest.mark.skipif(not exiftool._KEPT_OPEN_SUPPORTED, reason="the kept-open session is Linux-only")
def test_one_process_serves_many_files_and_survives_a_failed_one(scan, tmp_path):
    copy_exif_metadata(scan, _output(tmp_path / "a.tif"))
    session = exiftool.session()
    pid = session.pid
    with pytest.raises(exiftool.ExifToolError):
        copy_exif_metadata(tmp_path / "missing.tif", _output(tmp_path / "b.tif"))
    assert copy_exif_metadata(scan, _output(tmp_path / "c.tif")) is True
    assert exiftool.session() is session and session.pid == pid


_DARKTABLE_XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:darktable="http://darktable.sf.net/" darktable:history_end="46"/>'
    b"</rdf:RDF></x:xmpmeta>"
)


@pytest.mark.parametrize("suffix", [".jpg", ".png"])
def test_a_sharing_copy_keeps_the_camera_but_not_the_private_tags(tmp_path, suffix):
    import json

    from halide.io.raster import write_srgb_8bit_image

    positive = tmp_path / "positive.tif"
    tifffile.imwrite(
        positive,
        np.full((8, 8, 3), 0.25, dtype=np.float32),
        photometric="rgb",
        description='{"halide": "provenance"}',
        datetime="2026:09:27 12:00:00",
        extratags=[(271, "s", 0, "Canon", True), (272, "s", 0, "Canon EOS R6", True),
                   (269, "s", 0, "/home/someone/scans/positive.tif", True), (700, "B", 0, _DARKTABLE_XMP, True)],
    )
    shared = tmp_path / f"shared{suffix}"
    write_srgb_8bit_image(shared, np.full((8, 8, 3), 128, dtype=np.uint8))
    assert copy_exif_metadata(positive, shared, drop_icc=True, for_sharing=True) is True

    tags = json.loads(subprocess.run(["exiftool", "-j", "-G1", str(shared)], capture_output=True, text=True,
                                     check=True).stdout)[0]
    assert tags.get("IFD0:Model") == "Canon EOS R6"
    assert not any(key.endswith(":ImageDescription") or key.endswith(":DocumentName") for key in tags)
    assert not any(key.startswith("XMP-darktable:") for key in tags)
    assert tags.get("ICC_Profile:ProfileDescription", "").startswith("sRGB")  # its own profile survives
