"""End-to-end tests for the flat -> (external edit) -> `halide print` round trip, driving the real
CLI entry point against synthetic scans."""

import json

import numpy as np
import pytest

from halide.cli.main import main
from halide.io.icc import output_profile_bytes
from halide.io.tiff import read_tiff, write_tiff
from tests.unit.test_icc import LINEAR_TAGS, _para_tag, build_icc

MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


@pytest.fixture
def negative_tiff(tmp_path):
    # A continuous spread of densities (not a two-patch fixture) so the print fit's percentiles
    # measure a real range, like a real frame.
    rng = np.random.default_rng(0)
    base = np.array([0.094, 0.131, 0.050], dtype=np.float32)
    density = rng.uniform(0.0, 1.0, size=(48, 48, 1)).astype(np.float32)
    img = base * 10.0 ** (-density)
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


def _develop(negative, out, *extra):
    assert main(["invert", str(negative), str(out), *MANUAL, *extra]) == 0
    return read_tiff(out)


def _simulate_external_edit(flat_path, out_path, exposure_multiply):
    """What a darktable round trip does to the file, as far as halide can tell: a global exposure
    change, a float linear TIFF with a linear profile embedded, and halide's metadata gone."""
    scan = read_tiff(flat_path)
    write_tiff(out_path, scan.image * exposure_multiply, icc_profile=output_profile_bytes())
    assert "halide" not in (read_tiff(out_path).description or "")


def test_outputs_record_their_printing_decision(negative_tiff, tmp_path):
    printed = _develop(negative_tiff, tmp_path / "print.tif")
    record = json.loads(printed.description)["halide"]
    assert record["output"] == "print"
    assert 0 < record["contrast"] <= 1.0
    assert "exposure" in record and "white_balance" in record

    flat = _develop(negative_tiff, tmp_path / "flat.tif", "--output", "flat")
    record = json.loads(flat.description)["halide"]
    assert record["output"] == "flat" and record["linear_scale"] > 0


def test_untouched_flat_then_print_equals_direct_print(negative_tiff, tmp_path):
    direct = _develop(negative_tiff, tmp_path / "direct.tif")
    _develop(negative_tiff, tmp_path / "flat.tif", "--output", "flat")
    assert main(["print", str(tmp_path / "flat.tif"), str(tmp_path / "printed.tif")]) == 0
    printed = read_tiff(tmp_path / "printed.tif")
    assert printed.image == pytest.approx(direct.image, rel=1e-4, abs=1e-6)


def test_externally_exposure_adjusted_flat_still_prints_identically(negative_tiff, tmp_path):
    # The fitted print is invariant to a global multiply, so an exposure change made in darktable
    # (and the loss of halide's metadata) doesn't change the print.
    direct = _develop(negative_tiff, tmp_path / "direct.tif")
    _develop(negative_tiff, tmp_path / "flat.tif", "--output", "flat")
    _simulate_external_edit(tmp_path / "flat.tif", tmp_path / "edited.tif", exposure_multiply=1.7)
    assert main(["print", str(tmp_path / "edited.tif"), str(tmp_path / "printed.tif")]) == 0
    printed = read_tiff(tmp_path / "printed.tif")
    assert printed.image == pytest.approx(direct.image, rel=1e-4, abs=1e-6)


def test_pinned_exposure_reproduces_exactly_when_metadata_survives(negative_tiff, tmp_path):
    direct = _develop(negative_tiff, tmp_path / "direct.tif", "--exposure", "0.2", "--contrast", "0.7")
    _develop(negative_tiff, tmp_path / "flat.tif", "--output", "flat")
    assert main(["print", str(tmp_path / "flat.tif"), str(tmp_path / "p.tif"), "--exposure", "0.2", "--contrast", "0.7"]) == 0
    assert read_tiff(tmp_path / "p.tif").image == pytest.approx(direct.image, rel=1e-4, abs=1e-6)


def test_pinned_exposure_without_metadata_warns_and_fits_instead(negative_tiff, tmp_path, capsys):
    _develop(negative_tiff, tmp_path / "flat.tif", "--output", "flat")
    _simulate_external_edit(tmp_path / "flat.tif", tmp_path / "edited.tif", exposure_multiply=1.0)
    assert main(["print", str(tmp_path / "edited.tif"), str(tmp_path / "p.tif"), "--exposure", "0.2"]) == 0
    assert "can't be reproduced" in capsys.readouterr().out
    record = json.loads(read_tiff(tmp_path / "p.tif").description)["halide"]
    assert record["exposure"] != pytest.approx(0.2)


def test_print_refuses_an_already_printed_file(negative_tiff, tmp_path):
    _develop(negative_tiff, tmp_path / "print.tif")
    with pytest.raises(SystemExit, match="already a halide print"):
        main(["print", str(tmp_path / "print.tif"), str(tmp_path / "again.tif")])


def test_print_rejects_a_gamma_encoded_input(tmp_path):
    tags = dict(LINEAR_TAGS)
    for ch in ("rTRC", "gTRC", "bTRC"):
        tags[ch] = _para_tag(0, [2.2])
    path = tmp_path / "gamma.tif"
    write_tiff(path, np.full((8, 8, 3), 0.5, dtype=np.float32), icc_profile=build_icc(tags))
    with pytest.raises(SystemExit, match="not a linear tone curve"):
        main(["print", str(path), str(tmp_path / "out.tif")])


def test_print_directory_mode(negative_tiff, tmp_path):
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    for name in ("a", "b"):
        _develop(negative_tiff, flat_dir / f"{name}.tif", "--output", "flat")
    out_dir = tmp_path / "prints"
    assert main(["print", str(flat_dir), str(out_dir), "--quiet", "--workers", "1"]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["a.tif", "b.tif"]
