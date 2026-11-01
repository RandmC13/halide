import numpy as np
import pytest
from PIL import Image

from halide.cli.main import main
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def test_invert_then_export_end_to_end(tmp_path):
    negative_path = tmp_path / "negative.tiff"
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    write_tiff(negative_path, img, icc_profile=build_icc(LINEAR_TAGS))

    positive_path = tmp_path / "positive.tiff"
    assert (
        main(
            [
                "invert", str(negative_path), str(positive_path),
                "--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78",
            ]
        )
        == 0
    )

    delivery_path = tmp_path / "delivery.png"
    assert main(["export", str(positive_path), str(delivery_path)]) == 0

    with Image.open(delivery_path) as delivered:
        assert delivered.size == (16, 16)
        assert delivered.mode == "RGB"
        assert delivered.info.get("icc_profile") is not None


def _delivery(path):
    with Image.open(path) as img:
        return np.asarray(img).astype(int)


def test_export_converts_a_foreign_linear_profile_first(tmp_path, capsys):
    from halide.io.icc import convert_to_working_space, output_profile_bytes, parse_linear_rgb_profile

    rng = np.random.default_rng(3)
    pixels = rng.uniform(0.02, 0.8, (8, 8, 3)).astype(np.float32)
    foreign = tmp_path / "foreign.tiff"
    write_tiff(foreign, pixels, icc_profile=build_icc(LINEAR_TAGS))
    converted = tmp_path / "converted.tiff"
    write_tiff(converted, convert_to_working_space(pixels, parse_linear_rgb_profile(build_icc(LINEAR_TAGS))),
               icc_profile=output_profile_bytes())

    assert main(["export", str(foreign), str(tmp_path / "a.png")]) == 0
    assert main(["export", str(converted), str(tmp_path / "b.png")]) == 0
    assert "ACEScg" not in capsys.readouterr().out
    assert np.abs(_delivery(tmp_path / "a.png") - _delivery(tmp_path / "b.png")).max() <= 1


def test_export_refuses_untagged_and_garbage_profiles(tmp_path):
    untagged = tmp_path / "untagged.tiff"
    write_tiff(untagged, np.full((4, 4, 3), 0.18, dtype=np.float32))
    garbage = tmp_path / "garbage.tiff"
    write_tiff(garbage, np.full((4, 4, 3), 0.18, dtype=np.float32), icc_profile=b"not a profile")
    for path, text in ((untagged, "no embedded ICC profile"), (garbage, "unusable colour profile")):
        out = tmp_path / "out.png"
        with pytest.raises(SystemExit, match=text):
            main(["export", str(path), str(out)])
        assert not out.exists()


def test_bulk_export_fails_only_the_untagged_frame(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    write_tiff(in_dir / "a.tiff", np.full((4, 4, 3), 0.18, dtype=np.float32), icc_profile=build_icc(LINEAR_TAGS))
    write_tiff(in_dir / "b.tiff", np.full((4, 4, 3), 0.18, dtype=np.float32))
    out_dir = tmp_path / "out"
    assert main(["export", str(in_dir), str(out_dir), "--quiet"]) != 0
    assert [p.name for p in out_dir.glob("*.png")] == ["a.png"]


def _write_positive(path):
    write_tiff(path, np.full((4, 4, 3), 0.18, dtype=np.float32), icc_profile=build_icc(LINEAR_TAGS))


def test_bulk_export_converts_every_tiff_in_directory(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        _write_positive(in_dir / f"frame_{i:02d}.tiff")

    out_dir = tmp_path / "out"
    assert main(["export", str(in_dir), str(out_dir), "--quiet"]) == 0

    outputs = sorted(out_dir.glob("*.png"))
    assert len(outputs) == 3
    with Image.open(outputs[0]) as delivered:
        assert delivered.size == (4, 4)
        assert delivered.mode == "RGB"
        assert delivered.info.get("icc_profile") is not None


def test_bulk_export_respects_format_and_suffix(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _write_positive(in_dir / "frame_00.tiff")

    out_dir = tmp_path / "out"
    assert (
        main(
            ["export", str(in_dir), str(out_dir), "--format", "jpg", "--suffix", "_delivery", "--quiet"]
        )
        == 0
    )

    outputs = list(out_dir.glob("*.jpg"))
    assert len(outputs) == 1
    assert outputs[0].name == "frame_00_delivery.jpg"


def test_bulk_export_one_bad_frame_does_not_abort_the_rest(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _write_positive(in_dir / "frame_00.tiff")
    write_tiff(in_dir / "frame_01.tiff", np.zeros((4, 4, 3), dtype=np.float32))  # no ICC at all
    _write_positive(in_dir / "frame_02.tiff")

    out_dir = tmp_path / "out"
    # frame_01 has no ICC profile, which is only a warning (not an error) for export — sabotage it
    # further with unreadable pixel data instead, to force a genuine per-file failure.
    (in_dir / "frame_01.tiff").write_bytes(b"not a tiff")

    exit_code = main(["export", str(in_dir), str(out_dir), "--quiet"])
    assert exit_code == 1

    assert (out_dir / "frame_00.png").exists()
    assert (out_dir / "frame_02.png").exists()
    assert not (out_dir / "frame_01.png").exists()


def test_bulk_export_empty_directory_errors(tmp_path):
    in_dir = tmp_path / "empty_in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    assert main(["export", str(in_dir), str(out_dir)]) == 1


def test_bulk_export_runs_in_parallel_with_explicit_workers(tmp_path):
    # Exercises the real ProcessPoolExecutor path (not a fake executor) with more than one worker,
    # to catch anything that only works when everything stays in a single process (e.g. a worker
    # function or its arguments failing to pickle).
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(4):
        _write_positive(in_dir / f"frame_{i:02d}.tiff")

    out_dir = tmp_path / "out"
    assert main(["export", str(in_dir), str(out_dir), "--workers", "2", "--quiet"]) == 0
    assert len(list(out_dir.glob("*.png"))) == 4


def test_export_bulk_rerun_refuses_without_overwrite(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        _write_positive(in_dir / f"frame_{i:02d}.tiff")

    out_dir = tmp_path / "out"
    assert main(["export", str(in_dir), str(out_dir), "--quiet"]) == 0
    outputs = sorted(out_dir.glob("*.png"))
    before = {p: p.read_bytes() for p in outputs}

    with pytest.raises(SystemExit) as exc:
        main(["export", str(in_dir), str(out_dir), "--quiet"])
    message = str(exc.value.code)
    assert "--overwrite" in message and "--skip-existing" in message
    assert {p: p.read_bytes() for p in outputs} == before  # nothing was silently overwritten

    # --skip-existing develops nothing new (all 3 already exist) and leaves them untouched.
    assert main(["export", str(in_dir), str(out_dir), "--quiet", "--skip-existing"]) == 0
    assert {p: p.read_bytes() for p in outputs} == before

    # --overwrite re-runs and succeeds.
    assert main(["export", str(in_dir), str(out_dir), "--quiet", "--overwrite"]) == 0
    assert len(list(out_dir.glob("*.png"))) == 3
