import numpy as np
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


def test_export_warns_on_mismatched_profile(tmp_path, capsys):
    # A TIFF whose embedded profile is NOT ACEScg (the LINEAR_TAGS fixture profile) should still
    # export, but with a clear warning rather than silently mis-coloring the delivery image.
    path = tmp_path / "not_acescg.tiff"
    write_tiff(path, np.full((4, 4, 3), 0.18, dtype=np.float32), icc_profile=build_icc(LINEAR_TAGS))
    output = tmp_path / "out.png"
    assert main(["export", str(path), str(output)]) == 0
    assert "does not look like ACEScg" in capsys.readouterr().out
