"""The .cube reader: honest about what it was given, and the paper curve is read once per process."""

import numpy as np
import pytest

from halide.io.lut import load_1d_cube, load_paper_curve


def _write(tmp_path, text):
    path = tmp_path / "curve.cube"
    path.write_text(text)
    return path


def test_reads_size_domain_and_rows(tmp_path):
    cube = load_1d_cube(_write(tmp_path, "TITLE x\nLUT_1D_SIZE 3\nDOMAIN_MIN -1\nDOMAIN_MAX 2\n0 0 0\n0.5 0.5 0.5\n1 1 1\n"))
    assert cube.values.tolist() == [0.0, 0.5, 1.0]
    assert (cube.domain_min, cube.domain_max) == (-1.0, 2.0)


def test_lut_1d_size_must_match_the_rows(tmp_path):
    with pytest.raises(ValueError, match="LUT_1D_SIZE says 5 entries but the file has 2 rows"):
        load_1d_cube(_write(tmp_path, "LUT_1D_SIZE 5\n0 0 0\n1 1 1\n"))


def test_trailing_comment_on_a_row_is_ignored(tmp_path):
    cube = load_1d_cube(_write(tmp_path, "0 0 0 # first\n1 1 1\n"))
    assert cube.values.tolist() == [0.0, 1.0]


def test_empty_file_says_so(tmp_path):
    with pytest.raises(ValueError, match="no data rows"):
        load_1d_cube(_write(tmp_path, "# nothing here\n"))


def test_unreadable_line_names_the_line(tmp_path):
    with pytest.raises(ValueError, match="line 2"):
        load_1d_cube(_write(tmp_path, "0 0 0\nLUT_1D_INPUT_RANGE 0 3\n"))


def test_missing_file_is_a_plain_error(tmp_path):
    with pytest.raises(ValueError, match="couldn't read"):
        load_1d_cube(tmp_path / "nope.cube")


def test_degenerate_domain_is_refused(tmp_path):
    with pytest.raises(ValueError, match="DOMAIN_MAX"):
        load_1d_cube(_write(tmp_path, "DOMAIN_MIN 1\nDOMAIN_MAX 1\n0 0 0\n1 1 1\n"))


def test_3d_lut_is_refused(tmp_path):
    with pytest.raises(ValueError, match="3D LUT"):
        load_1d_cube(_write(tmp_path, "LUT_3D_SIZE 2\n0 0 0\n"))


def test_paper_curve_is_read_once_and_normalised():
    first = load_paper_curve()
    assert load_paper_curve() is first
    assert float(np.max(first.values)) == 1.0
