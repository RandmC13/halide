"""The develop animation's pictures: they start at the negative and end at the print."""

import numpy as np

from halide.gui.develop_animation import TRAY_HOLD, PrintFrames


def _frames():
    rng = np.random.default_rng(0)
    neg = rng.integers(0, 256, (6, 9, 3), dtype=np.uint8)
    pos = np.zeros((6, 9, 3), dtype=np.uint8)
    pos[:3] = 20  # shadows
    pos[3:] = 235  # highlights
    return neg, pos, PrintFrames(neg, pos)


def test_expose_starts_on_the_negative_and_ends_on_blank_safelit_paper():
    neg, _pos, f = _frames()
    assert np.array_equal(f.expose(0.0), neg)
    paper = f.expose(1.0)
    assert (paper == paper[0, 0]).all() and paper[0, 0, 0] > paper[0, 0, 1] > paper[0, 0, 2]
    assert np.array_equal(f.tray(0.0), paper)  # the tray starts where exposing ended


def test_shadows_come_up_before_highlights_and_the_tray_ends_where_lights_on_starts():
    _neg, _pos, f = _frames()
    blank = f.tray(0.0).astype(int)
    mid = f.tray(0.4).astype(int)
    assert (blank[:3] - mid[:3]).sum() > 0 and (blank[3:] - mid[3:]).sum() == 0
    assert np.array_equal(f.tray(1.0), f.lights(0.0))
    held = f.tray(TRAY_HOLD).astype(int)
    assert np.abs(held - f.tray(1.0).astype(int)).max() > 0  # held just short of full


def test_lights_on_ends_on_the_colour_print():
    _neg, pos, f = _frames()
    assert np.array_equal(f.lights(1.0), pos)
