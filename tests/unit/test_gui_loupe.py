"""The loupe: its readout is the density a click at the same spot would give, and it is round."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint

from tests.unit.test_gui_interaction import HIGHLIGHT, SHADOW, _click_image, app, window  # noqa: F401


def test_loupe_density_matches_the_point_a_click_creates(window):
    for spot in (SHADOW, HIGHLIGHT):
        window._on_image_hovered(*spot)
        shown = window._magnifier.readout_text()
        _click_image(window, spot)
        view = window.session.views()[-1]
        assert shown == f"D {view.green_density:.2f}"


def test_loupe_is_round_not_a_rectangle(window):
    window._on_image_hovered(*SHADOW)
    mask = window._magnifier.mask()
    assert not mask.contains(window._magnifier.rect().topLeft())  # the lens corner is cut away
    assert mask.contains(QPoint(62, 62))  # the lens centre
    assert mask.contains(QPoint(60, 135))  # the readout
