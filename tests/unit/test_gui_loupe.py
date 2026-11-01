"""The loupe: its readout is the density a click at the same spot would give, and it is painted
inside the image view beside the cursor (not a window of its own, which Wayland can neither place
exactly nor make round)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF

from halide.gui.main_window import loupe_rect, loupe_text
from tests.unit.test_gui_interaction import HIGHLIGHT, SHADOW, _click_image, app, window  # noqa: F401


def test_loupe_density_matches_the_point_a_click_creates(window):
    for spot in (SHADOW, HIGHLIGHT):
        window._on_image_hovered(*spot)
        shown = loupe_text(window.image_view._loupe[1])
        _click_image(window, spot)
        view = window.session.views()[-1]
        assert shown == f"D {view.green_density:.2f}"


def test_loupe_sits_beside_the_cursor_inside_the_image_view_and_goes_when_the_cursor_leaves(window):
    window._on_image_hovered(*HIGHLIGHT)
    lens = window.image_view.loupe_rect()
    cursor = window.image_view._loupe[2]
    assert lens is not None and QRectF(window.image_view.rect()).contains(lens.center())
    gap = ((lens.center().x() - cursor.x()) ** 2 + (lens.center().y() - cursor.y()) ** 2) ** 0.5 - lens.width() / 2
    assert 5 <= gap <= 20  # just clear of the cursor's tip, not across the frame
    window.image_view.left.emit()
    assert window.image_view.loupe_rect() is None


def test_loupe_flips_below_and_left_near_the_top_right_corner():
    bounds = QRectF(0, 0, 600, 400)
    normal = loupe_rect(QPointF(300, 300), bounds)
    assert normal.center().x() > 300 and normal.center().y() < 300  # above and to the right
    corner = loupe_rect(QPointF(590, 10), bounds)
    assert corner.center().x() < 590 and corner.center().y() > 10  # below and to the left
