"""The picker's stylesheet rules and the edge-print font."""

import pytest
from PySide6.QtGui import QFontDatabase, QFontInfo
from PySide6.QtWidgets import QApplication

from halide.gui import theme


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _blocks():
    return [b for b in theme.STYLESHEET.split("}") if b.strip()]


def test_default_button_is_transparent():
    block = next(b for b in _blocks() if b.strip().startswith("QPushButton {"))
    assert "background-color: transparent" in block


def test_amber_fills_only_selection():
    for b in _blocks():
        if theme.AMBER in b and theme.AMBER_HOVER not in b and theme.AMBER_ACTIVE not in b:
            selector = b.split("{")[0]
            assert any(k in selector for k in (":checked", ":hover", ":focus", "segment")), selector


def test_red_edge_only_in_primary_blocks():
    blocks = [b for b in _blocks() if theme.RED_EDGE in b]
    assert blocks and all('QPushButton[role="primary"]' in b for b in blocks)


def test_edge_family_is_registered(app):
    assert theme.EDGE_FAMILY in QFontDatabase.families()


def test_edge_font_resolves_to_condensed_bold(app):
    font = theme.edge_font(9)
    info = QFontInfo(font)
    assert info.family() == theme.EDGE_FAMILY and info.bold() and info.pixelSize() == 9
    assert "Condensed" in info.styleName()
