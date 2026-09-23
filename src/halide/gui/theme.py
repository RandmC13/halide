"""The GUI's shared visual identity — the Qt equivalent of `cli/console.py`. A QSS stylesheet plus
a couple of raw color constants for anything that needs to be painted by hand (the magnifier, custom
widgets) rather than styled declaratively.

Palette matches the CLI's own darkroom/film-photography language (see `cli/console.py`'s `Style`
class and the tank/enlarger animations there): a warm dark charcoal background, an amber accent for
ordinary controls, and — new in this rewrite — a red accent reserved for exactly one "primary action"
button per window (Save calibration profile / Develop), a cheap nod to the big red button on the
reference enlarger controller photo (`enlarger-controller.png`) without attempting its full
skeuomorphic skin (deferred, see the GUI redesign plan's final section).
"""

from __future__ import annotations

BACKGROUND = "#181512"
BACKGROUND_ALT = "#201c18"
BORDER = "#4a3c2e"
TEXT = "#e8e0d4"
TEXT_DIM = "#968c7c"
TEXT_WARNING = "#e6b450"

AMBER = "#c46420"
AMBER_HOVER = "#e07a2a"
AMBER_ACTIVE = "#f59137"

RED = "#c4342a"
RED_HOVER = "#e0453a"
RED_ACTIVE = "#f5584c"

# The single source of truth for "what color is a shadow/highlight pick" - the Shadow/Highlight
# mode buttons and the markers drawn on the image (gui/main_window.py::ImageView) both read these,
# so a button's color always tells you what color its picks will draw as, not just which one is
# currently active.
SHADOW_POINT_COLOR = "#ff5a5a"
SHADOW_POINT_MUTED = "#8a3a3a"
SHADOW_POINT_HOVER = "#b34848"

HIGHLIGHT_POINT_COLOR = "#5aa0ff"
HIGHLIGHT_POINT_MUTED = "#355a8a"
HIGHLIGHT_POINT_HOVER = "#4570b3"

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "DejaVu Sans", sans-serif;
    font-size: 13px;
}}

QDialog {{
    background-color: {BACKGROUND};
}}

QLabel {{
    background: transparent;
}}

QLabel[role="dim"] {{
    color: {TEXT_DIM};
}}

QLabel[role="warning"] {{
    color: {TEXT_WARNING};
}}

QPushButton {{
    background-color: {AMBER};
    color: {TEXT};
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: {AMBER_HOVER};
}}
QPushButton:pressed {{
    background-color: {AMBER_ACTIVE};
}}
QPushButton:disabled {{
    background-color: {BACKGROUND_ALT};
    color: {TEXT_DIM};
}}
QPushButton:checked {{
    background-color: {AMBER_ACTIVE};
    border: 2px solid {TEXT};
}}

QPushButton[role="primary"] {{
    background-color: {RED};
    font-weight: bold;
    padding: 10px 18px;
}}
QPushButton[role="primary"]:hover {{
    background-color: {RED_HOVER};
}}
QPushButton[role="primary"]:pressed {{
    background-color: {RED_ACTIVE};
}}
QPushButton[role="primary"]:disabled {{
    background-color: {BACKGROUND_ALT};
    color: {TEXT_DIM};
}}

QPushButton[role="shadow"] {{
    background-color: {SHADOW_POINT_MUTED};
}}
QPushButton[role="shadow"]:hover {{
    background-color: {SHADOW_POINT_HOVER};
}}
QPushButton[role="shadow"]:checked {{
    background-color: {SHADOW_POINT_COLOR};
    border: 2px solid {TEXT};
}}

QPushButton[role="highlight"] {{
    background-color: {HIGHLIGHT_POINT_MUTED};
}}
QPushButton[role="highlight"]:hover {{
    background-color: {HIGHLIGHT_POINT_HOVER};
}}
QPushButton[role="highlight"]:checked {{
    background-color: {HIGHLIGHT_POINT_COLOR};
    border: 2px solid {TEXT};
}}

QLineEdit {{
    background-color: {BACKGROUND_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    color: {TEXT};
}}

QFrame[role="imageBox"] {{
    background-color: {BACKGROUND_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QFrame[role="panel"] {{
    background-color: {BACKGROUND_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

QSlider::groove:horizontal {{
    background: {BACKGROUND_ALT};
    border: 1px solid {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {AMBER};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {AMBER_HOVER};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BACKGROUND_ALT};
}}
QCheckBox::indicator:checked {{
    background: {AMBER_ACTIVE};
}}

QToolButton {{
    background: transparent;
    border: none;
    color: {TEXT_DIM};
}}
QToolButton:hover {{
    color: {TEXT};
}}

QStatusBar {{
    background: {BACKGROUND_ALT};
    color: {TEXT_DIM};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: {BACKGROUND_ALT};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {AMBER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {AMBER_HOVER};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
"""


def apply(app) -> None:
    """Apply the stylesheet to a QApplication. Call once, right after construction."""
    app.setStyleSheet(STYLESHEET)
