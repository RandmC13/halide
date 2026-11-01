"""The GUI's shared visual identity — the Qt equivalent of `cli/console/`. A QSS stylesheet plus
a couple of raw color constants for anything that needs to be painted by hand (the magnifier, custom
widgets) rather than styled declaratively.

Palette matches the CLI's own darkroom/film-photography language (see `cli/console/`'s `Style`
class and the tank/enlarger animations there): a warm dark charcoal background, an amber accent for
ordinary controls, and and a red accent reserved for exactly one "primary action"
button per window (Save calibration profile / Develop), a cheap nod to the big red button on the
reference enlarger controller photo (`enlarger-controller.png`) without attempting its full
skeuomorphic skin (deferred, see the GUI redesign plan's final section).
"""

from __future__ import annotations

BACKGROUND = "#181512"
BACKGROUND_ALT = "#201c18"
BORDER = "#4a3c2e"
BORDER_DISABLED = "#2f2822"  # still reads as a button, clearly not one you can press
TEXT = "#e8e0d4"
TEXT_DIM = "#968c7c"
TEXT_WARNING = "#e6b450"

AMBER = "#c46420"
AMBER_HOVER = "#e07a2a"
AMBER_ACTIVE = "#f59137"

RED = "#c4342a"
RED_HOVER = "#e0453a"
RED_ACTIVE = "#f5584c"
RED_EDGE = "#7a1f18"  # the primary button's lower lip

# Neutral-point agreement (calibration/anchors.py bands): the markers on the image, the point list
# and the step wedge's ticks all read these, so a point is the same colour everywhere. AGREE_RED is
# a status colour, deliberately warmer and lighter than RED - that one stays reserved for the single
# primary action button per window.
AGREE_NONE = TEXT  # fewer than 3 points: nothing to agree with yet
AGREE_CALM = "#8fbf7f"
AGREE_AMBER = TEXT_WARNING
AGREE_RED = "#ff6b5a"

# Film, as the filmstrip and contact sheets draw it: black rebate, sprocket holes, orange edge print.
FILM_REBATE = "#0c0c0c"
FILM_SPROCKET = "#2c2620"
EDGE_PRINT = "#e0a060"
FILM_LOADING = "#1c1916"  # a filmstrip frame still developing
FILM_FAILED = "#3a1a18"  # ... and one that failed to load

# The condensed bold face film edge print is drawn in.
EDGE_FAMILY = "DejaVu Sans Condensed"

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
    background-color: transparent;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 13px;
}}
QPushButton:focus {{
    border: 2px solid {TEXT};
    padding: 4px 12px;  /* the ring's px come out of the padding: the button doesn't change size */
}}
QPushButton:hover {{
    border-color: {AMBER};
}}
QPushButton:pressed {{
    background-color: {BACKGROUND_ALT};
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER_DISABLED};
}}
QPushButton:checked {{
    background-color: {AMBER_ACTIVE};
    border: 2px solid {TEXT};
}}

QPushButton[role="primary"] {{
    background-color: {RED};
    font-weight: bold;
    border: none;
    border-bottom: 2px solid {RED_EDGE};
    padding: 10px 18px 8px 18px;
}}
QPushButton[role="primary"]:focus {{
    border: 2px solid {TEXT};
    border-bottom: 2px solid {RED_EDGE};
    padding: 8px 16px 8px 16px;
}}
QPushButton[role="primary"]:hover {{
    background-color: {RED_HOVER};
}}
QPushButton[role="primary"]:pressed {{
    background-color: {RED_ACTIVE};
    border-bottom: none;
    border-top: 2px solid {RED_EDGE};  /* the lip moves to the top: the label sits 1 px lower */
    padding: 9px 18px 9px 18px;
}}
QPushButton[role="primary"]:disabled {{
    background-color: {BACKGROUND_ALT};
    border-bottom-color: {BACKGROUND_ALT};
    color: {TEXT_DIM};
}}

QPushButton[role="segment"] {{
    background-color: {BACKGROUND_ALT};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 14px;
}}
QPushButton[role="segment"]:hover {{
    color: {TEXT};
}}
QPushButton[role="segment"]:focus {{
    border: 1px solid {AMBER_ACTIVE};
    padding: 4px 14px;
}}
QPushButton[role="segment"]:checked {{
    background-color: {AMBER};
    color: {TEXT};
    border: 1px solid {AMBER};
}}

QPushButton[role="remove"] {{
    background: transparent;
    color: {TEXT_DIM};
    border: none;
    padding: 0px 5px;
    border-radius: 3px;
}}
QPushButton[role="remove"]:focus {{
    border: none;
    padding: 0px 5px;
    color: {TEXT};
    background-color: {BORDER};
}}
QPushButton[role="remove"]:hover {{
    color: {TEXT};
    background-color: {BORDER};
}}

QLabel[role="section"] {{
    color: {TEXT_DIM};
    font-size: 12px;
    font-weight: 600;
}}

QLineEdit {{
    background-color: {BACKGROUND_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    color: {TEXT};
}}
QLineEdit:focus {{
    border: 1px solid {AMBER_ACTIVE};
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
    background: {TEXT_DIM};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:focus {{
    background: {AMBER};
}}
QSlider::handle:horizontal:hover {{
    background: {AMBER};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BACKGROUND_ALT};
}}
QCheckBox::indicator:focus {{
    border: 1px solid {AMBER_ACTIVE};
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
    background: {BORDER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {BACKGROUND_ALT};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
"""


def load_fonts() -> None:
    """Register the vendored DejaVu fonts so "DejaVu Sans" is the same everywhere; a missing file
    just leaves the system's own choice."""
    from importlib.resources import files

    from PySide6.QtGui import QFontDatabase

    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf"):
        path = files("halide.assets") / "fonts" / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))


def apply(app) -> None:
    """Apply the stylesheet to a QApplication. Call once, right after construction."""
    load_fonts()
    app.setStyleSheet(STYLESHEET)


def edge_font(pixel_size: int):
    """The bold condensed face for anything drawn as film edge print."""
    from PySide6.QtGui import QFont

    font = QFont(EDGE_FAMILY)
    font.setBold(True)  # the family has only a bold face registered
    font.setPixelSize(pixel_size)
    return font
