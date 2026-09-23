"""The GUI's shared visual identity — the dearpygui equivalent of `cli/console.py`. Nothing here
imports calibrate_screen/preview_screen/quick_pick; those import this, never the other way around.

Palette matches the CLI's own darkroom/film-photography language (see `cli/console.py`'s `Style`
class and the tank/enlarger animations there): a warm dark charcoal background instead of dearpygui's
stock cool grey, with an amber accent (the same family as `Style.ORANGE`'s ANSI 256 color 208, which
renders as roughly rgb(255, 135, 0) — used here at a couple of brightness levels for normal/hovered/
active states, the same "brighter = more active" logic the CLI's batch grid uses for its own
processing-state colors).
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

BACKGROUND = (24, 21, 18)
BACKGROUND_ALT = (32, 28, 24)  # child windows, frame backgrounds (input fields, combos)
BORDER = (74, 60, 46)
TEXT = (232, 224, 212)
TEXT_DIM = (150, 138, 124)
TEXT_WARNING = (230, 180, 80)  # matches console.Style.YELLOW's warning role

AMBER = (196, 100, 32)  # resting accent (buttons, checkmarks, active tab)
AMBER_HOVER = (224, 122, 42)
AMBER_ACTIVE = (245, 145, 55)

_THEME_TAG = "halide_theme"


def build_theme() -> int:
    """Create (or return the already-created) global theme tag. Idempotent so callers don't need
    to track whether this has already run in this process."""
    if dpg.does_item_exist(_THEME_TAG):
        return _THEME_TAG

    with dpg.theme(tag=_THEME_TAG):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BACKGROUND)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BACKGROUND_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BACKGROUND_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BACKGROUND)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BACKGROUND_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_Border, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT)

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BACKGROUND_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, BORDER)

            dpg.add_theme_color(dpg.mvThemeCol_Button, AMBER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, AMBER_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, AMBER_ACTIVE)

            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, AMBER_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, AMBER)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, AMBER_ACTIVE)

            dpg.add_theme_color(dpg.mvThemeCol_Tab, BACKGROUND_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, AMBER_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, AMBER)
            dpg.add_theme_color(dpg.mvThemeCol_Header, AMBER)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, AMBER_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, AMBER_ACTIVE)

            dpg.add_theme_color(dpg.mvThemeCol_Separator, BORDER)

    return _THEME_TAG


def apply() -> None:
    """Bind the theme globally. Call once, right after `dpg.create_context()` — every window/tab
    created afterward (including calibrate_screen reused standalone by quick_pick) picks it up."""
    dpg.bind_theme(build_theme())


def section_break(label: str | None = None) -> None:
    """A themed separator marking a new logical section of a screen — the GUI's equivalent of the
    CLI's `console.rule()`, used to break what used to be one flat vertical stack of widgets into
    visually distinct groups (load / pick / preview / save, etc.)."""
    dpg.add_spacer(height=6)
    if label:
        dpg.add_text(label.upper(), color=TEXT_DIM)
    dpg.add_separator()
    dpg.add_spacer(height=2)
