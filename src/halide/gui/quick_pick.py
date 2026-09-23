"""Standalone, blocking calibration picker for `halide invert --pick`: a one-shot alternative to
the persistent `halide calibrate` app, for a user who just wants to manually pick shadow/highlight
points for a single image without building a reusable named profile.

Reuses `calibrate_screen`'s widgets as-is (magnifier, markers, live preview, the auto-detection
overlay/comparison) — none of that is specific to the profile-building workflow, all of it is
directly useful for picking quickly and accurately. The only real difference is the entry point
itself: a blocking function that returns a value, rather than a screen inside the persistent
tabbed app (see `gui/app.py`).
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from halide.core.types import DensityProfile
from halide.gui import calibrate_screen, theme


def run_quick_pick(path: str) -> DensityProfile | None:
    """Open a standalone picker window pre-loaded with `path`, block until the user either clicks
    "Use these values for this run" (returns the picked DensityProfile) or closes the window
    without doing so (returns None).

    Uses a manual `render_dearpygui_frame` loop instead of `start_dearpygui` so this function can
    return a value once the user's done, rather than running until the process exits — new
    territory for this codebase's GUI code (every other entry point runs the full event loop and
    exits the process), verified via real interactive testing, not just code review.
    """
    dpg.create_context()
    theme.apply()
    dpg.create_viewport(title="halide · quick calibrate", width=1000, height=1050)
    dpg.setup_dearpygui()

    state = {"profile": None, "done": False}

    def on_continue(sender, app_data) -> None:
        state["profile"] = screen.live_profile
        state["done"] = True

    with dpg.window(tag="quick_pick_window"):
        screen = calibrate_screen.build(show_path_input=False)
        dpg.add_button(label="Use these values for this run", callback=on_continue)

    dpg.set_primary_window("quick_pick_window", True)
    dpg.show_viewport()
    # Must come after show_viewport(): load_image() forces a frame render to flush its "Loading..."
    # status onto screen before the blocking read, which needs a real, shown viewport window to
    # render into (found via a real GLFW null-window crash on this exact path).
    screen.load_image(path)
    while dpg.is_dearpygui_running() and not state["done"]:
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

    return state["profile"]
