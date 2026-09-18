"""Entry point for the halide GUI: anchor-frame calibration picker and single-frame preview,
as tabs in one lightweight window — not a full image editor, deliberately (see the project's
build plan for why: crop/dust/creative editing stays in darktable/GIMP/whatever you already use)."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from halide.gui import calibrate_screen, preview_screen


def main(initial_calibrate_path: str | None = None) -> None:
    dpg.create_context()
    dpg.create_viewport(title="halide", width=1000, height=1050)
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window"):
        with dpg.tab_bar():
            with dpg.tab(label="Calibrate"):
                screen = calibrate_screen.build()
            with dpg.tab(label="Preview"):
                preview_screen.build()

    if initial_calibrate_path:
        dpg.set_value("path_input", initial_calibrate_path)
        screen.load_image(initial_calibrate_path)

    dpg.set_primary_window("primary_window", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
