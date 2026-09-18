"""The anchor-frame calibration picker: load one frame, click a shadow-side and a highlight-side
neutral point, get live feedback, solve, and save a reusable DensityProfile — this is the upgrade
over the old script's crude 4x-downsampled matplotlib click-picker (see core/density.py and
calibration/auto.py docstrings for the color-science background this UI is built around).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from halide.calibration.profile_store import save_named_profile
from halide.core.density import solve_density_balance
from halide.gui.sampling import (
    downsample_for_display,
    patch_chroma,
    preview_stretch,
    snap_to_representative_pixel,
)
from halide.processing import ScanColorError, load_working_space_image

_IMAGE_TAG = "calibrate_image"
_TEXTURE_TAG = "calibrate_image_texture"
# Must fit inside the image_container child_window's fixed height (see build(), below) — a
# displayed image taller than its container silently makes the bottom portion unclickable rather
# than erroring, so keep these two in sync (found via interactive testing, see
# gui/sampling.py::downsample_for_display's docstring).
_IMAGE_CONTAINER_HEIGHT = 600
_MAX_DISPLAY_WIDTH = 950
_MAX_DISPLAY_HEIGHT = _IMAGE_CONTAINER_HEIGHT - 20


class CalibrateScreen:
    """Owns this screen's state. A DPG app is inherently callback-driven with shared UI state, so
    this is a thin stateful wrapper rather than a pure function — the actual color-science logic
    it calls into (solve_density_balance, snap_to_representative_pixel, ...) stays pure and
    separately unit-tested."""

    def __init__(self) -> None:
        self.full_image: np.ndarray | None = None  # working-space (ACEScg), full resolution
        self.stride: int = 1
        self.shadow_point: tuple[int, int, np.ndarray] | None = None  # (x, y, rgb), full-res coords
        self.highlight_point: tuple[int, int, np.ndarray] | None = None

    def _status(self, message: str) -> None:
        dpg.set_value("status_text", message)

    def _refresh_point_labels(self) -> None:
        for label, point, tag in (
            ("Shadow (dark neutral)", self.shadow_point, "shadow_text"),
            ("Highlight (bright neutral)", self.highlight_point, "highlight_text"),
        ):
            if point is None:
                dpg.set_value(tag, f"{label}: not picked")
                continue
            x, y, rgb = point
            chroma = patch_chroma(rgb)
            dpg.set_value(
                tag,
                f"{label}: pixel ({x},{y})  RGB=({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})  "
                f"chroma={chroma:.2f} (rough guide only — see note above)",
            )

    def load_image(self, path: str) -> None:
        if not path:
            self._status("Enter a TIFF path first.")
            return
        try:
            working = load_working_space_image(path)
        except ScanColorError as exc:
            self._status(f"Error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — surface any I/O error in the UI, don't crash it
            self._status(f"Error loading {path}: {exc}")
            return

        self.full_image = working
        self.shadow_point = None
        self.highlight_point = None

        display, stride = downsample_for_display(
            working, max_width=_MAX_DISPLAY_WIDTH, max_height=_MAX_DISPLAY_HEIGHT
        )
        self.stride = stride
        stretched = preview_stretch(display)

        height, width = stretched.shape[:2]
        rgba = np.dstack([stretched, np.ones((height, width), dtype=np.float32)]).ravel()

        if dpg.does_item_exist(_TEXTURE_TAG):
            dpg.delete_item(_TEXTURE_TAG)
        if dpg.does_item_exist(_IMAGE_TAG):
            dpg.delete_item(_IMAGE_TAG)

        dpg.add_raw_texture(
            width, height, rgba, tag=_TEXTURE_TAG, format=dpg.mvFormat_Float_rgba, parent="texture_registry"
        )
        dpg.add_image(_TEXTURE_TAG, tag=_IMAGE_TAG, parent="image_container")

        self._status(f"Loaded {path} ({working.shape[1]}x{working.shape[0]}, displayed at 1/{stride} scale)")
        self._refresh_point_labels()

    def on_image_click(self, sender, app_data) -> None:
        if self.full_image is None or not dpg.does_item_exist(_IMAGE_TAG):
            return
        if not dpg.is_item_hovered(_IMAGE_TAG):
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        origin_x, origin_y = dpg.get_item_rect_min(_IMAGE_TAG)
        display_x = int(mouse_x - origin_x)
        display_y = int(mouse_y - origin_y)

        h, w = self.full_image.shape[:2]
        full_x = max(0, min(w - 1, display_x * self.stride))
        full_y = max(0, min(h - 1, display_y * self.stride))

        x, y, rgb = snap_to_representative_pixel(self.full_image, full_x, full_y)

        if dpg.get_value("pick_mode") == "Shadow (dark neutral)":
            self.shadow_point = (x, y, rgb)
        else:
            self.highlight_point = (x, y, rgb)
        self._refresh_point_labels()

    def solve_and_save(self, sender, app_data) -> None:
        if self.shadow_point is None or self.highlight_point is None:
            self._status("Pick both a shadow and a highlight point first.")
            return

        shadow_rgb = tuple(float(v) for v in self.shadow_point[2])
        highlight_rgb = tuple(float(v) for v in self.highlight_point[2])
        try:
            profile = solve_density_balance(shadow_rgb, highlight_rgb)
        except ValueError as exc:
            self._status(f"Could not solve a profile: {exc}")
            return
        profile = replace(profile, source="anchor")

        dpg.set_value(
            "solved_text",
            f"white_balance={tuple(round(v, 4) for v in profile.white_balance)}  "
            f"density_scale={tuple(round(v, 4) for v in profile.density_scale)}",
        )

        name = dpg.get_value("profile_name")
        if not name:
            self._status("Profile solved. Enter a name above and click Solve & Save again to save it.")
            return
        path = save_named_profile(profile, name)
        self._status(f"Saved profile {name!r} to {path}")


def build(screen: CalibrateScreen | None = None) -> CalibrateScreen:
    """Adds this screen's widgets to whatever DPG container is currently open (call from within a
    `with dpg.tab(...):`/`with dpg.window(...):` block — relies on DPG's implicit container
    stack rather than an explicit parent tag, so it composes correctly when nested in tabs)."""
    screen = screen or CalibrateScreen()

    dpg.add_texture_registry(tag="texture_registry")

    with dpg.group():
        dpg.add_text(
            "Pick a shadow-side neutral point (a dark, genuinely neutral object) and a "
            "highlight-side neutral point (a bright, genuinely neutral object). One frame's worth "
            "of picks is solved once and reused for the whole roll.",
            wrap=700,
        )
        dpg.add_text(
            "Note: the chroma number below is a rough guide only, not proof of neutrality — a "
            "truly neutral scene object can still show real channel imbalance on the raw negative "
            "(that imbalance is exactly what density balance corrects for). Pick objects you know "
            "are actually neutral in real life, and avoid surfaces under noticeably colored or "
            "mixed lighting.",
            wrap=700,
            color=(230, 180, 80),
        )
        dpg.add_input_text(label="TIFF path", tag="path_input")
        dpg.add_button(label="Load", callback=lambda s, a: screen.load_image(dpg.get_value("path_input")))
        dpg.add_radio_button(
            ["Shadow (dark neutral)", "Highlight (bright neutral)"],
            tag="pick_mode",
            default_value="Shadow (dark neutral)",
        )
        with dpg.child_window(tag="image_container", height=_IMAGE_CONTAINER_HEIGHT):
            pass
        dpg.add_text("Shadow (dark neutral): not picked", tag="shadow_text")
        dpg.add_text("Highlight (bright neutral): not picked", tag="highlight_text")
        dpg.add_input_text(label="Save as (profile name)", tag="profile_name")
        dpg.add_button(label="Solve & Save", callback=screen.solve_and_save)
        dpg.add_text("", tag="solved_text")
        dpg.add_text("", tag="status_text")

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=screen.on_image_click)

    return screen
