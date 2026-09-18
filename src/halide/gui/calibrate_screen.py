"""The anchor-frame calibration picker: load one frame, click a shadow-side and a highlight-side
neutral point, get a live rendered preview, and optionally save a reusable DensityProfile — this
is the upgrade over the old script's crude 4x-downsampled matplotlib click-picker (see
core/density.py and calibration/auto.py docstrings for the color-science background this UI is
built around).

The displayed image is the RAW, UNINVERTED negative (see gui/sampling.py::preview_stretch) — on a
raw negative, brightness is backwards from the real scene: a real highlight (bright in life) is a
dense, DARK area on the raw negative, and a real shadow (dark in life) is a thin, LIGHT area. Every
user-facing label here says so explicitly — this used to be a real source of confusion (and a
plausible source of genuinely wrong calibration, not just an annoyance), see CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import replace

import dearpygui.dearpygui as dpg
import numpy as np

from halide.calibration.auto import DEFAULT_NEUTRAL_FRACTION, _neutral_candidate_mask, auto_density_balance
from halide.calibration.profile_store import save_named_profile
from halide.core.density import apply_density_balance, apply_white_balance, solve_density_balance
from halide.core.invert import invert
from halide.core.pipeline import run_pipeline
from halide.core.tone_render import estimate_exposure
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui.sampling import (
    apply_stretch,
    compute_stretch_bounds,
    display_to_full_res_coords,
    downsample_for_display,
    extract_magnifier_patch,
    full_res_to_display_coords,
    magnify_patch,
    patch_chroma,
    snap_to_representative_pixel,
)
from halide.io.raster import to_srgb_8bit
from halide.processing import ScanColorError, load_working_space_image

_TEXTURE_TAG = "calibrate_image_texture"
_DRAWLIST_TAG = "calibrate_drawlist"
_MAGNIFIER_TEXTURE_TAG = "calibrate_magnifier_texture"
_MAGNIFIER_IMAGE_TAG = "calibrate_magnifier_image"
_PREVIEW_TEXTURE_TAG = "calibrate_preview_texture"
_PREVIEW_IMAGE_TAG = "calibrate_preview_image"
_SHADOW_MARKER_TAG = "calibrate_shadow_marker"
_HIGHLIGHT_MARKER_TAG = "calibrate_highlight_marker"

# Must fit inside the image_container child_window's fixed height (see build(), below) — a
# displayed image taller than its container silently makes the bottom portion unclickable rather
# than erroring, so keep these two in sync (found via interactive testing, see
# gui/sampling.py::downsample_for_display's docstring).
_IMAGE_CONTAINER_HEIGHT = 420
_MAX_DISPLAY_WIDTH = 680
_MAX_DISPLAY_HEIGHT = _IMAGE_CONTAINER_HEIGHT - 20

_MAGNIFIER_RADIUS = 12
_MAGNIFIER_ZOOM = 8
_MAGNIFIER_SIZE = (2 * _MAGNIFIER_RADIUS + 1) * _MAGNIFIER_ZOOM  # 200

_PREVIEW_CONTAINER_HEIGHT = 220

# The single source of truth for these labels — used by the radio button, the click handler's mode
# check, and nowhere else, so there's exactly one place that can go stale.
_SHADOW_MODE = "Shadow point - a scene shadow (dark in real life); looks LIGHT on this raw negative"
_HIGHLIGHT_MODE = "Highlight point - a scene highlight (bright in real life); looks DARK on this raw negative"

_SHADOW_MARKER_COLOR = (255, 90, 90, 255)
_HIGHLIGHT_MARKER_COLOR = (90, 160, 255, 255)
_OVERLAY_TINT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class CalibrateScreen:
    """Owns this screen's state. A DPG app is inherently callback-driven with shared UI state, so
    this is a thin stateful wrapper rather than a pure function — the actual color-science logic
    it calls into (solve_density_balance, snap_to_representative_pixel, ...) stays pure and
    separately unit-tested."""

    def __init__(self) -> None:
        self.full_image: np.ndarray | None = None  # working-space (ACEScg), full resolution
        self.preview_working: np.ndarray | None = None  # downsampled, pre-stretch — display + live preview render
        self.stride: int = 1
        self.shadow_point: tuple[int, int, np.ndarray] | None = None  # (x, y, rgb), full-res coords
        self.highlight_point: tuple[int, int, np.ndarray] | None = None
        self.live_profile: DensityProfile | None = None  # solved automatically once both points are picked
        self.auto_profile: DensityProfile | None = None  # independent statistical estimate, for comparison
        self.auto_candidate_mask: np.ndarray | None = None  # (H, W) bool, same shape as preview_working
        self._stretch_bounds: tuple[float, float] = (0.0, 1.0)

    def _status(self, message: str) -> None:
        dpg.set_value("status_text", message)

    def _refresh_point_labels(self) -> None:
        for label, point, tag in (
            ("Shadow point", self.shadow_point, "shadow_text"),
            ("Highlight point", self.highlight_point, "highlight_text"),
        ):
            if point is None:
                dpg.set_value(tag, f"{label}: not picked")
                continue
            x, y, rgb = point
            chroma = patch_chroma(rgb)
            dpg.set_value(
                tag,
                f"{label}: pixel ({x},{y})  RGB=({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})  "
                f"chroma={chroma:.2f} (rough guide only - see note above)",
            )

    def _refresh_auto_comparison_text(self) -> None:
        if self.auto_profile is None:
            dpg.set_value("auto_comparison_text", "Auto-detected estimate: unavailable for this image.")
            return
        dpg.set_value(
            "auto_comparison_text",
            "Auto-detected estimate (for comparison, computed on the downsampled preview - may "
            "differ slightly from `halide invert --auto-density` on the full-resolution image): "
            f"white_balance={tuple(round(v, 4) for v in self.auto_profile.white_balance)}  "
            f"density_scale={tuple(round(v, 4) for v in self.auto_profile.density_scale)}",
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
        self.live_profile = None

        display, stride = downsample_for_display(
            working, max_width=_MAX_DISPLAY_WIDTH, max_height=_MAX_DISPLAY_HEIGHT
        )
        self.stride = stride
        self.preview_working = display
        self._stretch_bounds = compute_stretch_bounds(display)

        try:
            self.auto_profile = auto_density_balance(display)
            self.auto_candidate_mask = _neutral_candidate_mask(display, DEFAULT_NEUTRAL_FRACTION)
        except ValueError:
            # Auto-detection can legitimately fail (too few candidates) on an unusual image — this
            # is a comparison aid, not a required part of the picker, so degrade gracefully.
            self.auto_profile = None
            self.auto_candidate_mask = None

        self._upload_main_image()
        self._clear_preview()
        self._refresh_auto_comparison_text()
        self._status(f"Loaded {path} ({working.shape[1]}x{working.shape[0]}, displayed at 1/{stride} scale)")
        self._refresh_point_labels()

    def _build_main_image_rgba(self) -> tuple[np.ndarray, int, int]:
        stretched = apply_stretch(self.preview_working, *self._stretch_bounds)
        if dpg.get_value("auto_overlay_checkbox") and self.auto_candidate_mask is not None:
            stretched = stretched.copy()
            mask = self.auto_candidate_mask
            stretched[mask] = stretched[mask] * 0.6 + _OVERLAY_TINT * 0.4
        height, width = stretched.shape[:2]
        rgba = np.dstack([stretched, np.ones((height, width), dtype=np.float32)]).ravel()
        return rgba, width, height

    def _upload_main_image(self) -> None:
        """Tear down and recreate the main texture/drawlist — only called from load_image, where
        the image (and therefore its dimensions) is genuinely new. Deleting and recreating a GPU
        texture on every interaction (not just a new image) was a real crash: the overlay checkbox
        and repeated point-picking used to call this on every toggle/click, and segfaulted on real
        (non-Xvfb) hardware — see CLAUDE.md. Only reshape/recreate when the content might actually
        be a different size; anything that only changes pixel *values* uses _refresh_main_image."""
        rgba, width, height = self._build_main_image_rgba()

        if dpg.does_item_exist(_DRAWLIST_TAG):
            dpg.delete_item(_DRAWLIST_TAG)  # also deletes its child markers
        if dpg.does_item_exist(_TEXTURE_TAG):
            dpg.delete_item(_TEXTURE_TAG)

        dpg.add_raw_texture(
            width, height, rgba, tag=_TEXTURE_TAG, format=dpg.mvFormat_Float_rgba, parent="texture_registry"
        )
        dpg.add_drawlist(width, height, tag=_DRAWLIST_TAG, parent="image_container")
        dpg.draw_image(_TEXTURE_TAG, pmin=[0, 0], pmax=[width, height], parent=_DRAWLIST_TAG)
        self._redraw_markers()

    def _refresh_main_image(self) -> None:
        """Update the main image's pixel data in place, leaving the texture/drawlist/markers
        untouched — safe because the array's shape never changes here (only the overlay tint is
        toggled), unlike _upload_main_image which tears down and recreates GPU resources."""
        rgba, _, _ = self._build_main_image_rgba()
        dpg.set_value(_TEXTURE_TAG, rgba)

    def toggle_auto_overlay(self, sender, app_data) -> None:
        if self.preview_working is None:
            return
        self._refresh_main_image()

    def _draw_marker(self, tag: str, full_x: int, full_y: int, color: tuple[int, int, int, int]) -> None:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        display_x, display_y = full_res_to_display_coords(full_x, full_y, self.stride)
        dpg.draw_circle((display_x, display_y), radius=6, color=color, thickness=2, tag=tag, parent=_DRAWLIST_TAG)

    def _redraw_markers(self) -> None:
        for point, tag, color in (
            (self.shadow_point, _SHADOW_MARKER_TAG, _SHADOW_MARKER_COLOR),
            (self.highlight_point, _HIGHLIGHT_MARKER_TAG, _HIGHLIGHT_MARKER_COLOR),
        ):
            if point is not None:
                self._draw_marker(tag, point[0], point[1], color)

    def _clear_preview(self) -> None:
        if dpg.does_item_exist(_PREVIEW_IMAGE_TAG):
            dpg.delete_item(_PREVIEW_IMAGE_TAG)
        if dpg.does_item_exist(_PREVIEW_TEXTURE_TAG):
            dpg.delete_item(_PREVIEW_TEXTURE_TAG)
        dpg.set_value("solved_text", "")

    def on_image_click(self, sender, app_data) -> None:
        if self.full_image is None or not dpg.does_item_exist(_DRAWLIST_TAG):
            return
        if not dpg.is_item_hovered(_DRAWLIST_TAG):
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        origin_x, origin_y = dpg.get_item_rect_min(_DRAWLIST_TAG)
        display_x = int(mouse_x - origin_x)
        display_y = int(mouse_y - origin_y)

        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(display_x, display_y, self.stride, w, h)
        x, y, rgb = snap_to_representative_pixel(self.full_image, full_x, full_y)

        if dpg.get_value("pick_mode") == _SHADOW_MODE:
            self.shadow_point = (x, y, rgb)
            self._draw_marker(_SHADOW_MARKER_TAG, x, y, _SHADOW_MARKER_COLOR)
        else:
            self.highlight_point = (x, y, rgb)
            self._draw_marker(_HIGHLIGHT_MARKER_TAG, x, y, _HIGHLIGHT_MARKER_COLOR)
        self._refresh_point_labels()
        self._maybe_render_live_preview()

    def on_image_hover(self, sender, app_data) -> None:
        if self.full_image is None or not dpg.does_item_exist(_DRAWLIST_TAG):
            return
        if not dpg.is_item_hovered(_DRAWLIST_TAG):
            return

        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        origin_x, origin_y = dpg.get_item_rect_min(_DRAWLIST_TAG)
        display_x = int(mouse_x - origin_x)
        display_y = int(mouse_y - origin_y)

        h, w = self.full_image.shape[:2]
        full_x, full_y = display_to_full_res_coords(display_x, display_y, self.stride, w, h)

        patch = extract_magnifier_patch(self.full_image, full_x, full_y, radius=_MAGNIFIER_RADIUS)
        magnified = magnify_patch(apply_stretch(patch, *self._stretch_bounds), zoom=_MAGNIFIER_ZOOM)
        height, width = magnified.shape[:2]
        rgba = np.dstack([magnified, np.ones((height, width), dtype=np.float32)]).ravel()
        dpg.set_value(_MAGNIFIER_TEXTURE_TAG, rgba)

        raw_rgb = self.full_image[full_y, full_x]
        dpg.set_value(
            "hover_readout_text",
            f"pixel ({full_x},{full_y})  RGB=({raw_rgb[0]:.4f}, {raw_rgb[1]:.4f}, {raw_rgb[2]:.4f})",
        )

    def _maybe_render_live_preview(self) -> None:
        if self.shadow_point is None or self.highlight_point is None or self.preview_working is None:
            return

        shadow_rgb = tuple(float(v) for v in self.shadow_point[2])
        highlight_rgb = tuple(float(v) for v in self.highlight_point[2])
        try:
            profile = solve_density_balance(shadow_rgb, highlight_rgb)
        except ValueError as exc:
            self._status(f"Could not solve a preview profile: {exc}")
            return
        self.live_profile = replace(profile, source="anchor")

        dpg.set_value(
            "solved_text",
            f"Solved from your picks: white_balance={tuple(round(v, 4) for v in self.live_profile.white_balance)}  "
            f"density_scale={tuple(round(v, 4) for v in self.live_profile.density_scale)}",
        )
        self._render_preview()
        self._status("Preview updated. Enter a name and click Save Profile to keep this calibration.")

    def _render_preview(self) -> None:
        wb = apply_white_balance(self.preview_working, self.live_profile)
        db = apply_density_balance(wb, self.live_profile)
        positive = invert(db)
        tone_params = ToneCurveParams(exposure=estimate_exposure(positive), contrast=0.5)
        result = run_pipeline(self.preview_working, self.live_profile, tone_params)
        display_8bit = to_srgb_8bit(result)

        height, width = display_8bit.shape[:2]
        rgba = np.dstack(
            [display_8bit.astype(np.float32) / 255.0, np.ones((height, width), dtype=np.float32)]
        ).ravel()

        # The preview panel's dimensions never change while one image stays loaded (they're fixed
        # by self.preview_working, same array used every render) — so a re-pick just updates the
        # existing texture's pixel data instead of tearing down and recreating a GPU texture on
        # every click. Repeatedly deleting+recreating here (once per click) was a real crash on
        # real (non-Xvfb) hardware — see CLAUDE.md. Only the *first* successful pick pair for a
        # newly loaded image needs to create these items; _clear_preview (called from load_image)
        # deletes them so a genuinely new, differently-sized image creates fresh ones.
        if dpg.does_item_exist(_PREVIEW_TEXTURE_TAG):
            dpg.set_value(_PREVIEW_TEXTURE_TAG, rgba)
        else:
            dpg.add_raw_texture(
                width, height, rgba, tag=_PREVIEW_TEXTURE_TAG, format=dpg.mvFormat_Float_rgba, parent="texture_registry"
            )
            dpg.add_image(_PREVIEW_TEXTURE_TAG, tag=_PREVIEW_IMAGE_TAG, parent="calibrate_preview_container")

    def save_profile(self, sender, app_data) -> None:
        if self.live_profile is None:
            self._status(
                "Pick both a shadow and a highlight point first - a preview profile is solved "
                "automatically once both are picked."
            )
            return
        name = dpg.get_value("profile_name")
        if not name:
            self._status("Enter a profile name to save.")
            return
        path = save_named_profile(self.live_profile, name)
        self._status(f"Saved profile {name!r} to {path}")


def build(screen: CalibrateScreen | None = None, *, show_path_input: bool = True) -> CalibrateScreen:
    """Adds this screen's widgets to whatever DPG container is currently open (call from within a
    `with dpg.tab(...):`/`with dpg.window(...):` block — relies on DPG's implicit container
    stack rather than an explicit parent tag, so it composes correctly when nested in tabs).

    `show_path_input=False` hides the TIFF path field and Load button — for a caller (e.g.
    `gui/quick_pick.py`) that already knows the path and calls `screen.load_image(path)` itself;
    showing an empty, pre-irrelevant text field there would just be confusing."""
    screen = screen or CalibrateScreen()

    dpg.add_texture_registry(tag="texture_registry")

    with dpg.group():
        dpg.add_text(
            "This preview shows the RAW, uninverted negative - brightness is backwards from "
            "real life: a real highlight (bright in the scene) looks DARK/dense here, and a real "
            "shadow (dark in the scene) looks LIGHT/thin here. Pick two objects you know are "
            "neutral gray in real life - one that was in shadow or dim light, one that was "
            "brightly lit - by how they looked in the scene, not by how they look on this "
            "screen. One frame's worth of picks is solved once and reused for the whole roll.",
            wrap=700,
        )
        dpg.add_text(
            "Note: the chroma number below is a rough guide only, not proof of neutrality - a "
            "truly neutral scene object can still show real channel imbalance on the raw negative "
            "(that imbalance is exactly what density balance corrects for). Pick objects you know "
            "are actually neutral in real life, and avoid surfaces under noticeably colored or "
            "mixed lighting.",
            wrap=700,
            color=(230, 180, 80),
        )
        if show_path_input:
            dpg.add_input_text(label="TIFF path", tag="path_input")
            dpg.add_button(label="Load", callback=lambda s, a: screen.load_image(dpg.get_value("path_input")))
        dpg.add_radio_button([_SHADOW_MODE, _HIGHLIGHT_MODE], tag="pick_mode", default_value=_SHADOW_MODE)

        with dpg.group(horizontal=True):
            with dpg.child_window(tag="image_container", width=_MAX_DISPLAY_WIDTH + 20, height=_IMAGE_CONTAINER_HEIGHT):
                pass
            with dpg.group():
                with dpg.child_window(
                    tag="magnifier_container", width=_MAGNIFIER_SIZE + 20, height=_MAGNIFIER_SIZE + 20
                ):
                    zeros_rgba = np.zeros(_MAGNIFIER_SIZE * _MAGNIFIER_SIZE * 4, dtype=np.float32)
                    dpg.add_dynamic_texture(
                        _MAGNIFIER_SIZE, _MAGNIFIER_SIZE, zeros_rgba, tag=_MAGNIFIER_TEXTURE_TAG, parent="texture_registry"
                    )
                    dpg.add_image(_MAGNIFIER_TEXTURE_TAG, tag=_MAGNIFIER_IMAGE_TAG)
                dpg.add_text("Hover over the image to inspect a pixel.", tag="hover_readout_text", wrap=260)
                dpg.add_text("Shadow point: not picked", tag="shadow_text", wrap=260)
                dpg.add_text("Highlight point: not picked", tag="highlight_text", wrap=260)

        dpg.add_text("", tag="solved_text")
        dpg.add_text("", tag="auto_comparison_text", wrap=700)
        dpg.add_checkbox(
            label="Show auto-detected neutral-candidate overlay",
            tag="auto_overlay_checkbox",
            default_value=False,
            callback=screen.toggle_auto_overlay,
        )
        with dpg.child_window(tag="calibrate_preview_container", height=_PREVIEW_CONTAINER_HEIGHT):
            pass
        dpg.add_input_text(label="Save as (profile name)", tag="profile_name")
        dpg.add_button(label="Save Profile", callback=screen.save_profile)
        dpg.add_text("", tag="status_text")

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=screen.on_image_click)
        dpg.add_mouse_move_handler(callback=screen.on_image_hover)

    return screen
