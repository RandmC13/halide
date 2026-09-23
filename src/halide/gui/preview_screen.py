"""Single-frame preview: run the full pipeline on a downsampled frame with live tone-render
sliders, so you can see the effect of a calibration profile and tone settings before committing
to a full batch run.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
import numpy as np

from halide.calibration.profile_store import list_profiles, load_profile, resolve_profile_path
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.invert import invert
from halide.core.pipeline import run_pipeline
from halide.core.tone_render import estimate_exposure
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui import theme
from halide.gui.sampling import downsample_for_display, preview_stretch
from halide.io.raster import to_srgb_8bit
from halide.processing import ScanColorError, load_working_space_image

_IMAGE_TAG = "preview_image"
_TEXTURE_TAG = "preview_image_texture"
_IMAGE_PLACEHOLDER_TAG = "preview_image_placeholder"
_PROFILE_COMBO_TAG = "preview_profile_combo"
_FILE_DIALOG_TAG = "preview_file_dialog"
# Small, so slider drags re-render at interactive speed, and must fit inside
# preview_image_container's fixed height (see build()) — see calibrate_screen.py's matching
# constants for why these two must stay in sync.
_IMAGE_CONTAINER_HEIGHT = 600
_MAX_DISPLAY_WIDTH = 900
_MAX_DISPLAY_HEIGHT = _IMAGE_CONTAINER_HEIGHT - 20


class PreviewScreen:
    def __init__(self) -> None:
        self.working_image: np.ndarray | None = None  # downsampled, for fast interactive preview
        self.density_profile: DensityProfile | None = None

    def _status(self, message: str) -> None:
        dpg.set_value("preview_status_text", message)

    def load_image(self, path: str) -> None:
        if not path:
            self._status("Enter a TIFF path first.")
            return
        self._status(f"Loading {path}...")
        dpg.render_dearpygui_frame()  # flush that status onto screen before the blocking read below
        try:
            working = load_working_space_image(path)
        except ScanColorError as exc:
            self._status(f"Error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — surface I/O errors in the UI, don't crash it
            self._status(f"Error loading {path}: {exc}")
            return

        display, stride = downsample_for_display(
            working, max_width=_MAX_DISPLAY_WIDTH, max_height=_MAX_DISPLAY_HEIGHT
        )
        self.working_image = display
        self._status(f"Loaded {path} for preview ({display.shape[1]}x{display.shape[0]}, 1/{stride} scale)")
        # Seed a sensible starting exposure the moment both an image and a profile are available,
        # rather than making the user discover the Auto button before seeing a usable preview.
        self.reset_exposure_to_auto() if self.density_profile is not None else self.render()

    def load_profile_by_name(self, name_or_path: str) -> None:
        if not name_or_path:
            self._status("Enter a profile name or path first.")
            return
        try:
            path = resolve_profile_path(name_or_path)
            self.density_profile = load_profile(path)
        except (FileNotFoundError, ValueError) as exc:
            self._status(f"Error loading profile: {exc}")
            return
        self._status(f"Loaded profile {name_or_path!r}")
        self.reset_exposure_to_auto() if self.working_image is not None else self.render()

    def reset_exposure_to_auto(self, sender=None, app_data=None) -> None:
        if self.working_image is None or self.density_profile is None:
            self._status("Load an image and a profile first.")
            return
        try:
            wb = apply_white_balance(self.working_image, self.density_profile)
            db = apply_density_balance(wb, self.density_profile)
            positive = invert(db)
            dpg.set_value("preview_exposure", estimate_exposure(positive))
        except Exception as exc:  # noqa: BLE001 - surface it in the UI, don't crash the app
            self._status(f"Error estimating exposure: {exc}")
            return
        self.render()

    def render(self, sender=None, app_data=None) -> None:
        if self.working_image is None:
            self._status("Load an image first.")
            return
        if self.density_profile is None:
            self._status("Load a calibration profile first.")
            return

        try:
            tone_params = ToneCurveParams(
                mode="linear" if dpg.get_value("preview_linear") else "paper",
                exposure=dpg.get_value("preview_exposure"),
                contrast=dpg.get_value("preview_contrast"),
            )
            result = run_pipeline(self.working_image, self.density_profile, tone_params)
            display_8bit = (
                to_srgb_8bit(result) if tone_params.mode == "paper" else (preview_stretch(result) * 255).astype(np.uint8)
            )

            height, width = display_8bit.shape[:2]
            rgba = np.dstack(
                [display_8bit.astype(np.float32) / 255.0, np.ones((height, width), dtype=np.float32)]
            ).ravel()
        except Exception as exc:  # noqa: BLE001 - surface it in the UI, don't crash the app
            self._status(f"Error rendering preview: {exc}")
            return

        if dpg.does_item_exist(_IMAGE_PLACEHOLDER_TAG):
            dpg.delete_item(_IMAGE_PLACEHOLDER_TAG)
        if dpg.does_item_exist(_TEXTURE_TAG):
            dpg.delete_item(_TEXTURE_TAG)
        if dpg.does_item_exist(_IMAGE_TAG):
            dpg.delete_item(_IMAGE_TAG)
        dpg.add_raw_texture(
            width, height, rgba, tag=_TEXTURE_TAG, format=dpg.mvFormat_Float_rgba, parent="preview_texture_registry"
        )
        dpg.add_image(_TEXTURE_TAG, tag=_IMAGE_TAG, parent="preview_image_container")


def build(screen: PreviewScreen | None = None) -> PreviewScreen:
    """Adds this screen's widgets to whatever DPG container is currently open — see
    calibrate_screen.build's docstring for why there's no explicit parent parameter."""
    screen = screen or PreviewScreen()

    dpg.add_texture_registry(tag="preview_texture_registry")

    def _load_from_dialog(sender, app_data) -> None:
        path = app_data.get("file_path_name", "")
        if path:
            dpg.set_value("preview_path_input", path)
            screen.load_image(path)

    def _profile_names() -> list[str]:
        return [name for name, _ in list_profiles()]

    def _on_profile_selected(sender, app_data) -> None:
        dpg.set_value("preview_profile_input", app_data)
        screen.load_profile_by_name(app_data)

    def _refresh_profile_combo(sender=None, app_data=None) -> None:
        dpg.configure_item(_PROFILE_COMBO_TAG, items=_profile_names())

    with dpg.group():
        dpg.add_text(
            "Preview a calibration profile on this frame before running a full batch. The preview "
            "is downsampled for interactive speed - always confirm the final look on a full-"
            "resolution `halide invert`/`halide batch` run.",
            wrap=700,
        )

        theme.section_break("Load negative")
        with dpg.file_dialog(
            directory_selector=False, show=False, tag=_FILE_DIALOG_TAG,
            callback=_load_from_dialog, width=700, height=400,
        ):
            dpg.add_file_extension(".tif", color=theme.AMBER_HOVER)
            dpg.add_file_extension(".tiff", color=theme.AMBER_HOVER)
            dpg.add_file_extension(".*")
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="TIFF path", tag="preview_path_input", width=400)
            dpg.add_button(label="Browse...", callback=lambda s, a: dpg.show_item(_FILE_DIALOG_TAG))
            dpg.add_button(
                label="Load negative", callback=lambda s, a: screen.load_image(dpg.get_value("preview_path_input"))
            )

        theme.section_break("Load calibration")
        with dpg.group(horizontal=True):
            dpg.add_combo(
                _profile_names(), tag=_PROFILE_COMBO_TAG, label="Saved profiles",
                callback=_on_profile_selected, width=250,
            )
            dpg.add_button(label="Refresh", callback=_refresh_profile_combo)
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="Profile name or path", tag="preview_profile_input", width=400)
            dpg.add_button(
                label="Load calibration",
                callback=lambda s, a: screen.load_profile_by_name(dpg.get_value("preview_profile_input")),
            )

        theme.section_break("Render")
        dpg.add_checkbox(label="Linear output (skip tone curve)", tag="preview_linear", default_value=False, callback=screen.render)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="Exposure", tag="preview_exposure", default_value=0.0, min_value=-2.0, max_value=2.0, callback=screen.render)
            dpg.add_button(label="Auto", callback=screen.reset_exposure_to_auto)
        dpg.add_slider_float(label="Contrast", tag="preview_contrast", default_value=0.5, min_value=0.0, max_value=1.0, callback=screen.render)
        with dpg.child_window(tag="preview_image_container", height=_IMAGE_CONTAINER_HEIGHT):
            dpg.add_text(
                "Load a negative and a calibration profile above to render a preview here.",
                tag=_IMAGE_PLACEHOLDER_TAG,
                color=theme.TEXT_DIM,
                wrap=_MAX_DISPLAY_WIDTH,
            )
        dpg.add_text("", tag="preview_status_text")

    return screen
