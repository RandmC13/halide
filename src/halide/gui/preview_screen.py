"""Single-frame preview: run the full pipeline on a downsampled frame with live tone-render
sliders, so you can see the effect of a calibration profile and tone settings before committing
to a full batch run.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
import numpy as np

from halide.calibration.profile_store import load_profile, resolve_profile_path
from halide.core.density import apply_density_balance, apply_white_balance
from halide.core.invert import invert
from halide.core.pipeline import run_pipeline
from halide.core.tone_render import estimate_exposure
from halide.core.types import DensityProfile, ToneCurveParams
from halide.gui.sampling import downsample_for_display, preview_stretch
from halide.io.raster import to_srgb_8bit
from halide.processing import ScanColorError, load_working_space_image

_IMAGE_TAG = "preview_image"
_TEXTURE_TAG = "preview_image_texture"
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
        wb = apply_white_balance(self.working_image, self.density_profile)
        db = apply_density_balance(wb, self.density_profile)
        positive = invert(db)
        dpg.set_value("preview_exposure", estimate_exposure(positive))
        self.render()

    def render(self, sender=None, app_data=None) -> None:
        if self.working_image is None:
            self._status("Load an image first.")
            return
        if self.density_profile is None:
            self._status("Load a calibration profile first.")
            return

        tone_params = ToneCurveParams(
            mode="linear" if dpg.get_value("preview_linear") else "paper",
            exposure=dpg.get_value("preview_exposure"),
            contrast=dpg.get_value("preview_contrast"),
        )
        result = run_pipeline(self.working_image, self.density_profile, tone_params)
        display_8bit = to_srgb_8bit(result) if tone_params.mode == "paper" else (preview_stretch(result) * 255).astype(np.uint8)

        height, width = display_8bit.shape[:2]
        rgba = np.dstack(
            [display_8bit.astype(np.float32) / 255.0, np.ones((height, width), dtype=np.float32)]
        ).ravel()

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

    with dpg.group():
        dpg.add_text(
            "Preview a calibration profile on this frame before running a full batch. The preview "
            "is downsampled for interactive speed — always confirm the final look on a full-"
            "resolution `halide invert`/`halide batch` run.",
            wrap=700,
        )
        dpg.add_input_text(label="TIFF path", tag="preview_path_input")
        dpg.add_button(label="Load image", callback=lambda s, a: screen.load_image(dpg.get_value("preview_path_input")))
        dpg.add_input_text(label="Profile name or path", tag="preview_profile_input")
        dpg.add_button(
            label="Load profile", callback=lambda s, a: screen.load_profile_by_name(dpg.get_value("preview_profile_input"))
        )
        dpg.add_checkbox(label="Linear output (skip tone curve)", tag="preview_linear", default_value=False, callback=screen.render)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="Exposure", tag="preview_exposure", default_value=0.0, min_value=-2.0, max_value=2.0, callback=screen.render)
            dpg.add_button(label="Auto", callback=screen.reset_exposure_to_auto)
        dpg.add_slider_float(label="Contrast", tag="preview_contrast", default_value=0.5, min_value=0.0, max_value=1.0, callback=screen.render)
        with dpg.child_window(tag="preview_image_container", height=_IMAGE_CONTAINER_HEIGHT):
            pass
        dpg.add_text("", tag="preview_status_text")

    return screen
