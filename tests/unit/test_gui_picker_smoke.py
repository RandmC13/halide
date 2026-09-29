"""The picker end to end on Qt's offscreen platform: a synthetic three-frame roll, two points added,
both views, every drawer, a saved profile, close. Nothing here checks pixels or layout; it checks
that the whole session runs and that closing leaves no QThread running (Qt aborts the whole process
if a window is destroyed while one is)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import time

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from halide.calibration.profile_store import default_profiles_dir
from halide.gui import proof_window
from halide.gui.main_window import MainWindow, SaveProfileDialog
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _pump(app, until, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert until(), "timed out waiting for the picker"


def _write_frame(path, seed):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.001, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


def _frame_ready(window, index):
    return window._current == index and window.full_image is not None and window._frame_loader is None


def test_picker_session_from_load_to_saved_profile_to_clean_close(tmp_path, monkeypatch, app):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr("halide.gui.main_window.QMessageBox.warning", lambda *a, **k: pytest.fail("warned"))
    roll = tmp_path / "roll"
    roll.mkdir()
    for i in range(3):
        _write_frame(roll / f"frame_{i:02d}.tiff", seed=i)

    window = MainWindow()
    window.load_roll(roll)
    window.show()
    _pump(app, lambda: _frame_ready(window, 0))
    assert [f.path.name for f in window.session.frames] == [f"frame_{i:02d}.tiff" for i in range(3)]

    # A shadow pick on frame 0 (top half), a highlight pick on frame 1 (bottom half).
    window._on_image_clicked(8, 3)
    assert len(window.session.points) == 1
    window._select_frame(1)
    _pump(app, lambda: _frame_ready(window, 1))
    window._on_image_clicked(8, 12)
    assert len(window.session.points) == 2
    assert window.primary_button.isEnabled()  # two separated points: a fit exists

    # Negative <-> Positive both render; the Print drawer only works on the positive.
    assert not window.print_drawer.isEnabled()
    window.positive_button.click()
    app.processEvents()
    assert window._positive_mode() and window.print_drawer.isEnabled()

    # Each drawer opens, and opening one closes the others.
    drawers = window._accordion.drawers
    assert [d._title for d in drawers] == ["Extra information", "Print", "Details"]
    for drawer in drawers:
        drawer.header.click()
        app.processEvents()
        assert [d.header.isChecked() for d in drawers] == [d is drawer for d in drawers], drawer._title

    window.negative_button.click()
    app.processEvents()
    assert not window._positive_mode() and not window.print_drawer.isEnabled()

    # Save a profile through the real dialog.
    window.details_form.set_values({"film_stock": "Test Stock"})
    window._on_details_changed(window.details_form.values())
    dialog = SaveProfileDialog(window, window.session.profile_to_save(), window.session.sidecars(), "")
    dialog.saved.connect(window._on_saved)
    dialog._name_input.setText("smoke")
    dialog._on_save_clicked()
    saved = default_profiles_dir() / "smoke.json"
    assert saved.exists()
    data = json.loads(saved.read_text())
    assert len(data["white_balance"]) == 3 and len(data["density_scale"]) == 3
    assert len(data["anchors"]) == 2 and os.path.basename(data["roll"]) == "roll"
    assert data["film_stock"] == "Test Stock"
    assert window.status_label.full_text().startswith("Saved calibration profile 'smoke'")

    window.close()
    proof_window.wait_for_stopping_renderers()
    assert window._threads == set()
    assert not [t for t in window.findChildren(QThread) if t.isRunning()]

