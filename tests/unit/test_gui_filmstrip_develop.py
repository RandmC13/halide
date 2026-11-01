"""The filmstrip printing each frame as the roll develops, following the run's signals."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from halide.gui import filmstrip
from halide.gui.filmstrip import Filmstrip


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


_kept = []


def _strip():
    film = Filmstrip()
    _kept.append(film)  # the strip dies with its scroll area
    film.strip.set_frames([1, 2, 3])
    return film.strip


NEG = np.full((4, 6, 3), 200, dtype=np.uint8)
POS = np.full((4, 6, 3), 40, dtype=np.uint8)


def test_states_follow_the_signals_and_are_final_at_once_without_motion(app):
    strip = _strip()
    assert [strip.print_phase(i) for i in range(3)] == [None, None, None]  # queued: negatives
    strip.print_start(0, NEG, POS)
    strip.print_start(1, NEG, POS)
    assert strip.print_phase(0) == "tray"  # exposed at once, then held in the tray
    strip.print_done(0, True)
    strip.print_done(1, False, "bad scan")
    assert strip.print_phase(0) == "printed" and np.array_equal(strip._prints[0][3], POS)
    assert strip.print_phase(1) is None and strip.failure_at(strip.frame_rect(1).center()) == "bad scan"
    assert strip.print_phase(2) is None
    strip.grab()  # paints every state
    strip.clear_prints()
    assert strip.print_phase(0) is None


def test_with_motion_the_tray_holds_until_done(app, monkeypatch):
    monkeypatch.delenv("HALIDE_MOTION", raising=False)
    monkeypatch.setattr(filmstrip, "EXPOSE_MS", 40)
    monkeypatch.setattr(filmstrip, "TRAY_MS", 80)
    monkeypatch.setattr(filmstrip, "LIGHTS_MS", 40)
    strip = _strip()
    strip.print_start(0, NEG, POS)
    assert strip.print_phase(0) == "expose"

    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)

    pump(0.4)  # well past expose + the whole tray time
    assert strip.print_phase(0) == "tray"
    held = strip._prints[0][3]
    assert not np.array_equal(held, POS)
    strip.print_done(0, True)
    assert strip.print_phase(0) == "lights"
    pump(0.3)
    assert strip.print_phase(0) == "printed" and np.array_equal(strip._prints[0][3], POS)


def test_frames_left_mid_print_go_back_to_negatives_when_the_run_ends(app):
    strip = _strip()
    strip.print_start(0, NEG, POS)
    strip.print_start(1, NEG, POS)
    strip.print_done(0, True)
    strip.drop_unfinished()
    assert [strip.print_phase(i) for i in range(2)] == ["printed", None]
