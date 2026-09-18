import numpy as np
import pytest

from halide.core.density import solve_density_balance
from halide.core.pipeline import run_pipeline
from halide.core.types import ToneCurveParams

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def _synthetic_negative():
    # A tiny synthetic "negative": one pixel at each of the two calibration patches, plus a
    # pixel roughly halfway between them in density.
    mid_rgb = tuple((s + h) / 2 for s, h in zip(SHADOW_RGB, HIGHLIGHT_RGB))
    return np.array([[SHADOW_RGB, HIGHLIGHT_RGB, mid_rgb]], dtype=np.float64)


def test_run_pipeline_defaults_to_paper_tone_render():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    result = run_pipeline(_synthetic_negative(), profile)
    assert result.shape == (1, 3, 3)
    assert np.all(result >= 0.0) and np.all(result <= 1.0 + 1e-9)


def test_run_pipeline_calibration_points_become_neutral_after_full_pipeline():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    result = run_pipeline(_synthetic_negative(), profile, ToneCurveParams(mode="linear"))
    shadow_out, highlight_out = result[0, 0], result[0, 1]
    for out in (shadow_out, highlight_out):
        assert out[0] == pytest.approx(out[1], rel=1e-6)
        assert out[2] == pytest.approx(out[1], rel=1e-6)
    # And the shadow patch (higher negative transmittance) must invert to a *smaller* positive
    # value than the highlight patch — the pipeline must not reverse tonal order.
    assert shadow_out[1] < highlight_out[1]


def test_run_pipeline_linear_output_is_unbounded():
    profile = solve_density_balance(SHADOW_RGB, HIGHLIGHT_RGB)
    result = run_pipeline(_synthetic_negative(), profile, ToneCurveParams(mode="linear"))
    assert result.max() > 1.0
