import argparse

from halide.cli._calibration_args import resolve_tone_params
from halide.core.types import ToneCurveParams

_SAVED_TONE = ToneCurveParams(mode="paper", exposure=0.4, contrast=0.7)


def _args(exposure=None, contrast=None, linear_output=False) -> argparse.Namespace:
    return argparse.Namespace(exposure=exposure, contrast=contrast, linear_output=linear_output)


def test_explicit_cli_flag_wins_over_saved_tone():
    result = resolve_tone_params(_args(exposure=1.0, contrast=0.2), saved_tone=_SAVED_TONE)
    assert result.exposure == 1.0
    assert result.contrast == 0.2


def test_saved_tone_wins_over_built_in_default():
    result = resolve_tone_params(_args(), saved_tone=_SAVED_TONE)
    assert result.exposure == 0.4
    assert result.contrast == 0.7


def test_built_in_default_used_when_nothing_else_given():
    result = resolve_tone_params(_args(), saved_tone=None)
    assert result.exposure is None  # auto-computed per image
    assert result.contrast == 0.5


def test_partial_cli_override_still_falls_back_to_saved_tone_per_field():
    # Only --exposure given on the CLI - contrast should still come from the saved profile.
    result = resolve_tone_params(_args(exposure=-1.2), saved_tone=_SAVED_TONE)
    assert result.exposure == -1.2
    assert result.contrast == 0.7


def test_linear_output_is_cli_flag_only_never_inherited_from_saved_tone():
    result = resolve_tone_params(_args(linear_output=False), saved_tone=_SAVED_TONE)
    assert result.mode == "paper"
    result = resolve_tone_params(_args(linear_output=True), saved_tone=_SAVED_TONE)
    assert result.mode == "linear"
