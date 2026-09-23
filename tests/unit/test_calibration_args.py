import pytest
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
    assert result.exposure is None  # fitted per image
    assert result.contrast is None  # fitted per image (see core.tone_render.fit_print)


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


def test_output_flat_selects_linear_mode_and_linear_output_is_an_alias():
    args = _args()
    args.output_mode = "flat"
    assert resolve_tone_params(args).mode == "linear"
    args = _args(linear_output=True)
    args.output_mode = None
    assert resolve_tone_params(args).mode == "linear"
    args = _args()
    args.output_mode = "print"
    assert resolve_tone_params(args).mode == "paper"


def test_linear_output_conflicts_with_explicit_output_print():
    args = _args(linear_output=True)
    args.output_mode = "print"
    with pytest.raises(SystemExit):
        resolve_tone_params(args)
