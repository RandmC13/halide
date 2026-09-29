"""Shared CLI presentation layer: colours and symbols, prompts, the run sheet, banners, animations
and the top-level exception boundary used by the installed `halide` console script.

Deliberately separate from halide.batch.progress (which owns the batch-grid rendering itself) so
every command shares one consistent look and feel without either module pulling in the other's
concerns. This is the single place CLI-facing wording/color/interactivity policy lives, the same
role halide.processing plays for the actual processing pipeline (see its own module docstring).

The visual identity is deliberately themed around the darkroom process this tool automates
(developing/printing a negative) rather than being generic tool chrome — see VERB/VERB_PAST and
SPROCKET in style.py — but semantic clarity always wins over theme: success/failure stay ✓/✗,
colors stay in the same semantic slots (green=success, yellow=warning, red=error) everywhere.

The package is split by concern (style, prompts, runsheet, screens, animation, guard), and every
name is re-exported here, so `from halide.cli import console; console.warning(...)` and
`from halide.cli.console import RunSheet` keep working unchanged. Importing any of it must stay
cheap: no numpy/tifffile/Pillow/colour (tests/unit/test_cli_startup.py).
"""

from __future__ import annotations

# Tests reach through the package to these standard modules (e.g. monkeypatching
# console.shutil.get_terminal_size), as they did when this was one file.
import os  # noqa: F401
import shutil  # noqa: F401
import sys  # noqa: F401

from halide.cli.console.style import (  # noqa: F401
    ICON_FAIL,
    ICON_OK,
    ICON_WARN,
    NOUN,
    SPROCKET,
    Style,
    VERB,
    VERB_PAST,
    _ANSI_ESCAPE,
    _COLOR_CODES,
    _SOURCE_COLOR_NAME,
    _StyleMeta,
    cancelled,
    cancelled_message,
    dim,
    error,
    finish_message,
    framed,
    full_width_rule,
    human_bytes,
    human_time,
    interactive_output,
    plural,
    rule,
    rule_fitting,
    source_color,
    success,
    use_color,
    visible_width,
    warning,
)

from halide.cli.console.screens import (  # noqa: F401
    _banner,
    _rule_width,
    help_banner,
    welcome_screen,
)

from halide.cli.console.runsheet import (  # noqa: F401
    RunSheet,
)

from halide.cli.console.animation import (  # noqa: F401
    ENLARGER_FRAMES,
    ENLARGER_MIN_SIZE,
    MINI_SPINNERS,
    TANK_MIN_SIZE,
    _ENLARGER_CONE,
    _ENLARGER_DEVELOP_STAGES,
    _ENLARGER_LAMP,
    _ENLARGER_SWAP_STAGES,
    _ENLARGER_TRAY_BOTTOM,
    _ENLARGER_TRAY_TOP,
    _enlarger_frame,
    _terminal_fits,
    animation,
    random_mini_spinner_frames,
    slide_strings,
    spinner,
    tank_frames,
    themed_animation,
)

from halide.cli.console.prompts import (  # noqa: F401
    is_interactive,
    confirm,
    menu,
    prompt_line,
)

from halide.cli.console.guard import (  # noqa: F401
    _print_err,
    run_guarded,
)
