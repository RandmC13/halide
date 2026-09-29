"""Whole-screen text: the welcome screen, the help banner, and human-readable time/size."""

from __future__ import annotations

from halide.cli.console.style import Style, rule


def _rule_width(columns: int = 20) -> int:
    """The visible width of `rule(columns)` — matches its construction exactly (columns pairs of
    "▫ ", minus the one trailing space `.rstrip()` removes), so anything centered against this can
    never drift out of sync with `rule()` itself the way hand-guessed padding already has once."""
    return columns * 2 - 1


def _banner(tagline: str) -> str:
    return "\n".join([rule(), f"{Style.BOLD}{tagline}{Style.RESET}", rule()])


def welcome_screen() -> str:
    """Shown when `halide` is run with no subcommand — replaces argparse's bare "the following
    arguments are required: command" error with something that actually orients a non-programmer
    photographer, instead of just failing."""
    starts = [
        ("halide calibrate in_dir/", "pick neutral points, save a profile"),
        ("halide invert negative.tif positive.tif", "develop a single scan"),
        ("halide check in_dir/", "check a roll was scanned consistently"),
        ("halide batch in_dir/ out_dir/", "develop a whole roll"),
        ("halide batch in_dir/ --contact-sheet s.jpg", "preview settings as a contact sheet"),
    ]
    # One padding for the whole list, from its longest command, so the descriptions can't drift.
    pad = max(len(command) for command, _ in starts) + 3
    return "\n".join(
        [
            _banner("h a l i d e".center(_rule_width())),
            "develop scanned color negative film".center(_rule_width()),  # not bold, outside _banner
            rule(),
            "",
            "  New here? Start with:",
            *(f"    {command.ljust(pad)}{note}" for command, note in starts),
            "",
            "  Editing yourself? Develop flat, edit in darktable, then print:",
            "    halide invert negative.tif flat.tif --output flat",
            "    halide print flat_edited.tif print.tif",
            "",
            "  Run `halide --help` for the full command list",
        ]
    )


def help_banner() -> str:
    """A small themed banner printed before `halide --help`'s own (argparse-generated) output —
    deliberately printed directly rather than via a custom HelpFormatter: argparse's own
    add_subparsers() internally re-invokes the parser's formatter_class just to compute each
    subcommand's prog-prefix string (e.g. "halide invert"), completely unrelated to rendering
    user-facing help text — a HelpFormatter.format_help() override meant for the latter ends up
    hijacking the former too, corrupting every subcommand's usage line with this banner's own text.
    Printing it as a plain string before argparse ever runs sidesteps that internal reuse entirely."""
    plain_tagline = "halide · develop scanned color negative film"
    # The default rule (20 columns -> 39 visible chars) is narrower than this tagline (44 visible
    # chars) — centering against it would be a no-op (str.center() can't shrink below the string's
    # own length), so this banner uses a wider rule specifically sized to leave room either side.
    columns = 24
    width = _rule_width(columns)
    tagline = f"{Style.BOLD}halide{Style.RESET} · develop scanned color negative film"
    centered = tagline.center(width + len(Style.BOLD) + len(Style.RESET))
    assert len(plain_tagline) < width, "help_banner()'s rule must stay wider than its tagline"
    return f"{rule(columns)}\n{centered}\n{rule(columns)}\n"
