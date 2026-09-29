"""Unit tests for the pure/non-TTY-dependent logic in halide.cli.console: formatting helpers and
the run_guarded() exception boundary. The interactive (confirm/menu/spinner) TTY branches aren't
exercised here — sys.stdin/stdout under pytest are never a real terminal, so these tests are
already implicitly covering the non-interactive fallback path for confirm/menu, which is exactly
the behavior that must never block a script waiting on input. The overwrite policy itself
(`--overwrite`/`--skip-existing`/refuse) lives in cli/_output_policy.py and is tested in
tests/unit/test_output_policy.py — it replaced console.confirm_overwrite, whose non-interactive
"always proceed" was exactly the silent-overwrite bug D-1 fixed."""

from __future__ import annotations

import pytest

from halide.cli import console


def test_human_time_formats_seconds_minutes_hours():
    assert console.human_time(4.2) == "4.2s"
    assert console.human_time(65) == "1m05s"
    assert console.human_time(3725) == "1h02m"


def test_human_bytes_formats_units():
    assert console.human_bytes(500) == "500 B"
    assert console.human_bytes(2048) == "2.0 KB"
    assert console.human_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_confirm_non_tty_returns_default_without_blocking():
    assert console.confirm("proceed?", default=False) is False
    assert console.confirm("proceed?", default=True) is True


def test_menu_non_tty_returns_none():
    assert console.menu("pick one", [("a", "Option A"), ("b", "Option B")]) is None


def test_run_guarded_returns_main_funcs_exit_code():
    assert console.run_guarded(lambda argv: 0) == 0
    assert console.run_guarded(lambda argv: 1) == 1


def test_run_guarded_converts_string_system_exit_to_clean_exit_1(capsys):
    def main_func(argv):
        raise SystemExit("something went wrong")

    code = console.run_guarded(main_func)
    assert code == 1
    assert "something went wrong" in capsys.readouterr().err


def test_run_guarded_preserves_argparse_style_int_system_exit():
    def main_func(argv):
        raise SystemExit(2)

    assert console.run_guarded(main_func) == 2


def test_run_guarded_keyboard_interrupt_exits_130(capsys):
    def main_func(argv):
        raise KeyboardInterrupt

    assert console.run_guarded(main_func) == 130
    assert "Cancelled" in capsys.readouterr().err


def test_run_guarded_unexpected_error_is_short_by_default(capsys):
    def main_func(argv):
        raise ValueError("boom")

    code = console.run_guarded(main_func, argv=[])
    assert code == 1
    err = capsys.readouterr().err
    assert "boom" in err
    assert "--debug" in err


def test_run_guarded_debug_flag_reraises_full_exception():
    def main_func(argv):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        console.run_guarded(main_func, argv=["--debug"])


def test_slide_strings_endpoints_and_length():
    frames = console.slide_strings("AAAAA", "BBBBB", steps=6)
    assert len(frames) == 6
    assert frames[0] == "AAAAA"
    assert frames[-1] == "BBBBB"
    assert all(len(f) == 5 for f in frames)


def test_slide_strings_rejects_unequal_length():
    with pytest.raises(ValueError):
        console.slide_strings("AA", "BBB")


def test_mini_spinners_all_have_multiple_frames():
    assert len(console.MINI_SPINNERS) >= 4
    for glyphs in console.MINI_SPINNERS.values():
        assert len(glyphs) >= 2


def test_random_mini_spinner_frames_returns_styled_glyphs():
    frames = console.random_mini_spinner_frames()
    assert len(frames) >= 2
    assert all(isinstance(f, str) and f for f in frames)


def test_animation_degrades_immediately_when_terminal_too_small(monkeypatch):
    monkeypatch.setenv("COLUMNS", "10")
    monkeypatch.setenv("LINES", "5")
    art = ["line one\nline two", "line one\nline two"]
    anim = console.animation(art, label="x", min_size=(200, 50))
    assert anim.frames != art
    assert len(anim.frames) == 4


def test_animation_uses_given_frames_when_terminal_is_big_enough(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("LINES", "50")
    art = ["a", "b"]
    anim = console.animation(art, label="x", min_size=(10, 5))
    assert anim.frames == art


def test_themed_animation_returns_an_animation_instance():
    assert isinstance(console.themed_animation(["a"], "label", min_width=1, min_height=1), console.animation)


def test_framed_sizes_rules_to_short_text_and_to_the_terminal_for_long_output(monkeypatch):
    # The user's convention: one or two lines -> rules as wide as the text (neatly wrapping it);
    # longer output -> rules spanning the whole terminal line (a short rule looks cut off).
    import os

    from halide.cli import console

    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((101, 30)))
    short = console.framed(["halide · calibration picker"]).split("\n")  # 27 visible characters
    assert console.visible_width(short[0]) == 27
    long = console.framed(["a", "b", "c"]).split("\n")
    assert console.visible_width(long[0]) == 101
    assert long[0] == long[-1] and short[0] == short[-1]


def _plain(text: str) -> list[str]:
    from halide.cli import console

    return [console._ANSI_ESCAPE.sub("", line) for line in text.splitlines()]


def test_run_sheet_aligns_rows_stacks_labels_and_wraps_under_the_value_column(monkeypatch, capsys):
    import os

    from halide.cli import console

    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((60, 30)))
    with console.RunSheet() as sheet:
        sheet.row("Roll", "Roll16 · 37 frames → out")
        sheet.warn("Scans", "exported with 14 different raw white balances — shifts colour frame to frame")
        sheet.note("Scans", "details: halide check Roll16")
        sheet.row("Workers", "8")
    lines = _plain(capsys.readouterr().out)

    assert lines[0] == lines[-1] == _plain(console.full_width_rule())[0]  # full-width sprocket rules
    body = lines[1:-1]
    value_column = 2 + console.RunSheet.LABEL_WIDTH
    assert body[0].startswith("  Roll ") and body[0][value_column:] == "Roll16 · 37 frames → out"
    assert body[1].startswith("  Scans ") and body[1][value_column:].startswith("⚠ exported")
    # The warning wraps under its own text (past the icon), never back under the labels...
    assert body[2][: value_column + 2].strip() == "" and body[2][value_column + 2] != " "
    # ...and a second row with the same label stacks under it without repeating the label.
    assert body[3].strip() == "details: halide check Roll16" and body[3].startswith(" " * value_column)
    assert body[4].startswith("  Workers ")
    assert all(console.visible_width(line) < 60 for line in body)


def test_run_sheet_quiet_prints_only_warnings_and_no_rules(capsys):
    from halide.cli import console

    with console.RunSheet(quiet=True) as sheet:
        sheet.row("Roll", "Roll16")
        sheet.ok("Scans", "scanned consistently")
        with sheet.working("Calibration", "estimating…"):
            pass
        sheet.warn("Workers", "--workers 40 may exceed available memory")
    assert _plain(capsys.readouterr().out) == ["⚠ Warning: --workers 40 may exceed available memory"]


def test_run_sheet_prints_nothing_when_nothing_was_added(capsys):
    from halide.cli import console

    with console.RunSheet():
        pass
    assert capsys.readouterr().out == ""


def test_run_sheet_closes_its_frame_when_an_error_interrupts_it(capsys):
    from halide.cli import console

    with pytest.raises(SystemExit):
        with console.RunSheet() as sheet:
            sheet.row("Roll", "Roll16")
            raise SystemExit("no EXIF")
    lines = _plain(capsys.readouterr().out)
    assert len(lines) == 3 and lines[0] == lines[-1]


def test_run_sheet_follows_the_rule_sizing_convention(monkeypatch, capsys):
    # One or two lines: rules as wide as the text. More (or once a spinner has started it
    # printing): rules across the whole terminal — see framed().
    import os

    from halide.cli import console

    monkeypatch.setattr(console.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((100, 30)))
    with console.RunSheet() as sheet:
        sheet.row("Roll", "pos · 1 frame → out")
        sheet.row("Workers", "1")
    short = _plain(capsys.readouterr().out)
    text_width = max(map(console.visible_width, short[1:3]))
    assert len(short) == 4 and console.visible_width(short[0]) in (text_width, text_width + 1)

    with console.RunSheet() as sheet:
        sheet.row("Roll", "Roll16")
        with sheet.working("Calibration", "estimating…"):
            pass
        sheet.row("Calibration", "auto")
    streamed = _plain(capsys.readouterr().out)
    assert streamed[0] == streamed[-1] == _plain(console.full_width_rule())[0]
    assert [line.split()[0] for line in streamed[1:-1]] == ["Roll", "Calibration"]


def _use_color(monkeypatch, *, tty, env=()):
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env:
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(console.sys.stdout, "isatty", lambda: tty, raising=False)
    return console.use_color()


def test_use_color_follows_no_color_term_dumb_pipes_and_force_color(monkeypatch):
    assert _use_color(monkeypatch, tty=True) is True
    assert _use_color(monkeypatch, tty=False) is False
    assert _use_color(monkeypatch, tty=True, env=[("NO_COLOR", "1")]) is False
    assert _use_color(monkeypatch, tty=True, env=[("NO_COLOR", "")]) is False  # any value, even empty
    assert _use_color(monkeypatch, tty=True, env=[("TERM", "dumb")]) is False
    assert _use_color(monkeypatch, tty=False, env=[("FORCE_COLOR", "1")]) is True
    assert _use_color(monkeypatch, tty=True, env=[("NO_COLOR", "1"), ("FORCE_COLOR", "1")]) is True


def test_no_colour_output_has_no_escape_codes_from_any_helper(monkeypatch):
    _use_color(monkeypatch, tty=True, env=[("NO_COLOR", "1")])
    pieces = [
        console.success("ok"), console.error("bad"), console.warning("hmm"), console.dim("faint"),
        console.rule(), console.framed(["a", "b"]), console.help_banner(), console.welcome_screen(),
        console.source_color("auto"), "".join(console.tank_frames()), "".join(console.random_mini_spinner_frames()),
        console.cancelled(),
    ]
    sheet_lines = console.RunSheet()._lines("Label", "text", icon="x", icon_style=console.Style.RED)
    assert not any("\033" in piece for piece in pieces + sheet_lines)


def test_piped_output_has_no_escape_codes(monkeypatch, capsys):
    _use_color(monkeypatch, tty=False)
    with console.RunSheet() as sheet:
        sheet.warn("Scans", "careful")
        sheet.ok("Roll", "fine")
    assert "\033" not in capsys.readouterr().out
    assert "\033" not in console.success("x")


def test_colour_is_still_there_in_a_terminal(monkeypatch):
    _use_color(monkeypatch, tty=True)
    assert "\033[92m" in console.success("ok")


def test_plural_and_finish_and_cancel_wording():
    assert console.plural(1, "frame") == "1 frame" and console.plural(0, "frame") == "0 frames"
    assert console.finish_message("batch", 1, 1, 0, 3.0) == "Developed 1 frame in 3.0s"
    assert console.finish_message("export", 12, 12, 0, 8.0) == "Exported 12 files in 8.0s"
    assert console.finish_message("batch", 36, 37, 1, 42.0) == "Developed 36 of 37 frames (1 failed) in 42.0s"
    assert console.cancelled_message("batch", 2, 6) == "Cancelled - 2 of 6 frames developed; nothing half-written"


def test_run_guarded_cancel_uses_the_one_wording(capsys):
    def main(_argv):
        raise KeyboardInterrupt

    assert console.run_guarded(main, []) == 130
    assert "✗ Cancelled - nothing half-written" in capsys.readouterr().err


def test_run_cli_makes_stdout_line_buffered(monkeypatch):
    from halide.cli import main as cli_main

    calls = []

    class Out:
        def reconfigure(self, **kw):
            calls.append(kw)

        def isatty(self):
            return False

        def write(self, _t):
            return 0

        def flush(self):
            pass

    monkeypatch.setattr(cli_main.sys, "stdout", Out())
    monkeypatch.setattr(cli_main, "main", lambda argv=None: 0)
    monkeypatch.setenv("HALIDE_NO_COMPLETION", "1")
    assert cli_main.run_cli([]) == 0
    assert calls == [{"line_buffering": True}]


def test_verbs_name_what_each_command_does():
    """R-096: `export` says "Exporting" (it was "Printing" before a real print step existed)."""
    assert console.VERB["export"] == "Exporting" and console.VERB_PAST["export"] == "Exported"
    assert console.VERB["print"] == "Printing" and console.VERB_PAST["print"] == "Printed"
    assert console.VERB["invert"] == console.VERB["batch"] == "Developing"
