import pytest

from halide.cli._naming import TemplateError, parse_template, safe_field


def _r(t, **kw):
    base = dict(stem="IMG_0138", roll="Roll1", n=7, stock="Portra 400")
    base.update(kw)
    return parse_template(t).render(**base)


def test_fields_and_padding():
    assert _r("{roll}_{n:02}") == "Roll1_07"
    assert _r("{n:03}-{stem}") == "007-IMG_0138"
    assert _r("{n}") == "7"
    assert _r("{stock} {roll}") == "Portra 400 Roll1"


def test_missing_stock_renders_empty():
    assert _r("{roll}{stock}_{n:02}", stock=None) == "Roll1_07"


def test_unsafe_characters_become_dashes():
    assert safe_field('a/b\\c:d*e?f"g<h>i|j\x01k') == "a-b-c-d-e-f-g-h-i-j-k"
    assert _r("{stock}_{n}", stock="Kodak/Portra:400") == "Kodak-Portra-400_7"


@pytest.mark.parametrize("bad", ["{date}", "{n:x}", "{n:2}", "{roll:>5}", "a/{n}", "a\\{n}", "", "{stock}", "{n", "n}"])
def test_bad_templates_are_refused(bad):
    with pytest.raises(TemplateError) as err:
        parse_template(bad).render(stem="s", roll="r", n=1, stock=None)
    assert "{stem}" in str(err.value) and "{n:02}" in str(err.value)


def test_literal_braces_are_not_supported_but_text_is_kept():
    assert _r("Holiday1_{n:02}") == "Holiday1_07"


@pytest.mark.parametrize("only_stock", ["{stock}", "  {stock} ", "{stock}{stock}"])
def test_a_stock_only_template_is_refused_before_rendering(only_stock):
    # a profile without a film stock would give every frame an empty name
    with pytest.raises(TemplateError, match="empty name"):
        parse_template(only_stock)


@pytest.mark.parametrize("hidden", [".{n}", " .{roll}_{n}"])
def test_a_name_starting_with_a_dot_is_refused(hidden):
    # later folder runs skip dotfiles, so these outputs would vanish from the roll
    with pytest.raises(TemplateError, match="hidden"):
        parse_template(hidden)


def test_a_field_value_that_starts_the_name_with_a_dot_is_refused():
    with pytest.raises(TemplateError, match="hidden"):
        _r("{stock}_{n}", stock=".Portra")
