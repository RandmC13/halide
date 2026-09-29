"""core/ is pure: no file I/O, no printing, nothing imported from io/ at run time (CLAUDE.md)."""

import re
import tokenize
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "src" / "halide" / "core"

FORBIDDEN = [
    (r"\bopen\(", "open("),
    (r"\.read_text\(|\.read_bytes\(|\.write_text\(|\.write_bytes\(", "Path read/write"),
    (r"\bprint\(", "print("),
    (r"^\s*from halide\.io\b.*\bload_", "a load_ function imported from io"),
]


def _code_lines(path: Path):
    """Lines of code only: comments and docstrings can talk about these things freely. Found with
    tokenize, so a "#" inside a string can't cut a line short."""
    lines = path.read_text().splitlines()
    blank = []  # (row, start col, end col) spans to blank out
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            is_docstring_like = token.type == tokenize.STRING and token.string.lstrip("rbfRBF").startswith(('"""', "'''"))
            if token.type == tokenize.COMMENT or is_docstring_like:
                (r1, c1), (r2, c2) = token.start, token.end
                for row in range(r1, r2 + 1):
                    blank.append((row, c1 if row == r1 else 0, c2 if row == r2 else len(lines[row - 1])))
    for row, start, end in blank:
        text = lines[row - 1]
        lines[row - 1] = text[:start] + " " * (end - start) + text[end:]
    yield from enumerate(lines, start=1)


def test_core_does_no_file_io_or_printing():
    problems = []
    for path in sorted(CORE.glob("*.py")):
        for number, line in _code_lines(path):
            for pattern, what in FORBIDDEN:
                if re.search(pattern, line):
                    problems.append(f"{path.name}:{number}: {what}")
    assert not problems, problems


def test_core_imports_io_only_for_types():
    for path in sorted(CORE.glob("*.py")):
        for number, line in _code_lines(path):
            if re.match(r"^from halide\.io\b", line):  # top level (unindented) only
                raise AssertionError(f"{path.name}:{number}: runtime import from halide.io")
