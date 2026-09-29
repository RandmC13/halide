"""core/ is pure: no file I/O, no printing, nothing imported from io/ at run time (CLAUDE.md)."""

import re
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "src" / "halide" / "core"

FORBIDDEN = [
    (r"\bopen\(", "open("),
    (r"\.read_text\(|\.read_bytes\(|\.write_text\(|\.write_bytes\(", "Path read/write"),
    (r"\bprint\(", "print("),
    (r"^\s*from halide\.io\b.*\bload_", "a load_ function imported from io"),
]


def _code_lines(path: Path):
    """Lines of code only: comments and docstrings can talk about these things freely."""
    text = re.sub(r'"""(.*?)"""', lambda m: "\n" * m.group(0).count("\n"), path.read_text(), flags=re.S)
    for number, line in enumerate(text.splitlines(), start=1):
        yield number, line.split("#", 1)[0]


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
