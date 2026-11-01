"""Comments and docstrings in src/ describe the code, not the planning behind it."""

import ast
import io
import re
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# A review ID looks like 2.4-9: one digit, a dot, then two small integers, and not part of a
# longer number or range (0.86-1.00, 1.3-2.6 GiB, 0.8-0.9).
PATTERNS = [
    re.compile(r"(?<![\d.])\d\.\d{1,2}-\d{1,2}(?![\d.]|\s*(?:GiB|MiB|D\b))(?<!\b0\.\d-\d)"),
    re.compile(r"(?<![\w-])(?:F\d{1,3}|R-\d{1,3}|D[12])(?![\w.-])(?!\s*[0-9]*\.)"),
    re.compile(r"\bTask [A-Z]?\d"),
    re.compile(r"\bdocs/"),
    re.compile(r"CLAUDE\.md"),
    re.compile(r"\bround \d", re.IGNORECASE),
]
NOQA = re.compile(r"#\s*noqa:\s*F\d+")


def _docstring_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                lines.add(body[0].value.lineno)
    return lines


def _hits():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        doc_starts = _docstring_lines(ast.parse(text))
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                body = NOQA.sub("", tok.string)
            elif tok.type == tokenize.STRING and tok.start[0] in doc_starts:
                body = tok.string
            else:
                continue
            for offset, line in enumerate(body.splitlines()):
                if any(p.search(line) for p in PATTERNS):
                    found.append(f"{path.relative_to(SRC.parent)}:{tok.start[0] + offset}: {line.strip()}")
    return found


def test_no_planning_labels_in_comments_or_docstrings():
    hits = _hits()
    assert not hits, f"{len(hits)} planning labels in src/:\n" + "\n".join(hits)
