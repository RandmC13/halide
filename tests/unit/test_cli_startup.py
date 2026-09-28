"""Starting the CLI must not import the image pipeline's heavy dependencies. colour-science alone
(it drags in scipy) was ~0.95 s of a ~1.3 s `halide --help`; numpy, tifffile and Pillow another
~0.15 s. They're imported inside the functions that use them, and one stray module-level import
anywhere the CLI reaches would silently bring the slow start back — hence this test."""

import subprocess
import sys

HEAVY = ("numpy", "tifffile", "PIL", "colour", "scipy", "PySide6", "cupy")


def test_building_the_cli_parser_imports_no_heavy_dependencies():
    # A fresh interpreter: this test process has long since imported all of them.
    code = (
        "import sys\n"
        "from halide.cli.main import build_parser\n"
        "build_parser()\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ""
