"""`halide gpu` — tell the user whether they have an NVIDIA card halide could use, and add optional
GPU support (CuPy plus NVIDIA's CUDA runtime libraries, ~1 GB) on request. See CLAUDE.md's GPU
priority and docs/plans/gpu-acceleration.md §3.7: a plain `pip install halide` must never pull that
in, so this command is the one way halide offers it — and only when there's actually a card to
offer it for; nobody without an NVIDIA GPU should be pitched a ~1 GB download.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

# halide.device only reaches for ctypes/importlib.util at import time (cupy itself is imported
# lazily, deep inside resolve_device) — so importing it here, and even running `halide gpu` for
# status, stays fast and never needs cupy installed. See tests/unit/test_cli_startup.py, which
# building this command's parser must keep passing.
from halide import device as halide_device
from halide.cli import console


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install GPU support (about 1 GB download); asks first unless --yes is given",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="With --install, don't ask first (for scripts; still refuses with no usable NVIDIA "
        "driver)",
    )


def _cuda_version_str(cuda_version: int) -> str:
    # cuDriverGetVersion's own integer form: major*1000 + minor*10 (e.g. 13040 -> "13.4").
    return f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"


def run(args: argparse.Namespace) -> int:
    if args.install:
        return _install(args)
    return _status()


def _status() -> int:
    if halide_device.gpu_support_installed():
        device = halide_device.resolve_device("auto")
        if device.kind == "gpu":
            print(f"{device.name} — GPU support is installed and working")
            print("GPU is in use by default (--device auto)")
        else:
            # The carried-over minor from Task 3: fallback_reason can embed a raw CUDA error
            # string. Lead with a plain sentence, then show that as its own detail line.
            print("GPU support is installed, but the card isn't usable right now")
            print(f"    {device.fallback_reason or 'no reason was reported'}")
            print("halide will use the CPU until this is fixed (or use --device cpu to force it)")
        return 0

    driver = halide_device.detect_nvidia_driver()
    if driver is None:
        print("No NVIDIA GPU found. halide will run on the CPU")
        return 0

    name = driver.device_name or "An NVIDIA GPU"
    if halide_device.cupy_package_for(driver) is None:
        print(
            f"{name} found, but its driver is too old for GPU support (needs CUDA 12 or newer; "
            f"this one supports CUDA {_cuda_version_str(driver.cuda_version)}). Update the NVIDIA "
            "driver to use it"
        )
        return 0

    print(f"{name} found (driver supports CUDA {_cuda_version_str(driver.cuda_version)})")
    print("GPU support is not installed — add it with: halide gpu --install (about 1 GB download)")
    return 0


def _install(args: argparse.Namespace) -> int:
    if halide_device.gpu_support_installed():
        device = halide_device.resolve_device("auto")
        print("GPU support is already installed")
        if device.kind == "gpu":
            print(f"{device.name} — working, in use by default (--device auto)")
        else:
            print(f"It isn't usable right now: {device.fallback_reason or 'no reason was reported'}")
        return 0

    driver = halide_device.detect_nvidia_driver()
    if driver is None:
        print("No NVIDIA GPU found — nothing to install")
        return 1

    name = driver.device_name or "This card"
    package = halide_device.cupy_package_for(driver)
    if package is None:
        print(
            f"{name}'s driver is too old for GPU support (needs CUDA 12 or newer; this one "
            f"supports CUDA {_cuda_version_str(driver.cuda_version)}). Update the NVIDIA driver, "
            "then run `halide gpu --install` again"
        )
        return 1

    if importlib.util.find_spec("pip") is None:
        print(f"halide can't install GPU support automatically here — there's no pip in {sys.prefix}")
        print("Run whichever of these matches how halide itself was installed:")
        # Double quotes (the brackets need quoting in zsh): they work in POSIX shells, PowerShell
        # and cmd.exe alike — cmd.exe keeps single quotes as part of the argument.
        print(f'    {sys.executable} -m pip install "{package}"')
        print(f'    pipx inject halide "{package}"')
        print(f'    uv tool install --with "{package}" halide')
        return 1

    dist_name = package.split("[")[0]  # pip uninstall doesn't take the [ctk] extra
    cuda_str = _cuda_version_str(driver.cuda_version)
    print(f"GPU support for halide: {name} (driver supports CUDA {cuda_str})")
    print(
        f"This installs {package} into halide's Python environment ({sys.prefix}) — about 1 GB,\n"
        "mostly NVIDIA's CUDA libraries. To remove it later:\n"
        f"    {sys.executable} -m pip uninstall {dist_name}\n"
        "  then the NVIDIA libraries it brought with it (the packages named nvidia-... in\n"
        f"  `{sys.executable} -m pip list`), and free pip's downloaded copies with:\n"
        f"    {sys.executable} -m pip cache purge"
    )
    if not args.yes:
        if not console.is_interactive():
            print("Not installing — pass --yes to install without asking (no terminal to confirm in)")
            return 1
        if not console.confirm("Install now?", default=False):  # EOF (Ctrl-D) counts as "no"
            print("Not installed")
            return 0

    result = subprocess.run([sys.executable, "-m", "pip", "install", package])
    if result.returncode != 0:
        print("Install failed — see the pip output above")
        return 1

    # The current process may already have tried (and failed) `import cupy` earlier in this same
    # run, or loaded partial CUDA state some other way — re-check in a fresh interpreter rather
    # than trust anything this one has cached.
    check = subprocess.run(
        [sys.executable, "-m", "halide.cli.main", "gpu"], capture_output=True, text=True
    )
    print(check.stdout, end="")
    if check.returncode != 0:
        if check.stderr:
            print(check.stderr, end="" if check.stderr.endswith("\n") else "\n", file=sys.stderr)
        print("GPU support was installed, but checking it failed (the error is above). "
              "Run `halide gpu` to check again")
        return 1
    return 0
