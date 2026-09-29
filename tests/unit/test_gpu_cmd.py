"""`halide gpu` — find an NVIDIA card without CuPy installed, tell the user GPU support exists, and
install it on request (Task 6b, docs/plans/gpu-acceleration.md §3.7). CuPy plus NVIDIA's runtime
libraries is ~1 GB, so a plain `pip install halide` must never pull it in; this is the one place
that offers it, and only to someone who actually has a card.

Every scenario here is faked: `ctypes.CDLL`/`ctypes.WinDLL` for driver detection, `sys.modules` for
CuPy presence, and `subprocess.run`/`input` for the installer — so these tests pass on any machine,
including this sandbox (which has CuPy installed but no real NVIDIA driver: `import cupy` works,
but the first CUDA call raises `cudaErrorInsufficientDriver`, and `libcuda.so.1` doesn't exist for
`detect_nvidia_driver` either)."""

from __future__ import annotations

import ctypes
import importlib.machinery
import importlib.util
import re
import subprocess
import sys
import types

import numpy as np
import pytest

import halide.device as device_module
from halide.cli.main import build_parser, main
from halide.device import NvidiaDriver, cupy_package_for, detect_nvidia_driver, gpu_support_installed
from halide.io.tiff import write_tiff
from tests.unit.test_icc import LINEAR_TAGS, build_icc

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def _collapse(text: str) -> str:
    """Normalizes RunSheet's wrapped, hanging-indented rows to one line per logical row, so a
    long value (like the GPU hint) can be found by exact text regardless of where it wrapped."""
    return " ".join(text.split())


def _fake_installed_cupy() -> types.ModuleType:
    """A `sys.modules["cupy"]` stand-in real enough for `importlib.util.find_spec` (which needs a
    real `__spec__`, not None, or it raises `ValueError` instead of just answering "found")."""
    module = types.ModuleType("cupy")
    module.__spec__ = importlib.machinery.ModuleSpec("cupy", loader=None)
    return module


# ---------------------------------------------------------------------------
# detect_nvidia_driver (ctypes only, never raises)
# ---------------------------------------------------------------------------


def _fake_libcuda(*, name=b"NVIDIA GeForce RTX 3070", version=13040, init_ok=True, get_ok=True, name_ok=True):
    lib = types.SimpleNamespace()

    def cuDriverGetVersion(ptr):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = version
        return 0

    def cuInit(flags):
        return 0 if init_ok else 1

    def cuDeviceGet(ptr, ordinal):
        if not get_ok:
            return 1
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    def cuDeviceGetName(buf, size, dev):
        if not name_ok:
            return 1
        ctypes.memmove(buf, name + b"\x00", len(name) + 1)
        return 0

    lib.cuDriverGetVersion = cuDriverGetVersion
    lib.cuInit = cuInit
    lib.cuDeviceGet = cuDeviceGet
    lib.cuDeviceGetName = cuDeviceGetName
    return lib


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux", raising=False)


def test_detect_nvidia_driver_returns_none_when_library_is_missing(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: (_ for _ in ()).throw(OSError("not found")))
    assert detect_nvidia_driver() is None


def test_detect_nvidia_driver_reads_version_and_name(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _fake_libcuda())
    driver = detect_nvidia_driver()
    assert driver == NvidiaDriver(cuda_version=13040, device_name="NVIDIA GeForce RTX 3070")


def test_detect_nvidia_driver_keeps_version_when_name_lookup_fails(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _fake_libcuda(name_ok=False))
    driver = detect_nvidia_driver()
    assert driver == NvidiaDriver(cuda_version=13040, device_name=None)


def test_detect_nvidia_driver_keeps_version_when_cuinit_fails(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _fake_libcuda(init_ok=False))
    driver = detect_nvidia_driver()
    assert driver == NvidiaDriver(cuda_version=13040, device_name=None)


def test_detect_nvidia_driver_never_raises_on_error_codes(monkeypatch):
    """Even a library that reports failure for every call (bogus CUresults) must come back None or
    a partial result, never an exception."""

    def boom_get_version(ptr):
        return 999  # not CUDA_SUCCESS, and no value written

    lib = types.SimpleNamespace(cuDriverGetVersion=boom_get_version)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    assert detect_nvidia_driver() is None


def test_detect_nvidia_driver_never_raises_on_unexpected_exception(monkeypatch):
    def explodes(ptr):
        raise RuntimeError("garbage driver")

    lib = types.SimpleNamespace(cuDriverGetVersion=explodes)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    assert detect_nvidia_driver() is None


def test_detect_nvidia_driver_on_unsupported_platform_is_none(monkeypatch):
    monkeypatch.setattr(sys, "platform", "some-other-os", raising=False)
    assert detect_nvidia_driver() is None


# ---------------------------------------------------------------------------
# cupy_package_for / gpu_support_installed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cuda_version, expected",
    [(13040, "cupy-cuda13x[ctk]"), (12080, "cupy-cuda12x[ctk]"), (11080, None), (13000, "cupy-cuda13x[ctk]"), (12000, "cupy-cuda12x[ctk]")],
)
def test_cupy_package_for(cuda_version, expected):
    assert cupy_package_for(NvidiaDriver(cuda_version=cuda_version, device_name=None)) == expected


def test_gpu_support_installed_false_when_cupy_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert gpu_support_installed() is False


def test_gpu_support_installed_true_when_cupy_present(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_installed_cupy())
    assert gpu_support_installed() is True


def test_gpu_support_installed_does_not_import_cupy(monkeypatch):
    # A real installed cupy in this sandbox's venv must not actually get imported just to answer
    # "is it installed" -- that would cost the ~0.2s `import cupy` takes on every command.
    code = "import sys, halide.device; halide.device.gpu_support_installed(); print('cupy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# `halide gpu` status: the four machine cases
# ---------------------------------------------------------------------------


def test_gpu_command_is_registered():
    args = build_parser().parse_args(["gpu"])
    assert args.command == "gpu"


def test_status_no_driver_found(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(device_module, "detect_nvidia_driver", lambda: None)
    assert main(["gpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "No NVIDIA GPU found" in out


def test_status_driver_found_no_cupy(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=13040, device_name="NVIDIA GeForce RTX 3070"),
    )
    assert main(["gpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "NVIDIA GeForce RTX 3070" in out
    assert "GPU support is not installed — add it with: halide gpu --install (about 1 GB download)" in out


def test_status_driver_too_old_no_cupy(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=11020, device_name="NVIDIA GeForce GT 710"),
    )
    assert main(["gpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "too old" in out
    assert "halide gpu --install" not in out


def test_status_cupy_installed_and_working(monkeypatch, capsys):
    from halide.device import ComputeDevice

    monkeypatch.setitem(sys.modules, "cupy", _fake_installed_cupy())
    monkeypatch.setattr(
        device_module, "resolve_device",
        lambda requested, isolated=False: ComputeDevice(kind="gpu", name="NVIDIA GeForce RTX 3070", memory_free=1, memory_total=2),
    )
    assert main(["gpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "NVIDIA GeForce RTX 3070" in out
    assert "in use by default (--device auto)" in out


def test_status_cupy_installed_but_device_fails(monkeypatch, capsys):
    """The sandbox's own real case."""
    from halide.device import ComputeDevice

    monkeypatch.setitem(sys.modules, "cupy", _fake_installed_cupy())
    monkeypatch.setattr(
        device_module, "resolve_device",
        lambda requested, isolated=False: ComputeDevice(kind="cpu", fallback_reason="cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version"),
    )
    assert main(["gpu"]) == 0
    out = _strip(capsys.readouterr().out)
    assert "cudaErrorInsufficientDriver" in out


def test_status_real_sandbox_run_does_not_crash(capsys):
    """No monkeypatching: exercises this sandbox's real state (CuPy installed, no driver)."""
    code = main(["gpu"])
    out = _strip(capsys.readouterr().out)
    assert code == 0
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# `halide gpu --install`
# ---------------------------------------------------------------------------


@pytest.fixture
def old_driver_absent_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=13040, device_name="NVIDIA GeForce RTX 3070"),
    )
    # The installer asks whether pip exists; a venv made --without-pip says no. Fake "yes" so these
    # tests pass on any host (the no-pip test overrides this); everything else is looked up for real.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: object() if name == "pip" else real_find_spec(name, *a, **k),
    )


def test_install_declines_runs_nothing(monkeypatch, capsys, old_driver_absent_cupy):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["gpu", "--install"]) == 0
    assert calls == []
    out = _strip(capsys.readouterr().out)
    assert "Not installed" in out


def test_install_prompt_eof_counts_as_no(monkeypatch, capsys, old_driver_absent_cupy):
    """Ctrl-D (or a dropped terminal) at "Install now?" is a clean "not installed", not a traceback."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["gpu", "--install"]) == 0
    assert calls == []
    assert "Not installed" in _strip(capsys.readouterr().out)


def test_install_accepts_runs_pip_install_exactly(monkeypatch, capsys, old_driver_absent_cupy):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["gpu", "--install"]) == 0
    assert calls[0] == [sys.executable, "-m", "pip", "install", "cupy-cuda13x[ctk]"]
    out = _strip(capsys.readouterr().out)
    assert "GPU support for halide: NVIDIA GeForce RTX 3070 (driver supports CUDA 13.4)" in out
    assert f"{sys.executable} -m pip uninstall cupy-cuda13x" in out
    assert "about 1 GB" in out


def test_install_yes_flag_skips_prompt(monkeypatch, old_driver_absent_cupy):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fail_input(prompt=""):
        raise AssertionError("must not prompt when --yes is given")

    monkeypatch.setattr("builtins.input", fail_input)
    assert main(["gpu", "--install", "--yes"]) == 0
    assert calls[0] == [sys.executable, "-m", "pip", "install", "cupy-cuda13x[ctk]"]


def test_install_no_pip_prints_alternatives_and_runs_nothing(monkeypatch, capsys, old_driver_absent_cupy):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name in ("pip", "cupy") else object())
    assert main(["gpu", "--install", "--yes"]) == 1
    assert calls == []
    out = _strip(capsys.readouterr().out)
    # Double quotes: they work in POSIX shells, PowerShell and cmd.exe (which keeps single quotes
    # literally, so pip would be handed 'cupy-cuda13x[ctk]' with the quotes as part of the name).
    assert f'{sys.executable} -m pip install "cupy-cuda13x[ctk]"' in out
    assert 'pipx inject halide "cupy-cuda13x[ctk]"' in out
    assert 'uv tool install --with "cupy-cuda13x[ctk]"' in out
    assert "'" not in out.split("Run whichever")[1]


def test_install_non_interactive_without_yes_refuses(monkeypatch, capsys, old_driver_absent_cupy):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert main(["gpu", "--install"]) == 1
    assert calls == []
    out = _strip(capsys.readouterr().out)
    assert "--yes" in out


def test_install_driver_too_old_explains_and_exits_1(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=11020, device_name="NVIDIA GeForce GT 710"),
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    assert main(["gpu", "--install", "--yes"]) == 1
    assert calls == []
    out = _strip(capsys.readouterr().out)
    assert "too old" in out


def test_install_no_driver_found_runs_nothing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(device_module, "detect_nvidia_driver", lambda: None)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    assert main(["gpu", "--install", "--yes"]) == 1
    assert calls == []
    out = _strip(capsys.readouterr().out)
    assert "No NVIDIA GPU found" in out


def test_install_already_installed_does_nothing(monkeypatch, capsys):
    from halide.device import ComputeDevice

    monkeypatch.setitem(sys.modules, "cupy", _fake_installed_cupy())
    monkeypatch.setattr(
        device_module, "resolve_device", lambda requested, isolated=False: ComputeDevice(kind="gpu", name="NVIDIA GeForce RTX 3070")
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)) or None)
    assert main(["gpu", "--install"]) == 0
    assert calls == []
    out = _strip(capsys.readouterr().out)
    assert "already installed" in out


def test_install_pip_failure_reports_and_exits_1(monkeypatch, capsys, old_driver_absent_cupy):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=1, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(["gpu", "--install", "--yes"]) == 1
    out = _strip(capsys.readouterr().out)
    assert "failed" in out.lower()


def test_install_success_rechecks_in_a_fresh_subprocess(monkeypatch, capsys, old_driver_absent_cupy):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "install" in argv:
            return subprocess.CompletedProcess(argv, returncode=0, stdout="")
        return subprocess.CompletedProcess(argv, returncode=0, stdout="NVIDIA GeForce RTX 3070 — GPU support is installed and working.\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(["gpu", "--install", "--yes"]) == 0
    assert len(calls) == 2
    assert calls[0] == [sys.executable, "-m", "pip", "install", "cupy-cuda13x[ctk]"]
    assert calls[1][:3] == [sys.executable, "-m", "halide.cli.main"]
    assert "gpu" in calls[1]
    out = _strip(capsys.readouterr().out)
    assert "GPU support is installed and working" in out


def test_install_recheck_failure_shows_its_error_and_exits_1(monkeypatch, capsys, old_driver_absent_cupy):
    # pip succeeded, but the fresh `halide gpu` check itself failed (e.g. a broken CuPy import that
    # crashes it): say so plainly, show what it printed on stderr, and don't report success.
    def fake_run(argv, **kwargs):
        if "install" in argv:
            return subprocess.CompletedProcess(argv, returncode=0, stdout="")
        return subprocess.CompletedProcess(argv, returncode=1, stdout="",
                                           stderr="ImportError: libcudart.so.13: cannot open shared object file\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(["gpu", "--install", "--yes"]) == 1
    captured = capsys.readouterr()
    everything = _strip(captured.out + captured.err)
    assert "libcudart.so.13" in everything
    assert "installed, but" in everything.lower() and "halide gpu" in everything


# ---------------------------------------------------------------------------
# run sheet Compute row hint (cli/_run_sheet.py::compute_row)
# ---------------------------------------------------------------------------

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)


def _write_negative(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))


@pytest.fixture
def roll_dir(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(2):
        _write_negative(in_dir / f"frame_{i:02d}.tiff", seed=i)
    return in_dir


def test_run_sheet_shows_hint_when_driver_found_and_cupy_absent(roll_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=13040, device_name="NVIDIA GeForce RTX 3070"),
    )
    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--workers", "1"]
    assert main(args) == 0
    out = _strip(capsys.readouterr().out)
    sheet = _collapse(out[: out.index("Developing")])
    assert "CPU — NVIDIA GeForce RTX 3070 found; add GPU support with: halide gpu --install" in sheet


def test_run_sheet_says_just_cpu_when_no_driver(roll_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(device_module, "detect_nvidia_driver", lambda: None)
    out_dir = tmp_path / "out"
    args = ["batch", str(roll_dir), str(out_dir), "--auto-density-roll", "--workers", "1"]
    assert main(args) == 0
    out = _strip(capsys.readouterr().out)
    sheet = out[: out.index("Developing")]
    assert "add GPU support" not in sheet
    assert re.search(r"Compute\s+CPU\b", sheet)


# ---------------------------------------------------------------------------
# invert's once-per-machine hint
# ---------------------------------------------------------------------------


@pytest.fixture
def negative_tiff(tmp_path):
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[4:12, 4:12] = HIGHLIGHT_RGB
    path = tmp_path / "negative.tiff"
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return path


MANUAL = ["--rm", "2.28", "--bm", "1.47", "--rs", "1.32", "--bs", "0.78"]


@pytest.fixture
def gpu_hint_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("HALIDE_NO_GPU_HINT", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        device_module, "detect_nvidia_driver",
        lambda: NvidiaDriver(cuda_version=13040, device_name="NVIDIA GeForce RTX 3070"),
    )
    return tmp_path


def test_invert_shows_gpu_hint_once(negative_tiff, tmp_path, gpu_hint_home, capsys):
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL]) == 0
    out = _strip(capsys.readouterr().out)
    assert "NVIDIA GeForce RTX 3070 found; add GPU support with: halide gpu --install" in out

    output2 = tmp_path / "positive2.tiff"
    assert main(["invert", str(negative_tiff), str(output2), *MANUAL]) == 0
    out2 = _strip(capsys.readouterr().out)
    assert "add GPU support" not in out2


def test_invert_gpu_hint_respects_env_opt_out(negative_tiff, tmp_path, gpu_hint_home, monkeypatch, capsys):
    monkeypatch.setenv("HALIDE_NO_GPU_HINT", "1")
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL]) == 0
    out = _strip(capsys.readouterr().out)
    assert "add GPU support" not in out


def test_invert_no_hint_when_no_driver(negative_tiff, tmp_path, gpu_hint_home, monkeypatch, capsys):
    monkeypatch.setattr(device_module, "detect_nvidia_driver", lambda: None)
    output = tmp_path / "positive.tiff"
    assert main(["invert", str(negative_tiff), str(output), *MANUAL]) == 0
    out = _strip(capsys.readouterr().out)
    assert "add GPU support" not in out


def test_invert_no_card_never_stamps_and_reprobes_every_run(negative_tiff, tmp_path, gpu_hint_home, monkeypatch):
    """Review finding (fix round 1): the stamp is only written once a hint is actually shown, so a
    machine with no NVIDIA card at all never writes it, and `detect_nvidia_driver` (the ctypes
    probe) is called again on every single invert, indefinitely — deliberate, so a card added to
    the machine later still gets the hint once. Measured negligible (~0.11 ms median per call in a
    fresh process, this sandbox, no libcuda present) rather than papered over with an unconditional
    stamp; see ruling R8 in docs/plans/gpu-acceleration.md."""
    calls = []

    def fake_detect():
        calls.append(1)
        return None

    monkeypatch.setattr(device_module, "detect_nvidia_driver", fake_detect)
    stamp = tmp_path / "data" / "halide" / "gpu-hint.shown"

    output1 = tmp_path / "positive1.tiff"
    assert main(["invert", str(negative_tiff), str(output1), *MANUAL]) == 0
    assert calls == [1]
    assert not stamp.exists()

    output2 = tmp_path / "positive2.tiff"
    assert main(["invert", str(negative_tiff), str(output2), *MANUAL]) == 0
    assert calls == [1, 1]  # probed again — no stamp means no early exit
    assert not stamp.exists()


# ---------------------------------------------------------------------------
# `--device gpu` without CuPy names the install command (already covered by
# test_device.py/test_device_cli.py; re-checked here as part of this task's review focus)
# ---------------------------------------------------------------------------


def test_device_gpu_without_cupy_names_install(negative_tiff, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    output = tmp_path / "positive.tiff"
    with pytest.raises(SystemExit, match="halide gpu --install"):
        main(["invert", str(negative_tiff), str(output), *MANUAL, "--device", "gpu"])


def test_install_advice_names_the_nvidia_wheels_and_the_pip_cache(monkeypatch, capsys, old_driver_absent_cupy):
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, stdout=""))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    main(["gpu", "--install"])
    out = _strip(capsys.readouterr().out)
    assert "nvidia-" in out and "pip cache purge" in out


def test_status_never_prints_none_for_a_missing_reason(monkeypatch, capsys):
    from halide.device import ComputeDevice

    monkeypatch.setitem(sys.modules, "cupy", _fake_installed_cupy())
    monkeypatch.setattr(device_module, "resolve_device",
                        lambda requested, isolated=False: ComputeDevice(kind="cpu", fallback_reason=None))
    main(["gpu"])
    assert "None" not in _strip(capsys.readouterr().out)
