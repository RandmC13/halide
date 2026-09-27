"""The device path on a real GPU against the CPU path (docs/plans/gpu-acceleration.md, §3.3, D2).

Run with `pytest -m gpu` on a machine with CuPy and a working NVIDIA GPU; everywhere else the whole
module skips at collection. tests/unit/test_device_pipeline.py pins the same plumbing bit-for-bit on
a fake device; this checks the real arithmetic, which is close but not bit-identical (different
float rounding in the GPU's matmuls, logs and powers), held to the D2 tolerance:
  - float32 pixels within 1e-5 relative, or 1e-7 absolute (for values near zero);
  - fitted exposure / contrast within 1e-5; the flat output's linear_scale within 1e-5 relative.

The real scans (IMG_0151/0156/0156-nowb/0158.tif in the repo root, gitignored) are used when
present — they are what the tolerance is really about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.gpu

pytest.importorskip("cupy")

from halide.device import resolve_device  # noqa: E402

_DEVICE = resolve_device("auto")
if _DEVICE.kind != "gpu":
    pytest.skip(
        f"no usable GPU ({_DEVICE.fallback_reason or 'CuPy found no device'})", allow_module_level=True
    )

import halide.banding  # noqa: E402
from halide.core.types import DensityProfile, ToneCurveParams  # noqa: E402
from halide.io.icc import output_profile_bytes  # noqa: E402
from halide.io.tiff import read_tiff, read_tiff_description, write_tiff  # noqa: E402
from halide.processing import (  # noqa: E402
    Stage,
    export_delivery_image,
    print_scan,
    process_scan,
    read_provenance,
)
from tests.unit.test_icc import LINEAR_TAGS, build_icc  # noqa: E402

PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))
REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SCANS = [p for p in (REPO_ROOT / f"IMG_{n}.tif" for n in ("0151", "0156", "0156-nowb", "0158")) if p.exists()]

PIXEL_RTOL, PIXEL_ATOL = 1e-5, 1e-7
TONE_TOL = 1e-5


def _write_scan(path, shape=(37, 53, 3), seed=7, icc=None):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.01, 0.3, size=shape).astype(np.float32)
    image.flat[:5] = 0.0
    image.flat[-4:] = -0.001
    write_tiff(path, image, icc_profile=icc if icc is not None else build_icc(LINEAR_TAGS))
    return path


def _assert_pixels_close(gpu_path, cpu_path):
    gpu, cpu = read_tiff(gpu_path).image, read_tiff(cpu_path).image
    assert gpu.dtype == cpu.dtype == np.float32 and gpu.shape == cpu.shape
    np.testing.assert_allclose(gpu, cpu, rtol=PIXEL_RTOL, atol=PIXEL_ATOL)


def _assert_tone_close(gpu, cpu):
    if cpu is None:
        assert gpu is None
        return
    assert gpu.mode == cpu.mode
    if cpu.mode == "linear":
        assert type(gpu.linear_scale) is type(cpu.linear_scale)  # a host scalar, as on the CPU
        assert gpu.linear_scale == pytest.approx(cpu.linear_scale, rel=TONE_TOL)
    else:
        assert gpu.exposure == pytest.approx(cpu.exposure, abs=TONE_TOL)
        assert gpu.contrast == pytest.approx(cpu.contrast, abs=TONE_TOL)


def _run_both(scan, tmp_path, stage, profile, tone, gain=1.0):
    cpu = process_scan(scan, tmp_path / "cpu.tif", stage, profile, tone, scan_gain=gain)
    warnings = []
    gpu = process_scan(scan, tmp_path / "gpu.tif", stage, profile, tone, scan_gain=gain,
                       device=_DEVICE, on_warning=warnings.append)
    assert warnings == [], warnings  # a fallback would make this a CPU-vs-CPU comparison
    _assert_pixels_close(tmp_path / "gpu.tif", tmp_path / "cpu.tif")
    _assert_tone_close(gpu, cpu)
    if cpu is not None:
        assert read_provenance(read_tiff_description(tmp_path / "gpu.tif"))["device"] == "gpu"


CASES = [
    (Stage.FULL, ToneCurveParams(), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(mode="linear"), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(exposure=0.3, contrast=0.8), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(contrast=0.7), PROFILE, 1.37),
    (Stage.FULL, ToneCurveParams(), None, 1.0),
    (Stage.FULL, ToneCurveParams(mode="linear"), None, 1.37),
    (Stage.INVERT_ONLY, ToneCurveParams(), None, 1.0),
    (Stage.DENSITY_ONLY, ToneCurveParams(), PROFILE, 1.37),
]
CASE_IDS = ["print", "flat", "pinned", "pinned-grade+gain", "auto", "auto-flat+gain", "invert-only", "density-only+gain"]


@pytest.mark.parametrize("stage, tone, profile, gain", CASES, ids=CASE_IDS)
def test_process_scan_on_gpu_matches_cpu(tmp_path, stage, tone, profile, gain):
    _run_both(_write_scan(tmp_path / "neg.tif"), tmp_path, stage, profile, tone, gain)


@pytest.mark.parametrize("stage, tone, profile, gain", CASES[:3], ids=CASE_IDS[:3])
def test_one_row_bands_on_gpu(tmp_path, monkeypatch, stage, tone, profile, gain):
    monkeypatch.setattr(halide.banding, "DEVICE_BAND_BYTES", 1)
    _run_both(_write_scan(tmp_path / "neg.tif"), tmp_path, stage, profile, tone, gain)


@pytest.mark.parametrize("shape", [(2, 2, 3), (1, 9, 3)], ids=["2x2", "1x9"])
@pytest.mark.parametrize("tone", [ToneCurveParams(), ToneCurveParams(mode="linear")], ids=["print", "flat"])
def test_tiny_image_on_gpu(tmp_path, shape, tone):
    _run_both(_write_scan(tmp_path / "neg.tif", shape=shape), tmp_path, Stage.FULL, PROFILE, tone)


def test_halides_own_profile_on_gpu(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif", icc=output_profile_bytes())
    _run_both(scan, tmp_path, Stage.FULL, PROFILE, ToneCurveParams())


@pytest.mark.parametrize("tone", [ToneCurveParams(), ToneCurveParams(exposure=0.2)], ids=["fitted", "pinned-exposure"])
def test_print_scan_on_gpu_matches_cpu(tmp_path, tone):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "flat.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    cpu, _ = print_scan(tmp_path / "flat.tif", tmp_path / "cpu.tif", tone)
    warnings = []
    gpu, _ = print_scan(tmp_path / "flat.tif", tmp_path / "gpu.tif", tone, device=_DEVICE, on_warning=warnings.append)
    assert warnings == [], warnings
    _assert_pixels_close(tmp_path / "gpu.tif", tmp_path / "cpu.tif")
    _assert_tone_close(gpu, cpu)


def test_out_of_memory_on_gpu_falls_back_to_cpu(tmp_path, monkeypatch):
    """A real CuPy OutOfMemoryError mid-frame, raised from the device path: the frame still comes
    out, on the CPU, identical to a plain CPU run, with the warning."""
    import cupy

    import halide.processing

    real = halide.processing.negative_to_positive

    def oom_on_device(band, *args, **kwargs):
        if isinstance(band, cupy.ndarray):
            raise cupy.cuda.memory.OutOfMemoryError(1 << 40, 0, 0)
        return real(band, *args, **kwargs)

    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    monkeypatch.setattr(halide.processing, "negative_to_positive", oom_on_device)
    warnings = []
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=_DEVICE,
                 on_warning=warnings.append)
    assert np.array_equal(read_tiff(tmp_path / "gpu.tif").image, read_tiff(tmp_path / "cpu.tif").image)
    assert len(warnings) == 1 and "out of GPU memory" in warnings[0]


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scan", REAL_SCANS, ids=[p.stem for p in REAL_SCANS])
@pytest.mark.parametrize(
    "tone, profile",
    [(ToneCurveParams(), PROFILE), (ToneCurveParams(mode="linear"), PROFILE), (ToneCurveParams(), None)],
    ids=["print", "flat", "auto"],
)
def test_real_scan_on_gpu_matches_cpu(tmp_path, scan, tone, profile):
    _run_both(scan, tmp_path, Stage.FULL, profile, tone)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scan", REAL_SCANS, ids=[p.stem for p in REAL_SCANS])
def test_real_scan_print_on_gpu_matches_cpu(tmp_path, scan):
    process_scan(scan, tmp_path / "flat.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    cpu, _ = print_scan(tmp_path / "flat.tif", tmp_path / "cpu.tif", ToneCurveParams())
    warnings = []
    gpu, _ = print_scan(tmp_path / "flat.tif", tmp_path / "gpu.tif", ToneCurveParams(), device=_DEVICE,
                        on_warning=warnings.append)
    assert warnings == [], warnings  # a silent fallback would make this a CPU-vs-CPU comparison
    _assert_pixels_close(tmp_path / "gpu.tif", tmp_path / "cpu.tif")
    _assert_tone_close(gpu, cpu)
    assert read_provenance(read_tiff_description(tmp_path / "gpu.tif"))["device"] == "gpu"


# ---------------------------------------------------------------------------
# Task 6: export_delivery_image's device path (io/raster.py's namespace-generic to_srgb_8bit).
# Held to its own, 8-bit tolerance — the brief's bar, checked explicitly rather than folded into
# "<= 1 code value" so a systematic offset can't hide inside that check.
# ---------------------------------------------------------------------------

EXPORT_MAX_CODE_DIFF = 1
EXPORT_MIN_FRACTION_IDENTICAL = 0.999


def _png_pixels(path):
    with Image.open(path) as img:
        return np.asarray(img).astype(np.int16)


def _assert_export_close(gpu_png, cpu_png, label):
    gpu, cpu = _png_pixels(gpu_png), _png_pixels(cpu_png)
    assert gpu.shape == cpu.shape
    diff = np.abs(gpu - cpu)
    max_diff = int(diff.max())
    fraction_identical = 1.0 - np.count_nonzero(diff) / diff.size
    print(f"{label}: max code-value diff {max_diff}, {fraction_identical:.6%} of pixels identical")
    assert max_diff <= EXPORT_MAX_CODE_DIFF, f"{label}: max diff {max_diff} > {EXPORT_MAX_CODE_DIFF}"
    assert fraction_identical >= EXPORT_MIN_FRACTION_IDENTICAL, (
        f"{label}: only {fraction_identical:.4%} of pixels identical "
        f"(need >= {EXPORT_MIN_FRACTION_IDENTICAL:.1%})"
    )


def test_export_delivery_image_on_gpu_matches_cpu(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "positive.tif", Stage.FULL, PROFILE, ToneCurveParams())
    warning_cpu = export_delivery_image(tmp_path / "positive.tif", tmp_path / "cpu.png")
    warnings = []
    warning_gpu = export_delivery_image(
        tmp_path / "positive.tif", tmp_path / "gpu.png", device=_DEVICE, on_warning=warnings.append
    )
    assert warnings == [] and warning_cpu is None and warning_gpu is None
    _assert_export_close(tmp_path / "gpu.png", tmp_path / "cpu.png", "synthetic")


def test_export_out_of_memory_on_gpu_falls_back_to_cpu(tmp_path, monkeypatch):
    """A real CuPy OutOfMemoryError mid-conversion, raised from the device path: the delivery image
    still comes out, on the CPU, pixel-identical to a plain CPU export, with the warning."""
    import cupy

    import halide.processing

    real = halide.processing.to_srgb_8bit

    def oom_on_device(band, *args, **kwargs):
        if isinstance(band, cupy.ndarray):
            raise cupy.cuda.memory.OutOfMemoryError(1 << 40, 0, 0)
        return real(band, *args, **kwargs)

    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "positive.tif", Stage.FULL, PROFILE, ToneCurveParams())
    export_delivery_image(tmp_path / "positive.tif", tmp_path / "cpu.png")
    monkeypatch.setattr(halide.processing, "to_srgb_8bit", oom_on_device)
    warnings = []
    export_delivery_image(
        tmp_path / "positive.tif", tmp_path / "gpu.png", device=_DEVICE, on_warning=warnings.append
    )
    assert np.array_equal(_png_pixels(tmp_path / "gpu.png"), _png_pixels(tmp_path / "cpu.png"))
    assert len(warnings) == 1 and "out of GPU memory" in warnings[0]


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scan", REAL_SCANS, ids=[p.stem for p in REAL_SCANS])
def test_real_scan_export_on_gpu_matches_cpu(tmp_path, scan):
    process_scan(scan, tmp_path / "positive.tif", Stage.FULL, PROFILE, ToneCurveParams())
    warning_cpu = export_delivery_image(tmp_path / "positive.tif", tmp_path / "cpu.png")
    warnings = []
    warning_gpu = export_delivery_image(
        tmp_path / "positive.tif", tmp_path / "gpu.png", device=_DEVICE, on_warning=warnings.append
    )
    assert warnings == [] and warning_cpu is None and warning_gpu is None
    _assert_export_close(tmp_path / "gpu.png", tmp_path / "cpu.png", scan.stem)
