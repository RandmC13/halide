"""The device path on a real GPU against the CPU path (docs/plans/gpu-acceleration.md, §3.3, D2).

Run with `pytest -m gpu` on a machine with CuPy and a working NVIDIA GPU; everywhere else the whole
module skips at collection. tests/unit/test_device_pipeline.py pins the same plumbing bit-for-bit on
a fake device; this checks the real arithmetic, which is close but not bit-identical (different
float rounding in the GPU's matmuls, logs and powers), held to the D2 tolerance:
  - float32 pixels within 1e-5 relative, or 1e-7 absolute (for values near zero);
  - fitted exposure / contrast within 1e-5; the flat output's linear_scale within 1e-5 relative.

The real scans (IMG_0151/0156/0156-nowb/0158.tif in the repo root, gitignored) are used when
present — they are what the tolerance is really about.

The last section (Task B4) is different: the shared GPU service against the in-process GPU path,
both on the card, through the real batch code — held to bit-identity, not D2.
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


@pytest.fixture(autouse=True)
def _delete_outputs_even_on_failure(tmp_path):
    """Remove this test's output files as soon as it ends, pass or fail. Each real-scan test writes
    two or three ~130 MiB TIFFs, and pytest's "failed" retention (pyproject.toml) would keep a failed
    test's files — in /tmp, which is RAM on many Linux desktops — until the next run. Everything
    needed to diagnose a failure is in the assertion message, not the files."""
    yield
    for path in tmp_path.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)


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
    assert "exported this file" in warnings[0] and "developed" not in warnings[0]


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


# ---------------------------------------------------------------------------
# Task 8: auto calibration on the device frame (calibration/auto.py). Compared by tolerance only:
# the GPU's argsort orders exact luminance ties differently (the user's probe: 58% identical order),
# which can move a density bin's boundary by a pixel — close, never bit-identical.
# ---------------------------------------------------------------------------

PROFILE_RTOL = 1e-5


def _assert_profile_close(gpu, cpu):
    assert gpu.source == cpu.source == "auto"
    np.testing.assert_allclose(gpu.white_balance, cpu.white_balance, rtol=PROFILE_RTOL, atol=0)
    np.testing.assert_allclose(gpu.density_scale, cpu.density_scale, rtol=PROFILE_RTOL, atol=0)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scan", REAL_SCANS, ids=[p.stem for p in REAL_SCANS])
def test_real_scan_auto_density_balance_on_gpu_matches_cpu(scan):
    import cupy

    from halide.calibration.auto import auto_density_balance
    from halide.processing import load_working_space_image

    image = load_working_space_image(scan)
    cpu = auto_density_balance(image)
    gpu = auto_density_balance(cupy.asarray(image))
    _assert_profile_close(gpu, cpu)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
def test_real_scans_roll_auto_density_balance_on_gpu_matches_cpu():
    import cupy

    from halide.calibration.auto import roll_auto_density_balance
    from halide.processing import load_working_space_image

    frames = [load_working_space_image(scan)[::8, ::8].copy() for scan in REAL_SCANS]
    cpu = roll_auto_density_balance(frames)
    gpu = roll_auto_density_balance([cupy.asarray(f) for f in frames])
    _assert_profile_close(gpu, cpu)


# ---------------------------------------------------------------------------
# Task B4 (docs/plans/gpu-batch-throughput.md): the shared GPU service against the in-process GPU
# path (per-worker mode, HALIDE_GPU_SERVICE=0), both through the real batch code. Same device code
# on the same card, so these are held to bit-identity — not to D2: TIFF pixel bytes and provenance
# (whose "device" is "gpu" on both sides), and export PNG pixels. Any difference is a service bug —
# except the "auto-density"/"auto-density-flat" develop cases, where per-frame auto calibration's
# GPU argsort/percentile isn't guaranteed reproducible run to run (see
# _assert_tiffs_identical_diagnosing_nondeterminism below, which tells the two apart on a mismatch
# instead of assuming it's always the service).
#
# Real-scan outputs don't go to tmp_path: /tmp is RAM on the user's machine, with only a few GiB
# free, and each output is ~130 MiB. They go to a scratch folder inside the repo (gitignored),
# deleted after every test, pass or fail, and each batch is at most two real scans.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

from halide.batch.orchestrator import (  # noqa: E402
    SERVICE_ENV,
    BatchJob,
    batch_compute,
    run_batch,
    run_export_batch,
    run_print_batch,
)

SCRATCH_ROOT = REPO_ROOT / ".gpu-test-scratch"
REAL_PAIRS = [REAL_SCANS[i:i + 2] for i in range(0, len(REAL_SCANS), 2)]
REAL_PAIR_IDS = ["+".join(p.stem for p in pair) for pair in REAL_PAIRS]
_BATCH_WORKERS = 2
_MODES = ("service", "per_worker")


@pytest.fixture
def repo_scratch():
    """A folder inside the repo for real-scan outputs, removed when the test ends however it ends."""
    SCRATCH_ROOT.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="b4-", dir=SCRATCH_ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(OSError):  # another test's folder may still be in it
            SCRATCH_ROOT.rmdir()


def _batch_in_mode(monkeypatch, mode, jobs, workload, run):
    """Run one batch (`run(compute)`) with the GPU reached through `mode`: "service" (the shared GPU
    service, the default) or "per_worker" (HALIDE_GPU_SERVICE=0: each worker its own CUDA context,
    the in-process GPU path). Every frame must come out without an error or a CPU fallback."""
    import cupy

    # This process's own CuPy pool still holds frames' worth of blocks from the in-process tests
    # above (and a test's own GPU-developed inputs): hand them back before the batch needs the card.
    cupy.get_default_memory_pool().free_all_blocks()
    if mode == "service":
        monkeypatch.delenv(SERVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(SERVICE_ENV, "0")
    with batch_compute(jobs, _DEVICE, workload) as compute:
        assert compute.mode == mode, compute.fallback_reason
        results = run(compute)
    assert [(r.error, r.warning) for r in results] == [(None, None)] * len(jobs), (
        f"{mode}: {[(r.error, r.warning) for r in results]}"  # a fallback would compare CPU with GPU
    )


def _assert_tiffs_identical(service_path, in_process_path):
    service, in_process = read_tiff(service_path).image, read_tiff(in_process_path).image
    assert service.dtype == in_process.dtype == np.float32 and service.shape == in_process.shape
    if service.tobytes() != in_process.tobytes():
        diff = np.abs(service.astype(np.float64) - in_process.astype(np.float64))
        pytest.fail(f"{service_path.name}: not bit-identical — {np.count_nonzero(diff)} of {diff.size} values "
                    f"differ, max |diff| {diff.max():.3g}")
    service_record = read_provenance(read_tiff_description(service_path))
    in_process_record = read_provenance(read_tiff_description(in_process_path))
    assert service_record == in_process_record
    assert service_record["device"] == "gpu"


def _assert_pngs_identical(service_path, in_process_path):
    service, in_process = _png_pixels(service_path), _png_pixels(in_process_path)
    assert service.shape == in_process.shape
    if not np.array_equal(service, in_process):
        diff = np.abs(service - in_process)
        pytest.fail(f"{service_path.name}: not pixel-identical — {np.count_nonzero(diff)} of {diff.size} values "
                    f"differ, max {int(diff.max())} code values")


def _rerun_per_worker(monkeypatch, scans, folder, stage, profile, tone):
    """Re-run just the per-worker GPU path into a fresh folder — only ever called to diagnose a
    service-vs-per-worker mismatch that has already happened (see
    _assert_tiffs_identical_diagnosing_nondeterminism), never on the happy path."""
    rerun = folder / "per_worker_rerun"
    rerun.mkdir(exist_ok=True)
    jobs = [BatchJob(input_path=s, output_path=rerun / f"{s.stem}.tif") for s in scans]
    _batch_in_mode(monkeypatch, "per_worker", jobs, "develop", lambda compute: run_batch(
        jobs, stage, profile, tone, max_workers=_BATCH_WORKERS, device=_DEVICE, compute=compute))
    return rerun


def _assert_tiffs_identical_diagnosing_nondeterminism(per_worker_rerun, folder, profile, name):
    """service vs per-worker: on a mismatch with no explicit profile ("auto-density"/
    "auto-density-flat" — per-frame auto calibration, `calibration/auto.py`), the mismatch isn't
    automatically a service bug: CLAUDE.md records that GPU argsort/percentile ties aren't
    guaranteed to land the same way as the CPU's, and the same non-associative-reduction /
    thread-scheduling effect isn't guaranteed to reproduce identically run to run on the GPU either
    — the service and the per-worker path both run the *same* device code, but as two separate CUDA
    executions. So before blaming "a service bug", re-run the per-worker path a second time (cheap:
    only ever done once a mismatch has already happened) and check whether per-worker agrees with
    its own first run — if it doesn't, the GPU itself wasn't reproducible here and the mismatch is
    inconclusive, not evidence against the service specifically. `per_worker_rerun` is a
    zero-argument callable (not the folder itself) so this second run only actually happens once,
    even if more than one frame in the same batch mismatches."""
    # _assert_tiffs_identical raises a plain AssertionError for a shape/dtype/provenance mismatch,
    # but pytest.fail() (the pixel-mismatch path, the one nondeterminism could actually hit) raises
    # pytest.fail.Exception instead — a BaseException subclass, not an AssertionError — so both must
    # be caught here.
    try:
        _assert_tiffs_identical(folder / "service" / f"{name}.tif", folder / "per_worker" / f"{name}.tif")
    except (AssertionError, pytest.fail.Exception) as failure:
        if profile is not None:
            raise  # no explicit profile == no per-frame auto calibration == no known GPU nondeterminism source
        rerun = per_worker_rerun()
        try:
            _assert_tiffs_identical(rerun / f"{name}.tif", folder / "per_worker" / f"{name}.tif")
        except (AssertionError, pytest.fail.Exception):
            pytest.fail(
                f"{name}: service and per-worker disagree, but per-worker itself wasn't reproducible "
                f"across two runs on this card — looks like GPU sort/percentile nondeterminism in the "
                f"per-frame auto calibration, not a service bug. Original failure: {failure}"
            )
        pytest.fail(
            f"{name}: service disagrees with per-worker, and per-worker reproduced itself exactly "
            f"across two runs — looks like a real service bug, not GPU nondeterminism. Original "
            f"failure: {failure}"
        )


def _develop_both_ways(monkeypatch, scans, folder, stage, profile, tone):
    for mode in _MODES:
        (folder / mode).mkdir()
        jobs = [BatchJob(input_path=s, output_path=folder / mode / f"{s.stem}.tif") for s in scans]
        _batch_in_mode(monkeypatch, mode, jobs, "develop", lambda compute: run_batch(
            jobs, stage, profile, tone, max_workers=_BATCH_WORKERS, device=_DEVICE, compute=compute))
    rerun_cache: dict = {}

    def per_worker_rerun():
        if "path" not in rerun_cache:
            rerun_cache["path"] = _rerun_per_worker(monkeypatch, scans, folder, stage, profile, tone)
        return rerun_cache["path"]

    for s in scans:
        _assert_tiffs_identical_diagnosing_nondeterminism(per_worker_rerun, folder, profile, s.stem)


def _print_both_ways(monkeypatch, flats, folder):
    for mode in _MODES:
        (folder / mode).mkdir()
        jobs = [BatchJob(input_path=f, output_path=folder / mode / f"{f.stem}-print.tif") for f in flats]
        _batch_in_mode(monkeypatch, mode, jobs, "develop", lambda compute: run_print_batch(
            jobs, ToneCurveParams(), max_workers=_BATCH_WORKERS, device=_DEVICE, compute=compute))
    for f in flats:
        _assert_tiffs_identical(folder / "service" / f"{f.stem}-print.tif", folder / "per_worker" / f"{f.stem}-print.tif")


def _export_both_ways(monkeypatch, positives, folder):
    for mode in _MODES:
        (folder / mode).mkdir()
        jobs = [BatchJob(input_path=p, output_path=folder / mode / f"{p.stem}.png") for p in positives]
        _batch_in_mode(monkeypatch, mode, jobs, "export", lambda compute: run_export_batch(
            jobs, max_workers=_BATCH_WORKERS, device=_DEVICE, compute=compute))
    for p in positives:
        _assert_pngs_identical(folder / "service" / f"{p.stem}.png", folder / "per_worker" / f"{p.stem}.png")


BATCH_CASES = [
    (ToneCurveParams(), PROFILE),
    (ToneCurveParams(mode="linear"), PROFILE),
    (ToneCurveParams(), None),
    (ToneCurveParams(mode="linear"), None),
]
BATCH_CASE_IDS = ["print", "flat", "auto-density", "auto-density-flat"]


def _synthetic_roll(folder, n=3, shape=(64, 96, 3)):
    folder.mkdir()
    return [_write_scan(folder / f"f{i}.tif", shape=shape, seed=i + 11) for i in range(n)]


@pytest.mark.parametrize("tone, profile", BATCH_CASES, ids=BATCH_CASE_IDS)
def test_service_batch_matches_in_process_gpu_batch(tmp_path, monkeypatch, tone, profile):
    scans = _synthetic_roll(tmp_path / "in")
    _develop_both_ways(monkeypatch, scans, tmp_path, Stage.FULL, profile, tone)


def test_service_print_batch_matches_in_process_gpu_batch(tmp_path, monkeypatch):
    scans = _synthetic_roll(tmp_path / "in")
    flats = []
    for s in scans:
        flats.append(tmp_path / f"{s.stem}-flat.tif")
        process_scan(s, flats[-1], Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    _print_both_ways(monkeypatch, flats, tmp_path)


def test_service_export_batch_matches_in_process_gpu_batch(tmp_path, monkeypatch):
    scans = _synthetic_roll(tmp_path / "in")
    positives = []
    for s in scans:
        positives.append(tmp_path / f"{s.stem}-positive.tif")
        process_scan(s, positives[-1], Stage.FULL, PROFILE, ToneCurveParams())
    _export_both_ways(monkeypatch, positives, tmp_path)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scans", REAL_PAIRS, ids=REAL_PAIR_IDS)
@pytest.mark.parametrize("tone, profile", BATCH_CASES[:3], ids=BATCH_CASE_IDS[:3])
def test_real_scans_service_batch_matches_in_process_gpu_batch(repo_scratch, monkeypatch, scans, tone, profile):
    _develop_both_ways(monkeypatch, scans, repo_scratch, Stage.FULL, profile, tone)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scans", REAL_PAIRS, ids=REAL_PAIR_IDS)
def test_real_scans_service_print_batch_matches_in_process_gpu_batch(repo_scratch, monkeypatch, scans):
    flats = []
    for s in scans:
        flats.append(repo_scratch / f"{s.stem}-flat.tif")
        process_scan(s, flats[-1], Stage.FULL, PROFILE, ToneCurveParams(mode="linear"), device=_DEVICE)
    _print_both_ways(monkeypatch, flats, repo_scratch)


@pytest.mark.skipif(not REAL_SCANS, reason="no real scans (IMG_*.tif) in the repo root")
@pytest.mark.parametrize("scans", REAL_PAIRS, ids=REAL_PAIR_IDS)
def test_real_scans_service_export_batch_matches_in_process_gpu_batch(repo_scratch, monkeypatch, scans):
    positives = []
    for s in scans:
        positives.append(repo_scratch / f"{s.stem}-positive.tif")
        process_scan(s, positives[-1], Stage.FULL, PROFILE, ToneCurveParams(), device=_DEVICE)
    _export_both_ways(monkeypatch, positives, repo_scratch)
