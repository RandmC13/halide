"""process_scan / print_scan on a compute device (docs/plans/gpu-acceleration.md, §3.3).

There is no GPU here, so `device=ComputeDevice(kind="gpu")` runs on tests/unit/_fake_device.py,
injected where processing.py crosses to and from the device (`halide.device.to_device`/`to_host`)
and registered with core/'s array_namespace. The fake computes with numpy, so the device path must
be bit-identical to the CPU path: any difference is plumbing (a stage that skipped `xp`, a scalar
that came back from the device in a different type, a band that was missed), not arithmetic. The
real-GPU version of these checks, held to the D2 tolerance, is tests/gpu/test_gpu_parity.py.
"""

import json

import numpy as np
import pytest

import halide.banding
import halide.device
import halide.processing
from halide.core._xp import register_namespace
from halide.core.types import DensityProfile, ToneCurveParams
from halide.device import ComputeDevice
from halide.io.icc import output_profile_bytes
from halide.io.tiff import read_tiff, read_tiff_description, write_tiff
from halide.processing import Stage, print_scan, process_scan
from tests.unit import _fake_device
from tests.unit._fake_device import FakeDeviceArray, fake_xp
from tests.unit.test_icc import LINEAR_TAGS, build_icc

PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))


class OutOfMemoryError(MemoryError):
    """Named and based like cupy.cuda.memory.OutOfMemoryError (a MemoryError subclass)."""


@pytest.fixture
def uploads():
    """Every array the device path uploaded, in order."""
    return []


@pytest.fixture
def fake_gpu(monkeypatch, uploads):
    register_namespace(FakeDeviceArray, fake_xp)

    def to_device(a):
        uploads.append(a)
        return _fake_device.to_device(a)

    monkeypatch.setattr(halide.device, "to_device", to_device)
    monkeypatch.setattr(halide.device, "to_host", _fake_device.to_host)
    return ComputeDevice(kind="gpu", name="Fake GPU")


def _write_scan(path, shape=(37, 53, 3), seed=7, icc=None):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.01, 0.3, size=shape).astype(np.float32)
    image.flat[:5] = 0.0  # exercise the MIN_TRANSMITTANCE clamps
    image.flat[-4:] = -0.001
    write_tiff(path, image, icc_profile=icc if icc is not None else build_icc(LINEAR_TAGS))
    return path


def _pixels(path):
    return read_tiff(path).image


def _assert_same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def _provenance(path):
    return json.loads(read_tiff_description(path))["halide"]


CASES = [
    (Stage.FULL, ToneCurveParams(), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(mode="linear"), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(exposure=0.3, contrast=0.8), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(contrast=0.7), PROFILE, 1.37),
    (Stage.FULL, ToneCurveParams(), None, 1.0),  # per-frame auto calibration
    (Stage.FULL, ToneCurveParams(mode="linear"), None, 1.37),
    (Stage.INVERT_ONLY, ToneCurveParams(), None, 1.0),
    (Stage.DENSITY_ONLY, ToneCurveParams(), PROFILE, 1.37),
]
CASE_IDS = ["print", "flat", "pinned", "pinned-grade+gain", "auto", "auto-flat+gain", "invert-only", "density-only+gain"]


@pytest.mark.parametrize("stage, tone, profile, gain", CASES, ids=CASE_IDS)
def test_process_scan_on_device_matches_cpu_bit_for_bit(tmp_path, fake_gpu, uploads, stage, tone, profile, gain):
    scan = _write_scan(tmp_path / "neg.tif")
    cpu = process_scan(scan, tmp_path / "cpu.tif", stage, profile, tone, scan_gain=gain)
    warnings = []
    gpu = process_scan(scan, tmp_path / "gpu.tif", stage, profile, tone, scan_gain=gain,
                       device=fake_gpu, on_warning=warnings.append)
    assert warnings == []
    assert len(uploads) == 1  # really went through the device, once
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert gpu == cpu
    if cpu is not None and cpu.linear_scale is not None:
        # Brought back from the device as the CPU has it — a numpy scalar of the image's dtype,
        # not a device array and not a Python float (which changes the last bit of flat output).
        assert type(gpu.linear_scale) is type(cpu.linear_scale)
    if cpu is not None:
        gpu_record, cpu_record = _provenance(tmp_path / "gpu.tif"), _provenance(tmp_path / "cpu.tif")
        assert gpu_record.pop("device") == "gpu" and cpu_record.pop("device") == "cpu"
        assert gpu_record == cpu_record


def test_process_scan_on_device_thumbnail_only(tmp_path, fake_gpu):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, None, Stage.FULL, PROFILE, ToneCurveParams(), thumbnail_path=tmp_path / "cpu.png",
                 thumbnail_long_edge=20)
    process_scan(scan, None, Stage.FULL, PROFILE, ToneCurveParams(), thumbnail_path=tmp_path / "gpu.png",
                 thumbnail_long_edge=20, device=fake_gpu)
    from PIL import Image

    cpu, gpu = Image.open(tmp_path / "cpu.png"), Image.open(tmp_path / "gpu.png")
    assert np.array_equal(np.asarray(cpu), np.asarray(gpu))
    # The caption's record differs only in which device made it.
    assert {k: v.replace('"gpu"', '"cpu"') for k, v in gpu.info.items()} == cpu.info


def test_a_cpu_compute_device_takes_the_cpu_path(tmp_path, fake_gpu, uploads):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "out.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=ComputeDevice(kind="cpu"))
    assert uploads == []
    assert _provenance(tmp_path / "out.tif")["device"] == "cpu"


def test_provenance_records_the_device(tmp_path, fake_gpu):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu)
    assert _provenance(tmp_path / "cpu.tif")["device"] == "cpu"
    assert _provenance(tmp_path / "gpu.tif")["device"] == "gpu"


def _fail_on_device_call(monkeypatch, name, exc, on_call=2):
    """Make processing.<name> raise `exc` on its `on_call`-th call with a device array — a device
    failure partway through the frame. Host (CPU) calls pass straight through."""
    real = getattr(halide.processing, name)
    calls = {"device": 0}

    def wrapper(band, *args, **kwargs):
        if isinstance(band, FakeDeviceArray):
            calls["device"] += 1
            if calls["device"] == on_call:
                raise exc
        return real(band, *args, **kwargs)

    monkeypatch.setattr(halide.processing, name, wrapper)
    return calls


def test_device_out_of_memory_redoes_the_frame_on_cpu(tmp_path, fake_gpu, monkeypatch):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    # Several bands, so the failure lands after the device has already worked on part of the frame.
    monkeypatch.setattr(halide.banding, "DEVICE_BAND_BYTES", 53 * 3 * 4 * 5)
    calls = _fail_on_device_call(monkeypatch, "negative_to_positive", OutOfMemoryError("out of memory allocating"))

    warnings = []
    resolved = process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(),
                            device=fake_gpu, on_warning=warnings.append)
    assert calls["device"] == 2
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert resolved is not None
    assert len(warnings) == 1 and "out of GPU memory" in warnings[0]
    assert _provenance(tmp_path / "gpu.tif")["device"] == "cpu"  # says which path really made it


def test_any_device_error_redoes_the_frame_on_cpu(tmp_path, fake_gpu, monkeypatch):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    _fail_on_device_call(monkeypatch, "apply_tone", RuntimeError("cudaErrorLaunchFailure"), on_call=1)
    warnings = []
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"),
                 device=fake_gpu, on_warning=warnings.append)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert len(warnings) == 1 and "cudaErrorLaunchFailure" in warnings[0] and "CPU" in warnings[0]


def test_a_failed_download_rereads_the_scan(tmp_path, fake_gpu, monkeypatch):
    """A GPU error can surface only at the download (kernels run asynchronously), possibly after
    part of the host buffer was overwritten — then the scan is read again rather than trusted."""
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())

    def broken_download(a, out=None):
        if out is not None:
            out[: out.shape[0] // 2] = 12345.0  # half-written, then the error
        raise RuntimeError("cudaErrorIllegalAddress")

    monkeypatch.setattr(halide.device, "to_host", broken_download)
    warnings = []
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu,
                 on_warning=warnings.append)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert len(warnings) == 1


def test_device_fallback_without_on_warning_still_succeeds(tmp_path, fake_gpu, monkeypatch):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    _fail_on_device_call(monkeypatch, "negative_to_positive", OutOfMemoryError(), on_call=1)
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))


def test_the_decoded_host_buffer_is_the_download_target(tmp_path, fake_gpu, uploads, monkeypatch):
    """Host memory stays ~1 frame on the device path: the frame goes up once and comes back into
    the very buffer it was decoded into, which is what gets written."""
    scan = _write_scan(tmp_path / "neg.tif")
    decoded, written = [], []
    real_read, real_write = halide.processing.read_tiff, halide.processing.write_tiff

    def spy_read(path, *args, **kwargs):
        result = real_read(path, *args, **kwargs)
        decoded.append(result.image)
        return result

    def spy_write(path, image, *args, **kwargs):
        written.append(image)
        return real_write(path, image, *args, **kwargs)

    monkeypatch.setattr(halide.processing, "read_tiff", spy_read)
    monkeypatch.setattr(halide.processing, "write_tiff", spy_write)
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu)
    assert len(decoded) == 1 and len(written) == 1
    assert np.shares_memory(written[0], decoded[0]) and written[0].shape == decoded[0].shape
    assert uploads[0] is decoded[0]


@pytest.mark.parametrize("tone", [ToneCurveParams(), ToneCurveParams(exposure=0.2)], ids=["fitted", "pinned-exposure"])
def test_print_scan_on_device_matches_cpu(tmp_path, fake_gpu, uploads, tone):
    """Including undoing the flat file's recorded linear_scale, and the skip of ICC conversion for
    halide's own ACEScg profile."""
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "flat.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    cpu = print_scan(tmp_path / "flat.tif", tmp_path / "cpu.tif", tone)
    warnings = []
    gpu = print_scan(tmp_path / "flat.tif", tmp_path / "gpu.tif", tone, device=fake_gpu, on_warning=warnings.append)
    assert warnings == [] and len(uploads) == 1
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert gpu == cpu
    assert _provenance(tmp_path / "gpu.tif")["device"] == "gpu"
    assert _provenance(tmp_path / "cpu.tif")["device"] == "cpu"


def test_print_scan_on_device_converts_a_foreign_profile(tmp_path, fake_gpu):
    """A flat positive edited elsewhere comes back in another linear profile, without provenance."""
    flat = _write_scan(tmp_path / "flat.tif", seed=11)
    cpu = print_scan(flat, tmp_path / "cpu.tif", ToneCurveParams(exposure=0.1))
    gpu = print_scan(flat, tmp_path / "gpu.tif", ToneCurveParams(exposure=0.1), device=fake_gpu)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert gpu == cpu and gpu[1] is not None  # the "can't reproduce a pinned exposure" warning


def test_print_scan_device_error_redoes_on_cpu(tmp_path, fake_gpu, monkeypatch):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "flat.tif", Stage.FULL, PROFILE, ToneCurveParams(mode="linear"))
    cpu = print_scan(tmp_path / "flat.tif", tmp_path / "cpu.tif", ToneCurveParams())
    _fail_on_device_call(monkeypatch, "apply_tone", OutOfMemoryError(), on_call=1)
    warnings = []
    gpu = print_scan(tmp_path / "flat.tif", tmp_path / "gpu.tif", ToneCurveParams(), device=fake_gpu,
                     on_warning=warnings.append)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert gpu == cpu and "out of GPU memory" in warnings[0]
    assert _provenance(tmp_path / "gpu.tif")["device"] == "cpu"


@pytest.mark.parametrize("stage, tone, profile, gain", CASES, ids=CASE_IDS)
def test_one_row_bands_on_device(tmp_path, fake_gpu, monkeypatch, stage, tone, profile, gain):
    scan = _write_scan(tmp_path / "neg.tif")
    process_scan(scan, tmp_path / "cpu.tif", stage, profile, tone, scan_gain=gain)
    monkeypatch.setattr(halide.banding, "DEVICE_BAND_BYTES", 1)
    process_scan(scan, tmp_path / "gpu.tif", stage, profile, tone, scan_gain=gain, device=fake_gpu)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))


@pytest.mark.parametrize("shape", [(2, 2, 3), (1, 9, 3)], ids=["2x2", "1x9"])
@pytest.mark.parametrize("tone", [ToneCurveParams(), ToneCurveParams(mode="linear")], ids=["print", "flat"])
def test_tiny_image_on_device(tmp_path, fake_gpu, shape, tone):
    scan = _write_scan(tmp_path / "neg.tif", shape=shape)
    cpu = process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, tone)
    gpu = process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, tone, device=fake_gpu)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
    assert gpu == cpu


def test_scan_already_in_halides_own_profile_is_not_converted_on_device(tmp_path, fake_gpu):
    scan = _write_scan(tmp_path / "neg.tif", icc=output_profile_bytes())
    process_scan(scan, tmp_path / "cpu.tif", Stage.FULL, PROFILE, ToneCurveParams())
    process_scan(scan, tmp_path / "gpu.tif", Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu)
    _assert_same_bits(_pixels(tmp_path / "gpu.tif"), _pixels(tmp_path / "cpu.tif"))
