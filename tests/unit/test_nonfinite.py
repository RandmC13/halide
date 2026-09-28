"""F05: non-finite (NaN/inf) and non-positive pixels — a raw converter's export can contain a few
(a highlight-recovery artifact, a scanner glitch); this project's top priority is colorimetric
faithfulness, so they must be handled explicitly rather than silently poisoning a whole frame or
profile with NaN. See docs/investigations/codebase-review-evidence/2.1.md (2.1-2), 2.2.md (2.2-6).
"""

import json

import numpy as np
import pytest

import halide.device
from halide import gpu_service
from halide.calibration.auto import auto_density_balance
from halide.core._xp import register_namespace
from halide.core.tone_render import ResolvedTone
from halide.core.types import DensityProfile, ToneCurveParams
from halide.device import ComputeDevice
from halide.gpu_service import ServiceClient
from halide.io.tiff import read_tiff, write_tiff
from halide.processing import (
    DeviceJobFailed,
    DevelopRequest,
    ScanColorError,
    Stage,
    _read_scan,
    develop_request,
    print_scan,
    process_scan,
    provenance_json,
)
from tests.unit import _fake_device
from tests.unit._fake_device import FakeDeviceArray, fake_xp
from tests.unit.test_icc import LINEAR_TAGS, build_icc

SHADOW_RGB = (0.094, 0.131, 0.050)
HIGHLIGHT_RGB = (0.048, 0.054, 0.016)
PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))


def _write_negative(path, seed=1, bad_slice=None, bad_value=np.nan, bad_fraction=None):
    """A small synthetic negative TIFF with a real linear ICC profile — same shape as
    tests/unit/test_processing.py's own helper, plus the ability to inject non-finite/non-positive
    pixels for F05's tests. `bad_slice` sets one region of the array to `bad_value`; `bad_fraction`
    instead sets that fraction of the flattened array's leading elements to it (for the "too many"
    threshold test — a plain count-of-values fraction, matching the count `nan_to_num` cleans)."""
    rng = np.random.default_rng(seed)
    img = np.full((16, 16, 3), SHADOW_RGB, dtype=np.float32)
    img[8:16, :] = HIGHLIGHT_RGB
    img += rng.normal(scale=0.002, size=img.shape).astype(np.float32)
    if bad_slice is not None:
        img[bad_slice] = bad_value
    if bad_fraction is not None:
        flat = img.reshape(-1)
        flat[: int(flat.size * bad_fraction)] = bad_value
    write_tiff(path, img, icc_profile=build_icc(LINEAR_TAGS))
    return img


def _write_negative_file(path, seed=1):
    _write_negative(path, seed=seed)
    return path


def test_one_nan_pixel_invert_succeeds_and_warns(tmp_path):
    scan = tmp_path / "nan.tif"
    # One bad raw value corrupts all 3 of that pixel's channels once the ICC conversion's matrix
    # multiply mixes them (a real effect, not a test artifact — see 2.1-2) — but it's still one
    # *pixel* (one spot on the frame), which is what the warning counts and reports (R7).
    _write_negative(scan, seed=1, bad_slice=(0, 0, 0), bad_value=np.nan)
    out = tmp_path / "out.tif"
    warnings = []
    resolved = process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams(), on_warning=warnings.append)

    assert resolved is not None
    assert out.exists()
    result = read_tiff(out)
    assert np.isfinite(result.image).all()
    assert len(warnings) == 1
    assert "1 pixel wasn't a valid number (NaN/inf) and was treated as clear film" in warnings[0]
    assert "check the raw converter's export" in warnings[0]
    assert str(scan) in warnings[0]


def test_one_inf_pixel_flat_output_is_finite(tmp_path):
    scan = tmp_path / "inf.tif"
    # A whole pixel (all 3 channels) set to +inf: also exercises the ICC matmul turning inf into
    # NaN for the *other* channels of the working-space result (2.1-2's actual failure mode).
    _write_negative(scan, seed=2, bad_slice=(3, 3, slice(None)), bad_value=np.inf)
    out = tmp_path / "flat.tif"
    resolved = process_scan(
        scan, out, Stage.FULL, PROFILE, ToneCurveParams(mode="linear"), on_warning=lambda msg: None
    )

    assert resolved is not None
    assert resolved.mode == "linear"
    result = read_tiff(out)
    assert np.isfinite(result.image).all()


def test_over_one_percent_nonfinite_fails_frame_with_plain_message(tmp_path):
    scan = tmp_path / "broken.tif"
    _write_negative(scan, seed=3, bad_fraction=0.02)  # 2% of values NaN: well over the 1% cutoff
    out = tmp_path / "out.tif"

    with pytest.raises(ScanColorError) as excinfo:
        process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams())

    message = str(excinfo.value)
    assert "% of its pixels aren't valid numbers - this export looks broken" in message
    assert "re-export it from the raw converter" in message
    assert str(scan) in message
    assert not out.exists()  # atomic: a frame that fails must leave no output behind


def test_print_scan_also_cleans_nonfinite_pixels_and_warns(tmp_path):
    # print_scan takes the same ICC-conversion path as process_scan (a flat positive re-read
    # through the same working-space boundary), so it needs the same handling.
    flat = tmp_path / "flat.tif"
    _write_negative(flat, seed=4, bad_slice=(1, 1, 1), bad_value=np.nan)
    out = tmp_path / "print.tif"
    warnings = []

    resolved, _ = print_scan(flat, out, ToneCurveParams(), on_warning=warnings.append)

    assert resolved is not None
    result = read_tiff(out)
    assert np.isfinite(result.image).all()
    assert len(warnings) == 1
    assert "1 pixel wasn't a valid number (NaN/inf)" in warnings[0]  # see test_one_nan_pixel... above


# --- Device / GPU service (R8: bad input isn't a device failure; R7: warnings must travel back
# through the service without printing) --------------------------------------------------------


@pytest.fixture
def fake_gpu(monkeypatch):
    """A ComputeDevice whose to_device/to_host are the strict fake (numpy underneath) — both for
    the in-process device path and for a real gpu_service._serving(...), which runs the identical
    run_device_job through this same halide.device module."""
    register_namespace(FakeDeviceArray, fake_xp)
    monkeypatch.setattr(halide.device, "to_device", _fake_device.to_device)
    monkeypatch.setattr(halide.device, "to_host", _fake_device.to_host)
    return ComputeDevice(kind="gpu", name="Fake GPU")


def test_over_one_percent_nonfinite_on_device_raises_plainly_no_gpu_wording_no_retry(tmp_path, fake_gpu):
    # R8: run_device_job must not wrap a ScanColorError as a DeviceJobFailed — that would print
    # "the GPU failed (...)" and then retry on the CPU, which would just raise the identical error
    # a second time. It must fail once, plainly, exactly as the CPU-only path does.
    scan = tmp_path / "broken.tif"
    _write_negative(scan, seed=8, bad_fraction=0.02)
    out = tmp_path / "out.tif"
    warnings = []

    with pytest.raises(ScanColorError) as excinfo:
        process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams(), device=fake_gpu, on_warning=warnings.append)

    assert "aren't valid numbers - this export looks broken" in str(excinfo.value)
    assert warnings == []  # no "the GPU failed" fallback warning — it never fell back
    assert not out.exists()


def test_over_one_percent_nonfinite_through_service_raises_plainly_no_retry(tmp_path, fake_gpu):
    # Same R8 requirement, through the GPU service: the client must raise the ScanColorError the
    # service sends back as-is, not a DeviceJobFailed.
    scan = tmp_path / "broken.tif"
    _write_negative(scan, seed=9, bad_fraction=0.02)
    host, source_profile = _read_scan(scan)
    request = DevelopRequest(source_profile=source_profile, scan_gain=1.0, density_profile=PROFILE,
                             stage=Stage.FULL, tone_params=ToneCurveParams())

    from halide.shared_frames import new_frame

    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        with new_frame(host.shape, host.dtype) as frame:
            frame.array[...] = host
            with pytest.raises(ScanColorError) as excinfo:
                client.develop(frame, request, name=str(scan))
            assert not isinstance(excinfo.value, DeviceJobFailed)
        assert "aren't valid numbers - this export looks broken" in str(excinfo.value)
        # An input error is a whole, in-step reply: F06's "any interruption kills the client" rule
        # must not catch it, so the same client goes on to develop the next frame.
        good, _ = _read_scan(_write_negative_file(tmp_path / "good.tif", seed=10))
        with new_frame(good.shape, good.dtype) as frame:
            frame.array[...] = good
            reply = client.develop(frame, request, name="good.tif")
        assert reply.resolved is not None


def test_service_reports_nonfinite_warning_without_printing(tmp_path, capsys, fake_gpu):
    # R7 (this fix round): the warning must travel back in the reply, not be printed from inside
    # the spawned/threaded service process — nothing here is a photographer's terminal.
    scan = tmp_path / "nan.tif"
    _write_negative(scan, seed=10, bad_slice=(0, 0, 0), bad_value=np.nan)
    host, source_profile = _read_scan(scan)
    request = DevelopRequest(source_profile=source_profile, scan_gain=1.0, density_profile=PROFILE,
                             stage=Stage.FULL, tone_params=ToneCurveParams())

    from halide.shared_frames import new_frame

    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        with new_frame(host.shape, host.dtype) as frame:
            frame.array[...] = host
            reply = client.develop(frame, request, name=str(scan))

    assert len(reply.warnings) == 1
    assert "1 pixel wasn't a valid number (NaN/inf)" in reply.warnings[0]
    assert str(scan) in reply.warnings[0]
    out, err = capsys.readouterr()
    assert out == "" and err == ""  # the service printed nothing


def test_process_scan_through_service_re_emits_the_warning_via_on_warning(tmp_path, capsys, fake_gpu):
    # The full path a batch worker takes (processing.process_scan(..., service=...)): the worker's
    # own on_warning must receive the message exactly as the in-process paths deliver it, and the
    # service itself must still print nothing.
    scan = tmp_path / "nan.tif"
    _write_negative(scan, seed=11, bad_slice=(2, 2, 2), bad_value=np.nan)
    out = tmp_path / "out.tif"
    warnings = []

    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        resolved = process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams(), on_warning=warnings.append,
                                service=client, shm_prefix=None)

    assert resolved is not None
    assert out.exists()
    assert len(warnings) == 1
    assert "1 pixel wasn't a valid number (NaN/inf)" in warnings[0]
    assert str(scan) in warnings[0]
    out_text, err_text = capsys.readouterr()
    assert out_text == "" and err_text == ""


# --- Auto calibration (calibration/auto.py) ------------------------------------------------------
# verified.md's 2.2-6 synthetic reproduction: a 300x400 film-like frame with a known density scale
# (1.3/1/0.78), 0.1% of pixels set to a non-positive value moved the solved scale to ~(0.95,1,1.05)
# before this fix — a silent, badly-wrong profile.

_KNOWN_PROFILE = DensityProfile(white_balance=(1.0, 1.0, 1.0), density_scale=(1.3, 1.0, 0.78))


def _synthetic_film_frame(seed=0):
    """A 300x400 frame of plausible film-like transmittance values (positive, varied, each channel
    at a different characteristic density) that `_KNOWN_PROFILE` could plausibly have come from —
    enough spread and enough genuinely-neutral-looking pixels for auto_density_balance to recover a
    profile close to `_KNOWN_PROFILE` on clean input."""
    rng = np.random.default_rng(seed)
    density = rng.uniform(0.2, 1.2, size=(300, 400, 1)).astype(np.float32)  # shared "scene" density
    channel_scale = np.array([1.0, 1.0 / 1.3, 1.0 / 0.78], dtype=np.float32)  # invert of density_scale
    transmittance = (10.0 ** -density) * channel_scale
    return transmittance.astype(np.float32)


def _solved_scale(image):
    return np.asarray(auto_density_balance(image).density_scale)


def test_auto_density_ignores_nonpositive_pixels():
    clean = _synthetic_film_frame()
    clean_scale = _solved_scale(clean)

    dirtied = clean.copy()
    flat = dirtied.reshape(-1, 3)
    n_bad = max(1, int(flat.shape[0] * 0.001))
    flat[:n_bad] = -0.001

    dirtied_scale = _solved_scale(dirtied)
    assert dirtied_scale == pytest.approx(clean_scale, abs=0.01)


def test_auto_density_single_nan_gives_finite_profile():
    image = _synthetic_film_frame(seed=1)
    image[0, 0, :] = np.nan

    profile = auto_density_balance(image)

    assert np.all(np.isfinite(profile.white_balance))
    assert np.all(np.isfinite(profile.density_scale))


# --- Guards (core/tone_render.py, processing.py) -------------------------------------------------


def test_provenance_rejects_nan():
    resolved = ResolvedTone(mode="paper", exposure=float("nan"), contrast=0.8)
    with pytest.raises(ValueError):
        provenance_json(resolved, None)


def test_provenance_accepts_finite_values():
    resolved = ResolvedTone(mode="paper", exposure=0.1, contrast=0.8)
    text = provenance_json(resolved, PROFILE)
    assert json.loads(text)["halide"]["exposure"] == pytest.approx(0.1)


# --- No regression on clean input -----------------------------------------------------------------


def test_clean_input_bit_identical_to_before(tmp_path):
    """A hash of a clean synthetic frame's invert output, pinned so this whole change can't have
    touched clean-input behaviour (the hard constraint: nan_to_num must only ever run on a band
    that actually has a non-finite value)."""
    import hashlib

    scan = tmp_path / "clean.tif"
    _write_negative(scan, seed=5)
    out = tmp_path / "out.tif"
    process_scan(scan, out, Stage.FULL, PROFILE, ToneCurveParams())
    digest = hashlib.sha256(read_tiff(out).image.tobytes()).hexdigest()
    assert digest == "b3fd5cd6ade915bad200a8c1602d6a6616a67c9f387fffa9f16374cc13d5035c"
