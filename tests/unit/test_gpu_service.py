"""The GPU service process (halide/gpu_service.py, docs/plans/gpu-batch-throughput.md Task B2).

There is no GPU here, so the service runs in one of three ways:
  - a `"cpu"`-kind service in a real separate (spawned) process — proves the IPC and shared-memory
    plumbing across real processes, since its numpy path must match calling the CPU functions
    directly bit for bit;
  - the strict fake device (tests/unit/_fake_device.py) serving on threads of this process
    (`gpu_service._serving`) — proves the device code the service runs is unchanged;
  - a `"gpu"`-kind service in a real process whose child installs the fake as its GPU
    (`running_service(..., initializer=_fake_device.install_as_gpu)`) — proves the parent never
    imports CuPy.
Frames are small (/dev/shm is 64 MiB in this sandbox); full-size, real-GPU checks are Task B4.
"""

import os
import pickle
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import psutil
import pytest

import halide.banding
import halide.device
import halide.processing
from halide import gpu_service
from halide.core._xp import register_namespace
from halide.core.types import DensityProfile, Stage, ToneCurveParams
from halide.device import ComputeDevice
from halide.gpu_service import DevelopReply, PrintReply, ServiceClient, ServiceUnavailable, running_service
from halide.io.icc import output_profile_bytes
from halide.io.raster import srgb_8bit_from_acescg
from halide.io.tiff import write_tiff
from halide.processing import (
    DevelopRequest,
    DeviceFailure,
    DeviceJobFailed,
    ExportRequest,
    PrintRequest,
    _read_scan,
    develop_request,
    export_fallback,
    fall_back_to_cpu,
    print_request,
)
from halide.shared_frames import new_frame
from tests.unit import _fake_device
from tests.unit._fake_device import FakeDeviceArray, fake_xp
from tests.unit.test_icc import LINEAR_TAGS, build_icc

PROFILE = DensityProfile(white_balance=(1.0, 1.2, 1.5), density_scale=(1.0, 1.05, 1.1))

DEVELOP_CASES = [
    (Stage.FULL, ToneCurveParams(), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(mode="linear"), PROFILE, 1.37),
    (Stage.FULL, ToneCurveParams(exposure=0.3, contrast=0.8), PROFILE, 1.0),
    (Stage.FULL, ToneCurveParams(), None, 1.0),  # per-frame auto calibration
    (Stage.INVERT_ONLY, ToneCurveParams(), None, 1.0),
    (Stage.DENSITY_ONLY, ToneCurveParams(), PROFILE, 1.37),
]
DEVELOP_IDS = ["print", "flat+gain", "pinned", "auto", "invert-only", "density-only+gain"]


class OutOfMemoryError(MemoryError):
    """Named and based like cupy.cuda.memory.OutOfMemoryError (a MemoryError subclass)."""


# --- helpers run inside a spawned service child (module-level so spawn can find them) -----------


def _slow_develop_in_child(arrived: str) -> None:
    """Initializer (bound with functools.partial): every develop request in the service first
    creates the file `arrived` — so a test knows its request really is in flight — then sleeps for
    a minute: a live but stuck service."""
    def slow(*args, **kwargs):
        Path(arrived).touch()
        time.sleep(60)

    halide.processing.develop_request = slow


def _late_export_in_child(arrived: str) -> None:
    """Initializer: every export request signals `arrived`, stalls past the client's timeout, then
    writes its output buffer anyway — a stuck service that comes back to life."""
    def late(frame, out, request, band_bytes=None):
        Path(arrived).touch()
        time.sleep(1.5)
        out[...] = 77

    halide.processing.export_request = late


class _Abort(BaseException):
    """Not an Exception: what a stray SystemExit/KeyboardInterrupt from request code looks like."""


def _fail_at_startup() -> None:
    """Initializer: the service can't start (as when CuPy is installed but unusable)."""
    raise RuntimeError("cudaErrorInsufficientDriver: no usable GPU")


def _slow_startup(arrived: str) -> None:
    """Initializer (bound with functools.partial): touches `arrived` then sleeps well past any
    test's patience — a driver probe that hangs before the service ever reports ready. Used to
    check that a `cancel` callback bounds running_service's startup wait instead of it blocking for
    the full _STARTUP_TIMEOUT."""
    Path(arrived).touch()
    time.sleep(60)


# --- fixtures and helpers ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_leftover_segments():
    """Every test must leave /dev/shm exactly as it found it."""
    before = set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()
    yield
    after = set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()
    assert after - before == set()


@pytest.fixture(scope="module")
def cpu_service():
    with running_service("cpu") as address:
        yield address


@pytest.fixture
def uploads():
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


def _write_scan(path, shape=(37, 53, 3), seed=7):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.01, 0.3, size=shape).astype(np.float32)
    image.flat[:5] = 0.0
    image.flat[-4:] = -0.001
    write_tiff(path, image, icc_profile=build_icc(LINEAR_TAGS))
    return path


def _positive(shape=(37, 53, 3), seed=5):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 1.5, size=shape).astype(np.float32)
    image.flat[:3] = -0.02
    return image


def _assert_same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def _develop_through(client, scan, stage, tone, profile, gain):
    """Develop `scan` directly on the CPU and through `client`; returns (expected, got, reply)."""
    host, source_profile = _read_scan(scan)
    request = DevelopRequest(source_profile=source_profile, scan_gain=gain, density_profile=profile,
                             stage=stage, tone_params=tone)
    expected_frame = host.copy()
    expected = develop_request(expected_frame, request)
    with new_frame(host.shape, host.dtype) as frame:
        frame.array[...] = host
        reply = client.develop(frame, request)
        got = frame.array.copy()
    return (expected_frame, expected), got, reply


def _check_develop(expected, got, reply):
    (expected_frame, (resolved, profile)) = expected
    assert isinstance(reply, DevelopReply)
    _assert_same_bits(got, expected_frame)
    assert reply.resolved == resolved and reply.profile == profile
    if resolved is not None and resolved.linear_scale is not None:
        assert type(reply.resolved.linear_scale) is type(resolved.linear_scale)


def _print_through(client, tmp_path):
    host, source_profile = _read_scan(_write_scan(tmp_path / "flat.tif", seed=11))
    request = PrintRequest(source_profile=source_profile, scale=1.7, print_params=ToneCurveParams(exposure=0.2))
    expected_frame = host.copy()
    expected = print_request(expected_frame, request)
    with new_frame(host.shape, host.dtype) as frame:
        frame.array[...] = host
        reply = client.print_(frame, request)
        got = frame.array.copy()
    assert isinstance(reply, PrintReply)
    _assert_same_bits(got, expected_frame)
    assert reply.resolved == expected


def _export_through(client):
    image = _positive()
    with new_frame(image.shape, image.dtype) as frame, new_frame(image.shape, np.uint8) as out:
        frame.array[...] = image
        assert client.export(frame, out, ExportRequest()) is None
        _assert_same_bits(frame.array, image)  # export only ever reads its input
        got = out.array.copy()
    _assert_same_bits(got, srgb_8bit_from_acescg(image))


def _service_alive(pid):
    """Whether `pid` is still running. A killed child stays a zombie until its parent reaps it (which
    running_service's join does), so a zombie counts as gone."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_for(path, seconds=30):
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)


posix_signals = pytest.mark.skipif(not hasattr(signal, "SIGKILL") or not hasattr(signal, "SIGSTOP"),
                                   reason="needs POSIX SIGKILL/SIGSTOP")


# --- a "cpu"-kind service in a real process: the plumbing ---------------------------------------


@pytest.mark.parametrize("stage, tone, profile, gain", DEVELOP_CASES, ids=DEVELOP_IDS)
def test_cpu_service_develops_bit_identical_to_direct_cpu_call(tmp_path, cpu_service, stage, tone, profile, gain):
    with ServiceClient(cpu_service) as client:
        _check_develop(*_develop_through(client, _write_scan(tmp_path / "neg.tif"), stage, tone, profile, gain))


def test_cpu_service_prints_bit_identical_to_direct_cpu_call(tmp_path, cpu_service):
    with ServiceClient(cpu_service) as client:
        _print_through(client, tmp_path)


def test_cpu_service_exports_bit_identical_to_direct_cpu_call(cpu_service):
    with ServiceClient(cpu_service) as client:
        _export_through(client)


def test_one_client_many_requests_and_several_clients(tmp_path, cpu_service):
    scan = _write_scan(tmp_path / "neg.tif")
    with ServiceClient(cpu_service) as first, ServiceClient(cpu_service) as second:
        for client in (first, second, first, second):
            _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))


def test_service_address_is_picklable(cpu_service):
    """B3 hands it to pool workers."""
    assert pickle.loads(pickle.dumps(cpu_service)) == cpu_service


# --- the strict fake device, serving on threads of this process: the device code ----------------


@pytest.mark.parametrize("stage, tone, profile, gain", DEVELOP_CASES, ids=DEVELOP_IDS)
def test_fake_gpu_service_develops_bit_identical_to_cpu(tmp_path, fake_gpu, uploads, stage, tone, profile, gain):
    scan = _write_scan(tmp_path / "neg.tif")
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        expected, got, reply = _develop_through(client, scan, stage, tone, profile, gain)
    _check_develop(expected, got, reply)
    assert len(uploads) == 1  # really went through the (fake) device, once


def test_fake_gpu_service_prints_and_exports_bit_identical_to_cpu(tmp_path, fake_gpu, uploads):
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        _print_through(client, tmp_path)
        _export_through(client)
    assert len(uploads) == 2


def test_fake_gpu_service_one_row_bands(tmp_path, fake_gpu, monkeypatch):
    monkeypatch.setattr(halide.banding, "DEVICE_BAND_BYTES", 1)
    scan = _write_scan(tmp_path / "neg.tif")
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))
        _export_through(client)


# --- request errors come back as replies --------------------------------------------------------


def test_a_request_error_is_a_reply_and_the_service_keeps_serving(tmp_path, cpu_service):
    request = DevelopRequest(source_profile=None, scan_gain=1.0, density_profile=PROFILE, stage=Stage.FULL,
                             tone_params=ToneCurveParams())
    with ServiceClient(cpu_service) as client:
        with new_frame((4, 4, 3), np.float32) as frame:
            bogus = frame.__class__(name="halide-no-such-segment", shape=frame.shape, dtype=frame.dtype,
                                    array=frame.array)
            with pytest.raises(DeviceJobFailed) as failed:
                client.develop(bogus, request)
        failure = failed.value.failure
        assert failure.type_name == "FileNotFoundError" and not failure.host_touched
        # ... and the same connection, and the service, carry on.
        _check_develop(*_develop_through(client, _write_scan(tmp_path / "neg.tif"), Stage.FULL,
                                         ToneCurveParams(), PROFILE, 1.0))


def test_device_out_of_memory_comes_back_as_a_reply(tmp_path, fake_gpu, monkeypatch):
    real = halide.processing.negative_to_positive

    def failing(band, *args, **kwargs):
        if isinstance(band, FakeDeviceArray):
            raise OutOfMemoryError("out of memory allocating 181 MiB")
        return real(band, *args, **kwargs)

    monkeypatch.setattr(halide.processing, "negative_to_positive", failing)
    scan = _write_scan(tmp_path / "neg.tif")
    host, source_profile = _read_scan(scan)
    request = DevelopRequest(source_profile, 1.0, PROFILE, Stage.FULL, ToneCurveParams())
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        with new_frame(host.shape, host.dtype) as frame:
            frame.array[...] = host
            with pytest.raises(DeviceJobFailed) as failed:
                client.develop(frame, request)
            _assert_same_bits(frame.array, host)  # not downloaded: still the decoded scan
        failure = failed.value.failure
        assert failure.out_of_memory and not failure.host_touched
        assert failure.warning(scan) == f"{scan}: out of GPU memory — developed this frame on the CPU instead"
        # The service survived its request's failure.
        monkeypatch.setattr(halide.processing, "negative_to_positive", real)
        _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))


def test_a_failed_download_is_reported_as_touching_the_frame(tmp_path, fake_gpu, monkeypatch):
    def broken_download(a, out=None):
        if out is not None:
            out[: out.shape[0] // 2] = 12345.0  # half-written, then the error
        raise RuntimeError("cudaErrorIllegalAddress")

    monkeypatch.setattr(halide.device, "to_host", broken_download)
    scan = _write_scan(tmp_path / "neg.tif")
    host, source_profile = _read_scan(scan)
    request = DevelopRequest(source_profile, 1.0, PROFILE, Stage.FULL, ToneCurveParams())
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address) as client:
        with new_frame(host.shape, host.dtype) as frame:
            frame.array[...] = host
            with pytest.raises(DeviceJobFailed) as failed:
                client.develop(frame, request)
    failure = failed.value.failure
    assert failure.host_touched and not failure.out_of_memory
    assert failure.warning(scan) == (
        f"{scan}: the GPU failed (RuntimeError: cudaErrorIllegalAddress) — developed this frame on the CPU instead"
    )


# --- the worker-side helper: today's fallback rules, from a reply -----------------------------


def test_fall_back_to_cpu_keeps_an_untouched_frame(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif")
    host, _ = _read_scan(scan)
    warnings = []
    failure = DeviceFailure(type_name="OutOfMemoryError", message="oom", out_of_memory=True, host_touched=False)
    assert fall_back_to_cpu(scan, host, failure, warnings.append) is host
    assert warnings == [f"{scan}: out of GPU memory — developed this frame on the CPU instead"]


def test_fall_back_to_cpu_rereads_a_touched_frame_into_a_fresh_buffer(tmp_path):
    """Fresh, not the frame it was given: a service that timed out may still be writing into it."""
    scan = _write_scan(tmp_path / "neg.tif")
    host, _ = _read_scan(scan)
    decoded = host.copy()
    host[:10] = 12345.0
    warnings = []
    failure = DeviceFailure(type_name="RuntimeError", message="boom", out_of_memory=False, host_touched=True)
    again = fall_back_to_cpu(scan, host, failure, warnings.append, action="printed this frame")
    assert again is not host and not np.shares_memory(again, host)
    _assert_same_bits(again, decoded)
    assert warnings == [f"{scan}: the GPU failed (RuntimeError: boom) — printed this frame on the CPU instead"]


def test_device_failure_matches_todays_fallback_wording(tmp_path):
    for exc in (OutOfMemoryError("x"), RuntimeError("cudaErrorLaunchFailure"), RuntimeError("")):
        for action in ("developed this frame", "exported this file"):
            assert DeviceFailure.from_exception(exc).warning("f.tif", action) == \
                halide.processing._gpu_fallback_message("f.tif", exc, action)


# --- a dead, stuck, or stopping service never hangs a worker ------------------------------------


@posix_signals
def test_killed_service_makes_the_next_request_raise_promptly(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif")
    with running_service("cpu") as address:
        with ServiceClient(address) as client:
            _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))
            os.kill(address.pid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while _service_alive(address.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            start = time.monotonic()
            with pytest.raises(ServiceUnavailable) as dead:
                _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
            assert time.monotonic() - start < 5
            assert dead.value.failure.host_touched  # it may have died mid-download: re-read
            # Dead for good: no reconnecting, no waiting.
            with pytest.raises(ServiceUnavailable):
                _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
        # A new client can't reach it either.
        with ServiceClient(address) as late, pytest.raises(ServiceUnavailable) as unreachable:
            _develop_through(late, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
        assert not unreachable.value.failure.host_touched  # never got as far as sending


def test_stuck_service_times_out(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif")
    arrived = tmp_path / "arrived"
    with running_service("cpu", initializer=partial(_slow_develop_in_child, str(arrived))) as address:
        with ServiceClient(address, timeout=0.5) as client:
            start = time.monotonic()
            with pytest.raises(ServiceUnavailable) as stuck:
                _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
            assert time.monotonic() - start < 5
            assert stuck.value.failure.host_touched
            start = time.monotonic()
            with pytest.raises(ServiceUnavailable):  # the connection is treated as dead
                _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
            assert time.monotonic() - start < 0.5


def test_leaving_running_service_stops_it_with_a_request_in_flight(tmp_path):
    scan = _write_scan(tmp_path / "neg.tif")
    outcome = []

    def worker(address):
        with ServiceClient(address) as client:
            try:
                _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
                outcome.append("replied")
            except ServiceUnavailable:
                outcome.append("unavailable")

    arrived = tmp_path / "arrived"
    with running_service("cpu", initializer=partial(_slow_develop_in_child, str(arrived))) as address:
        thread = threading.Thread(target=worker, args=(address,))
        thread.start()
        _wait_for(arrived)  # the request is now sleeping inside the service
        assert outcome == []
        start = time.monotonic()
    assert time.monotonic() - start < 10
    assert not _service_alive(address.pid)
    thread.join(10)
    assert not thread.is_alive() and outcome == ["unavailable"]
    if address.family == "AF_UNIX":
        assert not os.path.exists(address.address)  # its socket file is gone too


@posix_signals
def test_first_connect_to_a_frozen_service_is_bounded(tmp_path):
    """A stopped (SIGSTOP) process still gets its Unix socket connections completed by the kernel,
    so only the authkey handshake can notice it isn't answering — it must give up too."""
    scan = _write_scan(tmp_path / "neg.tif")
    with running_service("cpu") as address:
        os.kill(address.pid, signal.SIGSTOP)
        try:
            with ServiceClient(address, timeout=1.0) as client:
                start = time.monotonic()
                with pytest.raises(ServiceUnavailable) as frozen:
                    _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
                assert time.monotonic() - start < 5
                assert not frozen.value.failure.host_touched  # the request was never sent
        finally:
            os.kill(address.pid, signal.SIGCONT)
        # Thawed, it serves new clients again.
        with ServiceClient(address) as client:
            _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))


def test_a_stuck_export_never_hands_back_its_output_buffer(tmp_path):
    """The service can write `out` after the client has given up on it; the fallback must go
    into a buffer the worker owns."""
    arrived = tmp_path / "arrived"
    image = _positive()
    expected = srgb_8bit_from_acescg(image)
    warnings = []
    with running_service("cpu", initializer=partial(_late_export_in_child, str(arrived))) as address:
        with ServiceClient(address, timeout=0.3) as client, \
                new_frame(image.shape, image.dtype) as frame, new_frame(image.shape, np.uint8) as out:
            frame.array[...] = image
            with pytest.raises(ServiceUnavailable) as stuck:
                client.export(frame, out, ExportRequest())
            assert stuck.value.failure.host_touched
            result = export_fallback("pos.tif", frame.array, stuck.value.failure, warnings.append)
            assert not np.shares_memory(result, out.array)
            _wait_for(arrived)
            deadline = time.monotonic() + 10
            while not (out.array == 77).all():  # the late write really happens...
                assert time.monotonic() < deadline
                time.sleep(0.05)
            _assert_same_bits(result, expected)  # ...and doesn't reach what the worker delivers
    assert len(warnings) == 1 and "exported this file on the CPU instead" in warnings[0]


def test_export_fallback_converts_into_a_fresh_buffer():
    image = _positive()
    warnings = []
    failure = DeviceFailure(type_name="OutOfMemoryError", message="", out_of_memory=True)
    result = export_fallback("pos.tif", image, failure, warnings.append)
    _assert_same_bits(result, srgb_8bit_from_acescg(image))
    assert warnings == ["pos.tif: out of GPU memory — exported this file on the CPU instead"]


def test_a_base_exception_in_a_request_is_a_reply_and_the_service_keeps_serving(tmp_path, fake_gpu, monkeypatch):
    real = halide.processing.negative_to_positive

    def aborting(band, *args, **kwargs):
        if isinstance(band, FakeDeviceArray):  # only in the service; the CPU reference runs as usual
            raise _Abort("stray interrupt")
        return real(band, *args, **kwargs)

    monkeypatch.setattr(halide.processing, "negative_to_positive", aborting)
    scan = _write_scan(tmp_path / "neg.tif")
    with gpu_service._serving(fake_gpu) as address, ServiceClient(address, timeout=10) as client:
        with pytest.raises(DeviceJobFailed) as failed:
            _develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0)
        assert failed.value.failure.type_name == "_Abort" and failed.value.failure.host_touched
        monkeypatch.setattr(halide.processing, "negative_to_positive", real)
        _check_develop(*_develop_through(client, scan, Stage.FULL, ToneCurveParams(), PROFILE, 1.0))


def test_a_service_that_cannot_start_raises_in_the_parent():
    start = time.monotonic()
    with pytest.raises(ServiceUnavailable, match="cudaErrorInsufficientDriver"):
        with running_service("gpu", initializer=_fail_at_startup):
            pytest.fail("should not have started")
    assert time.monotonic() - start < 60


def test_a_cancel_callback_bounds_a_hung_startup(tmp_path):
    """Finding 4, final whole-branch review: closing the picker's contact sheet window while the
    service is still starting must not wait out the full _STARTUP_TIMEOUT (120 s) on a hung driver
    probe. `cancel` is polled every _STARTUP_POLL_INTERVAL while running_service waits for the
    service to report ready; setting it makes the wait give up promptly, and `_stop`'s own bounded
    termination (it SIGTERMs a process stuck outside its stop-signal wait) is what actually ends the
    still-starting child, not a graceful reply from it."""
    arrived = tmp_path / "arrived"
    cancelled = threading.Event()
    outcome = []

    def worker():
        try:
            with running_service("cpu", initializer=partial(_slow_startup, str(arrived)), cancel=cancelled.is_set):
                outcome.append("started")
        except ServiceUnavailable as exc:
            outcome.append(str(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_for(arrived)  # the child is now stuck inside its startup hook, before ever reporting ready
    started = time.monotonic()
    cancelled.set()
    thread.join(15)
    assert not thread.is_alive()
    assert time.monotonic() - started < 15  # bounded, not the 60 s sleep or the 120 s startup timeout
    assert outcome == ["the GPU service's startup was cancelled"]


def test_no_cancel_callback_behaves_as_before(tmp_path):
    """cancel=None (every production caller except the GUI) must still time out the same way it
    always did — a plain, uncancellable startup wait."""
    scan_start = time.monotonic()
    with running_service("cpu") as address:
        assert address.kind == "cpu"
    assert time.monotonic() - scan_start < 10


_PARENT_SCRIPT = """
import sys
import numpy as np
from halide.core.types import DensityProfile, Stage, ToneCurveParams
from halide.gpu_service import ServiceClient, running_service
from halide.processing import DevelopRequest, develop_request
from halide.shared_frames import new_frame
from tests.unit import _fake_device

image = np.random.default_rng(1).uniform(0.01, 0.3, size=(16, 24, 3)).astype(np.float32)
request = DevelopRequest(None, 1.0, None, Stage.FULL, ToneCurveParams())
expected = image.copy()
develop_request(expected, request)
with running_service("gpu", initializer=_fake_device.install_as_gpu) as address, ServiceClient(address) as client:
    with new_frame(image.shape, image.dtype) as frame:
        frame.array[...] = image
        client.develop(frame, request)
        same = np.array_equal(frame.array.view(np.uint8), expected.view(np.uint8))
print("developed-identically" if same else "differs", "cupy" in sys.modules)
"""


def test_gpu_service_keeps_cupy_out_of_the_parent():
    """The child resolves (here: fakes) the GPU itself; the parent only ever sees an address. Run in
    a fresh interpreter: CuPy may be installed, and other tests in this session (tests/gpu's
    import-time device probe) import it into the pytest process itself."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", _PARENT_SCRIPT], cwd=root, capture_output=True, text=True,
                            timeout=120, env={**os.environ, "PYTHONPATH": str(root)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["developed-identically", "False"]
    assert result.stderr == ""
