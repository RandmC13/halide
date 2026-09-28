"""One GPU process shared by CPU-only batch workers — Part B of docs/plans/gpu-batch-throughput.md,
for the reasons in docs/investigations/gpu-batch-throughput.md: a GPU batch was limited by host RAM,
because every GPU worker carried its own CUDA context and NVIDIA's libraries (~0.8 GiB each), while
the GPU itself sat idle ~90% of the time. Here one service process holds the only CUDA context and
develops frames one at a time; the workers do the file work (decode, compress, write, exiftool) and
hand frames over through shared memory (halide/shared_frames.py) — never through the pipe.

How a request goes:
  - the worker decodes a scan straight into a `SharedFrame` and sends its descriptor (name, shape,
    dtype) plus a picklable request (processing.DevelopRequest / PrintRequest / ExportRequest);
  - the service attaches to the segment, uploads it, runs exactly the device code the in-process
    GPU path runs (processing.run_device_job / run_device_export), downloads into the same segment,
    and replies with the ResolvedTone/profile;
  - an error inside a request (out of GPU memory, a driver error) comes back as a reply carrying a
    processing.DeviceFailure; the client raises processing.DeviceJobFailed, and the worker redoes the
    frame on the CPU exactly as today (processing.fall_back_to_cpu), including re-reading the scan
    when the failure came during the download (an export: processing.export_fallback, into a fresh
    output buffer);
  - a dead, unreachable or stuck service raises ServiceUnavailable instead — promptly for a dead one
    (the connection breaks), after REQUEST_TIMEOUT for a stuck one — and the worker develops on the
    CPU from then on.

Processes and threads: the service is started with the **spawn** start method, so CUDA is only
ever initialised inside it — never in the parent or in a worker pool's forkserver (a forked CUDA
context is unusable). It resolves its own device (`halide.device.resolve_device`), so this module
imports nothing heavy (no numpy, no CuPy) in the parent at all. Inside the service, each client
connection is read on its own thread, and one compute thread runs requests one at a time: a frame
already fills the GPU, so running two at once would only make both slower.

Transport: `multiprocessing.connection` — a Unix socket on Linux, a named pipe on Windows — with a
random authkey. Each worker holds one connection (one ServiceClient).
"""

from __future__ import annotations

import functools
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import connection, get_context

from halide.interrupts import children_ignore_terminal_signals, ignore_terminal_signals

# How long a worker waits for one reply before treating the service as stuck. Generous: a frame
# takes ~0.14 s of GPU time, but the queue may hold one frame per worker ahead of it, and a first
# request can pay CuPy's kernel compilation. Only a genuinely hung service ever reaches it.
REQUEST_TIMEOUT = 300.0

# How long the service gets to start (spawn a Python, import numpy/colour/CuPy, make a CUDA
# context — a few seconds cold) before running_service gives up on it.
_STARTUP_TIMEOUT = 120.0

# How often running_service checks its optional `cancel` callback while waiting for the service to
# report ready. Short enough that a caller closing mid-startup (e.g. the picker's contact sheet
# window) isn't stuck waiting out the full _STARTUP_TIMEOUT on a hung driver.
_STARTUP_POLL_INTERVAL = 0.2

# How long stopping the service waits for its own wake-up connection to the accept loop.
_WAKE_TIMEOUT = 2.0

# How long the service gets to exit on its own when asked to stop, before it's terminated. It
# stops without finishing requests in flight, so this only has to cover interpreter shutdown.
_STOP_GRACE = 3.0


class ServiceUnavailable(RuntimeError):
    """The service is dead, unreachable, stuck, or never started: the caller develops on the CPU,
    and a ServiceClient that raised this stays dead (it never reconnects).

    `host_touched`: whether the service may have written, or may still write, into the request's
    shared buffers (the frame for develop/print, `out` for export) — True once the request was sent
    (it may have died mid-download, or be stuck and write later), False when it never got that far.
    See `failure`; the fallback helpers never reuse such a buffer."""

    def __init__(self, message: str, host_touched: bool = False):
        super().__init__(message)
        self.host_touched = host_touched

    @property
    def failure(self):
        """This as a processing.DeviceFailure, for processing.fall_back_to_cpu — so a worker
        handles a dead service and a failed request with the same two lines."""
        from halide.processing import DeviceFailure

        return DeviceFailure(type_name=type(self).__name__, message=str(self), out_of_memory=False,
                             host_touched=self.host_touched)


@dataclass(frozen=True)
class ServiceAddress:
    """Where a running service listens: everything a worker process needs to connect, as plain
    picklable data (B3 hands it to pool workers). `pid` is the service process, for diagnostics
    and tests; nothing in a worker needs it. `kind` is the device it develops on ("gpu"; "cpu" only
    in tests) — what a frame it developed records as its provenance "device"."""

    address: str | tuple
    family: str
    authkey: bytes
    pid: int
    kind: str = "gpu"


@dataclass(frozen=True)
class DevelopReply:
    """processing.develop_request's result: the tone actually used (None for Stage.DENSITY_ONLY)
    and the density profile actually used (the per-frame auto estimate when none was given).

    `warnings`: any messages the job produced (F05's non-finite-pixel notice, so far) — the service
    runs in its own process, with nothing of its own to print them to usefully, so they travel back
    here instead and the client's caller re-emits them through its own on_warning/BatchResult path,
    exactly as the in-process CPU/GPU paths do directly."""

    resolved: object  # ResolvedTone | None — typed loosely to keep processing out of this import
    profile: object  # DensityProfile
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrintReply:
    resolved: object  # ResolvedTone
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------------------
# The client (a worker's side)
# ---------------------------------------------------------------------------------------------


class ServiceClient:
    """One worker's connection to the service. Connects on first use; used by one thread at a time
    (guarded by a lock anyway). Every method either returns the reply, raises
    processing.DeviceJobFailed (the request failed on the device — the service carries on), or
    raises ServiceUnavailable (the service is gone — and so is this client, for good).

    `frame` stays the caller's shared-memory segment throughout: the service writes the developed
    frame back into it, so after develop/print_ the caller's `frame.array` holds the result."""

    def __init__(self, address: ServiceAddress, *, timeout: float = REQUEST_TIMEOUT) -> None:
        self._address = address
        self._timeout = timeout
        self._conn = None
        self._dead: str | None = None  # why, once this client has given up on the service
        self._lock = threading.Lock()

    @property
    def device_kind(self) -> str:
        """What the service develops on — see ServiceAddress.kind."""
        return self._address.kind

    def develop(self, frame, request, name: str | None = None) -> DevelopReply:
        """`name` (typically the input path) is only for F05's non-finite-pixel warning message —
        the service has no other way to know it, since `request` carries none of process_scan's own
        arguments."""
        (resolved, profile), warnings = self._request(("develop", frame.descriptor(), request, name))
        return DevelopReply(resolved=resolved, profile=profile, warnings=tuple(warnings))

    def print_(self, frame, request, name: str | None = None) -> PrintReply:
        resolved, warnings = self._request(("print", frame.descriptor(), request, name))
        return PrintReply(resolved=resolved, warnings=tuple(warnings))

    def export(self, frame, out, request) -> None:
        """`frame` is only ever read; the service writes `out`. After a failure `out` can't be
        trusted — partly written, and a service that stopped answering may still be writing into
        it — so the CPU fallback goes into a buffer the worker owns (processing.export_fallback)."""
        self._request(("export", frame.descriptor(), out.descriptor(), request))

    def close(self) -> None:
        with self._lock:
            self._drop_connection()

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _request(self, message):
        with self._lock:
            if self._dead is not None:
                raise ServiceUnavailable(self._dead)
            conn = self._connect()
            try:
                conn.send(message)
                # poll, not a bare recv: a live but stuck service must not hang a worker forever.
                if not conn.poll(self._timeout):
                    raise TimeoutError(f"no reply within {self._timeout:g} s")
                status, payload = conn.recv()
            except (OSError, EOFError, TimeoutError) as exc:
                # It may have died mid-download, or be stuck and still write later: none of this
                # request's shared buffers can be trusted any more.
                raise self._give_up(f"the GPU service stopped responding ({_describe(exc)})",
                                    host_touched=True) from exc
            except BaseException as exc:
                # F06: anything else here — Ctrl-C (KeyboardInterrupt) during poll/recv, above all —
                # leaves a reply still owed on this connection, so the next request would read this
                # one's reply and its own frame would be written out undeveloped. Any doubt means
                # dead, as for a timeout; the original exception carries on (the caller is
                # cancelling, not falling back).
                self._give_up(f"a request to the GPU service was interrupted ({_describe(exc)})",
                              host_touched=True)
                raise
        if status == "ok":
            return payload
        if status == "input_error":
            # R8: bad input data (e.g. F05's >1% non-finite pixels), not a device/service problem —
            # `payload` is the original exception (ScanColorError, picklable — a plain message, no
            # CuPy/device state), raised here exactly as the CPU path would raise it itself: no
            # DeviceJobFailed, no "GPU failed" wording, no CPU retry from the caller.
            raise payload
        from halide.processing import DeviceJobFailed

        raise DeviceJobFailed(payload)

    def _connect(self):
        if self._conn is None:
            address = self._address
            try:
                self._conn = _connect(address, self._timeout)
            except (OSError, EOFError, TimeoutError, connection.AuthenticationError) as exc:
                raise self._give_up(f"the GPU service can't be reached ({_describe(exc)})",
                                    host_touched=False) from exc
        return self._conn

    def _give_up(self, reason: str, host_touched: bool) -> ServiceUnavailable:
        self._dead = reason
        self._drop_connection()
        return ServiceUnavailable(reason, host_touched=host_touched)

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None


class _Deadline:
    """A connection whose recv_bytes gives up (TimeoutError) once `deadline` has passed — so the
    stdlib's authkey handshake, which only ever calls recv_bytes/send_bytes on the connection it's
    given, can't block forever on a service that accepted the connection but never answers (a
    frozen process: the kernel completes a Unix socket's connect without it)."""

    def __init__(self, conn, deadline: float):
        self._conn = conn
        self._deadline = deadline

    def recv_bytes(self, maxlength=None):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0 or not self._conn.poll(remaining):
            raise TimeoutError("no answer to the connection handshake")
        return self._conn.recv_bytes(maxlength)

    def send_bytes(self, buf):
        # A handshake message is ~100 bytes: it always fits the socket/pipe buffer, never blocks.
        self._conn.send_bytes(buf)


def _connect(address: ServiceAddress, timeout: float):
    """connection.Client(address, authkey=...), with the handshake under one `timeout`. The
    connect itself is bounded already: a Unix socket's connect doesn't wait for the service
    process, and Windows' PipeClient retries for at most ~20 s. The handshake is the stdlib's own
    (answer_challenge then deliver_challenge, as Client does), run against a _Deadline."""
    conn = connection.Client(address.address, family=address.family)  # no authkey: no handshake yet
    try:
        bounded = _Deadline(conn, time.monotonic() + timeout)
        connection.answer_challenge(bounded, address.authkey)
        connection.deliver_challenge(bounded, address.authkey)
    except BaseException:
        conn.close()
        raise
    return conn


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# ---------------------------------------------------------------------------------------------
# The service (runs inside the spawned process — or, in tests, on threads of the test process)
# ---------------------------------------------------------------------------------------------


class _Server:
    """Accepts connections, reads each on its own thread, and runs every request on one compute
    thread. Replies are sent by the connection's own thread, so each connection is only ever used
    by one thread."""

    def __init__(self, listener: connection.Listener, address: ServiceAddress, device) -> None:
        self._listener = listener
        self._address = address
        self._device = device
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._closing = threading.Event()
        self._accepting = threading.Thread(target=self._accept_loop, name="halide-service-accept", daemon=True)
        self._computing = threading.Thread(target=self._compute_loop, name="halide-service-compute", daemon=True)

    def start(self) -> None:
        self._accepting.start()
        self._computing.start()

    def stop(self) -> None:
        """Stop accepting and computing. Doesn't wait for a request in flight (the caller then
        exits the process, or is a test that has already closed its clients)."""
        self._closing.set()
        # Wake the accept loop with one last connection of our own: closing a listener doesn't
        # reliably interrupt an accept() already blocked on it (not on Linux, not on Windows pipes).
        try:
            _connect(self._address, _WAKE_TIMEOUT).close()
        except (OSError, EOFError, TimeoutError, connection.AuthenticationError):
            pass
        self._accepting.join(5)
        self._listener.close()
        self._jobs.put(None)
        self._computing.join(1)

    def _accept_loop(self) -> None:
        while not self._closing.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError, connection.AuthenticationError):
                if self._closing.is_set():
                    return
                continue  # a client that failed the handshake or hung up during it
            if self._closing.is_set():
                conn.close()
                return
            threading.Thread(target=self._serve_connection, args=(conn,), name="halide-service-conn",
                             daemon=True).start()

    def _serve_connection(self, conn) -> None:
        try:
            while True:
                message = conn.recv()
                done = threading.Event()
                slot: list = []
                self._jobs.put((message, slot, done))
                done.wait()
                conn.send(slot[0])
        except (OSError, EOFError):
            pass  # the worker went away (normally: its ServiceClient closed)
        finally:
            conn.close()

    def _compute_loop(self) -> None:
        while (job := self._jobs.get()) is not None:
            message, slot, done = job
            try:
                reply = self._handle(message)
            except BaseException as exc:  # noqa: BLE001
                # _handle already turns every Exception into a reply; this is the rest (a stray
                # SystemExit/KeyboardInterrupt from request code — SIGINT itself is ignored in the
                # service). Answered like any failure, and the loop carries on: dying here would
                # strand every connection thread waiting on its reply. Not re-raised — genuine
                # interpreter shutdown doesn't come through here (daemon threads are just stopped).
                # host_touched: where it struck isn't known, so the frame isn't trusted.
                from halide.processing import DeviceFailure

                reply = ("failed", DeviceFailure.from_exception(exc, host_touched=True))
            slot.append(reply)
            done.set()

    def _handle(self, message):
        from halide import processing
        from halide.processing import DeviceFailure, DeviceJobFailed, ScanColorError
        from halide.shared_frames import attach_frame

        try:
            kind = message[0]
            if kind == "develop" or kind == "print":
                _, descriptor, request, name = message
                job = processing.develop_request if kind == "develop" else processing.print_request
                with attach_frame(*descriptor) as host:
                    return "ok", self._run(host, job, request, name)
            if kind == "export":
                _, descriptor, out_descriptor, request = message
                with attach_frame(*descriptor) as host, attach_frame(*out_descriptor) as out:
                    self._run_export(host, out, request)
                return "ok", None
            raise ValueError(f"unknown request {kind!r}")
        except ScanColorError as exc:
            # R8: bad input data (e.g. F05's >1% non-finite pixels), not a device/service problem —
            # sent back as-is (see ServiceClient._request's "input_error" handling) so the worker
            # raises exactly the error the CPU path would: no DeviceJobFailed wrapping, no "GPU
            # failed"/service wording, and no CPU retry from the caller.
            return "input_error", exc
        except DeviceJobFailed as failed:
            return "failed", failed.failure
        except Exception as exc:  # noqa: BLE001 — a bad request fails that request, never the service
            return "failed", DeviceFailure.from_exception(exc)

    def _run(self, host, job, request, name):
        """Run `job` (develop_request/print_request), collecting any warnings it produces (F05's
        non-finite-pixel notice, so far) instead of letting them print inside this process — the
        service has nothing of its own a photographer would see to print them to; the client's
        caller re-emits them through its own on_warning/BatchResult path. Returns (job's result,
        warnings)."""
        from halide import processing

        collected: list[str] = []
        bound_job = functools.partial(job, name=name, on_warning=collected.append)
        if self._device.kind == "gpu":
            return processing.run_device_job(host, bound_job, request), collected
        # A "cpu" service works in place on the shared frame itself, so any failure may have
        # left it half-developed.
        try:
            return bound_job(host, request), collected
        except processing.ScanColorError:
            raise  # R8: bad input data, not a device problem — see _handle
        except BaseException as exc:  # noqa: BLE001 — see _compute_loop
            raise processing.DeviceJobFailed(processing.DeviceFailure.from_exception(exc, host_touched=True)) from exc

    def _run_export(self, host, out, request) -> None:
        from halide import processing

        if self._device.kind == "gpu":
            processing.run_device_export(host, out, request)
        else:
            processing.export_request(host, out, request)


def _new_address() -> tuple[str | tuple, str, bytes]:
    """A fresh listening address and authkey. running_service makes it in the parent, not the
    service: a Unix socket's path then lives in the parent's own temporary folder (removed when the
    parent exits), and the parent can remove the socket file itself after a service that was killed
    couldn't."""
    family = connection.default_family
    return connection.arbitrary_address(family), family, os.urandom(32)


@contextmanager
def _serving(device, where: tuple[str | tuple, str, bytes] | None = None) -> Iterator[ServiceAddress]:
    """Serve on threads of *this* process with an already-resolved `device` — how the spawned
    service runs, and how tests run the service against the strict fake device in-process.
    `where` is (address, family, authkey), made by _new_address; a fresh one if not given."""
    address, family, authkey = where if where is not None else _new_address()
    listener = connection.Listener(address, family=family, authkey=authkey)
    service_address = ServiceAddress(address=listener.address, family=family, authkey=authkey, pid=os.getpid(),
                                     kind=device.kind)
    server = _Server(listener, service_address, device)
    server.start()
    try:
        yield service_address
    finally:
        server.stop()


def _service_main(device_kind: str, where: tuple[str | tuple, str, bytes],
                  initializer: Callable[[], None] | None, status) -> None:
    """The spawned service process. Reports ("ready", address) or ("error", text) on `status`,
    then serves until the parent says stop — or goes away (EOF), so a crashed parent never leaves
    a service holding the GPU."""
    # Ctrl-C and SIGHUP (the terminal closing) reach the whole process group; the parent decides
    # when the service stops (halide.interrupts). Already ignored from start-up when the parent
    # launched it inside children_ignore_terminal_signals; this covers any other launch.
    ignore_terminal_signals()
    try:
        if initializer is not None:
            initializer()
        from halide import device as _device
        from halide import processing  # noqa: F401 — imported now, so the first request doesn't pay for it

        device = _device.resolve_device(device_kind)
        if device.kind != device_kind:
            raise RuntimeError(f"asked for a {device_kind} service, got a {device.kind} device")
        serving = _serving(device, where)
        address = serving.__enter__()
    except BaseException as exc:  # noqa: BLE001 — reported to the parent, which falls back
        # One line, not a traceback: it ends up on the run sheet as the reason for falling back.
        status.send(("error", _describe(exc)))
        return
    status.send(("ready", address))
    try:
        status.recv()
    except (OSError, EOFError):
        pass
    serving.__exit__(None, None, None)


def _wait_for_ready(status, cancel: Callable[[], bool] | None) -> ServiceAddress:
    """Block until the service reports ready (or fails), polling in short slices so an optional
    `cancel` callback is checked regularly instead of one blocking `poll(_STARTUP_TIMEOUT)` — a
    caller that wants to give up early (the picker's contact sheet window, closed while a hung
    driver is still being probed) would otherwise be stuck for the full _STARTUP_TIMEOUT. Raises
    ServiceUnavailable on a timeout, a cancellation, or the service itself failing to start; the
    caller's `finally` still runs `_stop` either way, which is what actually terminates a service
    stuck inside a slow (or hung) `initializer`/device probe."""
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while True:
        if cancel is not None and cancel():
            raise ServiceUnavailable("the GPU service's startup was cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceUnavailable(f"the GPU service didn't start within {_STARTUP_TIMEOUT:g} s")
        try:
            if status.poll(min(_STARTUP_POLL_INTERVAL, remaining)):
                break
        except (OSError, EOFError) as exc:
            raise ServiceUnavailable(f"the GPU service exited while starting ({_describe(exc)})") from exc
    try:
        state, detail = status.recv()
    except (OSError, EOFError) as exc:
        raise ServiceUnavailable(f"the GPU service exited while starting ({_describe(exc)})") from exc
    if state != "ready":
        raise ServiceUnavailable(f"the GPU service couldn't start: {detail}")
    return detail


@contextmanager
def running_service(device_kind: str, *, initializer: Callable[[], None] | None = None,
                    cancel: Callable[[], bool] | None = None) -> Iterator[ServiceAddress]:
    """Start the service process (`device_kind` "gpu" normally; "cpu" runs the numpy path, for
    tests), yield its address, and stop it on the way out — even with requests in flight: those
    clients get ServiceUnavailable. Raises ServiceUnavailable if it can't start (CuPy unusable, a
    driver problem...), so the caller can fall back to per-worker GPU mode.

    `initializer`: a picklable, module-level function the child runs before anything else — a test
    hook (tests/unit/_fake_device.py's `install_as_gpu` makes the child's "GPU" the strict fake).
    Production code never passes one.

    `cancel`: an optional no-argument callable polled while the service is starting
    (_wait_for_ready) — True gives up on startup promptly (bounded by _STARTUP_POLL_INTERVAL)
    instead of waiting out the full _STARTUP_TIMEOUT. `_stop` below still terminates the spawned
    process either way. Used by the GUI (gui/proof_window.py) so closing the window while the
    service is still starting doesn't hang the close on a slow or hung driver probe."""
    context = get_context("spawn")
    where = _new_address()
    parent_status, child_status = context.Pipe()
    process = context.Process(target=_service_main, args=(device_kind, where, initializer, child_status),
                              name="halide-gpu-service", daemon=True)
    with children_ignore_terminal_signals():
        process.start()
    child_status.close()
    try:
        address = _wait_for_ready(parent_status, cancel)
        yield address
    finally:
        _stop(process, parent_status, where)


def _stop(process, status, where: tuple[str | tuple, str, bytes]) -> None:
    try:
        status.send("stop")
    except (OSError, EOFError, ValueError):
        pass  # already gone
    process.join(_STOP_GRACE)
    if process.is_alive():
        process.terminate()
        process.join(_STOP_GRACE)
    if process.is_alive():
        process.kill()
        process.join()
    status.close()
    process.close()
    # A killed service can't remove its own Unix socket file; do it for it.
    address, family, _ = where
    if family == "AF_UNIX" and isinstance(address, str):
        try:
            os.unlink(address)
        except OSError:
            pass
