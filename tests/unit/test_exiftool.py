"""ExifToolSession and copy_exif_metadata's use of it, against a fake exiftool.

The fake is a small Python script named `exiftool` on a PATH of its own. It speaks the part of
exiftool's `-stay_open True -@ -` protocol halide uses (argument lines up to `-executeNNN`, then a
reply ending `{readyNNN}`), and it also runs one-shot like `exiftool ARGS...`, so the fallback path
goes through the same fake. Every command it receives is logged as one JSON line (pid, mode, args)
to the file in $FAKE_EXIFTOOL_LOG. Behaviour switches (per fake *process*, counting its commands
from 1): FAIL_ON=<n> fails its nth command the way exiftool does; DIE_AFTER=<n> exits abruptly on
command n+1 (0: on the first); HANG=1 never answers; NO_STATUS=1 ignores -echo3, like an exiftool
older than 12.10.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import pytest

from halide.io import exiftool
from halide.io.tiff import copy_exif_metadata

# Linux only: the kept-open session is Linux-only (exiftool._KEPT_OPEN_SUPPORTED), and the fake is a
# shebang script. The non-Linux behaviour is tested here by switching that flag off.
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the kept-open session is Linux-only")

FAKE = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, time

    LOG = os.environ["FAKE_EXIFTOOL_LOG"]
    FAIL_ON = int(os.environ.get("FAIL_ON", "0"))
    DIE_AFTER = os.environ.get("DIE_AFTER")
    HANG = os.environ.get("HANG") == "1"
    NO_STATUS = os.environ.get("NO_STATUS") == "1"

    def log(mode, args):
        with open(LOG, "a") as f:
            f.write(json.dumps({{"pid": os.getpid(), "mode": mode, "args": args}}) + "\\n")

    def write_file(args, n):
        # The file argument is the last one, as in halide's command. Like the real exiftool, refuse
        # to overwrite a temporary file that a killed exiftool left behind.
        dest = args[-1]
        tmp = dest + "_exiftool_tmp"
        if os.path.exists(tmp):
            return ["Error: Temporary file already exists: " + tmp, "    0 image files updated"], 1
        if n == FAIL_ON:
            return ["Error: fake failure - " + dest, "    0 image files updated",
                    "    1 files weren't updated due to errors"], 1
        return ["    1 image files updated"], 0

    argv = sys.argv[1:]
    if "-stay_open" not in argv:
        log("oneshot", argv)
        lines, status = write_file(argv, 1 if FAIL_ON == 1 else -1)
        print("\\n".join(lines))
        sys.exit(status)

    common = argv[argv.index("-common_args") + 1:] if "-common_args" in argv else []
    log("start", argv)
    n = 0
    args = []
    for line in sys.stdin:
        arg = line.rstrip("\\n")
        if arg.startswith("-execute"):
            n += 1
            log("session", args)
            if DIE_AFTER is not None and n > int(DIE_AFTER):
                os._exit(3)
            if HANG:
                open(args[-1] + "_exiftool_tmp", "w").close()
                time.sleep(3600)
            lines, status = write_file(args, n)
            full = args + common
            if "-echo3" in full and not NO_STATUS:
                lines.append(full[full.index("-echo3") + 1].replace("${{status}}", str(status)))
            print("\\n".join(lines))
            print("{{ready" + arg[len("-execute"):] + "}}", flush=True)
            args = []
        elif args[-1:] == ["-stay_open"] and arg == "False":
            log("stop", [])
            sys.exit(0)
        else:
            args.append(arg)
    # Like the real exiftool: end of input doesn't end -stay_open, it keeps polling for more.
    while True:
        time.sleep(0.01)
    """
)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake exiftool first (and alone) on PATH; returns a reader for its command log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "exiftool"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(0o755)
    log = tmp_path / "exiftool.log"
    log.touch()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_EXIFTOOL_LOG", str(log))
    for name in ("FAIL_ON", "DIE_AFTER", "HANG", "NO_STATUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(exiftool, "DEFAULT_TIMEOUT", 10.0)
    exiftool._reset()
    yield lambda: [json.loads(line) for line in log.read_text().splitlines()]
    exiftool._reset()


def _files(tmp_path, count=1, name="out{}.tif"):
    source = tmp_path / "scan.tif"
    source.write_bytes(b"source")
    dests = []
    for i in range(count):
        dest = tmp_path / name.format(i)
        dest.write_bytes(b"output")
        dests.append(dest)
    return source, dests


def _expected_args(source, dest, drop_icc):
    args = ["-TagsFromFile", str(source), "-all:all", "--ExifImageWidth", "--ExifImageHeight"]
    if drop_icc:
        args.append("--icc_profile")
    return args + ["-overwrite_original", str(dest)]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it has exited all the same.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_many_copies_share_one_exiftool_process(fake, tmp_path):
    source, dests = _files(tmp_path, 5)
    for dest in dests:
        assert copy_exif_metadata(source, dest, drop_icc=True) is True
    commands = [entry for entry in fake() if entry["mode"] == "session"]
    assert len(commands) == 5
    assert len({entry["pid"] for entry in commands}) == 1
    assert sum(entry["mode"] == "start" for entry in fake()) == 1


@pytest.mark.parametrize("drop_icc", [True, False])
def test_command_is_exactly_todays_argument_list(fake, tmp_path, drop_icc):
    source, (dest,) = _files(tmp_path)
    copy_exif_metadata(source, dest, drop_icc=drop_icc)
    (command,) = [entry for entry in fake() if entry["mode"] == "session"]
    assert command["args"] == _expected_args(source, dest, drop_icc)


def test_session_is_started_with_charset_and_status_as_common_arguments(fake, tmp_path):
    source, (dest,) = _files(tmp_path)
    copy_exif_metadata(source, dest)
    (start,) = [entry for entry in fake() if entry["mode"] == "start"]
    assert start["args"][:4] == ["-stay_open", "True", "-@", "-"]
    common = start["args"][start["args"].index("-common_args") + 1:]
    # -charset filename=utf8 on every platform: it doesn't change a byte on Linux (checked against
    # the real exiftool, tests/integration/test_exiftool_real.py) and is needed on Windows.
    assert common == ["-charset", "filename=utf8", "-echo3", "{status=${status}}"]


def test_exiftools_own_failure_raises_and_propagates(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIL_ON", "2")
    source, dests = _files(tmp_path, 3)
    assert copy_exif_metadata(source, dests[0]) is True
    with pytest.raises(exiftool.ExifToolError, match="fake failure"):
        copy_exif_metadata(source, dests[1])
    # A failed file is that file's failure, not the session's: the same process carries on.
    assert copy_exif_metadata(source, dests[2]) is True
    commands = [entry for entry in fake() if entry["mode"] == "session"]
    assert len({entry["pid"] for entry in commands}) == 1
    assert not any(entry["mode"] == "oneshot" for entry in fake())


def test_failure_is_recognised_without_the_status_echo(fake, tmp_path, monkeypatch):
    # exiftool before 12.10 prints -echo3's ${status} literally; fall back to reading its report.
    monkeypatch.setenv("NO_STATUS", "1")
    monkeypatch.setenv("FAIL_ON", "2")
    source, dests = _files(tmp_path, 2)
    assert copy_exif_metadata(source, dests[0]) is True
    with pytest.raises(exiftool.ExifToolError):
        copy_exif_metadata(source, dests[1])


def test_session_run_raises_on_failure_directly(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIL_ON", "1")
    source, (dest,) = _files(tmp_path)
    session = exiftool.session()
    assert session is not None
    with pytest.raises(exiftool.ExifToolError):
        session.run(_expected_args(source, dest, False))


def test_oneshot_failure_still_raises_like_today(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIL_ON", "1")
    source, (dest,) = _files(tmp_path, name="odd\nname{}.tif")
    with pytest.raises(subprocess.CalledProcessError):
        copy_exif_metadata(source, dest)


def test_a_session_that_died_is_restarted_once(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("DIE_AFTER", "1")
    source, dests = _files(tmp_path, 3)
    for dest in dests:
        assert copy_exif_metadata(source, dest) is True
    log = fake()
    starts = [entry["pid"] for entry in log if entry["mode"] == "start"]
    # First process: copy 1, dies on copy 2. Second: copy 2 again, dies on copy 3. Third: copy 3.
    assert len(starts) == 3
    assert not any(entry["mode"] == "oneshot" for entry in log)
    assert not any(_alive(pid) for pid in starts[:-1])


def test_a_restart_that_also_dies_falls_back_to_oneshot_for_good(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("DIE_AFTER", "0")
    source, dests = _files(tmp_path, 3)
    for dest in dests:
        assert copy_exif_metadata(source, dest) is True
    log = fake()
    # One session, one restart, then one-shot for every copy (no restart per frame).
    assert sum(entry["mode"] == "start" for entry in log) == 2
    oneshots = [entry for entry in log if entry["mode"] == "oneshot"]
    assert [entry["args"] for entry in oneshots] == [_expected_args(source, d, False) for d in dests]


def test_a_hung_session_times_out_and_the_copy_falls_back_to_oneshot(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("HANG", "1")
    monkeypatch.setattr(exiftool, "DEFAULT_TIMEOUT", 0.5)
    source, (dest,) = _files(tmp_path)
    started = time.monotonic()
    assert copy_exif_metadata(source, dest) is True
    assert time.monotonic() - started < 5
    log = fake()
    starts = [entry["pid"] for entry in log if entry["mode"] == "start"]
    assert len(starts) == 2  # the hung one and its restart, which hung too
    assert [entry["mode"] for entry in log][-1] == "oneshot"
    # Hung processes are killed, and the temporary file each left behind is removed (the one-shot
    # exiftool would otherwise refuse the file, as the real one does).
    assert not any(_alive(pid) for pid in starts)
    assert not Path(str(dest) + "_exiftool_tmp").exists()


def test_a_path_with_a_newline_uses_oneshot(fake, tmp_path):
    source, (dest,) = _files(tmp_path, name="odd\nname{}.tif")
    assert copy_exif_metadata(source, dest) is True
    log = fake()
    assert [entry["mode"] for entry in log] == ["oneshot"]
    assert log[0]["args"] == _expected_args(source, dest, False)


@pytest.mark.parametrize("name", [" leading{}.tif", "trailing{}.tif ", "#hash{}.tif"])
def test_arguments_an_argument_file_would_alter_use_oneshot(fake, tmp_path, monkeypatch, name):
    # exiftool strips leading white space and the line end from argument-file lines, and treats a
    # line starting with '#' as a comment. Relative paths make these reachable.
    monkeypatch.chdir(tmp_path)
    source, (dest,) = _files(tmp_path, name=name)
    relative = Path(dest.name)
    assert copy_exif_metadata(source, relative) is True
    assert [entry["mode"] for entry in fake()] == ["oneshot"]


def test_no_exiftool_on_path_returns_false_and_starts_nothing(fake, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    source, (dest,) = _files(tmp_path)
    assert copy_exif_metadata(source, dest) is False
    assert exiftool.session() is None
    assert fake() == []


def test_close_ends_the_process(fake, tmp_path):
    source, (dest,) = _files(tmp_path)
    copy_exif_metadata(source, dest)
    session = exiftool.session()
    (start,) = [entry for entry in fake() if entry["mode"] == "start"]
    session.close()
    assert fake()[-1]["mode"] == "stop"
    assert not _alive(start["pid"])
    session.close()  # idempotent


def test_a_new_process_gets_its_own_session(fake, tmp_path, monkeypatch):
    # A forked child inherits the module global; it must start its own exiftool, never write to
    # (or close) its parent's.
    source, dests = _files(tmp_path, 2)
    copy_exif_metadata(source, dests[0])
    parent = exiftool.session()
    monkeypatch.setattr(exiftool, "_session_pid", -1)
    child = exiftool.session()
    assert child is not parent
    copy_exif_metadata(source, dests[1])
    starts = [entry["pid"] for entry in fake() if entry["mode"] == "start"]
    assert len(starts) == 2
    assert _alive(starts[0])  # the "parent's" is untouched
    parent.close()


def test_a_real_forked_child_starts_its_own_session_and_leaves_the_parents_alone(fake, tmp_path):
    source, dests = _files(tmp_path, 3)
    assert copy_exif_metadata(source, dests[0])
    parent = exiftool.session()
    parent_exiftool = parent.pid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # "multi-threaded, fork() may deadlock"
        child = os.fork()
    if child == 0:  # the child: copy one file, then exit without atexit, like a killed worker
        code = 1
        try:
            mine = exiftool.session()
            if mine is not parent and copy_exif_metadata(source, dests[1]) and mine.pid != parent_exiftool:
                code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    # The parent's exiftool is still its own: alive, not stopped, and still answering.
    assert exiftool.session() is parent and _alive(parent_exiftool)
    assert copy_exif_metadata(source, dests[2])
    log = fake()
    by_parent = [entry["args"][-1] for entry in log if entry["pid"] == parent_exiftool and entry["mode"] == "session"]
    assert by_parent == [str(dests[0]), str(dests[2])]  # the child never wrote to it
    assert not any(entry["mode"] == "stop" and entry["pid"] == parent_exiftool for entry in log)
    by_child = [entry for entry in log if entry["mode"] == "session" and entry["pid"] != parent_exiftool]
    assert [entry["args"][-1] for entry in by_child] == [str(dests[1])]
    # The child died without atexit; its own exiftool went with it (parent-death signal).
    deadline = time.monotonic() + 5
    while _alive(by_child[0]["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(by_child[0]["pid"])


def test_without_a_parent_death_guarantee_every_copy_is_oneshot(fake, tmp_path, monkeypatch):
    # Off Linux nothing makes a kept-open exiftool die with a terminated worker, so none is started.
    monkeypatch.setattr(exiftool, "_KEPT_OPEN_SUPPORTED", False)
    source, dests = _files(tmp_path, 2)
    for dest in dests:
        assert copy_exif_metadata(source, dest, drop_icc=True) is True
    assert exiftool.session() is None
    log = fake()
    assert [entry["mode"] for entry in log] == ["oneshot", "oneshot"]
    assert [entry["args"] for entry in log] == [_expected_args(source, d, True) for d in dests]


def test_the_session_is_closed_at_interpreter_exit(fake, tmp_path):
    source, (dest,) = _files(tmp_path)
    code = (
        "from halide.io.tiff import copy_exif_metadata\n"
        f"assert copy_exif_metadata({str(source)!r}, {str(dest)!r})\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy(), timeout=30)
    log = fake()
    assert [entry["mode"] for entry in log] == ["start", "session", "stop"]
    deadline = time.monotonic() + 5
    while _alive(log[0]["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(log[0]["pid"])


def test_exiftool_does_not_outlive_a_killed_worker(fake, tmp_path):
    # exiftool's -stay_open never exits on end of input (it polls), so a worker killed without
    # running atexit (the GUI terminates its pool; the OOM killer) would leave it running forever.
    source, (dest,) = _files(tmp_path)
    code = (
        "import os, signal\n"
        "from halide.io.tiff import copy_exif_metadata\n"
        f"assert copy_exif_metadata({str(source)!r}, {str(dest)!r})\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), timeout=30)
    (start,) = [entry for entry in fake() if entry["mode"] == "start"]
    deadline = time.monotonic() + 5
    while _alive(start["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    orphaned = _alive(start["pid"])
    if orphaned:
        os.kill(start["pid"], 9)  # don't leave it polling forever when this fails
    assert not orphaned
