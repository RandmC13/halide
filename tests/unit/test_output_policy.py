"""Unit tests for halide.cli._output_policy — the D-1 overwrite policy (ask in a terminal, refuse
in a script, --overwrite/--skip-existing to choose; output == input is always a hard error)."""

from __future__ import annotations

import pytest

from halide.cli import console
from halide.cli._output_policy import (
    OutputPolicy,
    check_not_input,
    resolve_bulk_jobs,
    resolve_existing,
)


def test_output_equal_to_input_is_refused_even_with_overwrite(tmp_path):
    scan = tmp_path / "a.tif"
    scan.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        check_not_input([(scan, tmp_path / "." / "a.tif")])
    message = str(exc.value.code)
    assert "is the scan itself" in message
    assert "--suffix" in message  # default: the caller is assumed to support --suffix


def test_check_not_input_suggest_suffix_false_ends_with_output_path_hint(tmp_path):
    # invert (and single-file print/export) have no --suffix flag — their hint must not mention it.
    scan = tmp_path / "a.tif"
    scan.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        check_not_input([(scan, tmp_path / "." / "a.tif")], suggest_suffix=False)
    message = str(exc.value.code)
    assert "is the scan itself" in message
    assert "--suffix" not in message
    assert message.endswith("Choose a different output path.")


def test_output_equal_to_input_via_symlink_is_refused(tmp_path):
    scan = tmp_path / "a.tif"
    scan.write_bytes(b"x")
    link = tmp_path / "link.tif"
    link.symlink_to(scan)
    with pytest.raises(SystemExit) as exc:
        check_not_input([(scan, link)])
    assert "is the scan itself" in str(exc.value.code)


def test_check_not_input_allows_a_different_file(tmp_path):
    scan = tmp_path / "a.tif"
    scan.write_bytes(b"x")
    out = tmp_path / "b.tif"
    check_not_input([(scan, out)])  # must not raise


def _pairs(tmp_path, n, existing):
    """n (input, output) pairs; `existing` output indices actually exist on disk."""
    pairs = []
    for i in range(n):
        src = tmp_path / f"in_{i}.tif"
        out = tmp_path / f"out_{i}.tif"
        src.write_bytes(b"x")
        if i in existing:
            out.write_bytes(b"y")
        pairs.append((src, out))
    return pairs


def test_existing_outputs_non_interactive_refuses_and_names_both_flags(tmp_path):
    pairs = _pairs(tmp_path, 2, existing={0})
    with pytest.raises(SystemExit) as exc:
        resolve_existing(pairs, OutputPolicy.ASK, interactive=False)
    message = str(exc.value.code)
    assert "1 of 2" in message
    assert "--overwrite" in message
    assert "--skip-existing" in message


def test_existing_outputs_skip_existing_drops_them(tmp_path):
    pairs = _pairs(tmp_path, 3, existing={0, 2})
    kept = resolve_existing(pairs, OutputPolicy.SKIP_EXISTING, interactive=False)
    assert kept == [pairs[1]]


def test_existing_outputs_overwrite_keeps_all(tmp_path):
    pairs = _pairs(tmp_path, 3, existing={0, 2})
    kept = resolve_existing(pairs, OutputPolicy.OVERWRITE, interactive=True)
    assert kept == pairs


def test_existing_outputs_interactive_asks_once_for_the_whole_roll(tmp_path, monkeypatch):
    pairs = _pairs(tmp_path, 3, existing={0, 1, 2})
    prompts = []

    def fake_confirm(prompt, default=False):
        prompts.append(prompt)
        return True

    monkeypatch.setattr(console, "confirm", fake_confirm)
    kept = resolve_existing(pairs, OutputPolicy.ASK, interactive=True)
    assert kept == pairs
    assert len(prompts) == 1
    assert "3 of 3" in prompts[0]


def test_interactive_decline_writes_nothing(tmp_path, monkeypatch):
    pairs = _pairs(tmp_path, 2, existing={0})
    monkeypatch.setattr(console, "confirm", lambda prompt, default=False: False)
    with pytest.raises(SystemExit) as exc:
        resolve_existing(pairs, OutputPolicy.ASK, interactive=True)
    assert "Nothing was written" in str(exc.value.code)
    # Nothing beyond what already existed on disk was written.
    assert not pairs[1][1].exists()


def test_no_existing_outputs_never_prompts(tmp_path, monkeypatch):
    pairs = _pairs(tmp_path, 2, existing=set())

    def fail_confirm(*args, **kwargs):
        raise AssertionError("must not prompt when nothing exists")

    monkeypatch.setattr(console, "confirm", fail_confirm)
    kept = resolve_existing(pairs, OutputPolicy.ASK, interactive=True)
    assert kept == pairs


def test_single_pair_wording_is_not_the_plural_one_of_one_form(tmp_path):
    pairs = _pairs(tmp_path, 1, existing={0})
    with pytest.raises(SystemExit) as exc:
        resolve_existing(pairs, OutputPolicy.ASK, interactive=False)
    message = str(exc.value.code)
    assert "1 of 1" not in message
    assert str(pairs[0][1]) in message


def _jobs(tmp_path, n, existing):
    """n BatchJob-like objects; `existing` output indices already exist on disk."""
    from halide.batch.orchestrator import BatchJob

    jobs = []
    for i in range(n):
        src = tmp_path / f"in_{i}.tif"
        out = tmp_path / f"out_{i}.tif"
        src.write_bytes(b"x")
        if i in existing:
            out.write_bytes(b"y")
        jobs.append(BatchJob(input_path=src, output_path=out))
    return jobs


class _Args:
    def __init__(self, *, overwrite=False, skip_existing=False):
        self.overwrite = overwrite
        self.skip_existing = skip_existing


def test_resolve_bulk_jobs_overwrite_keeps_everything(tmp_path):
    jobs = _jobs(tmp_path, 3, existing={0, 2})
    kept, skipped, kept_outputs = resolve_bulk_jobs(jobs, _Args(overwrite=True), interactive=False)
    assert kept == jobs
    assert skipped == 0
    assert kept_outputs == {job.output_path for job in jobs}


def test_resolve_bulk_jobs_skip_existing_drops_the_right_jobs(tmp_path):
    jobs = _jobs(tmp_path, 3, existing={0, 2})
    kept, skipped, kept_outputs = resolve_bulk_jobs(jobs, _Args(skip_existing=True), interactive=False)
    assert kept == [jobs[1]]
    assert skipped == 2
    assert kept_outputs == {jobs[1].output_path}


def test_resolve_bulk_jobs_refuses_output_equal_to_input(tmp_path):
    from halide.batch.orchestrator import BatchJob

    scan = tmp_path / "a.tif"
    scan.write_bytes(b"x")
    jobs = [BatchJob(input_path=scan, output_path=scan)]
    with pytest.raises(SystemExit) as exc:
        resolve_bulk_jobs(jobs, _Args(), interactive=False)
    assert "is the scan itself" in str(exc.value.code)


def test_resolve_bulk_jobs_folds_extra_pairs_into_one_prompt(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path, 2, existing={0})
    extra_out = tmp_path / "sheet.jpg"
    extra_out.write_bytes(b"z")  # the sheet already exists too

    prompts = []
    monkeypatch.setattr(console, "confirm", lambda prompt, default=False: (prompts.append(prompt), True)[1])

    kept, skipped, kept_outputs = resolve_bulk_jobs(
        jobs, _Args(), interactive=True, extra_pairs=[(extra_out, extra_out)]
    )
    assert len(prompts) == 1
    assert "2 of 3" in prompts[0]  # both job outputs' existing member plus the sheet, one decision
    assert kept == jobs  # confirmed -> everything proceeds, including the sheet
    assert extra_out in kept_outputs
    assert skipped == 0
