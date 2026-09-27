"""tests/conftest.py's guard against a real GPU service: on for every test, except those marked
`gpu` (the real-GPU checks in tests/gpu, which compare the service on the real card). Neither test
here needs a GPU or starts anything — they only look at which `_start_service` the test sees."""

import pytest

import halide.batch.orchestrator as orchestrator

_REAL_START_SERVICE = orchestrator._start_service  # read at import, before any fixture patches it


def test_an_ordinary_test_cannot_start_a_real_gpu_service():
    assert orchestrator._start_service is not _REAL_START_SERVICE
    with pytest.raises(Exception, match="isn't started in tests"):
        orchestrator._start_service("gpu", None)


@pytest.mark.gpu
def test_a_gpu_marked_test_gets_the_real_service_starter():
    assert orchestrator._start_service is _REAL_START_SERVICE
