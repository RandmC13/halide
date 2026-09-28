"""halide.interrupts: the signal helpers behind orderly cancelling (F12, review 2.4-9)."""

import signal

import pytest

from halide import interrupts


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")
def test_cancel_on_hangup_and_term_raises_keyboard_interrupt_and_restores():
    before = signal.getsignal(signal.SIGTERM)
    with interrupts.cancel_on_hangup_and_term():
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_handler_installed_from_c_is_left_alone_on_restore(monkeypatch):
    """signal.signal returns None for a handler installed from C: there is nothing to put back,
    and passing None back would raise TypeError on the way out."""
    calls = []

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        return None

    monkeypatch.setattr(interrupts.signal, "signal", fake_signal)
    with interrupts.cancel_on_hangup_and_term():
        pass
    assert all(handler is interrupts._raise_interrupt for _, handler in calls)
