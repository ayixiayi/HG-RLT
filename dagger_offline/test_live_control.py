import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dagger_offline.arbitration import Ownership
from dagger_offline.live_control import Controller, PolicyWorker, TickLease


class Worker:
    busy = False
    result = None

    def submit(self, ticket, observation):
        self.busy = True
        self.ticket = ticket
        self.observation = observation

    def poll(self):
        result, self.result = self.result, None
        if result is not None:
            self.busy = False
        return result


class Relative:
    def anchor(self, leader, qpos):
        self.last = qpos.copy()

    def step(self, leader):
        return self.last + 0.001


def leader():
    return {side: np.zeros(7) for side in ("left", "right")}


def propose(controller, now, **kwargs):
    return controller.propose(
        now=now, observed_at=now, observation={"frame": now}, **kwargs
    )


@pytest.mark.parametrize("late", [np.ones((4, 14)), RuntimeError("late failure")])
def test_takeover_cancels_pending_even_late_error(late):
    worker = Worker()
    control = Controller(Relative(), worker, now=0)
    control.control("p", now=0.1, qpos=np.zeros(14))
    propose(control, 0.2)
    ticket = worker.ticket
    qpos = np.arange(14) / 100
    control.control("h", now=0.3, qpos=qpos, leader=leader(), sampled_at=0.3)
    worker.result = ticket, late
    np.testing.assert_array_equal(
        propose(control, 0.31, leader=leader(), sampled_at=0.3), qpos
    )
    assert control.owner is Ownership.HUMAN
    assert control.pending is None


def test_horizon_replan_and_explicit_fresh_handback():
    worker = Worker()
    control = Controller(Relative(), worker, now=0, horizon=2, replan_lead=0)
    assert propose(control, 0.1) is None
    assert not worker.busy
    control.control("p", now=0.2, qpos=np.zeros(14))
    assert control.propose(now=0.3, observed_at=0.2, observation={}) is None
    assert not worker.busy
    propose(control, 0.31)
    worker.result = worker.ticket, np.arange(56).reshape(4, 14)
    assert propose(control, 0.32) is None  # acceptance precedes a fresh image
    np.testing.assert_array_equal(propose(control, 0.325), np.arange(14))
    np.testing.assert_array_equal(propose(control, 0.33), np.arange(14, 28))
    assert propose(control, 0.34) is None
    # Exhaustion creates a HOLD boundary: replan uses a frame after it.
    assert propose(control, 0.345) is None
    assert worker.busy and worker.ticket.observed_at == 0.345
    control.control("s", now=0.35, qpos=np.zeros(14))
    worker.result = worker.ticket, np.ones((2, 14))
    assert propose(control, 0.36) is None
    assert not worker.busy
    control.control("h", now=0.4, qpos=np.zeros(14), leader=leader(), sampled_at=0.4)
    control.control("p", now=0.41, qpos=np.zeros(14))
    assert control.propose(now=0.42, observed_at=0.4, observation={}) is None
    assert not worker.busy
    propose(control, 0.43)
    assert worker.busy


def test_continuous_policy_prefetches_and_never_emits_gap_between_chunks():
    worker = Worker()
    control = Controller(Relative(), worker, now=0, horizon=4, replan_lead=2)
    control.control("p", now=0.1, qpos=np.zeros(14))
    assert propose(control, 0.11) is None
    worker.result = worker.ticket, np.arange(4 * 14).reshape(4, 14)
    assert propose(control, 0.12) is None  # first chunk waits for a fresh image
    emitted = []
    for i in range(8):
        if i == 3:
            prefetch = worker.ticket
            assert prefetch.continuous
            assert prefetch.observed_at == pytest.approx(0.15)
            worker.result = prefetch, 1000 + np.arange(4 * 14).reshape(4, 14)
        if i == 5:
            worker.result = worker.ticket, 2000 + np.arange(4 * 14).reshape(4, 14)
        emitted.append(propose(control, 0.13 + i / 100))
    np.testing.assert_array_equal(emitted[0], np.arange(14))
    np.testing.assert_array_equal(emitted[1], np.arange(14, 28))
    # remaining == 2 at the third tick: prefetch from its image, then action 2.
    np.testing.assert_array_equal(emitted[2], np.arange(28, 42))
    # Reply before tick 3: action 2 already ran after the image, so skip 1.
    np.testing.assert_array_equal(emitted[3], 1000 + np.arange(14, 28))
    assert all(value is not None for value in emitted)
    assert control.owner is Ownership.POLICY


def test_continuous_switch_skips_actions_run_after_prefetch_image():
    worker = Worker()
    control = Controller(Relative(), worker, now=0, horizon=4, replan_lead=2)
    control.control("p", now=0.1, qpos=np.zeros(14))
    propose(control, 0.11)
    worker.result = worker.ticket, np.zeros((4, 14))
    propose(control, 0.12)
    propose(control, 0.13)
    propose(control, 0.14)
    propose(control, 0.15)  # prefetch observed here, action 2 sent after it
    prefetch = worker.ticket
    worker.result = prefetch, 1 + np.arange(4)[:, None] * np.ones((1, 14))
    # Reply arrives before action 3: skip action 0 (executed as old action 2).
    np.testing.assert_array_equal(propose(control, 0.16), np.full(14, 2.0))
    np.testing.assert_array_equal(propose(control, 0.17), np.full(14, 3.0))


@pytest.mark.parametrize("stamp,now", [(0.1, 0.5), (0.19, 0.21), (0.23, 0.22)])
def test_bad_leader_samples_latch_fault(stamp, now):
    control = Controller(Relative(), Worker(), now=0)
    control.control("h", now=0.2, qpos=np.zeros(14), leader=leader(), sampled_at=0.2)
    with pytest.raises((ValueError, RuntimeError)):
        propose(control, now, leader=leader(), sampled_at=stamp)
    assert control.owner is Ownership.FAULT
    with pytest.raises(RuntimeError):
        control.control("p", now=now, qpos=np.zeros(14))


@pytest.mark.parametrize("stamp", [-1, 0.3])
def test_invalid_takeover_sample_also_faults(stamp):
    control = Controller(Relative(), Worker(), now=0)
    with pytest.raises(ValueError):
        control.control(
            "h", now=0.2, qpos=np.zeros(14), leader=leader(), sampled_at=stamp
        )
    assert control.owner is Ownership.FAULT


def test_acknowledgement_reseeds_and_copies():
    relative = Relative()
    control = Controller(relative, Worker(), now=0)
    control.control("h", now=0.1, qpos=np.zeros(14), leader=leader(), sampled_at=0.1)
    applied = np.full(14, 0.2)
    control.acknowledged(applied)
    applied[:] = 9
    np.testing.assert_allclose(
        propose(control, 0.12, leader=leader(), sampled_at=0.12), 0.201
    )


def test_blocked_policy_does_not_block_control_poll_or_close():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Client:
        def reset(self):
            calls.append(("reset", threading.get_ident()))

        def infer_chunk(self, observation):
            calls.append(("infer", threading.get_ident()))
            entered.set()
            assert release.wait(3)
            return np.zeros((2, 14))

        def close(self):
            calls.append(("close", threading.get_ident()))
            closed.set()

    def factory():
        calls.append(("construct", threading.get_ident()))
        return Client()

    worker = PolicyWorker(factory)
    try:
        control = Controller(Relative(), worker, now=0)
        control.control("p", now=0.1, qpos=np.zeros(14))
        propose(control, 0.2)
        assert entered.wait(1)
        start = time.monotonic()
        assert worker.poll() is None
        control.control(
            "h", now=0.3, qpos=np.zeros(14), leader=leader(), sampled_at=0.3
        )
        np.testing.assert_array_equal(
            propose(control, 0.31, leader=leader(), sampled_at=0.3), np.zeros(14)
        )
        assert time.monotonic() - start < 0.2
        start = time.monotonic()
        worker.close()
        assert time.monotonic() - start < 0.3
        assert not closed.is_set()
    finally:
        release.set()
        worker.thread.join(2)
    assert closed.is_set() and not worker.thread.is_alive()
    assert [name for name, _ in calls] == ["construct", "reset", "infer", "close"]
    assert {ident for _, ident in calls} == {worker.thread.ident}
    assert worker.thread.ident != threading.get_ident()


def test_tick_lease_only_renews_on_pulse_and_rejects_unexpected_hold():
    calls = []
    state = {
        "state": "safe_hold",
        "hold_confirmed": True,
        "session_id": None,
        "session_api_version": 1,
        "strict_session": True,
    }
    safety = SimpleNamespace(
        status=lambda: state.copy(),
        acquire_session=lambda session: (
            calls.append(("begin", session)),
            state.update(session_id=session),
        ),
        heartbeat=lambda name, session: calls.append((name, session)) or state.copy(),
        safe_hold=lambda reason, **kw: state.copy(),
        activate=lambda session: state.update(state="active"),
    )
    lease = TickLease(safety, "test")
    lease.begin()
    before = list(calls)
    time.sleep(0.12)
    assert calls == before
    lease.activate()
    state["state"] = "safe_hold"
    with pytest.raises(RuntimeError, match="Unexpected daemon HOLD"):
        lease.pulse()
    after = list(calls)
    with pytest.raises(RuntimeError, match="Latched"):
        lease.activate()
    assert calls == after
