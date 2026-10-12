import math

import pytest

from dagger_offline.arbitration import Arbiter, Ownership


def action(offset=0):
    return [offset + i / 100 for i in range(14)]


def policy(arbiter, now=1):
    request = arbiter.request_policy(observed_at=now, now=now)
    assert arbiter.accept_policy(request, [action(), action(1)], now=now)
    return request


def test_starts_hold_and_requires_new_observation():
    arbiter = Arbiter(now=0, human_timeout=1, observation_timeout=2)
    assert arbiter.owner is Ownership.HOLD
    assert arbiter.next_action(now=0) is None
    with pytest.raises(ValueError):
        arbiter.request_policy(observed_at=0, now=1)
    request = arbiter.request_policy(observed_at=1, now=1)
    assert arbiter.owner is Ownership.HOLD
    assert arbiter.next_action(now=1) is None
    assert arbiter.accept_policy(request, [action()], now=1)
    assert arbiter.owner is Ownership.POLICY
    assert arbiter.next_action(now=1) == tuple(action())
    assert arbiter.next_action(now=1) is None
    assert arbiter.owner is Ownership.HOLD


@pytest.mark.parametrize("transition", ["takeover", "hold", "fault"])
def test_transition_invalidates_requests_and_whole_chunk(transition):
    arbiter = Arbiter(now=0)
    old = policy(arbiter)
    if transition == "takeover":
        arbiter.takeover(action(2), sampled_at=2, now=2)
    else:
        getattr(arbiter, transition)(now=2)
    assert arbiter.generation > old.generation
    assert not arbiter.accept_policy(old, [action(99)], now=2)
    expected = tuple(action(2)) if transition == "takeover" else None
    assert arbiter.next_action(now=2) == expected


def test_human_return_requires_hold_new_request_and_new_observation():
    arbiter = Arbiter(now=0)
    old = policy(arbiter)
    arbiter.takeover(action(2), sampled_at=2, now=2)
    with pytest.raises(RuntimeError):
        arbiter.request_policy(observed_at=2, now=2)
    arbiter.hold(now=3)
    with pytest.raises(ValueError):
        arbiter.request_policy(observed_at=3, now=3)
    request = arbiter.request_policy(observed_at=4, now=4)
    assert not arbiter.accept_policy(old, [action(99)], now=4)
    assert arbiter.accept_policy(request, [action(3)], now=4)
    assert arbiter.next_action(now=4) == tuple(action(3))


def test_new_request_rejects_out_of_order_and_duplicate_responses():
    arbiter = Arbiter(now=0)
    first = arbiter.request_policy(observed_at=1, now=1)
    second = arbiter.request_policy(observed_at=2, now=2)
    assert not arbiter.accept_policy(first, [action(99)], now=2)
    assert arbiter.accept_policy(second, [action(2)], now=2)
    assert not arbiter.accept_policy(second, [action(99)], now=2)
    assert arbiter.next_action(now=2) == tuple(action(2))


def test_human_expiry_boundary_faults_and_does_not_auto_recover():
    arbiter = Arbiter(now=0, human_timeout=1)
    arbiter.takeover(action(), sampled_at=1, now=1)
    assert arbiter.next_action(now=2) == tuple(action())
    assert arbiter.next_action(now=2.001) is None
    assert arbiter.owner is Ownership.FAULT
    with pytest.raises(RuntimeError):
        arbiter.update_human(action(1), sampled_at=3, now=3)
    with pytest.raises(RuntimeError):
        arbiter.hold(now=3)
    with pytest.raises(RuntimeError):
        arbiter.takeover(action(), sampled_at=3, now=3)
    arbiter.reset_fault(now=3)
    assert arbiter.owner is Ownership.HOLD
    with pytest.raises(ValueError):
        arbiter.request_policy(observed_at=3, now=3)


@pytest.mark.parametrize("sampled_at", [0, 3, math.nan])
def test_stale_future_or_invalid_human_sample_fails_closed(sampled_at):
    arbiter = Arbiter(now=0, human_timeout=1)
    policy(arbiter)
    with pytest.raises(ValueError):
        arbiter.takeover(action(), sampled_at=sampled_at, now=2)
    assert arbiter.owner is Ownership.FAULT
    assert arbiter.next_action(now=2) is None


@pytest.mark.parametrize("bad", [[0] * 7, [0] * 13, [0] * 15, [math.nan] * 14])
def test_entire_14d_command_is_validated_before_any_output(bad):
    arbiter = Arbiter(now=0)
    request = arbiter.request_policy(observed_at=1, now=1)
    with pytest.raises(ValueError):
        arbiter.accept_policy(request, [action(), bad], now=1)
    assert arbiter.owner is Ownership.FAULT
    assert arbiter.next_action(now=1) is None


def test_human_update_is_atomic_copied_and_monotonic():
    arbiter = Arbiter(now=0)
    arbiter.takeover(action(), sampled_at=1, now=1)
    target = action(5)
    arbiter.update_human(target, sampled_at=2, now=2)
    target[0] = 999
    assert arbiter.next_action(now=2) == tuple(action(5))
    with pytest.raises(ValueError):
        arbiter.update_human(action(99), sampled_at=1.9, now=2)
    assert arbiter.owner is Ownership.FAULT


def test_delayed_policy_and_expired_queued_policy_emit_nothing():
    arbiter = Arbiter(now=0, observation_timeout=1)
    request = arbiter.request_policy(observed_at=1, now=1)
    assert not arbiter.accept_policy(request, [action()], now=2.01)
    assert arbiter.owner is Ownership.HOLD
    policy(arbiter, now=3)
    assert arbiter.next_action(now=4) == tuple(action())
    assert arbiter.next_action(now=4.01) is None
    assert arbiter.owner is Ownership.HOLD


def test_policy_requests_cannot_reuse_or_reverse_observation_time():
    arbiter = Arbiter(now=0)
    policy(arbiter, now=2)
    with pytest.raises(ValueError):
        arbiter.request_policy(observed_at=1.5, now=2)
    # A rejected prefetch leaves the executing chunk untouched.
    assert arbiter.owner is Ownership.POLICY and arbiter.pending is None
    assert arbiter.next_action(now=2) == tuple(action())


def chunk(start, length=10):
    return [action(start + i) for i in range(length)]


def test_prefetch_replaces_remaining_chunk_without_hold_and_skips_executed():
    arbiter = Arbiter(now=0)
    first = arbiter.request_policy(observed_at=1, now=1)
    assert arbiter.accept_policy(first, chunk(0, 5), now=1)
    generation = arbiter.generation
    assert arbiter.next_action(now=1.0) == tuple(action(0))
    prefetch = arbiter.request_policy(observed_at=1.05, now=1.05)
    assert prefetch.continuous and arbiter.owner is Ownership.POLICY
    with pytest.raises(RuntimeError, match="already outstanding"):
        arbiter.request_policy(observed_at=1.06, now=1.06)
    # Two actions run after the prefetch image while inference is in flight.
    assert arbiter.next_action(now=1.07) == tuple(action(1))
    assert arbiter.next_action(now=1.1) == tuple(action(2))
    assert arbiter.accept_policy(prefetch, chunk(100), now=1.12)
    assert arbiter.owner is Ownership.POLICY
    assert arbiter.generation == generation + 1
    assert arbiter.remaining == 8
    assert arbiter.next_action(now=1.13) == tuple(action(102))


def test_late_prefetch_resumes_after_exhaustion_hold():
    arbiter = Arbiter(now=0)
    first = arbiter.request_policy(observed_at=1, now=1)
    assert arbiter.accept_policy(first, chunk(0, 2), now=1)
    prefetch = arbiter.request_policy(observed_at=1.01, now=1.01)
    assert arbiter.next_action(now=1.02) == tuple(action(0))
    assert arbiter.next_action(now=1.03) == tuple(action(1))
    assert arbiter.next_action(now=1.04) is None
    assert arbiter.owner is Ownership.HOLD and arbiter.pending is prefetch
    assert arbiter.accept_policy(prefetch, chunk(100), now=1.2)
    assert arbiter.next_action(now=1.21) == tuple(action(102))


def test_prefetch_too_late_or_invalidated_falls_back_to_hold():
    arbiter = Arbiter(now=0, max_skip=2)
    first = arbiter.request_policy(observed_at=1, now=1)
    assert arbiter.accept_policy(first, chunk(0), now=1)
    prefetch = arbiter.request_policy(observed_at=1.01, now=1.01)
    for i in range(3):
        arbiter.next_action(now=1.02 + i / 100)
    assert not arbiter.accept_policy(prefetch, chunk(100), now=1.1)
    assert arbiter.owner is Ownership.HOLD and arbiter.pending is None

    first = arbiter.request_policy(observed_at=1.2, now=1.2)
    assert arbiter.accept_policy(first, chunk(0), now=1.2)
    prefetch = arbiter.request_policy(observed_at=1.21, now=1.21)
    arbiter.takeover(action(), sampled_at=1.22, now=1.22)
    assert not arbiter.accept_policy(prefetch, chunk(100), now=1.23)
    assert arbiter.owner is Ownership.HUMAN


def test_time_and_transition_contracts():
    arbiter = Arbiter(now=0)
    with pytest.raises(RuntimeError):
        arbiter.update_human(action(), sampled_at=1, now=1)
    with pytest.raises(RuntimeError):
        arbiter.reset_fault(now=1)
    with pytest.raises(ValueError):
        arbiter.next_action(now=0)
    with pytest.raises(ValueError):
        arbiter.next_action(now=math.inf)
    assert arbiter.next_action(now=1) is None


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan])
def test_timeouts_must_be_finite_positive(timeout):
    with pytest.raises(ValueError):
        Arbiter(now=0, human_timeout=timeout)
