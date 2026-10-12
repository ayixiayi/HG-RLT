"""CPU-only command ownership; no hardware access or background threads.

One control-loop thread must serialize every call and the subsequent command
send. All timestamps must use the same monotonic clock. Returned actions are
proposals for the existing safety filter, not evidence of physical execution.
None means no proposal: the caller must implement safe HOLD/FAULT behavior and
retain its watchdog, emergency stop, limits, and whole-task safety budgets.
"""

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Ownership(Enum):
    POLICY = "POLICY"
    HUMAN = "HUMAN"
    HOLD = "HOLD"
    FAULT = "FAULT"


@dataclass(frozen=True)
class PolicyRequest:
    """Opaque in-process ticket; return the exact object with its response.

    continuous: issued while a policy chunk was executing (prefetch). Its reply
    replaces the remaining chunk without HOLD, skipping the actions already
    executed since the observation.
    """

    generation: int
    observed_at: float
    continuous: bool = False


class Arbiter:
    def __init__(
        self,
        *,
        now: float,
        human_timeout: float = 1.0,
        observation_timeout: float = 2.0,
        max_skip: int = 12,
    ):
        for value in (human_timeout, observation_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if type(max_skip) is not int or max_skip < 0:
            raise ValueError("max_skip must be a non-negative integer")
        self._max_skip = max_skip
        self._executed_since_request = 0
        self._human_timeout = human_timeout
        self._observation_timeout = observation_timeout
        self._now = now
        self._barrier = now
        self._last_observation = now
        self._owner = Ownership.HOLD
        self._generation = 0
        self._pending = None
        self._chunk = deque()
        self._observed_at = None
        self._human_action = None
        self._sampled_at = None

    @property
    def owner(self) -> Ownership:
        return self._owner

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def remaining(self) -> int:
        """Unexecuted actions of the current policy chunk (0 unless POLICY)."""
        return len(self._chunk) if self._owner is Ownership.POLICY else 0

    @property
    def pending(self) -> PolicyRequest | None:
        return self._pending

    def peek(self, count: int) -> list[tuple[float, ...]]:
        """The next ``count`` actions of the current policy chunk (no consume)."""
        if self._owner is not Ownership.POLICY or len(self._chunk) < count:
            raise ValueError("policy chunk shorter than requested")
        return list(self._chunk)[:count]

    def replace_chunk(self, actions) -> None:
        """Substitute the remaining policy chunk (RL actor output), same generation."""
        if self._owner is not Ownership.POLICY:
            raise RuntimeError("only a running policy chunk can be replaced")
        try:
            chunk = deque(self._action(action) for action in actions)
            if not chunk:
                raise ValueError("replacement chunk must not be empty")
        except (TypeError, ValueError, OverflowError) as error:
            self._transition(Ownership.FAULT)
            raise ValueError("invalid replacement chunk") from error
        self._chunk = chunk

    def _transition(self, owner: Ownership) -> None:
        self._generation += 1
        self._owner = owner
        self._barrier = self._now
        self._pending = None
        self._chunk.clear()
        self._observed_at = None
        self._human_action = None
        self._sampled_at = None

    def _tick(self, now: float) -> None:
        if not math.isfinite(now) or now < self._now:
            self._transition(Ownership.FAULT)
            raise ValueError("now must be finite and nondecreasing")
        self._now = now
        if (
            self._owner is Ownership.HUMAN
            and now - self._sampled_at > self._human_timeout
        ):
            self._transition(Ownership.FAULT)

    @staticmethod
    def _action(values: Iterable[float]) -> tuple[float, ...]:
        action = tuple(float(value) for value in values)
        if len(action) != 14 or not all(math.isfinite(value) for value in action):
            raise ValueError("actions must have exactly 14 finite values")
        return action

    def _human_sample(self, action, sampled_at: float) -> tuple[float, ...]:
        try:
            if (
                not math.isfinite(sampled_at)
                or sampled_at > self._now
                or self._now - sampled_at > self._human_timeout
                or (self._sampled_at is not None and sampled_at <= self._sampled_at)
            ):
                raise ValueError("human sample must be fresh and strictly newer")
            return self._action(action)
        except (TypeError, ValueError, OverflowError) as error:
            self._transition(Ownership.FAULT)
            raise ValueError("invalid human sample") from error

    def takeover(self, action, *, sampled_at: float, now: float) -> None:
        self._tick(now)
        if self._owner is Ownership.FAULT:
            raise RuntimeError("reset the latched fault before takeover")
        target = self._human_sample(action, sampled_at)
        self._transition(Ownership.HUMAN)
        self._human_action = target
        self._sampled_at = sampled_at

    def update_human(self, action, *, sampled_at: float, now: float) -> None:
        self._tick(now)
        if self._owner is not Ownership.HUMAN:
            raise RuntimeError("human updates require human ownership")
        target = self._human_sample(action, sampled_at)
        self._human_action = target
        self._sampled_at = sampled_at

    def hold(self, *, now: float) -> None:
        self._tick(now)
        if self._owner is Ownership.FAULT:
            raise RuntimeError("reset the latched fault before HOLD")
        self._transition(Ownership.HOLD)

    def fault(self, *, now: float) -> None:
        self._tick(now)
        self._transition(Ownership.FAULT)

    def reset_fault(self, *, now: float) -> None:
        self._tick(now)
        if self._owner is not Ownership.FAULT:
            raise RuntimeError("reset requires a latched fault")
        self._transition(Ownership.HOLD)

    def request_policy(self, *, observed_at: float, now: float) -> PolicyRequest:
        """From HOLD: start policy after handback. From POLICY: prefetch.

        A prefetch keeps executing the current chunk; only one request may be
        outstanding.
        """
        self._tick(now)
        if self._owner not in (Ownership.HOLD, Ownership.POLICY):
            raise RuntimeError("policy requests require explicit HOLD handback")
        continuous = self._owner is Ownership.POLICY
        if continuous and self._pending is not None:
            raise RuntimeError("a policy prefetch is already outstanding")
        if (
            not math.isfinite(observed_at)
            or observed_at <= self._barrier
            or observed_at <= self._last_observation
            or observed_at > now
            or now - observed_at > self._observation_timeout
        ):
            if not continuous:
                self._pending = None
                self._generation += 1
            raise ValueError(
                "observation must be fresh and newer than HOLD/last request"
            )
        self._last_observation = observed_at
        self._executed_since_request = 0
        if continuous:
            self._pending = PolicyRequest(self._generation, observed_at, True)
            return self._pending
        # Replace any pending request, without advancing the handback barrier:
        # this observation was already captured after it.
        self._generation += 1
        self._chunk.clear()
        self._observed_at = None
        self._pending = PolicyRequest(self._generation, observed_at)
        return self._pending

    def accept_policy(self, request: PolicyRequest, actions, *, now: float) -> bool:
        """False: the reply is unrelated or too late; the caller stays/goes HOLD."""
        self._tick(now)
        if (
            request is not self._pending
            or self._pending is None
            or self._owner not in (Ownership.HOLD, Ownership.POLICY)
        ):
            return False
        skip = self._executed_since_request if request.continuous else 0
        if (
            now - request.observed_at > self._observation_timeout
            or skip > self._max_skip
        ):
            self._transition(Ownership.HOLD)
            return False
        try:
            chunk = deque(self._action(action) for action in actions)
            if len(chunk) <= skip:
                raise ValueError("policy chunk must outlast the executed actions")
        except (TypeError, ValueError, OverflowError) as error:
            self._transition(Ownership.FAULT)
            raise ValueError("invalid policy chunk") from error
        for _ in range(skip):
            chunk.popleft()
        self.last_skip = skip
        if request.continuous:
            # New chunk generation: the sender resets its per-chunk budget.
            self._generation += 1
        self._pending = None
        self._chunk = chunk
        self._observed_at = request.observed_at
        self._owner = Ownership.POLICY
        return True

    def next_action(self, *, now: float) -> tuple[float, ...] | None:
        self._tick(now)
        if self._owner is Ownership.HUMAN:
            return self._human_action
        if self._owner is Ownership.POLICY:
            if now - self._observed_at > self._observation_timeout:
                self._transition(Ownership.HOLD)
                return None
            if not self._chunk:
                # Prefetch reply is late: HOLD, but keep the outstanding request
                # so its reply can still resume (with the executed actions skipped).
                pending = self._pending
                self._transition(Ownership.HOLD)
                self._pending = pending
                return None
            self._executed_since_request += 1
            return self._chunk.popleft()
        return None
