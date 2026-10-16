"""Single-threaded HG-DAgger ownership with an inference-only worker.

No worker can send robot commands. The caller serializes controls, proposal,
safe filtering, send and acknowledgement on one thread. All clocks are monotonic.
"""

import queue
import threading

import numpy as np

from dagger_offline.arbitration import Arbiter, Ownership


class PolicyWorker:
    """One outstanding request; a late result is discarded by the arbiter.

    Construct/use/close the websocket client exclusively on this worker thread.
    Factory and infer must have transport timeouts. Shutdown never blocks HOLD.
    """

    def __init__(self, factory):
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.busy = False
        self.thread = threading.Thread(target=self._run, args=(factory,), daemon=True)
        self.thread.start()

    def _run(self, factory):
        client = None
        try:
            while not self.stopping.is_set():
                try:
                    ticket, observation = self.requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if client is None:
                        client = factory()
                        client.reset()
                    result = client.infer_chunk(observation)
                    token = getattr(client, "last_rl_token", None)
                    if token is not None:
                        result = (result, token)  # RLT round: actions + RL token
                except Exception as exc:  # noqa: BLE001 - forward failure to the owning control thread
                    result = exc
                self.results.put_nowait((ticket, result))
        finally:
            if client is not None:
                client.close()

    def submit(self, ticket, observation):
        if self.busy or self.stopping.is_set():
            raise RuntimeError("Policy worker is not available")
        # Caller transfers ownership of an independent observation snapshot.
        self.busy = True
        self.requests.put_nowait((ticket, observation))

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=0.1)


class Controller:
    """Explicit h takeover / p handback / s hold; faults require a new episode.

    `relative` exposes anchor(leader_dict, follower), step(leader_dict), and
    last (the acknowledged command, not the unexecuted IK proposal).
    """

    def __init__(
        self,
        relative,
        worker,
        *,
        now,
        horizon=30,
        timeout=0.2,
        replan_lead=8,
        blend=0,
        rl=None,
        rl_lead=None,
    ):
        if not isinstance(horizon, int) or not 1 <= horizon <= 30:
            raise ValueError("execution horizon must be 1..30")
        if not isinstance(replan_lead, int) or not 0 <= replan_lead < horizon:
            raise ValueError("replan lead must be 0 (no prefetch) .. horizon-1")
        self.arbiter = Arbiter(now=now, human_timeout=timeout, blend=blend)
        self.relative, self.worker = relative, worker
        self.horizon, self.timeout = horizon, timeout
        # Request the next chunk when this many actions remain, so consecutive
        # chunks execute without an inference HOLD (RhOS async_replan_lead=8).
        self.replan_lead = replan_lead
        self.autonomous = False
        self.human_stamp = None
        self.pending = None
        self.last_key = "s"
        self.ready_after = now
        self._accepted = []
        # RLT: `rl` decides an actor chunk from (token, proprio, reference).
        self.rl, self.rl_active = rl, False
        # RL chunks prefetch their next decision rl_lead steps before the end
        # (default: the run's config lead); the steps still running then are
        # recorded as the decision's committed actions.
        if rl is not None:
            rl_lead = rl.config.lead if rl_lead is None else rl_lead
            if rl_lead != rl.config.lead:
                raise ValueError("rl_lead must equal the RLT run's lead")
        self.rl_lead = rl_lead or 0
        self._decisions = []
        self._qpos = self._previous_qpos = None
        self._request_state = {}

    @property
    def owner(self):
        return self.arbiter.owner

    def control(self, key, *, now, qpos, leader=None, sampled_at=None):
        if key not in {"h", "p", "s", "r"}:
            raise ValueError("Unknown ownership control")
        if key == "r" and self.rl is None:
            raise ValueError("RL mode needs an RLT agent (--rlt)")
        if key == "h":
            if self.owner is Ownership.HUMAN:
                return  # Key-repeat must not silently reanchor a moving leader.
            try:
                self._fresh_leader(leader, sampled_at, now)
                self.relative.anchor(leader, qpos)
            except Exception:
                self.autonomous = False
                self.pending = None
                self.arbiter.fault(now=now)
                raise
            # First human target is exactly the measured anchor; no IK jump.
            self.arbiter.takeover(qpos, sampled_at=sampled_at, now=now)
            self.human_stamp = sampled_at
            self.autonomous = False
        else:
            self.arbiter.hold(now=now)
            self.autonomous = key in {"p", "r"}
        self.rl_active = key == "r"
        self._request_state.clear()
        self.pending = None
        self.last_key = key
        self.ready_after = now

    def fault(self, *, now):
        self.autonomous = False
        self.rl_active = False
        self.pending = None
        self.arbiter.fault(now=now)

    def _fresh_leader(self, leader, stamp, now):
        if (
            stamp is None
            or not np.isfinite(stamp)
            or not 0 <= now - stamp <= self.timeout
        ):
            raise ValueError("Leader sample is stale or has invalid timestamp")
        if not isinstance(leader, dict) or set(leader) != {"left", "right"}:
            raise ValueError("Both leader arms are required")
        for vector in leader.values():
            value = np.asarray(vector)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError("Leader requires finite 7-D samples")

    def propose(
        self, *, now, observed_at, observation, leader=None, sampled_at=None, qpos=None
    ):
        try:
            if qpos is not None:
                self._previous_qpos = self._qpos if self._qpos is not None else qpos
                self._qpos = np.asarray(qpos, dtype=float).copy()
            result = self.worker.poll()
            if result is not None:
                ticket, actions = result
                token = None
                if isinstance(actions, tuple):
                    actions, token = actions
                # Old request failures cannot cancel a subsequent intervention.
                if ticket is self.pending:
                    self.pending = None
                    if isinstance(actions, Exception):
                        raise RuntimeError("Policy request failed") from actions
                    values = np.asarray(actions)
                    if (
                        values.ndim != 2
                        or values.shape[1:] != (14,)
                        or len(values) < self.horizon
                    ):
                        raise ValueError(
                            "Policy must supply the complete execution horizon (T,14)"
                        )
                    accepted = self.arbiter.accept_policy(
                        ticket, values[: self.horizon], now=now
                    )
                    if accepted:
                        self._accepted.append(
                            {
                                "observed_at": ticket.observed_at,
                                "accepted_at": now,
                                "actions": np.array(values, dtype=float, copy=True),
                                "continuous": ticket.continuous,
                                "skip": self.arbiter.last_skip,
                            }
                        )
                    if accepted and self.rl_active:
                        self._rl_decide(ticket, token, now, values)
                    self._request_state.pop(ticket, None)
                    if accepted and not ticket.continuous:
                        self.ready_after = now
                    # A late reply leaves HOLD; autonomous mode re-requests
                    # from a fresh observation below.
            if self.pending is not None and now - self.pending.observed_at > 2.0:
                raise RuntimeError("Policy request deadline exceeded")
            # Do not consume the first action against a pre-transition image.
            # The next post-boundary observation pairs with action zero, so a
            # complete 30-step chunk really yields 30 training rows.
            if (
                self.owner in {Ownership.HUMAN, Ownership.POLICY}
                and observed_at <= self.ready_after
            ):
                return None
            if self.owner is Ownership.HUMAN:
                self._fresh_leader(leader, sampled_at, now)
                if sampled_at < self.human_stamp:
                    raise ValueError("Leader timestamp regressed")
                if sampled_at > self.human_stamp:
                    target = self.relative.step(leader)
                    self.arbiter.update_human(target, sampled_at=sampled_at, now=now)
                    self.human_stamp = sampled_at
            if (
                self.owner is Ownership.POLICY
                and (self.rl_lead if self.rl_active else self.replan_lead)
                and 0
                < self.arbiter.remaining
                <= (self.rl_lead if self.rl_active else self.replan_lead)
                and self.pending is None
                and not self.worker.busy
            ):
                # Prefetch from this tick's observation BEFORE taking this
                # tick's action: that action already runs after the image, so
                # the reply skips it along with every later executed action.
                if observed_at > now:
                    raise ValueError("Future observation timestamp")
                try:
                    ticket = self.arbiter.request_policy(
                        observed_at=observed_at, now=now
                    )
                except ValueError:
                    ticket = None  # Retry prefetch with the next fresh frame.
                if ticket is not None:
                    self.pending = ticket
                    self._remember_state(ticket)
                    self.worker.submit(ticket, observation)
            action = self.arbiter.next_action(now=now)
            if self.owner is Ownership.FAULT:
                raise RuntimeError("Latched control fault")
            if (
                action is None
                and self.owner is Ownership.HOLD
                and self.autonomous
                and not self.worker.busy
            ):
                # A handback cannot use a camera observation from before HOLD.
                if observed_at > now:
                    raise ValueError("Future observation timestamp")
                try:
                    ticket = self.arbiter.request_policy(
                        observed_at=observed_at, now=now
                    )
                except ValueError:
                    return None  # Remain HOLD until a fresh post-boundary frame.
                self.pending = ticket
                self._remember_state(ticket)
                self.worker.submit(ticket, observation)
            return None if action is None else np.asarray(action)
        except Exception:
            self.fault(now=now)
            raise

    def _remember_state(self, ticket):
        # The actor's state is the proprioception at the observation the VLA
        # (and its RL token) saw, not at reply time, plus the steps of the
        # running chunk already fixed then (this tick's included).
        if self._qpos is not None:
            committed = []
            if self.rl_active and self.owner is Ownership.POLICY:
                count = min(self.arbiter.remaining, self.rl_lead)
                committed = self.arbiter.peek(count) if count else []
            self._request_state[ticket] = (
                self._qpos.copy(),
                self._previous_qpos.copy(),
                np.asarray(committed, dtype=float).reshape(-1, 14),
            )

    def _rl_decide(self, ticket, token, now, values):
        if token is None:
            raise RuntimeError("RL mode needs a model that serves rl_token (RLT round)")
        if ticket not in self._request_state:
            raise RuntimeError("RL decision lacks the observation's joint state")
        qpos, previous, committed = self._request_state[ticket]
        chunk = self.rl.chunk
        # Both the VLA reference and the actor chunk are aligned to the
        # observation; the first `skip` steps already ran (committed steps).
        reference = np.asarray(values[:chunk], dtype=float)
        skip = int(self.arbiter.last_skip)
        actions, record = self.rl.decide(
            token, qpos, previous, reference, committed, skip=skip
        )
        if skip < chunk:
            self.arbiter.replace_chunk(actions[skip:])
        else:
            self.arbiter.hold(now=now)  # the reply outlived its chunk: decide anew
        self._decisions.append(
            {
                **record,
                "skip": skip,
                "observed_at": ticket.observed_at,
                "accepted_at": now,
                "generation": self.arbiter.generation,
            }
        )

    def take_decisions(self):
        """RL actor decisions since the last call (for the recording)."""
        decisions, self._decisions = self._decisions, []
        return decisions

    def take_accepted(self):
        """Accepted chunks since the last call (complete reply, before skip)."""
        accepted, self._accepted = self._accepted, []
        return accepted

    def acknowledged(self, action):
        self.relative.last = np.asarray(action, dtype=np.float32).copy()


class TickLease:
    """Use the existing daemon leases, renewed ONLY by a progressing loop.

    A blocked sensor, IK, recorder or RPC must not be hidden by background
    heartbeats. No motor enable, gravity switch, homing or daemon startup here.
    """

    def __init__(self, safety, session_id):
        self.safety, self.session_id = safety, session_id
        self.active = False
        self.attempted = False
        self.failed = False

    def require_hold(self, status):
        if (
            status.get("state") != "safe_hold"
            or status.get("hold_confirmed") is not True
        ):
            raise RuntimeError("Existing confirmed SAFE_HOLD is required")

    @staticmethod
    def require_contract(status):
        if status.get("session_api_version") != 1:
            raise RuntimeError("Daemon requires exclusive session_api_version=1")

    def begin(self):
        status = self.safety.status()
        self.require_contract(status)
        self.require_hold(status)
        # The daemon decides ownership atomically, including abandoned leases.
        # Mark the attempt first: an ACK loss may still leave us owning HOLD.
        self.attempted = True
        self.safety.acquire_session(self.session_id)
        self.pulse()
        self.hold("HG-DAgger starts in HOLD")

    def check(self):
        if self.failed:
            raise RuntimeError("Latched daemon lease failure")
        status = self.safety.status()
        if (
            status.get("session_id") != self.session_id
            or status.get("strict_session") is not True
            or status.get("session_api_version") != 1
            or status.get("state") not in {"safe_hold", "active"}
            or status.get("stale_clients")
            or self.active
            and status.get("state") != "active"
        ):
            self.failed = True
            raise RuntimeError(
                "Unexpected daemon HOLD, lease expiry or session change; no automatic reactivation"
            )
        if not self.active and status.get("state") != "safe_hold":
            self.failed = True
            raise RuntimeError("Unexpected active daemon")
        return status

    def pulse(self):
        # Check BEFORE renewing: renewal must not erase evidence of expiration.
        self.check()
        for name in ("parent", "collector"):
            status = self.safety.heartbeat(name, self.session_id)
            if not status or status.get("state") in {"fault", "disarmed"}:
                self.failed = True
                raise RuntimeError("Daemon lease rejected or faulted")
        self.check()

    def activate(self):
        self.pulse()
        if not self.active:
            self.safety.activate(self.session_id)
            self.active = True
            self.check()

    def hold(self, reason):
        status = self.safety.safe_hold(reason, session_id=self.session_id)
        self.require_hold(status)
        if status.get("session_id") != self.session_id:
            raise RuntimeError("Daemon ownership changed during HOLD")
        self.active = False

    def close(self):
        if self.attempted:
            status = self.safety.status()
            if status.get("session_id") != self.session_id:
                raise RuntimeError("Cannot close another daemon session")
            self.hold("HG-DAgger exit; no home/reset")
            status = self.safety.release_session(self.session_id)
            self.require_hold(status)
            if (
                status.get("session_id") is not None
                or status.get("strict_session") is not False
            ):
                raise RuntimeError("Daemon did not confirm exclusive session release")
            self.attempted = False
