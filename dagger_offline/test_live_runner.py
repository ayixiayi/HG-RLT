"""Hardware-free tests of the actual run_robot loop and Pi0.5 data pipeline.

The fake daemon models the REQUIRED lease contract, not proof that a remote
daemon implements it. No hardware modules, sockets or GPU model are constructed.
"""

import importlib.util
import json
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dagger_offline.arbitration import Ownership
from dagger_offline.live_control import Controller, TickLease
from dagger_offline.live_recording import AsyncEpisodeRecorder, EpisodeRecorder
from dagger_offline.live_robot import GuardedSender
from dagger_offline.round_data import prepare_round, read_source

spec = importlib.util.spec_from_file_location(
    "hg_runner", Path(__file__).resolve().parents[1] / "Dagger-Practice/run_robot.py"
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Clock:
    value = 10.0
    tick = 0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        self.tick += 1
        assert self.tick < 400, "loop failed to stop"


class Safety:
    def __init__(self, h):
        self.h = h
        self.state = "safe_hold"
        self.session = None
        self.beats = {}
        self.activations = []
        self.holds = []
        self.begin_error = None
        self.hold_error = False
        self.release_error = None

    def status(self):
        now = self.h.clock.monotonic()
        stale = [
            k for k in ("parent", "collector") if now - self.beats.get(k, -1) > 0.2
        ]
        if self.session and stale:
            self.state = "safe_hold"
        return {
            "state": self.state,
            "session_id": self.session,
            "stale_clients": stale,
            "hold_confirmed": self.state == "safe_hold",
            "session_api_version": 1,
            "strict_session": self.session is not None,
            "joint_limits": np.tile([-3, 3], (14, 1)).tolist(),
        }

    def acquire_session(self, session):
        if self.begin_error == "before":
            raise RuntimeError("begin rejected")
        if self.session is not None:
            raise RuntimeError("daemon occupied")
        self.session = session
        self.beats = {k: self.h.clock.monotonic() for k in ("parent", "collector")}
        if self.begin_error == "after":
            raise TimeoutError("begin ACK lost")

    def release_session(self, session):
        assert session == self.session and self.state == "safe_hold"
        if self.release_error == "before":
            raise TimeoutError("release unavailable")
        self.session = None
        if self.release_error == "after":
            raise TimeoutError("release ACK lost")
        return self.status()

    def heartbeat(self, name, session):
        assert session == self.session
        self.beats[name] = self.h.clock.monotonic()
        return self.status()

    def activate(self, session):
        assert session == self.session and not self.status()["stale_clients"]
        self.state = "active"
        self.activations.append(self.h.clock.tick)
        self.h.hook("activate")

    def safe_hold(self, reason, *, session_id):
        assert session_id == self.session
        self.holds.append(reason)
        if self.hold_error:
            raise TimeoutError("HOLD RPC failed")
        self.state = "safe_hold"
        return self.status()


class Worker:
    busy = False

    def __init__(self, h):
        self.h, self.requests = h, []
        self.delay = 0.04
        self.failure = None

    def submit(self, ticket, observation):
        assert not self.busy
        self.busy = True
        self.ticket, self.obs = ticket, observation
        self.ready = self.h.clock.monotonic() + self.delay
        self.requests.append((ticket, observation))

    def poll(self):
        if not self.busy or self.h.clock.monotonic() < self.ready:
            return None
        self.busy = False
        actions = np.tile(self.obs["state"], (30, 1))
        actions[:, 0] += np.arange(1, 31) * 0.001
        actions[:, 7] -= np.arange(1, 31) * 0.0015
        return self.ticket, self.failure if self.failure is not None else actions


class Relative:
    anchors = 0

    def anchor(self, leader, qpos):
        self.last = qpos.copy()
        self.anchors += 1

    def step(self, leader):
        result = self.last.copy()
        result[0] -= 0.0005
        result[7] += 0.0007
        return result


class Harness:
    def __init__(self, path, keys=None, offset=0):
        self.path, self.keys = (
            path,
            dict(keys or {0: ["p"], 38: ["h"], 74: ["p"], 110: ["k"]}),
        )
        self.clock = Clock()
        self.clock.value += offset
        self.qpos = np.linspace(-0.3, 0.3, 14)
        self.qpos[[6, 13]] = [0.4, 0.65]
        self.sent, self.inbox = [], []
        self.hook = lambda stage: None
        self.safety = Safety(self)
        self.worker = Worker(self)
        self.relative = Relative()
        self.control = Controller(
            self.relative, self.worker, now=self.clock.monotonic()
        )
        self.sender = GuardedSender(
            self,
            np.tile([-3, 3], (14, 1)),
            self.qpos,
            session_id="synthetic-session",
            planner=self.plan,
            backpressure=lambda a, q, **kw: (a, ()),
        )
        self.lease = TickLease(self.safety, "synthetic-session")
        self.leader_state = {"left": np.zeros(7), "right": np.ones(7) * 0.1}
        self.leaders = None

    def plan(self, reference, target, limits, **kwargs):
        self.hook("plan")
        # An intentionally different filter from raw proposals makes accidental
        # raw-action training fail numeric assertions below.
        return SimpleNamespace(
            applied_action=reference + np.clip(target - reference, -0.0008, 0.0008),
            violations=[],
        )

    def command_joint_state(self, command, *, session_id):
        assert session_id == self.safety.session
        assert self.safety.state == "active"
        self.sent.append((self.clock.tick, self.control.owner, command.copy()))
        self.hook("ack")
        self.qpos = command.copy()
        self.clock.value += 0.001
        return {"applied_joint_state": command.copy()}

    def get_joint_state(self):
        return self.qpos.copy()

    def capture(self):
        now = self.clock.monotonic()
        images = {
            k: np.full((24, 32, 3), v, np.uint8)
            for k, v in (
                ("main", [205, 30, 10]),
                ("left", [20, 180, 40]),
                ("right", [10, 25, 200]),
            )
        }
        cameras = {"main": now + 0.0001, "left": now + 0.0002, "right": now + 0.0003}
        self.clock.value += 0.001
        snap = runner.Snapshot(
            self.qpos.copy(),
            images,
            cameras,
            now + 0.0004,
            now + 0.0008,
            now - np.linspace(0.003, 0.008, 14),
            {side: value.copy() for side, value in self.leader_state.items()},
            {"left": now, "right": now + 0.0005},
        )
        self.snapshot = snap
        self.hook("capture")
        return snap

    def poll(self):
        self.hook("poll")
        keys, self.inbox = self.inbox + self.keys.pop(self.clock.tick, []), []
        return keys

    def run(self, recorder_factory=EpisodeRecorder, *, review=None, metadata=None):
        with recorder_factory(
            self.path,
            task_name="choose_the_heavier_one",
            metadata={
                "synthetic": True,
                "offset": self.clock.value,
                **(metadata or {}),
            },
        ) as recorder:
            self.recorder = recorder
            runner.run_loop(
                control=self.control,
                sender=self.sender,
                lease=self.lease,
                recorder=recorder,
                capture=self.capture,
                poll=self.poll,
                build_observation=lambda s: {
                    "state": s.qpos.copy(),
                    "frame": min(s.camera_times.values()),
                },
                clock=self.clock,
                review=review,
                leaders=self.leaders,
            )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_full_chunks_hold_recording_and_first_action_alignment(
    tmp_path, capsys, asynchronous
):
    h = Harness(tmp_path / "loop.h5")
    # Fake time runs the entire episode without wall-clock sleeps. Let those
    # synthetic frames queue; production capacity stays 30, tested separately.
    h.run(
        partial(AsyncEpisodeRecorder, queue_size=256)
        if asynchronous
        else EpisodeRecorder
    )
    displayed = capsys.readouterr().out
    assert "HOLD · 保持" in displayed and "等待模型结果" in displayed
    assert (
        "POLICY_RUNNING · 策略控制" in displayed
        and "INTERVENTION · 人工跟随" in displayed
    )
    assert "已确认 HOLD 并释放控制权；记录已保存" in displayed
    assert "\033[" not in displayed
    with h5py.File(h.path) as f:
        synced, states, actions, labels, valid, _ = read_source(f, "current", 100)
        assert len(states) == 110  # Includes every HOLD frame, not only sends.
        assert np.sum(valid & (labels == "POLICY_RUNNING")) == 66
        assert np.sum(valid & (labels == "INTERVENTION")) == 35
        assert not valid[labels == "HOLD"].any()
        from deploy.recording import io as rio

        # Prefetch keeps each handback one continuous policy run across chunk
        # boundaries (30-step chunks), with no inference HOLD between them.
        assert [
            b - a for a, b in rio.true_ranges(valid & (labels == "POLICY_RUNNING"))
        ] == [34, 32]
        # First accepted policy target is paired with the first post-transition
        # observation, not dropped or shifted one step by latest-before joining.
        first = np.flatnonzero(valid & (labels == "POLICY_RUNNING"))[0]
        assert actions[first, 0] == pytest.approx(-0.3 + 0.0008)
        assert actions[first, 7] == pytest.approx(
            np.linspace(-0.3, 0.3, 14)[7] - 0.0008
        )
        np.testing.assert_allclose(actions[valid], np.array([a for _, _, a in h.sent]))
        assert np.all(f["pairing/command_ack_ns"][:][valid] > synced.base_ts_ns[valid])
        assert np.all(f["pairing/camera_ns"][:, 1] != synced.base_ts_ns)
        assert f.attrs["sample_alignment"] == "observation_command_pair_v2"
        assert f["pairing/follower_received_ns"].shape == (110, 14)
        assert np.all(
            f["pairing/follower_received_ns"][:, 13]
            < f["pairing/follower_received_ns"][:, 0]
        )
        assert not np.allclose(f["diagnostics/raw_action"][0], actions[first])
    assert h.control.owner is Ownership.HOLD and not h.lease.active
    assert h.safety.session is None
    assert all(obs["frame"] == ticket.observed_at for ticket, obs in h.worker.requests)


class Followers:
    def __init__(self, h, continuous=True):
        self.h, self.calls, self.continuous = h, [], continuous

    def follow(self, qpos):
        self.calls.append((self.h.clock.tick, "follow", np.asarray(qpos).copy()))

    def passive(self):
        self.calls.append((self.h.clock.tick, "passive", None))

    def raise_if_failed(self):
        pass


@pytest.mark.parametrize("cancel", [False, True])
def test_takeover_follows_h_after_the_countdown_with_leaders_aligned(
    tmp_path, cancel
):
    keys = {0: ["p"], 40: ["h"], 130: ["p"], 160: ["k"]}
    if cancel:
        keys[60] = ["s"]
    h = Harness(tmp_path / "countdown.h5", keys)
    h.leaders = Followers(h)

    def hook(stage):
        if stage == "poll":
            # Triggers no longer gate anything: a closed trigger at h is fine.
            for side in ("left", "right"):
                h.leader_state[side][6] = 0.1

    h.hook = hook
    h.run()
    human = [t for t, owner, _ in h.sent if owner is Ownership.HUMAN]
    if cancel:
        assert not human
        assert not any(40 < t < 130 for t, _, _ in h.sent)
    else:
        took = min(t for t, kind, _ in h.leaders.calls if kind == "passive")
        assert not any(40 < t < took for t, _, _ in h.sent)  # HOLD while armed
        assert 40 + 59 <= took <= 40 + 62  # 2 s at 30 Hz
        assert human and min(human) >= took and max(human) < 130
        assert all(
            kind == "passive" for t, kind, _ in h.leaders.calls if took <= t < 130
        )
        follows = [t for t, kind, _ in h.leaders.calls if kind == "follow"]
        assert follows and all(t < took or t >= 130 for t in follows)
        # The countdown leaders follow the full 14-D robot state (grippers too).
        assert all(q.shape == (14,) for t, kind, q in h.leaders.calls if kind == "follow")
    with h5py.File(h.path) as f:
        reasons = f["dagger_events/reason"].asstr()[:]
        assert "takeover armed" in reasons


def test_on_demand_leaders_only_align_while_takeover_is_armed(tmp_path):
    h = Harness(
        tmp_path / "on-demand.h5", {0: ["p"], 40: ["h"], 130: ["p"], 160: ["k"]}
    )
    h.leaders = Followers(h, continuous=False)
    h.run()
    follows = [t for t, kind, _ in h.leaders.calls if kind == "follow"]
    assert follows and min(follows) >= 40 and max(follows) < 40 + 62
    assert any(owner is Ownership.HUMAN for _, owner, _ in h.sent)
    assert any(owner is Ownership.POLICY and t > 130 for t, owner, _ in h.sent)


@pytest.mark.parametrize("stage", ["capture", "plan", "activate"])
def test_pending_stop_at_each_blocking_boundary_prevents_send(tmp_path, stage):
    h = Harness(tmp_path / "preempt.h5", {0: ["h"], 5: ["k"]})
    fired = False

    def hook(where):
        nonlocal fired
        if where == stage and h.clock.tick == 1 and not fired:
            fired = True
            h.inbox.append("s")

    h.hook = hook
    h.run()
    assert fired and not h.sent


def test_checkpoint_key_and_repeated_takeover_do_not_interrupt(tmp_path):
    h = Harness(tmp_path / "keys.h5", {0: ["h"], 2: ["h", "h"], 8: ["k"]})
    fired = False

    def hook(stage):
        nonlocal fired
        if stage == "plan" and h.clock.tick == 3 and not fired:
            h.inbox.append("c")
            fired = True

    h.hook = hook
    h.run()
    assert [t for t, _, _ in h.sent] == list(range(1, 8))
    assert h.relative.anchors == 1 and h.safety.activations == [1]
    with h5py.File(h.path) as f:
        assert len(f["dagger_checkpoints/timestamp_ns"]) == 1


def test_late_inference_cannot_override_takeover_and_handback_is_fresh(tmp_path):
    h = Harness(tmp_path / "slow.h5", {0: ["p"], 3: ["h"], 25: ["p"], 70: ["k"]})
    h.worker.delay = 0.45
    h.run()
    assert all(owner is Ownership.HUMAN for t, owner, _ in h.sent if 3 <= t < 25)
    assert any(owner is Ownership.POLICY for _, owner, _ in h.sent)
    assert h.worker.requests[1][0].observed_at > 10 + 25 / 30


@pytest.mark.parametrize("failure", [None, "hold", "lease", "deadline", "human"])
def test_policy_rejection_waits_for_operator_and_preserves_corrections(
    tmp_path, failure
):
    from rhospolicy.inference.yam_act import apply_tracking_backpressure, plan_action

    h = Harness(tmp_path / "rejection.h5", {0: ["p"], 15: ["h"], 52: ["p"], 86: ["k"]})
    poll = h.worker.poll

    def policy():
        result = poll()
        if result is not None and len(h.worker.requests) == 1:
            result[1][3:, 4] += 0.31  # Above the raw-jump limit.
        return result

    h.worker.poll = policy
    h.sender.backpressure = apply_tracking_backpressure
    rejected = []

    def plan(*args, **kwargs):
        result = plan_action(*args, **kwargs)
        if result.violations:
            rejected.append(h.clock.tick)
            h.inbox.append("p")  # A queued pre-rejection handback must be discarded.
            if failure == "hold":
                h.safety.hold_error = True
            elif failure == "lease":
                h.safety.state = "safe_hold"
            elif failure == "deadline":
                h.clock.value += 0.11
        return result

    h.sender.plan = plan
    if failure == "human":
        h.keys = {0: ["h"], 86: ["k"]}
        h.relative.step = lambda leader: (
            h.sender.feedback + np.eye(14)[4] * (0.31 if h.sent else 0)
        )
    if failure is not None:
        with pytest.raises((RuntimeError, BaseExceptionGroup)):
            h.run()
        assert rejected  # Must exercise planning rejection, not a later travel fault.
        assert h.control.owner is Ownership.FAULT
        with h5py.File(h.path) as f:
            assert not f.attrs["usable"] and not f.attrs["complete"]
        return

    h.run()
    assert len(rejected) == 1
    assert not any(rejected[0] <= tick <= 15 for tick, _, _ in h.sent)
    # No implicit inference retry in HOLD: after the rejected first chunk the
    # next request (and its prefetches) follow the explicit handback at tick 52.
    assert len(h.worker.requests) >= 2
    assert all(t.observed_at > 10 + 52 / 30 for t, _ in h.worker.requests[1:])
    assert h.safety.session is None
    with h5py.File(h.path) as f:
        _, _, actions, labels, valid, _ = read_source(f, "current", 100)
        assert f.attrs["usable"] and f.attrs["complete"]
        reasons = f["dagger_events/reason"].asstr()[:]
        assert any("prediction jump 0.310000 exceeds 0.300000" in r for r in reasons)
        assert "FAULT" not in f["dagger_events/controller_state"].asstr()[:]
        assert not valid[rejected[0]]  # Rejected raw action is diagnostics only.
        assert np.sum(valid & (labels == "INTERVENTION")) == 36
        from deploy.recording import io as rio

        assert [
            b - a for a, b in rio.true_ranges(valid & (labels == "POLICY_RUNNING"))
        ] == [3, 30]
        np.testing.assert_allclose(actions[valid], np.array([a for _, _, a in h.sent]))


@pytest.mark.parametrize(
    "fault",
    [
        "camera",
        "leader",
        "regression",
        "follower_stale",
        "follower_future",
        "follower_regression",
        "state",
        "ack",
        "stall",
        "lease",
        "deadline",
        "eof",
        "signal",
        "cold",
    ],
)
def test_faults_stop_and_journal_without_publishing(tmp_path, fault):
    h = Harness(tmp_path / f"{fault}.h5", {0: ["h"], 80: ["k"]})
    if fault == "cold":
        h.keys = {0: ["p"], 90: ["k"]}
        h.worker.delay = 20

    def hook(stage):
        if stage == "capture" and fault == "regression":
            h.snapshot.leader_times["left"] -= 0.1
        if h.clock.tick != 3:
            return
        if stage == "capture":
            if fault == "camera":
                raise OSError("camera disconnected")
            if fault == "leader":
                h.snapshot.leader_times["right"] -= 0.3
            if fault == "regression":
                # Left is still the minimum; checking only min(left,right)
                # would miss this right-arm timestamp regression.
                h.snapshot.leader_times["right"] -= 0.04
            if fault == "state":
                h.snapshot.qpos[7] = np.nan
            if fault == "follower_stale":
                h.snapshot.follower_times[13] -= 0.2
            if fault == "follower_future":
                h.snapshot.follower_times[0] += 0.2
            if fault == "follower_regression":
                h.snapshot.follower_times[10] -= 0.04
            if fault == "lease":
                h.safety.state = "safe_hold"
            if fault == "stall":
                h.clock.value += 0.3
            if fault == "eof":
                raise EOFError("terminal closed")
            if fault == "signal":
                raise KeyboardInterrupt("SIGTERM/SIGINT handler")
        if stage == "plan" and fault == "deadline":
            h.clock.value += 0.11
        if stage == "ack" and fault == "ack":
            raise TimeoutError("command ACK lost")

    h.hook = hook
    with pytest.raises(
        (RuntimeError, OSError, ValueError, EOFError, KeyboardInterrupt)
    ):
        h.run()
    assert h.control.owner is Ownership.FAULT and h.safety.state == "safe_hold"
    assert not h.lease.active
    assert not any(t > 3 for t, _, _ in h.sent)
    assert len(h.safety.activations) <= 1
    with h5py.File(h.path) as f:
        assert not f.attrs["usable"] and not f.attrs["complete"]
        assert f["dagger_events/controller_state"][-1].decode() == "FAULT"


@pytest.mark.parametrize("mode", ["before", "after", "other", "hold_failure"])
def test_start_and_cleanup_failure_ownership(tmp_path, mode):
    h = Harness(tmp_path / f"start-{mode}.h5")
    if mode == "other":
        h.safety.session = "somebody-else"
    elif mode == "hold_failure":
        h.safety.hold_error = True
    else:
        h.safety.begin_error = mode
    with pytest.raises((RuntimeError, TimeoutError, BaseExceptionGroup)):
        h.run()
    assert not h.sent and not h.safety.activations
    if mode in {"before", "other"}:
        assert not h.safety.holds  # Do not manipulate an unowned session.
    else:
        assert h.safety.holds
    with h5py.File(h.path) as f:
        assert not f.attrs["usable"]
        assert f["dagger_events/controller_state"][-1].decode() == "FAULT"


@pytest.mark.parametrize("mode", ["quit", "whole", "partial"])
def test_save_and_discard_through_operator_keys(tmp_path, mode):
    keys = {0: ["h"], 12: ["q" if mode == "quit" else "k"]}
    if mode == "partial":
        keys[4] = ["c"]
    if mode != "quit":
        keys[8] = ["x"]
    h = Harness(tmp_path / f"discard-{mode}.h5", keys)
    h.run()
    with h5py.File(h.path) as f:
        assert bool(f.attrs["usable"]) == (mode == "partial")
        if mode == "partial":
            *_, valid, _ = read_source(f, "current", 100)
            assert valid[1:4].all() and not valid[5:].any()


@pytest.mark.parametrize("loss", ["before", "after"])
def test_release_failure_never_publishes_recording(tmp_path, loss):
    h = Harness(tmp_path / "release.h5", {0: ["h"], 8: ["k"]})
    h.safety.release_error = loss
    with pytest.raises((TimeoutError, BaseExceptionGroup)):
        h.run()
    with h5py.File(h.path) as f:
        assert not f.attrs["usable"] and not f.attrs["complete"]
        assert f["dagger_events/controller_state"][-1].decode() == "FAULT"
    assert h.safety.state == "safe_hold"


@pytest.mark.parametrize("outcome", ["success", "fail", "discard", "interrupted"])
def test_result_review_waits_in_released_hold_and_persists_outcome(tmp_path, outcome):
    h = Harness(tmp_path / "labelled.h5", {0: ["h"], 8: ["k"]})

    def review():
        assert h.safety.state == "safe_hold" and h.safety.session is None
        assert h.control.owner is Ownership.HOLD
        h.clock.value += 30  # Operator may take arbitrarily long to decide.
        if outcome == "interrupted":
            raise EOFError("operator terminal closed during review")
        return outcome

    if outcome == "interrupted":
        with pytest.raises(EOFError):
            h.run(review=review)
    else:
        h.run(review=review)
    assert max(tick for tick, _, _ in h.sent) < 8
    with h5py.File(h.path) as recording:
        assert recording.attrs["outcome"] == (
            "aborted" if outcome == "interrupted" else outcome
        )
        assert bool(recording.attrs["usable"]) == (outcome in ("success", "fail"))
        assert bool(recording.attrs["complete"]) == (outcome != "interrupted")


@pytest.mark.parametrize("writer_class", [EpisodeRecorder, AsyncEpisodeRecorder])
def test_confirmed_save_survives_slow_hold_and_release(tmp_path, writer_class):
    h = Harness(tmp_path / "slow-release.h5", {0: ["h"], 8: ["k"]})
    close = h.lease.close

    def slow_close():
        close()
        # Aggregate latency of status/HOLD/release, not one over-timeout RPC.
        h.clock.value += 0.24

    h.lease.close = slow_close
    h.run(writer_class)
    assert h.control.owner is Ownership.HOLD and h.safety.session is None
    with h5py.File(h.path) as f:
        assert f.attrs["usable"] and f.attrs["complete"]
        assert "FAULT" not in f["dagger_events/controller_state"].asstr()[:]
        assert len(f["data/q_left"]) == 8


@pytest.mark.parametrize("during_plan", [False, True])
def test_explicit_stop_does_not_expire_during_confirmed_hold(tmp_path, during_plan):
    h = Harness(tmp_path / "slow-hold.h5", {0: ["h"], 8: ["k"]})
    hold = h.safety.safe_hold
    if not during_plan:
        h.keys[3] = ["s"]

    def slow_hold(reason, *, session_id):
        if h.clock.tick == 3:
            h.clock.value += 0.08  # Within the 100ms safety RPC timeout.
        return hold(reason, session_id=session_id)

    def hook(stage):
        if stage == "capture":
            for side in h.snapshot.leader_times:
                h.snapshot.leader_times[side] -= 0.12  # Fresh under the 200ms guard.
        if during_plan and stage == "plan" and h.clock.tick == 3:
            h.inbox.append("s")

    h.safety.safe_hold = slow_hold
    h.hook = hook
    h.run(AsyncEpisodeRecorder)
    assert [tick for tick, _, _ in h.sent] == [1, 2]
    assert h.control.owner is Ownership.HOLD and h.safety.session is None
    with h5py.File(h.path) as f:
        assert f.attrs["usable"] and f.attrs["complete"]
        assert "FAULT" not in f["dagger_events/controller_state"].asstr()[:]


def test_interrupt_after_commit_does_not_claim_recording_unpublished(tmp_path, capsys):
    class InterruptedRecorder(AsyncEpisodeRecorder):
        def finish(self, *, keep):
            super().finish(keep=keep)
            raise KeyboardInterrupt("interrupted after storage committed")

    h = Harness(tmp_path / "committed.h5", {0: ["h"], 8: ["k"]})
    with pytest.raises(KeyboardInterrupt):
        h.run(InterruptedRecorder)
    with h5py.File(h.path) as f:
        assert f.attrs["usable"] and f.attrs["complete"]
    output = capsys.readouterr().out
    assert "保存结果未确认" in output
    assert "本条记录未发布为训练数据" not in output


@pytest.mark.parametrize("fault", [None, "capture", "write", "journal"])
def test_async_disk_wait_only_after_confirmed_hold_and_release(tmp_path, capsys, fault):
    entered, resume, released = (threading.Event() for _ in range(3))

    class BlockedWriter(EpisodeRecorder):
        def sample(self, *args, **kwargs):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test storage gate was not released")
            if fault == "write":
                raise OSError("disk full during save")
            super().sample(*args, **kwargs)

        def abort(self, timestamp_ns, reason):
            if fault == "journal":
                raise OSError("journal unavailable")
            super().abort(timestamp_ns, reason)

    h = Harness(tmp_path / "drain.h5", {0: ["h"], 8: ["k"]})
    release = h.safety.release_session

    def release_session(session):
        status = release(session)
        released.set()
        return status

    def hook(stage):
        if stage == "capture" and h.clock.tick == 1:
            assert entered.wait(2)
        if fault in {"capture", "journal"} and stage == "capture" and h.clock.tick == 3:
            raise OSError("camera disconnected during slow write")

    h.safety.release_session = release_session
    h.hook = hook
    errors = []

    def run():
        try:
            h.run(partial(AsyncEpisodeRecorder, writer_factory=BlockedWriter))
        except BaseException as exc:  # noqa: BLE001 - assert thread failures below
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(2) and released.wait(2)
        status = h.safety.status()
        assert status["state"] == "safe_hold" and status["hold_confirmed"]
        assert status["session_id"] is None and thread.is_alive()
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(errors) == int(fault is not None)
    if fault == "capture":
        assert isinstance(errors[0], OSError)
    elif fault == "write":
        assert isinstance(errors[0], RuntimeError) and "disk full" in str(errors[0])
    elif fault == "journal":
        assert isinstance(errors[0], BaseExceptionGroup)
        assert [str(exc) for exc in errors[0].exceptions] == [
            "camera disconnected during slow write",
            "journal unavailable",
        ]
    else:
        output = capsys.readouterr().out
        assert output.index("正在等待落盘") < output.index("记录已保存")
    with h5py.File(h.path) as f:
        assert bool(f.attrs["usable"]) == (not fault)
        assert bool(f.attrs["complete"]) == (not fault)
        if fault != "write":
            # Already queued ACKs remain useful for diagnosing a camera fault.
            assert len(f["data/q_left"]) == (3 if fault else 8)
        events = f["dagger_events/controller_state"].asstr()[:].tolist()
        assert events.count("FAULT") == int(fault in {"capture", "write"})


def test_async_writer_failure_during_planning_prevents_motor_command(tmp_path):
    entered, fail = threading.Event(), threading.Event()

    class FailedWriter(EpisodeRecorder):
        def sample(self, *args, **kwargs):
            entered.set()
            if not fail.wait(5):
                raise TimeoutError("test failure gate was not released")
            raise OSError("disk full")

    h = Harness(tmp_path / "disk-failure.h5", {0: ["h"], 8: ["k"]})
    reached_planning = False

    def hook(stage):
        nonlocal reached_planning
        if stage == "plan" and h.clock.tick == 1:
            reached_planning = True
            assert entered.wait(2)
            fail.set()
            deadline = time.monotonic() + 2
            while True:
                try:
                    h.recorder.raise_if_failed()
                except RuntimeError:
                    return
                assert time.monotonic() < deadline, "writer did not report failure"
                time.sleep(0.001)

    h.hook = hook
    try:
        with pytest.raises(RuntimeError, match="recorder failed: disk full"):
            h.run(partial(AsyncEpisodeRecorder, writer_factory=FailedWriter))
    finally:
        fail.set()
    assert reached_planning and not h.sent
    assert h.control.owner is Ownership.FAULT
    assert h.safety.state == "safe_hold" and h.safety.session is None
    with h5py.File(h.path) as f:
        assert not f.attrs["usable"] and not f.attrs["complete"]
        assert f["dagger_events/controller_state"][-1].decode() == "FAULT"


def test_keyboard_drains_typeahead_without_waiting_for_another_key(monkeypatch):
    import os
    import pty
    import select
    import termios

    master, slave = pty.openpty()
    try:
        with os.fdopen(os.dup(slave), "r") as terminal:
            monkeypatch.setattr(runner.sys, "stdin", terminal)
            original = termios.tcgetattr(slave)
            with runner.keyboard() as poll:
                os.write(master, b"Hsp")
                assert select.select([slave], [], [], 1)[0]
                assert poll() == ["h", "s", "p"]
                assert poll() == []
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def test_execute_rejects_unattended_before_hardware_import(monkeypatch):
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(RuntimeError, match="onsite"):
        runner.execute(None, None, None, None, None, None)
    with pytest.raises(RuntimeError, match="onsite"):
        runner.prepare_start(None, None, None)


@pytest.mark.parametrize(
    "distance,residual,confirmation,error",
    [
        (1.25, 0.035, "", None),
        (0.0, 0.041, "", "feedback did not settle"),
        (1.25, 0.0, "cancel", "已取消归零"),
    ],
)
def test_prepare_start_full_range_feedback_and_confirmation(
    tmp_path, monkeypatch, distance, residual, confirmation, error
):
    from dagger_offline import homing

    h = Harness(tmp_path / "unused.h5")
    h.qpos[:] = 0
    h.qpos[8] = distance
    h.qpos[[6, 13]] = [0.3, 0.7]
    bias = np.zeros(14)
    bias[3] = residual
    h.get_joint_state = lambda: h.qpos.copy() + bias
    initial = h.get_joint_state()
    h.close = h.safety.close = lambda: None
    monkeypatch.setattr("gello.safety.rpc.SafetyClient", lambda **kw: h.safety)
    monkeypatch.setattr("gello.zmq_core.robot_node.ZMQClientRobot", lambda **kw: h)
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: confirmation)
    monkeypatch.setattr(
        homing,
        "move_to_start",
        partial(homing.move_to_start, clock=h.clock.monotonic, sleep=h.clock.sleep),
    )
    profile = SimpleNamespace(
        network=SimpleNamespace(host="127.0.0.1", safety_port=1, hardware_port=2)
    )
    target = np.array([0, 0, 0, 0, 0, 0, 1] * 2, dtype=float)
    prepare = partial(
        runner.prepare_start,
        SimpleNamespace(output=tmp_path / "prepare.h5"),
        profile,
        {"state": target.tolist()},
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            prepare()
    else:
        prepare()
    assert h.safety.state == "safe_hold" and h.safety.session is None
    if confirmation:
        assert not h.sent
    else:
        commands = np.array([entry[2] for entry in h.sent])
        np.testing.assert_allclose(commands[-1], target, atol=1e-12)
        steps = np.abs(np.diff(np.vstack([initial, commands]), axis=0))
        assert np.max(steps[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]) <= 0.01 + 1e-12
        assert np.max(steps[:, [6, 13]]) <= 0.02 + 1e-12


@pytest.mark.parametrize(
    "tty,no_color,term,colored",
    [
        (True, False, "xterm", True),
        (False, False, "xterm", False),
        (True, True, "xterm", False),
        (True, False, "dumb", False),
    ],
)
def test_console_color_is_optional_and_logs_remain_plain(
    monkeypatch, tty, no_color, term, colored
):
    import io

    from dagger_offline.console import message

    stream = io.StringIO()
    stream.isatty = lambda: tty
    monkeypatch.setenv("TERM", term)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    message("模式", "HOLD · 保持", level="warning", stream=stream)
    output = stream.getvalue()
    assert "[模式]" in output and "HOLD · 保持" in output
    assert ("\033[" in output) is colored


@pytest.mark.parametrize(
    "blocked", ["unattended", "denied", "safety_port", "hardware_port"]
)
def test_daemon_start_rejects_without_enabling(tmp_path, monkeypatch, blocked):
    import runpy
    import socket

    daemon = runpy.run_path(str(Path(runner.__file__).with_name("daemon.py")))
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(isatty=lambda: blocked != "unattended")
    )
    monkeypatch.setattr(
        "builtins.input", lambda _: "no" if blocked == "denied" else "HG-DAEMON-ON-SITE"
    )
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        ports = {"safety_port": 0, "hardware_port": 0}
        if blocked in ports:
            ports[blocked] = occupied.getsockname()[1]
        profile = SimpleNamespace(network=SimpleNamespace(**ports))
        with pytest.raises(RuntimeError, match="onsite|not authorized|occupied"):
            daemon["start"](tmp_path, profile)


@pytest.mark.parametrize(
    "state,owner,answer,expected",
    [
        ("fault", "faulted-owner", "off", ("disarm", "faulted-owner")),
        ("fault", None, "off", ("disarm", "fresh")),
        ("safe_hold", None, "off", ("begin+disarm", "fresh")),
        ("active", "live-owner", "off", "仍在控制"),
        ("safe_hold", None, "no", "not authorized"),
    ],
)
def test_daemon_stop_uses_off_api_and_waits_for_exit(
    tmp_path, monkeypatch, capsys, state, owner, answer, expected
):
    import runpy

    daemon = runpy.run_path(str(Path(runner.__file__).with_name("daemon.py")))
    calls = []
    status = {"state": state, "pid": None, "reason": "x", "session_id": owner}
    safety = SimpleNamespace(
        status=lambda: status,
        begin_session=lambda s: calls.append(("begin", s)),
        disarm=lambda s, c: calls.append(("disarm", s, c)),
        close=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "gello.safety.rpc",
        SimpleNamespace(SafetyClient=lambda **kw: safety),
    )
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: answer)
    monkeypatch.setitem(
        daemon["stop"].__globals__,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="fresh")),
    )
    profile = SimpleNamespace(
        network=SimpleNamespace(host="127.0.0.1", safety_port=6002)
    )
    if isinstance(expected, str):
        with pytest.raises(RuntimeError, match=expected):
            daemon["stop"](profile)
        assert not any(call[0] == "disarm" for call in calls)
        return
    daemon["stop"](profile)
    kind, session = expected
    disarms = [call for call in calls if call[0] == "disarm"]
    assert disarms == [("disarm", session, "off")]
    assert (("begin", session) in calls) == (kind == "begin+disarm")
    assert "faulted-owner" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["ready", "legacy", "feedback_missing", "fault"])
def test_daemon_check_is_read_only_and_requires_actual_capabilities(
    monkeypatch, capsys, mode
):
    import runpy

    daemon = runpy.run_path(str(Path(runner.__file__).with_name("daemon.py")))
    channels = {"left": "can_follower_l", "right": "can_follower_r"}
    status = {
        "state": "fault" if mode == "fault" else "safe_hold",
        "hold_confirmed": True,
        "session_api_version": None if mode == "legacy" else 1,
        "protocol_version": 2,
        "hardware_port": 6001,
        "channels": channels,
        "watchdogs": {side: {i: 8000 for i in range(1, 8)} for side in channels},
        "session_id": "must-not-be-printed",
    }
    safety = SimpleNamespace(status=lambda: status, close=lambda: None)
    robot = SimpleNamespace(
        get_joint_sample_capabilities=lambda: {
            "supported": mode != "feedback_missing",
            "api_version": 1,
            "num_dofs": 14,
            "time_base": "host_monotonic_can_receive",
        },
        close=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "gello.safety.rpc",
        SimpleNamespace(SafetyClient=lambda **kw: safety),
    )
    monkeypatch.setitem(
        sys.modules,
        "gello.zmq_core.robot_node",
        SimpleNamespace(ZMQClientRobot=lambda **kw: robot),
    )
    profile = SimpleNamespace(
        network=SimpleNamespace(host="127.0.0.1", hardware_port=6001, safety_port=6002),
        arms={
            side: SimpleNamespace(can_channel=channel)
            for side, channel in channels.items()
        },
    )
    if mode == "ready":
        daemon["check"](profile)
    else:
        with pytest.raises(RuntimeError, match="session_api|timestamps|SAFE_HOLD"):
            daemon["check"](profile)
    output = capsys.readouterr().out
    assert "must-not-be-printed" not in output
    assert ("CONTRACT OK" in output) == (mode == "ready")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("checkpoint", "checkpoints/pi05_yam_heavier_0928/39999"),
        ("port", 8768),
        ("host", "0.0.0.0"),
        ("action_space", "eef"),
        ("instruction", "different_task"),
        ("direct_policy_targets", True),
    ],
)
def test_preflight_rejects_wrong_baseline_before_adapter_import(
    tmp_path, monkeypatch, field, bad
):
    import yaml

    config = {
        "train_config": "pi05_yam_heavier",
        "repo_id": "local/balance_0922_train",
        "port": 8767,
        "host": "127.0.0.1",
        "action_space": "joints",
        "instruction": "choose_the_heavier_one",
        "checkpoint": "checkpoints/pi05_yam_heavier/pi05_yam_20k_20260924_023553/39999",
    }
    config[field] = bad
    deployment = tmp_path / "configs/deployments/pi05_yam_heavier.yaml"
    deployment.parent.mkdir(parents=True)
    deployment.write_text(yaml.safe_dump(config))
    replies = iter([runner.REVISION, ""])
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: next(replies)
    )
    monkeypatch.setattr(runner.sys, "path", list(sys.path))
    with pytest.raises(ValueError, match="old baseline"):
        runner.preflight(
            SimpleNamespace(
                rhos_root=tmp_path, abc=tmp_path, output=tmp_path / "unused.h5"
            )
        )


@pytest.mark.parametrize("missing", ["pyrealsense2", "client_server.ws.model_client"])
def test_preflight_checks_lazily_loaded_runtime_dependencies(
    tmp_path, monkeypatch, missing
):
    import builtins

    import yaml

    deployment = tmp_path / "configs/deployments/pi05_yam_heavier.yaml"
    deployment.parent.mkdir(parents=True)
    deployment.write_text(
        yaml.safe_dump(
            {
                "train_config": "pi05_yam_heavier",
                "repo_id": "local/balance_0922_train",
                "port": 8767,
                "host": "127.0.0.1",
                "action_space": "joints",
                "instruction": "choose_the_heavier_one",
                "checkpoint": "checkpoints/pi05_yam_heavier/pi05_yam_20k_20260924_023553/39999",
            }
        )
    )
    replies = iter([runner.REVISION, ""])
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: next(replies)
    )
    monkeypatch.setattr(runner.sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(f"Unavailable: {name}", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    output = tmp_path / "unused.h5"
    with pytest.raises(ModuleNotFoundError, match=f"Unavailable: {missing}"):
        runner.preflight(
            SimpleNamespace(rhos_root=tmp_path, abc=tmp_path, output=output)
        )
    assert not output.exists()


@pytest.mark.parametrize("sealed", [False, True])
def test_execute_startup_failure_closes_owned_readers_and_journals(
    tmp_path, monkeypatch, sealed
):
    import dagger_offline.deployment as deployment_module
    import dagger_offline.live_robot as robot_module

    closed = []
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("Recording asked for a magic word")
    )
    safety = SimpleNamespace(
        status=lambda: {
            "joint_limits": np.tile([-3, 3], (14, 1)),
            "session_api_version": 1,
        },
        close=lambda: closed.append("safety"),
    )
    robot = SimpleNamespace(
        get_joint_sample=lambda: {"joint_state": np.zeros(14)},
        get_joint_sample_capabilities=lambda: {
            "supported": True,
            "api_version": 1,
            "num_dofs": 14,
            "time_base": "host_monotonic_can_receive",
        },
        close=lambda: closed.append("robot"),
    )
    modules = {
        "gello.cameras.realsense_rig": {"CameraRig": None},
        "gello.safety.rpc": {"SafetyClient": lambda **kw: safety},
        "gello.zmq_core.robot_node": {"ZMQClientRobot": lambda **kw: robot},
        "rhospolicy.inference.yam_act": {
            "unpack_action": np.asarray,
            "validate_joint_limits": np.asarray,
        },
        "rhospolicy.inference.yam_pi05": {
            "Pi05PolicyClient": None,
            "build_pi05_observation": None,
            "wait_for_camera_bundle": None,
        },
        "run_pi05_heavier": {"require_start": lambda *args: None},
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    calls = []

    def leader(config, **_options):
        calls.append(config)
        if len(calls) == 2:
            raise OSError("second leader unavailable")
        return SimpleNamespace(close=lambda: closed.append("left"), port="/dev/l")

    monkeypatch.setattr(robot_module, "Leader", leader)
    config = tmp_path / "arm.yaml"
    config.write_text("{}")
    profile = SimpleNamespace(
        network=SimpleNamespace(host="unused", safety_port=1, hardware_port=2),
        arms={k: SimpleNamespace(config_path=config) for k in ("left", "right")},
    )
    path = tmp_path / "startup.h5"
    if sealed:

        def wrong_model(*args, **kwargs):
            raise ValueError("Policy identity mismatch")

        monkeypatch.setattr(deployment_module, "SealedPolicyClient", wrong_model)
        with pytest.raises(ValueError, match="identity"):
            runner.execute(
                SimpleNamespace(output=path),
                {"instruction": "synthetic", "host": "127.0.0.1", "port": 8769},
                profile,
                None,
                None,
                {"sealed_policy_id": "a" * 64},
            )
        assert not path.exists() and not closed and not calls
        return
    with pytest.raises(OSError, match="second leader"):
        runner.execute(
            SimpleNamespace(output=path),
            {"instruction": "synthetic", "host": "127.0.0.1", "port": 8767},
            profile,
            None,
            None,
            {},
        )
    assert closed == ["left", "robot", "safety"]
    with h5py.File(path) as f:
        assert not f.attrs["usable"] and not f.attrs["complete"]
        assert f["dagger_events/controller_state"][-1].decode() == "FAULT"


def test_dropped_camera_tick_splits_instead_of_stitching_policy_windows(tmp_path):
    from deploy.recording import io as rio

    h = Harness(tmp_path / "gap.h5")
    capture = h.capture

    def gapped():
        if h.clock.tick == 10:
            h.clock.value += 0.06
        return capture()

    h.capture = gapped
    h.run()
    with h5py.File(h.path) as f:
        _, _, _, labels, valid, _ = read_source(f, "current", 100)
        assert not valid[10] and valid[9] and valid[11]
        assert [
            b - a for a, b in rio.true_ranges(valid & (labels == "POLICY_RUNNING"))
        ] == [6, 27, 32]


@pytest.mark.parametrize(
    "damage",
    [
        "no_ack",
        "old_camera",
        "late_ack",
        "subframe_boundary",
        "old_follower",
        "future_follower",
        "regressed_follower",
    ],
)
def test_paired_timing_revalidated_at_export(tmp_path, damage):
    h = Harness(tmp_path / "timing.h5")
    h.run()
    with h5py.File(h.path, "r+") as f:
        if damage == "no_ack":
            f["pairing/action_valid"][20] = False
        elif damage == "old_camera":
            f["pairing/camera_ns"][20, 2] = f["pairing/camera_ns"][19, 2]
        elif damage == "late_ack":
            f["pairing/command_ack_ns"][20] += 100_000_000
        elif damage == "old_follower":
            f["pairing/follower_received_ns"][20, 13] -= 101_000_000
        elif damage == "future_follower":
            f["pairing/follower_received_ns"][20, 6] = (
                f["pairing/state_received_ns"][20] + 1
            )
        elif damage == "regressed_follower":
            f["pairing/follower_received_ns"][20, 7] = (
                f["pairing/follower_received_ns"][19, 7] - 1
            )
        else:
            events = f["dagger_events"]
            at = int(f["timestamps/top"][20])
            times = np.r_[
                events["timestamp_ns"][:], np.array([at + 1, at + 2], dtype=np.uint64)
            ]
            order = np.argsort(times, kind="stable")
            states = np.r_[events["controller_state"][:], [b"HOLD", b"POLICY_RUNNING"]]
            for name, values in (("timestamp_ns", times), ("controller_state", states)):
                events[name].resize(len(times), axis=0)
                events[name][:] = values[order]
        _, _, _, labels, valid, _ = read_source(f, "current", 100)
        assert labels[20] == "POLICY_RUNNING" and not valid[20]
        assert valid[19] and valid[22]


def _openpi_cache():
    from openpi.shared import download

    cache = Path(download.get_cache_dir())
    if not (cache / "big_vision/paligemma_tokenizer.model").is_file():
        pytest.fail(f"OpenPI tokenizer missing from cache {cache}; tests stay offline")
    return cache


def test_record_export_lerobot_and_real_openpi_training_batch(tmp_path, monkeypatch):
    """No optional skip: run in the LeRobot/OpenPI CPU environment."""
    from dagger_offline.collection import run_collection
    from dagger_offline.lerobot_bridge import convert_round
    from dagger_offline.training import (
        MixtureSampler,
        WindowDataset,
        build_config,
        check_data,
        create_loader,
    )
    from dagger_offline.workflow import assign_splits

    records = []
    for i, split in enumerate(("train", "val")):
        h = Harness(tmp_path / f"previous-{split}.h5", offset=i * 100)
        # Exercise the live async recorder all the way to the actual CPU batch.
        # Capacity only differs because this harness advances time without sleep.
        h.run(partial(AsyncEpisodeRecorder, queue_size=256))
        records.append(
            {
                "path": str(h.path),
                "group": f"synthetic-scene-{i}",
                "role": "previous",
                "split": split,
            }
        )
    attempts = iter(((200, "success"), (300, "fail")))

    def record_episode(path, group):
        offset, outcome = next(attempts)
        Harness(path, offset=offset).run(
            partial(AsyncEpisodeRecorder, queue_size=256),
            review=lambda: outcome,
            metadata={"collection": {"group": group}},
        )

    records_json = run_collection(
        tmp_path / "collection", record_episode, episodes=2, prompt=lambda _: ""
    )
    collected = json.loads(records_json.read_text())
    assert [row["outcome"] for row in collected] == ["success", "fail"]
    assert all("split" not in row for row in collected)
    records.extend(
        assign_splits(
            [
                {**row, "path": str(records_json.parent / row["path"])}
                for row in collected
            ]
        )
    )
    stats = tmp_path / "parent_norm.json"
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    k: {
                        "mean": [0.0] * 14,
                        "std": [1.0] * 14,
                        "q01": [-2.0] * 14,
                        "q99": [2.0] * 14,
                    }
                    for k in ("state", "actions")
                }
            }
        )
    )
    root = tmp_path / "round"
    manifest = prepare_round(records, root, stats, image_size=(24, 32))
    counts = {
        c: sum(
            e["windows"]
            for e in manifest["episodes"]
            if e["component"] == c and e["split"] == "train"
        )
        for c in ("previous", "intervention", "autonomous")
    }
    # Continuous policy runs: one handback yields several 30-frame windows.
    assert counts == {"previous": 14, "intervention": 6, "autonomous": 8}
    # Every exported slice comes from real loop output and a single interval;
    # HOLD observations are retained in HDF5, never stitched into a chunk.
    for e in manifest["episodes"]:
        a, b = e["frame_range"]
        with h5py.File(e["source"]) as f:
            _, _, commands, labels, valid, _ = read_source(f, "current", 100)
            assert valid[a:b].all() and len(set(labels[a:b])) == 1
            rows = np.fromfile(
                root / e["path"] / "states_actions.bin", dtype=np.float64
            ).reshape(-1, 28)
            np.testing.assert_array_equal(rows[:, 14:], commands[a:b])
    output = tmp_path / "lerobot"
    convert_round(root / "round.json", output)
    path = output / "lerobot_round.json"
    ds = WindowDataset(path, "train")
    assert [b - a for a, b in ds.component_ranges] == [14, 6, 8]
    samples = list(MixtureSampler(ds, seed=37, start=0, stop=10000))
    frequencies = [
        sum(a <= x < b for x in samples) / 10000 for a, b in ds.component_ranges
    ]
    np.testing.assert_allclose(frequencies, [0.8, 0.1, 0.1], atol=0.015)
    for a, b in ds.component_ranges:
        for index in (a, b - 1):
            row = ds[index]
            assert row["action"].shape == (30, 14) and not row["action_is_pad"].any()
    cfg = build_config(
        path,
        tmp_path / "store",
        tmp_path / "parent/params",
        name="synthetic_r1",
        exp_name="cpu",
        batch_size=2,
    )
    assert check_data(cfg, path) == {"train": counts, "val": counts}
    obs, actions = next(iter(create_loader(cfg, path, start_step=0, num_batches=1)))
    assert actions.shape == (2, 30, 32) and obs.state.shape == (2, 32)
    assert all(np.asarray(v).all() for v in obs.image_masks.values())
    idx = next(iter(MixtureSampler(ds, seed=cfg.seed, start=0, stop=2)))
    row = ds[idx]
    expected = row["action"].numpy().copy()
    arms = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    expected[:, arms] -= row["observation.state"].numpy()[arms]
    np.testing.assert_allclose(np.asarray(actions[0, :, :14]), expected / 2, atol=1e-6)
    assert (output / "norm_stats.json").read_bytes() == stats.read_bytes()

    # Complete the handoff into deployment using real Orbax commit metadata and
    # OpenPI's asset writer, but tiny synthetic params instead of GPU training.
    import orbax.checkpoint as ocp
    from openpi.shared import normalize
    from openpi.training import checkpoints
    from scripts import train

    from dagger_offline.deployment import seal, serving_config
    from dagger_offline.training import run_training

    parent = tmp_path / "parent"
    (parent / "params").mkdir(parents=True)
    (parent / "assets/local/balance_0922_train").mkdir(parents=True)
    (parent / "assets/local/balance_0922_train/norm_stats.json").write_bytes(
        stats.read_bytes()
    )

    # Exercise the public lab/school commands, not only their individual modules.
    # The parent payload is deliberately tiny: this proves packaging and training
    # inputs, not actual Pi0.5 optimization or model quality.
    import os
    import shutil
    import subprocess
    import tarfile

    from dagger_offline.test_oss_transfer import make_fake_oss
    from dagger_offline.workflow import main as workflow

    (parent / "params/synthetic").write_bytes(b"not-real-pi05-weights")
    for split in ("train", "val"):
        (output / f"previous/{split}/meta/split.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "source_root_relative": "../../../baseline_source",
                    "source_episode_indices": [0, 2, 4]
                    if split == "train"
                    else [1, 3, 5],
                }
            )
        )
    video = next((output / "previous/train/videos").rglob("*.mp4"))
    external_video = tmp_path / "external-bc.mp4"
    video.rename(external_video)
    video.symlink_to(os.path.relpath(external_video, video.parent))
    code_root = Path(__file__).resolve().parents[1]
    prepared = tmp_path / "prepared"
    prepare_args = [
        "prepare-data",
        "--records",
        str(records_json),
        "--output",
        str(prepared),
        "--parent-checkpoint",
        str(parent),
        "--previous-train",
        str(output / "previous/train"),
        "--previous-val",
        str(output / "previous/val"),
        "--batch-size",
        "2",
    ]
    environment = {
        **os.environ,
        "HG_DATA_PYTHON": sys.executable,
        "HG_TRAIN_PYTHON": sys.executable,
        # Hermetic: reuse this process's OpenPI asset cache (tokenizer) instead
        # of the code root's, so the entrypoint never downloads from GCS.
        "OPENPI_DATA_HOME": str(_openpi_cache()),
    }
    ossutil, objects = make_fake_oss(tmp_path / "fake-oss")
    # The installed entrypoint owns relative paths even from another terminal cwd.
    installed_args = prepare_args.copy()
    for flag in ("--records", "--output"):
        index = installed_args.index(flag) + 1
        installed_args[index] = os.path.relpath(installed_args[index], code_root)
    result = subprocess.run(
        ["bash", str(code_root / "hgrlt"), *installed_args],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU 检查通过" in result.stdout and "(2, 30, 32)" in result.stdout
    prepared_manifest = prepared / "lerobot/combined.json"
    assert json.loads(prepared_manifest.read_text())["split_config"] == {
        "method": "group_random",
        "val_fraction": 0.2,
        "seed": 42,
    }
    assert [
        b - a for a, b in WindowDataset(prepared_manifest, "train").component_ranges
    ] == [14, 6, 8]
    with pytest.raises(FileExistsError):
        workflow(prepare_args)
    leaked = tmp_path / "leaked"
    with pytest.raises(ValueError, match="group"):
        workflow(
            [
                "prepare-data",
                "--train",
                f"same-scene={records[2]['path']}",
                "--val",
                f"same-scene={records[3]['path']}",
                "--output",
                str(leaked),
                "--parent-checkpoint",
                str(parent),
                "--previous-train",
                str(output / "previous/train"),
                "--previous-val",
                str(output / "previous/val"),
            ]
        )
    assert not leaked.exists()
    archive_path = tmp_path / "r1.tar"
    workflow(
        [
            "pack-training",
            "--round",
            str(prepared),
            "--parent-checkpoint",
            str(parent),
            "--output",
            str(archive_path),
        ]
    )
    with pytest.raises(FileExistsError):
        workflow(
            [
                "pack-training",
                "--round",
                str(prepared),
                "--parent-checkpoint",
                str(parent),
                "--output",
                str(archive_path),
            ]
        )
    result = subprocess.run(
        [
            "bash",
            str(code_root / "Dagger-Practice/lab.sh"),
            "upload-training",
            "--archive",
            str(archive_path),
            "--ossutil",
            str(ossutil),
            "--destination",
            "oss://test-bucket/training/r1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    release = next((objects / "test-bucket/training").rglob("RELEASE.json"))
    source_uri = "oss://" + release.parent.relative_to(objects).as_posix()
    # Bootstrap is a single file: it must run without the project or venv imports.
    bootstrap = tmp_path / "school-bootstrap.py"
    shutil.copyfile(release.parent / "oss_transfer.py", bootstrap)
    moved = tmp_path / "school-server"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(bootstrap),
            "download",
            "--kind",
            "training",
            "--source",
            source_uri,
            "--destination",
            str(moved),
            "--ossutil",
            str(ossutil),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (moved / "docs/OPERATOR_GUIDE.md").read_bytes() == (
        code_root / "docs/OPERATOR_GUIDE.md"
    ).read_bytes()
    assert not list(moved.rglob("*.h5"))
    assert not (moved / "Dagger-Practice/run_robot.py").exists()
    sums = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=moved, capture_output=True, text=True
    )
    assert sums.returncode == 0, sums.stdout + sums.stderr
    moved_manifest = moved / "round/lerobot/combined.json"
    payload = json.loads(moved_manifest.read_text())
    assert all(
        (moved_manifest.parent / e["path"]).resolve().is_relative_to(moved)
        for e in payload["datasets"]
    )
    assert (
        moved_manifest.parent / payload["parent_checkpoint"]
    ).resolve() == moved / "parent"
    assert (
        moved_manifest.parent / "norm_stats.json"
    ).read_bytes() == stats.read_bytes()
    # Hide original datasets, raw recordings, stats, and the video symlink target.
    # The moved bundle must decode images and produce correct actions without them.
    environment["HG_OPENPI"] = os.environ.get(
        "HG_OPENPI", str(code_root / "vendor/openpi")
    )
    environment["HG_LEROBOT"] = os.environ.get(
        "HG_LEROBOT", str(code_root / "vendor/lerobot")
    )
    environment["OPENPI_DATA_HOME"] = str(moved / "cache/openpi")
    hidden = []
    try:
        for source in (
            output,
            prepared,
            external_video,
            parent,
            records_json.parent,
            stats,
            *(
                Path(record["path"])
                for record in records
                if record["role"] == "previous"
            ),
        ):
            target = source.with_name(source.name + "-hidden")
            source.rename(target)
            hidden.append((source, target))
        result = subprocess.run(
            [
                "bash",
                str(moved / "Dagger-Practice/train.sh"),
                "check",
                "--batch-size",
                "2",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "CPU 检查通过" in result.stdout and str(moved / "parent") in result.stdout
        )
        # The packed copy alone also carries the RLT Stage 1 code.
        result = subprocess.run(
            [
                "bash",
                str(moved / "Dagger-Practice/train.sh"),
                "check",
                "--batch-size",
                "2",
                "--rl-token",
                "--rl-alpha",
                "0",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RL token" in result.stdout and "VLA 冻结" in result.stdout
        moved_row = WindowDataset(moved_manifest, "train")[idx]
        np.testing.assert_array_equal(moved_row["action"], row["action"])
        np.testing.assert_array_equal(
            moved_row["observation.images.main"], row["observation.images.main"]
        )
        result = subprocess.run(
            [
                "bash",
                str(moved / "Dagger-Practice/train.sh"),
                "train",
                "--batch-size",
                "2",
            ],
            cwd=moved,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode != 0 and "训练必须使用 GPU JAX" in result.stderr
        assert not (moved / "training").exists()
    finally:
        for source, target in reversed(hidden):
            target.rename(source)

    without_parent = tmp_path / "data-only.tar.gz"
    workflow(
        [
            "pack-training",
            "--round",
            str(prepared),
            "--parent-checkpoint",
            str(parent),
            "--without-parent",
            "--output",
            str(without_parent),
        ]
    )
    with tarfile.open(without_parent) as archive:
        assert not any(name.startswith("parent/") for name in archive.getnames())
        assert "round/lerobot/combined.json" in archive.getnames()
        assert (
            json.load(archive.extractfile("round/lerobot/combined.json"))[
                "parent_checkpoint"
            ]
            is None
        )

    def save_tiny_checkpoint(config):
        data = config.data.create(config.assets_dirs, config.model)
        manager, _ = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=None, overwrite=False, resume=False
        )
        with manager:
            manager.save(
                499,
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeSave({"params": {"synthetic": np.arange(7)}}),
                    assets=checkpoints.CallbackSave(
                        lambda directory: normalize.save(
                            directory / data.asset_id, data.norm_stats
                        )
                    ),
                ),
            )
            manager.wait_until_finished()

    monkeypatch.setattr(train, "main", save_tiny_checkpoint)
    run_training(cfg, path)
    deployment = seal(
        cfg.checkpoint_dir / "499", tmp_path / "sealed-r1.json", model_round=1
    )
    assert deployment["run"]["inference"]["asset_id"] == "lerobot"
    inference = serving_config(deployment)
    serving_data = inference.data.create(inference.assets_dirs, inference.model)
    assert serving_data.asset_id == "lerobot"  # Not the synthetic repo_id local/dagger.
    np.testing.assert_array_equal(serving_data.norm_stats["actions"].q01, [-2.0] * 14)
    assert inference.model.pi05 and inference.model.action_horizon == 30

    # School -> OSS -> lab. Export no optimizer state and serve only the relocated
    # round; original school files become unreachable before the next-round check.
    model_archive = tmp_path / "model-r1.tar"
    model_target = tmp_path / "lab-model"
    for arguments in (
        [
            "pack-model",
            "--checkpoint",
            str(cfg.checkpoint_dir / "499"),
            "--output",
            str(model_archive),
            "--model-round",
            "1",
        ],
        [
            "upload-model",
            "--archive",
            str(model_archive),
            "--ossutil",
            str(ossutil),
            "--destination",
            "oss://test-bucket/models/r1",
        ],
    ):
        result = subprocess.run(
            ["bash", str(moved / "Dagger-Practice/train.sh"), *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    release = next((objects / "test-bucket/models").rglob("RELEASE.json"))
    result = subprocess.run(
        [
            "bash",
            str(code_root / "Dagger-Practice/lab.sh"),
            "download-model",
            "--source",
            "oss://" + release.parent.relative_to(objects).as_posix(),
            "--destination",
            str(model_target),
            "--ossutil",
            str(ossutil),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from dagger_offline.deployment import load_deployment

    source_identity = deployment["policy_id"]
    cfg.checkpoint_dir.rename(cfg.checkpoint_dir.with_name("school-unreachable"))
    deployment = load_deployment(model_target / "deployment.json")
    assert deployment["policy_id"] == source_identity

    # Round two uses the exported checkpoint as its parent but keeps the exact
    # original normalization bytes and every earlier data component in the 80% pool.
    next_sources = []
    for split, record in zip(("train", "val"), records[2:], strict=True):
        source = tmp_path / f"next-{split}.h5"
        shutil.copyfile(record["path"], source)
        with h5py.File(source, "r+") as handle:
            handle.attrs["synthetic_round"] = 2
        next_sources.extend([f"--{split}", f"next-{split}={source}"])
    next_manifest = workflow(
        [
            "prepare-data",
            *next_sources,
            "--output",
            str(tmp_path / "round-two"),
            "--previous-round",
            str(prepared),
            "--parent-checkpoint",
            deployment["checkpoint"],
            "--batch-size",
            "2",
        ]
    )
    assert (next_manifest.parent / "norm_stats.json").read_bytes() == stats.read_bytes()
    assert [
        b - a for a, b in WindowDataset(next_manifest, "train").component_ranges
    ] == [28, 6, 8]
    next_archive = workflow(
        [
            "pack-training",
            "--round",
            str(next_manifest),
            "--output",
            str(tmp_path / "r2-training.tar"),
        ]
    )
    with tarfile.open(next_archive) as archive:
        # No --parent-checkpoint here: preserve the improved model selected by
        # prepare-data, not the old baseline with identical normalization.
        assert "parent/params/_METADATA" in archive.getnames()
        assert "parent/params/synthetic" not in archive.getnames()

    # Exercise the actual serving entrypoint and RhOS Pi_05 image/action adapter.
    # Replace only the heavyweight policy factory, not the adapter or websocket.
    import asyncio

    import openpi
    from client_server.ws.model_server import PolicyServer
    from openpi.policies import policy_config
    from XPolicyLab.policy.Pi_05 import model as pi05_adapter

    from dagger_offline.deployment import INSTRUCTION, SealedPolicyClient, serve

    qpos = np.linspace(-0.25, 0.35, 14).astype(np.float32)
    colors = {
        "cam_high": [203, 12, 45],
        "cam_left_wrist": [9, 180, 23],
        "cam_right_wrist": [14, 51, 222],
    }

    class SyntheticPolicy:
        def infer(self, observation):
            assert observation["prompt"] == INSTRUCTION
            for camera, color in colors.items():
                image = observation["images"][camera]
                assert image.shape == (3, 8, 12) and image.dtype == np.uint8
                np.testing.assert_array_equal(image[:, 0, 0], color)
            result = np.tile(observation["state"], (30, 1))
            result[:, 0] += np.arange(30) * 0.0007
            result[:, 7] -= np.arange(30) * 0.0011
            return {"actions": result}

    def policy_factory(config, checkpoint_dir, *, norm_stats):
        assert str(checkpoint_dir) == deployment["checkpoint"]
        assert config.name == cfg.name
        np.testing.assert_array_equal(norm_stats["actions"].q99, [2.0] * 14)
        return SyntheticPolicy()

    async def serve_one_request(server):
        assert server.config.host == "127.0.0.1" and server.config.port == 8769
        server.config.port = 0  # Ephemeral test socket; never touch a user's service.
        await server.start()
        try:

            def client_session():
                client = SealedPolicyClient(server.url, deployment["policy_id"])
                try:
                    client.reset()
                    result = client.infer_chunk(
                        {
                            "state": qpos,
                            "instruction": INSTRUCTION,
                            "images": {
                                k: np.tile(np.array(v, dtype=np.uint8), (8, 12, 1))
                                for k, v in colors.items()
                            },
                        }
                    )
                    np.testing.assert_allclose(result[0], qpos)
                    expected_last = qpos.copy()
                    expected_last[0] += 0.0203
                    expected_last[7] -= 0.0319
                    np.testing.assert_allclose(result[29], expected_last, atol=1e-7)
                finally:
                    client.close()

            await asyncio.to_thread(client_session)
        finally:
            await server.stop()

    monkeypatch.setattr(policy_config, "create_trained_policy", policy_factory)
    monkeypatch.setattr(PolicyServer, "serve_forever", serve_one_request)
    monkeypatch.setattr(sys, "path", list(sys.path))
    serve(
        model_target / "deployment.json",
        Path(pi05_adapter.__file__).resolve().parents[3],
        Path(openpi.__file__).resolve().parents[2],
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_policy_chunks_and_failure_type_are_recorded(tmp_path, asynchronous):
    h = Harness(tmp_path / "chunks.h5", {0: ["p"], 80: ["k"]})
    h.run(
        partial(AsyncEpisodeRecorder, queue_size=256)
        if asynchronous
        else EpisodeRecorder,
        review=lambda: ("fail", "grasp"),
    )
    with h5py.File(h.path) as f:
        assert f.attrs["outcome"] == "fail" and f.attrs["failure_type"] == "grasp"
        chunks = f["policy_chunks"]
        count = len(chunks["observed_ns"])
        assert count >= 2 and chunks["actions"].shape == (count, 30, 14)
        continuous = chunks["continuous"][:]
        assert not continuous[0] and continuous[1:].all()  # one handback
        assert chunks["skip"][0] == 0 and (chunks["skip"][1:] >= 1).all()
        assert np.all(chunks["accepted_ns"][:] >= chunks["observed_ns"][:])
        requested = {int(t.observed_at * 1_000_000_000) for t, _ in h.worker.requests}
        assert set(chunks["observed_ns"][:].tolist()) <= requested


def test_review_asks_failure_type_only_for_failures():
    from types import SimpleNamespace as NS

    keys = iter([["x"], ["2"], ["9"], ["3"]])
    clock = NS(sleep=lambda _s: None)
    assert runner.review_outcome(lambda: next(keys), clock) == ("fail", "drop")
    keys = iter([["1"]])
    assert runner.review_outcome(lambda: next(keys), clock) == "success"


def test_failure_type_only_labels_failures(tmp_path):
    with EpisodeRecorder(tmp_path / "x.h5", task_name="t", metadata={}) as rec:
        with pytest.raises(ValueError, match="failure_type"):
            rec.finish(keep=True, outcome="success", failure_type="grasp")


class TokenWorker(Worker):
    """RLT round model: each reply also carries an RL token."""

    def poll(self):
        result = super().poll()
        if result is None or isinstance(result[1], Exception):
            return result
        ticket, actions = result
        token = np.full(8, float(ticket.observed_at), np.float32)
        return ticket, (actions, token)


def test_rlt_episode_replay_learner_and_hot_reload(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dagger_offline import rlt, rlt_learn
    from dagger_offline.live_control import Controller

    config = rlt.RLTConfig(token_dim=8, hidden=32, batch=32)
    run = tmp_path / "rlt"
    recordings = tmp_path / "recordings"

    def episode(name, keys):
        path = recordings / "episodes" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        h = Harness(path, keys)
        h.worker = TokenWorker(h)
        agent = rlt.RLAgent(config, run / "actor", max_deviation=0.15)
        h.control = Controller(
            h.relative, h.worker, now=h.clock.monotonic(), rl=agent
        )
        h.run(
            review=lambda: "success",
            metadata={"collection": {"group": name}, "rlt": {"policy_id": "m1"}},
        )
        return h

    h = episode("episode-000001.h5", {0: ["p"], 40: ["r"], 130: ["k"]})
    with h5py.File(h.path) as f:
        d = f["rl_decisions"]
        count = len(d["accepted_ns"])
        assert count >= 5
        assert d["reference"].shape == (count, 10, 14) and d["actor"].shape == (
            count,
            10,
            14,
        )
        assert d["token"].shape == (count, 8) and set(d["actor_version"][:]) == {0}
        assert np.all(np.diff(d["accepted_ns"][:].astype(np.int64)) > 0)
        # Actor stays inside the deviation bound around the VLA reference.
        assert np.all(
            np.abs(d["actor"][:] - d["reference"][:])[..., [0, 1, 2]] <= 0.15 + 1e-9
        )
        # Synchronous chunks: nothing runs between a state and its decision.
        observed = d["observed_ns"][:].astype(np.int64)
        accepted = d["accepted_ns"][:].astype(np.int64)
        sent = f["pairing/command_sent_ns"][:]
        valid = f["pairing/action_valid"][:].astype(bool)
        for o, a in zip(observed[1:], accepted[1:], strict=True):
            assert not np.any(valid & (sent >= o) & (sent < a))
        # Every chunk but the last ran completely before the next observation.
        for i in range(count - 1):
            ran = valid & (sent >= accepted[i]) & (sent < accepted[i + 1])
            assert ran.sum() == config.chunk
    rows = rlt_learn.episode_transitions(h.path, config)
    assert rows and len(rows) <= count
    assert [r["done"] for r in rows[:-1]] == [0.0] * (len(rows) - 1)
    assert rows[-1]["done"] == 1.0 and rows[-1]["reward"] == 1.0
    assert all(0 < r["discount"] <= 1 for r in rows)
    rlt.bind_run(run, "m1")
    with pytest.raises(ValueError, match="另一个模型"):
        rlt.bind_run(run, "m2")
    version = rlt_learn.learn(recordings, run, config, min_transitions=1, once=True)
    assert version == 1 and (run / "actor/actor.npz").exists()
    assert (run / "replay/episode-000001.npz").exists()

    # Second episode: the collector picks the published actor up by itself.
    # RL for the hard part, VLA finishes (p): the RL run keeps the credit.
    h2 = episode("episode-000002.h5", {0: ["r"], 90: ["p"], 140: ["k"]})
    with h5py.File(h2.path) as f:
        assert set(f["rl_decisions/actor_version"][:]) == {1}
    rows = rlt_learn.episode_transitions(h2.path, config)
    assert rows[-1]["done"] == 1.0 and rows[-1]["reward"] == 1.0
    # A human takeover after the RL run removes it.
    h3 = episode("episode-000003.h5", {0: ["r"], 90: ["h"], 140: ["k"]})
    rows = rlt_learn.episode_transitions(h3.path, config)
    assert rows[-1]["done"] == 1.0 and rows[-1]["reward"] == 0.0


def test_r_without_rlt_is_ignored_and_rl_needs_token(tmp_path):
    from dagger_offline import rlt
    from dagger_offline.live_control import Controller

    h = Harness(tmp_path / "plain.h5", {0: ["r"], 20: ["k"]})
    h.run()
    assert not h.sent  # r did nothing without --rlt
    h = Harness(tmp_path / "notoken.h5", {0: ["r"], 60: ["k"]})
    h.control = Controller(
        h.relative,
        h.worker,
        now=h.clock.monotonic(),
        rl=rlt.RLAgent(rlt.RLTConfig(token_dim=8), tmp_path / "none"),
    )
    with pytest.raises((RuntimeError, BaseExceptionGroup)):
        h.run()


def _rl_harness(path, keys, run, config, rl_lead=0):
    from dagger_offline import rlt
    from dagger_offline.live_control import Controller

    path.parent.mkdir(parents=True, exist_ok=True)
    h = Harness(path, keys)
    h.worker = TokenWorker(h)
    h.control = Controller(
        h.relative,
        h.worker,
        now=h.clock.monotonic(),
        rl=rlt.RLAgent(config, run / "actor"),
        rl_lead=rl_lead,
    )
    return h


def test_rejection_drops_a_queued_r_and_stays_held(tmp_path):
    from dagger_offline import rlt

    config = rlt.RLTConfig(token_dim=8, hidden=32)
    h = _rl_harness(
        tmp_path / "rej.h5", {0: ["r"], 80: ["k"]}, tmp_path / "run", config
    )
    plan = h.plan
    rejected = []

    def reject_once(reference, target, limits, **kwargs):
        if not rejected and h.control.owner is Ownership.POLICY:
            rejected.append(h.clock.tick)
            h.inbox.extend(["r", "r"])  # queued start + key repeat
            return SimpleNamespace(applied_action=reference, violations=["injected"])
        return plan(reference, target, limits, **kwargs)

    h.sender.plan = reject_once
    h.run()
    assert rejected
    assert not any(t >= rejected[0] for t, _, _ in h.sent)


def test_discarded_rl_segment_never_reaches_replay(tmp_path):
    pytest.importorskip("jax")
    from dagger_offline import rlt, rlt_learn

    config = rlt.RLTConfig(token_dim=8, hidden=32, batch=32)
    path = tmp_path / "rec/episodes/episode-000001.h5"
    # c, RL runs, x discards it, RL again, success.
    h = _rl_harness(
        path,
        {0: ["c"], 2: ["r"], 60: ["x"], 70: ["r"], 140: ["k"]},
        tmp_path / "run",
        config,
    )
    h.run(review=lambda: "success", metadata={"collection": {"group": "g"}})
    with h5py.File(path) as f:
        start, end = f["discard_segments/start_ns"][0], f["discard_segments/end_ns"][0]
        observed = f["rl_decisions/observed_ns"][:]
    inside = int(np.sum((observed >= start) & (observed < end)))
    assert inside > 0
    rows = rlt_learn.episode_transitions(path, config)
    assert 0 < len(rows) <= len(observed) - inside
    # Exactly one successful terminal: the run after the discard.
    assert sum(r["reward"] for r in rows) == 1.0


def test_learner_restart_resumes_and_refuses_to_overwrite_without_state(tmp_path):
    pytest.importorskip("jax")
    from dagger_offline import rlt, rlt_learn

    config = rlt.RLTConfig(token_dim=8, hidden=32, batch=32)
    run, recordings = tmp_path / "run", tmp_path / "rec"
    for i in (1, 2):
        h = _rl_harness(
            recordings / f"episodes/episode-00000{i}.h5",
            {0: ["r"], 90: ["k"]},
            run,
            config,
        )
        h.run(review=lambda: "success", metadata={"collection": {"group": str(i)}})
        version = rlt_learn.learn(recordings, run, config, min_transitions=1, once=True)
        assert version == i  # the restarted learner continues the version
    # Interrupted between saving the state and publishing the actor: the
    # restart publishes the actor from the saved state.
    h = _rl_harness(
        recordings / "episodes/episode-000003.h5", {0: ["r"], 90: ["k"]}, run, config
    )
    h.run(review=lambda: "success", metadata={"collection": {"group": "3"}})
    publish = rlt.publish_actor

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    rlt.publish_actor = interrupted
    try:
        with pytest.raises(KeyboardInterrupt):
            rlt_learn.learn(recordings, run, config, min_transitions=1, once=True)
    finally:
        rlt.publish_actor = publish
    assert rlt.load_actor(run / "actor", config)[1] == 2
    assert rlt_learn.learn(recordings, run, config, min_transitions=1, once=True) == 3
    assert rlt.load_actor(run / "actor", config)[1] == 3
    state = (run / "learner.pkl").read_bytes()
    (run / "learner.pkl").unlink()
    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        rlt_learn.learn(recordings, run, config, min_transitions=1, once=True)
    (run / "learner.pkl").write_bytes(state)
    assert rlt_learn.learn(recordings, run, config, min_transitions=1, once=True) == 3


def test_prefetched_rl_recording_is_refused_by_the_replay(tmp_path):
    pytest.importorskip("jax")
    from dagger_offline import rlt, rlt_learn

    config = rlt.RLTConfig(token_dim=8, hidden=32, batch=32)
    path = tmp_path / "rec/episodes/episode-000001.h5"
    h = _rl_harness(path, {0: ["r"], 90: ["k"]}, tmp_path / "run", config, rl_lead=3)
    h.run(review=lambda: "success", metadata={"collection": {"group": "g"}})
    with pytest.raises(ValueError, match="rl_lead=0"):
        rlt_learn.episode_transitions(path, config)
