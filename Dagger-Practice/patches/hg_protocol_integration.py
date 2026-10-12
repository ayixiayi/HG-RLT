"""Explicit protocol test suite; real RPC + run_loop with synthetic motors only.

Requires the patched RhOSPolicy on PYTHONPATH, plus the HG runner and pinned ABC.
Servers bind ephemeral localhost/inproc endpoints; no device constructors run.
"""

import threading
import uuid
from contextlib import contextmanager
from functools import partial

import h5py
import numpy as np
import pytest
import zmq
from gello.safety.controller import SafetyController
from gello.safety.rpc import SafetyClient, SafetyRPCServer
from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

from dagger_offline.live_control import TickLease
from dagger_offline.live_recording import AsyncEpisodeRecorder
from dagger_offline.round_data import read_source
from dagger_offline.test_live_runner import Harness


@contextmanager
def daemon(h):
    class Motors:
        harness = h
        q = h.qpos.copy()
        stale = False
        sends = 0

        def num_dofs(self):
            return 14

        def get_joint_limits(self):
            return np.tile([-3, 3], (14, 1))

        def get_joint_state(self):
            return self.q.copy()

        def get_joint_sample_capabilities(self):
            return {
                "supported": True,
                "api_version": 1,
                "time_base": "host_monotonic_can_receive",
            }

        def get_joint_sample(self):
            received = self.harness.clock.monotonic() - np.linspace(0.003, 0.008, 14)
            if self.stale:
                received[13] -= 0.2
            return {"joint_state": self.q.copy(), "received_monotonic_s": received}

        def command_joint_state(self, q):
            self.q = q.copy()
            self.harness.qpos = q.copy()
            self.sends += 1
            self.harness.clock.value += 0.001

    motors = Motors()
    controller = SafetyController(
        motors, clock=lambda: motors.harness.clock.monotonic()
    )
    context = zmq.Context()
    endpoint = f"inproc://hg-{uuid.uuid4().hex}"
    safety_server = SafetyRPCServer(controller, endpoint=endpoint, context=context)
    robot_server = ZMQServerRobot(
        controller,
        port=0,
        host="127.0.0.1",
        # Match yam_safety_daemon.py: late wrong-owner sends must NOT invoke this.
        on_error=lambda exc: controller.enter_safe_hold(f"RPC exception: {exc}"),
    )
    port = int(
        robot_server._socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1]
    )
    threads = [threading.Thread(target=s.serve) for s in (safety_server, robot_server)]
    for t in threads:
        t.start()
    safety = SafetyClient(endpoint=endpoint, context=context, timeout_ms=100)
    robot = ZMQClientRobot(host="127.0.0.1", port=port, timeout_ms=100)
    try:
        yield controller, motors, safety, robot
    finally:
        robot.close()
        safety.close()
        for server in (safety_server, robot_server):
            server.stop()
        for t in threads:
            t.join(2)
            assert not t.is_alive()
        robot_server.close()
        safety_server.close()
        context.term()


def attach(h, motors, safety, robot, session):
    motors.harness = h
    h.qpos = motors.q.copy()
    h.sender.reanchor(h.qpos)
    h.sender.robot = robot
    h.sender.session_id = session
    h.lease = TickLease(safety, session)
    capture = h.capture

    def sample():
        snap = capture()
        snap.state_requested = h.clock.monotonic()
        measured = robot.get_joint_sample()
        snap.state_received = h.clock.monotonic()
        snap.qpos = measured["joint_state"]
        snap.follower_times = measured["received_monotonic_s"]
        return snap

    h.capture = sample


def test_real_protocol_two_recordings_release_reacquire_and_stale_owner(tmp_path):
    h = Harness(tmp_path / "first.h5")
    with daemon(h) as (controller, motors, safety, robot):
        assert robot.get_joint_sample_capabilities()["supported"]
        attach(h, motors, safety, robot, "first")
        # Fake time advances without sleeping; allow a whole synthetic episode
        # to queue while the real HDF5 worker runs. Production capacity is 30.
        h.run(partial(AsyncEpisodeRecorder, queue_size=256))
        assert safety.status()["session_id"] is None
        h2 = Harness(tmp_path / "second.h5", offset=10)
        attach(h2, motors, safety, robot, "second")

        def old_sender(stage):
            if stage == "plan":
                sent = motors.sends
                with pytest.raises(RuntimeError, match="SessionOwnershipError"):
                    robot.command_joint_state(np.ones(14), session_id="first")
                assert motors.sends == sent
                assert safety.status()["state"] == "active"
                with pytest.raises(RuntimeError, match="exclusive"):
                    safety.begin_session("legacy-intruder")

        h2.hook = old_sender
        h2.run(partial(AsyncEpisodeRecorder, queue_size=256))
        assert safety.status()["session_id"] is None
        assert controller.status()["hold_confirmed"]
        for path in (h.path, h2.path):
            with h5py.File(path) as f:
                _, _, _, labels, valid, _ = read_source(f, "current", 100)
                assert f.attrs["usable"] and f.attrs["complete"]
                assert np.sum(valid & (labels == "POLICY_RUNNING")) == 60
                assert np.sum(valid & (labels == "INTERVENTION")) == 35


def test_real_protocol_feedback_fault_cannot_be_cleared_by_cleanup(tmp_path):
    h = Harness(tmp_path / "stale.h5", {0: ["h"], 8: ["k"]})
    with daemon(h) as (controller, motors, safety, robot):
        attach(h, motors, safety, robot, "stale")
        sent_before_fault = []

        def fail(stage):
            if stage == "plan" and h.clock.tick == 3:
                motors.stale = True  # becomes stale AFTER the runner's snapshot
                sent_before_fault.append(motors.sends)

        h.hook = fail
        with pytest.raises(BaseExceptionGroup):
            h.run(AsyncEpisodeRecorder)
        assert controller.status()["state"] == "fault"
        assert not controller.status()["hold_confirmed"]
        assert motors.sends == sent_before_fault[0]
        with h5py.File(h.path) as f:
            assert not f.attrs["usable"] and not f.attrs["complete"]
            assert f["dagger_events/controller_state"][-1].decode() == "FAULT"


def test_real_protocol_rejected_policy_keeps_session_for_human_correction(tmp_path):
    from rhospolicy.inference.yam_act import apply_tracking_backpressure, plan_action

    h = Harness(tmp_path / "correction.h5", {0: ["p"], 15: ["h"], 52: ["k"]})
    h.sender.plan, h.sender.backpressure = plan_action, apply_tracking_backpressure
    poll = h.worker.poll

    def policy():
        result = poll()
        if result is not None:
            result[1][3:, 4] += 0.21  # Above the current 0.20-rad raw-jump guard.
        return result

    h.worker.poll = policy
    held = []
    with daemon(h) as (controller, motors, safety, robot):
        attach(h, motors, safety, robot, "correction")

        def observe_hold(stage):
            if stage == "capture" and h.clock.tick == 10:
                status = safety.status()
                assert status["state"] == "safe_hold" and status["hold_confirmed"]
                assert status["session_id"] == "correction" and status["strict_session"]
                held.append(True)

        h.hook = observe_hold
        h.run(partial(AsyncEpisodeRecorder, queue_size=256))
        assert held and safety.status()["session_id"] is None
        assert controller.status()["hold_confirmed"]
    with h5py.File(h.path) as f:
        _, _, _, labels, valid, _ = read_source(f, "current", 100)
        assert f.attrs["usable"] and f.attrs["complete"]
        assert np.sum(valid & (labels == "POLICY_RUNNING")) == 3
        assert np.sum(valid & (labels == "INTERVENTION")) == 36
        assert not valid[7:16].any()
        assert any(
            "prediction jump" in reason
            for reason in f["dagger_events/reason"].asstr()[:]
        )
