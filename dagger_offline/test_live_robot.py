from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dagger_offline.live_control import Controller
from dagger_offline.live_robot import (
    ARMS,
    HUMAN_ARM_VELOCITY,
    POLICY_CHUNK_TRAVEL,
    POLICY_STEP,
    SEGMENT_TRAVEL,
    TASK_TRAVEL,
    GuardedSender,
    PolicyTargetRejected,
    TrackingStalled,
    relative_controller,
)


class Robot:
    def __init__(self):
        self.sent = []
        self.offset = 0.0

    def command_joint_state(self, command, *, session_id):
        assert session_id == "guard-test"
        self.sent.append(command.copy())
        return {"applied_joint_state": command + self.offset}


def sender():
    robot = Robot()
    calls = []

    def plan(reference, target, limits, **kwargs):
        calls.append((reference.copy(), target.copy(), kwargs))
        return SimpleNamespace(applied_action=reference + 0.003, violations=[])

    def pressure(action, qpos, **kwargs):
        return action.copy(), None

    return (
        GuardedSender(
            robot,
            np.tile([-10, 10], (14, 1)),
            np.zeros(14),
            session_id="guard-test",
            planner=plan,
            backpressure=pressure,
        ),
        robot,
        calls,
    )


def test_applied_ack_not_raw_proposal_is_reference():
    guard, robot, calls = sender()
    robot.offset = 1e-7  # within daemon comparison tolerance, still use actual ACK
    target = np.ones(14)
    applied = guard.send(target, np.zeros(14), generation=1, policy=True)
    np.testing.assert_allclose(applied, 0.0030001)
    np.testing.assert_array_equal(guard.reference, applied)
    applied[:] = 99
    guard.send(target * 2, np.zeros(14), generation=1, policy=True)
    np.testing.assert_allclose(calls[1][0], 0.0030001)
    np.testing.assert_array_equal(calls[1][2]["previous_action"], target)
    assert calls[1][2]["dt"] == 1 / 30
    assert len(robot.sent) == 2


@pytest.mark.parametrize("ack", [np.zeros(13), np.full(14, np.nan), np.ones(14)])
def test_bad_ack_rejected_without_reseeding(ack):
    guard, robot, _ = sender()
    robot.command_joint_state = lambda command, **kw: {"applied_joint_state": ack}
    with pytest.raises(RuntimeError, match="acknowledgement"):
        guard.send(np.ones(14), np.zeros(14), generation=1, policy=False)
    np.testing.assert_array_equal(guard.reference, np.zeros(14))
    np.testing.assert_array_equal(guard.commanded, np.zeros(14))


def test_unsafe_proposal_never_reaches_backpressure_or_robot():
    guard, robot, _ = sender()
    guard.plan = lambda *args, **kwargs: SimpleNamespace(
        violations=["joint limit"], applied_action=np.zeros(14)
    )
    guard.backpressure = lambda *args, **kwargs: pytest.fail(
        "unsafe proposal reached tracking"
    )
    with pytest.raises(RuntimeError, match="joint limit"):
        guard.send(np.ones(14), np.zeros(14), generation=1, policy=False)
    assert not robot.sent


@pytest.mark.parametrize(
    "policy,limit", [(True, POLICY_STEP), (False, HUMAN_ARM_VELOCITY / 30)]
)
@pytest.mark.parametrize("excess", [0, 1e-6])
def test_tracking_backpressure_slew_bound(policy, limit, excess):
    guard, robot, _ = sender()
    guard.backpressure = lambda *args, **kwargs: (np.full(14, limit + excess), None)
    if not excess:
        guard.send(np.ones(14), np.zeros(14), generation=1, policy=policy)
        assert len(robot.sent) == 1
    else:
        with pytest.raises(RuntimeError, match="slew"):
            guard.send(np.ones(14), np.zeros(14), generation=1, policy=policy)
        assert not robot.sent


@pytest.mark.parametrize("policy", [False, True])
def test_human_and_policy_follow_demonstration_speed(policy):
    # Demonstrations move up to ~0.05 rad/frame (p95); neither path clips that.
    robot = Robot()
    initial = np.zeros(14)
    guard = GuardedSender(
        robot, np.tile([-3, 3], (14, 1)), initial, session_id="guard-test"
    )
    target = initial.copy()
    target[0], target[8] = 0.03, -0.045
    applied = guard.send(target, initial, generation=1, policy=policy)
    np.testing.assert_allclose(applied[[0, 8]], [0.03, -0.045])
    target[0], target[8] = 0.03 + 0.08, -0.045 - 0.07
    applied = guard.send(target, applied, generation=1, policy=policy)
    np.testing.assert_allclose(applied[[0, 8]], [0.03 + 0.05, -0.045 - 0.05])


def test_full_policy_horizon_handles_collected_target_steps_but_rejects_large_jump():
    robot = Robot()
    guard = GuardedSender(
        robot, np.tile([-10, 10], (14, 1)), np.zeros(14), session_id="guard-test"
    )
    target = np.zeros(14)
    steps = round(POLICY_CHUNK_TRAVEL / POLICY_STEP)
    for index in range(1, steps + 1):
        # Proposals change by ~0.06 rad; each sent step is slew-limited.
        target[0], target[8] = index * 0.06, -index * 0.07
        applied = guard.send(target, guard.feedback, generation=1, policy=True)
        guard.observe(applied)
    assert len(robot.sent) == steps
    np.testing.assert_allclose(
        applied[[0, 8]], [POLICY_CHUNK_TRAVEL, -POLICY_CHUNK_TRAVEL], atol=1e-9
    )
    with pytest.raises(RuntimeError, match="chunk travel"):
        guard.send(target, guard.feedback, generation=1, policy=True)
    guard.send(target, guard.feedback, generation=2, policy=True)
    target[0] += 0.20001
    with pytest.raises(PolicyTargetRejected, match="prediction jump"):
        guard.send(target, guard.feedback, generation=2, policy=True)
    assert len(robot.sent) == steps + 1


def test_actual_lifetime_budget_survives_reanchor_and_generation():
    guard, _robot, _ = sender()
    guard.send(np.zeros(14), np.zeros(14), generation=0, policy=True)
    moves = int(TASK_TRAVEL / 0.99)
    for index in range(1, moves + 1):
        guard.new_segment()
        qpos = np.zeros(14)
        qpos[ARMS] = 0.99 if index % 2 else 0
        guard.observe(qpos)
        guard.reanchor(qpos)
        guard.send(qpos, qpos, generation=index, policy=True)
    assert np.allclose(guard.actual[ARMS], moves * 0.99)
    qpos = qpos.copy()
    qpos[ARMS] += TASK_TRAVEL - moves * 0.99 + 0.01
    with pytest.raises(RuntimeError, match="Whole-task actual"):
        guard.observe(qpos)


def test_command_lifetime_budget_survives_reanchor_and_generation():
    guard, robot, _ = sender()
    # Accumulate real accepted commands; alternating direction avoids huge poses.
    guard.plan = lambda reference, *args, **kwargs: SimpleNamespace(
        applied_action=reference + (0.004 if len(robot.sent) % 2 == 0 else -0.004),
        violations=[],
    )
    sends = round(TASK_TRAVEL / 0.004) - 1
    for index in range(sends):
        guard.new_segment()
        guard.reanchor(guard.reference)
        guard.send(np.zeros(14), guard.reference, generation=index, policy=True)
    assert np.allclose(guard.commanded[ARMS], sends * 0.004)
    guard.reanchor(guard.reference)
    guard.backpressure = lambda action, *args, **kwargs: (
        guard.reference + 0.0045,
        None,
    )
    with pytest.raises(RuntimeError, match="Whole-task commanded"):
        guard.send(np.zeros(14), guard.reference, generation=sends, policy=True)
    assert len(robot.sent) == sends


def test_segment_budget_survives_policy_generation_and_hold():
    guard, robot, _ = sender()
    guard.send(np.zeros(14), np.zeros(14), generation=-1, policy=True)
    moves = int(SEGMENT_TRAVEL / 0.99)
    for i in range(moves):
        qpos = np.zeros(14)
        qpos[0] = 0.99 if i % 2 == 0 else 0
        guard.observe(qpos)
        guard.reanchor(qpos)
        guard.send(qpos, qpos, generation=i, policy=True)
    qpos = qpos.copy()
    qpos[0] += SEGMENT_TRAVEL - moves * 0.99 + 0.01
    with pytest.raises(RuntimeError, match="Segment actual"):
        guard.observe(qpos)
    assert len(robot.sent) == moves + 1


def test_limits_rechecked_after_backpressure_and_final_preemption():
    guard, robot, _ = sender()
    guard.limits[0, 0] = -0.049
    guard.backpressure = lambda *args, **kwargs: (np.zeros(14), ())
    with pytest.raises(RuntimeError, match="joint limits"):
        guard.send(np.ones(14), guard.reference, generation=1, policy=False)
    guard.limits[0, 0] = -10
    assert (
        guard.send(
            np.ones(14),
            guard.reference,
            generation=1,
            policy=False,
            before_send=lambda: False,
        )
        is None
    )
    assert not robot.sent


def test_tracking_stall_survives_chunk_generation():
    guard, robot, _ = sender()
    guard.backpressure = lambda *args, **kwargs: (np.zeros(14), (0,))
    for i in range(89):
        guard.send(np.ones(14), guard.feedback, generation=i, policy=True)
    with pytest.raises(TrackingStalled, match="joint L1 stalled"):
        guard.send(np.ones(14), guard.feedback, generation=90, policy=True)
    assert len(robot.sent) == 89
    # Recoverable: after the HOLD reanchor the operator may retry at once.
    guard.reanchor(guard.feedback)
    guard.send(np.ones(14), guard.feedback, generation=91, policy=True)
    assert len(robot.sent) == 90


def test_chunk_command_budget_survives_reanchor_but_new_generation_resets():
    guard, robot, _ = sender()
    arm_step = np.zeros(14)
    arm_step[ARMS] = 0.003  # grippers stay inside [0, 1]
    guard.plan = lambda reference, *args, **kwargs: SimpleNamespace(
        applied_action=reference + arm_step, violations=[]
    )
    sends = int(POLICY_CHUNK_TRAVEL / 0.003) - 1
    for _ in range(sends):
        guard.send(np.ones(14), guard.reference, generation=1, policy=True)
        guard.reanchor(guard.reference)
    with pytest.raises(RuntimeError, match="chunk travel"):
        # sends*0.003 + 0.004 exceeds the chunk budget regardless of rounding.
        guard.backpressure = lambda action, *args, **kwargs: (
            guard.reference + arm_step * 4 / 3,
            None,
        )
        guard.send(np.ones(14), guard.reference, generation=1, policy=True)
    assert len(robot.sent) == sends
    guard.send(np.ones(14), guard.reference, generation=2, policy=True)
    assert len(robot.sent) == sends + 1
    assert np.all(guard.commanded[ARMS] > POLICY_CHUNK_TRAVEL)


def test_chunk_actual_budget_blocks_policy_not_human():
    guard, robot, _ = sender()
    guard.send(np.ones(14), np.zeros(14), generation=1, policy=True)
    moved = np.zeros(14)
    moved[ARMS] = POLICY_CHUNK_TRAVEL + 0.01
    guard.observe(moved)
    guard.reanchor(guard.feedback)
    with pytest.raises(RuntimeError, match="chunk travel"):
        guard.send(np.ones(14), guard.feedback, generation=1, policy=True)
    assert len(robot.sent) == 1
    guard.send(np.ones(14), guard.feedback, generation=1, policy=False)
    moved = guard.feedback.copy()
    moved[0] += 0.7
    guard.observe(moved)
    guard.reanchor(moved)
    guard.send(np.ones(14), guard.feedback, generation=2, policy=True)
    expected = np.zeros(14)
    expected[ARMS] = POLICY_CHUNK_TRAVEL + 0.01
    np.testing.assert_allclose(guard.actual, expected)


def test_long_human_correction_does_not_consume_policy_travel():
    robot = Robot()
    initial = np.zeros(14)
    initial[[6, 13]] = [0.3, 0.6]
    guard = GuardedSender(
        robot, np.tile([-3, 3], (14, 1)), initial, session_id="guard-test"
    )
    for index in range(240):
        target = initial.copy()
        if index % 2 == 0:
            target[[0, 8]] = [0.04, -0.03]
        applied = guard.send(target, guard.feedback, generation=1, policy=False)
        guard.observe(applied)
    # 9.6 rad of accepted manual movement, without a segment reset. Handback
    # must still have its policy budget, including measured and commanded travel.
    target = initial.copy()
    target[[0, 8]] = [0.02, -0.025]
    applied = guard.send(target, guard.feedback, generation=2, policy=True)
    guard.observe(applied)
    np.testing.assert_allclose(applied[[0, 8]], [0.02, -0.025])
    np.testing.assert_allclose(applied[[6, 13]], [0.3, 0.6])


@pytest.mark.parametrize("policy,bound", [(False, 0.145), (True, 0.075)])
def test_policy_tracking_headroom_is_tighter_than_manual(policy, bound):
    robot = Robot()
    initial = np.zeros(14)
    initial[[6, 13]] = [0.3, 0.6]
    guard = GuardedSender(
        robot, np.tile([-3, 3], (14, 1)), initial, session_id="guard-test"
    )
    target = initial.copy()
    target[[0, 8]] = [0.4, -0.3]
    for _ in range(30):
        applied = guard.send(target, initial, generation=1, policy=policy)
    np.testing.assert_allclose(applied[[0, 8]], [bound, -bound])


def test_official_relative_stationary_anchor_continuous_gripper_and_ack_seed(
    monkeypatch,
):
    from deploy.robot.control import mink_ik

    model = (
        Path(__file__).resolve().parents[1]
        / "vendor/i2rt-model/i2rt/robot_models/yam/yam.xml"
    )
    assert model.is_file()
    monkeypatch.setattr(mink_ik, "find_yam_model", lambda: model)
    follower = np.array(
        [0.2, 0.5, 0.7, 0.3, 0.4, -0.2, 0.35, -0.3, 0.7, 0.5, -0.4, 0.2, 0.3, 0.6]
    )
    leader = {
        "left": np.array([-0.1, 0.6, 0.4, 0.1, 0.3, 0.2, 0.2]),
        "right": np.array([0.4, 0.4, 0.8, 0.2, 0.5, -0.1, 0.3]),
    }
    relative = relative_controller(follower)
    relative.anchor(leader, follower)
    for _ in range(3):
        np.testing.assert_allclose(relative.step(leader), follower, atol=2e-3)
    moved = {side: value.copy() for side, value in leader.items()}
    for value in moved.values():
        value[6] += 0.11
    result = relative.step(moved)
    np.testing.assert_allclose(result[[6, 13]], follower[[6, 13]] + 0.11, atol=1e-6)
    assert np.all(result[[6, 13]] < 1)  # official base implementation snaps open
    acknowledged = follower.copy()
    acknowledged[ARMS] += 0.001
    control = Controller(relative, SimpleNamespace(), now=0)
    control.acknowledged(acknowledged)
    seeds = []
    for solver in relative.solvers.values():

        def solve(position, orientation, seed):
            seeds.append(seed.copy())
            return seed

        monkeypatch.setattr(solver, "solve", solve)
    relative.step(leader)
    np.testing.assert_allclose(seeds[0], acknowledged[:6], atol=1e-7)
    np.testing.assert_allclose(seeds[1], acknowledged[7:13], atol=1e-7)


class ServoPacket:
    """Fake Dynamixel protocol-2 packet handler over a register table."""

    def __init__(self, models, modes):
        self.regs = {i: {11: modes.get(i, 3), 38: 1750, 64: 0} for i in models}
        self.models = models
        self.writes = []

    def ping(self, _port, dxl_id):
        return self.models[dxl_id], 0, 0

    def read1ByteTxRx(self, _port, dxl_id, address):
        return self.regs[dxl_id][address], 0, 0

    read2ByteTxRx = read1ByteTxRx

    def _write(self, _port, dxl_id, address, value):
        self.regs[dxl_id][address] = value
        self.writes.append((dxl_id, address, value))
        return 0, 0

    write1ByteTxRx = write2ByteTxRx = write4ByteTxRx = _write


def fake_leader_modules(monkeypatch, models, modes=None):
    import sys
    import threading
    import time

    packet = ServoPacket(models, modes or {})

    class SyncWrite:
        def __init__(self, _port, handler, address, _size):
            self.handler, self.address, self.params = handler, address, {}

        def addParam(self, dxl_id, data):
            self.params[dxl_id] = int.from_bytes(bytes(data), "little")
            return True

        def txPacket(self):
            for dxl_id, value in self.params.items():
                self.handler._write(None, dxl_id, self.address, value)
            return 0

    class Driver:
        def __init__(self, ids, port, use_fake_fallback):
            assert not use_fake_fallback
            self._ids, self._lock, self._baudrate = ids, threading.Lock(), 57600
            self._packetHandler, self._portHandler = packet, object()
            self.raw = np.full(len(ids), np.pi)
            self.closed = False

        def set_torque_mode(self, enable):
            assert not enable

        def get_joint_sample(self):
            return self.raw.copy(), time.monotonic()

        def close(self):
            self.closed = True

    class Port:
        opened = []

        def __init__(self, path):
            self.path = path

        def openPort(self):
            Port.opened.append(self.path)
            return True

        def setBaudRate(self, _baud):
            return True

        def closePort(self):
            pass

    for name, attrs in {
        "gello.dynamixel.driver": {"DynamixelDriver": Driver},
        "dynamixel_sdk.port_handler": {"PortHandler": Port},
        "dynamixel_sdk.packet_handler": {"PacketHandler": lambda _v: packet},
        "dynamixel_sdk.robotis_def": {"COMM_SUCCESS": 0},
        "dynamixel_sdk.group_sync_write": {"GroupSyncWrite": SyncWrite},
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    return packet


LEADER_CONFIG = {
    "agent": {
        "port": "/dev/fake-leader",
        "start_joints": [0, 0, 0, 0, 0, 0, 1],
        "dynamixel_config": {
            "joint_ids": [1, 2, 3, 4, 5, 6],
            "joint_offsets": [np.pi] * 6,
            "joint_signs": [1, -1, -1, -1, 1, 1],
            "gripper_config": [7, 158.0, 200.0],
        },
    }
}


def test_actuated_leader_selects_mode_follows_with_inverse_map_and_releases(
    monkeypatch,
):
    from dagger_offline.live_robot import (
        ADDR_GOAL_CURRENT,
        ADDR_GOAL_POSITION,
        ADDR_OPERATING_MODE,
        ADDR_TORQUE_ENABLE,
        LEADER_FOLLOW_CURRENT_MA,
        LEADER_MODE,
        Leader,
    )

    models = {i: 1200 for i in range(1, 8)}
    models[2] = 1030  # an XM430 shoulder uses 2.69 mA current units
    packet = fake_leader_modules(monkeypatch, models, {3: LEADER_MODE})
    packet.regs[4][38] = 900  # a lower Current Limit caps the goal current
    leader = Leader(LEADER_CONFIG, actuated=True)
    regs = packet.regs
    assert all(regs[i][ADDR_OPERATING_MODE] == LEADER_MODE for i in range(1, 7))
    assert regs[7][ADDR_OPERATING_MODE] == 3  # trigger servo untouched
    assert (3, ADDR_OPERATING_MODE, LEADER_MODE) not in packet.writes
    assert regs[1][ADDR_GOAL_CURRENT] == LEADER_FOLLOW_CURRENT_MA
    assert regs[2][ADDR_GOAL_CURRENT] == round(LEADER_FOLLOW_CURRENT_MA / 2.69)
    assert regs[4][ADDR_GOAL_CURRENT] == 900
    assert all(regs[i][ADDR_TORQUE_ENABLE] == 0 for i in range(1, 8))

    target = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.9])
    leader.follow(target)
    raw = target[:6] * leader.signs[:6] + leader.offsets[:6]
    hold = round(np.pi * 2048 / np.pi)
    position_writes = [w for w in packet.writes if w[1] == ADDR_GOAL_POSITION]
    # First the present pose (no snap), then the follower target.
    assert [v for i, _, v in position_writes[:6]] == [hold] * 6
    assert [regs[i][ADDR_GOAL_POSITION] for i in range(1, 7)] == [
        round(float(r) * 2048 / np.pi) & 0xFFFFFFFF for r in raw
    ]
    assert all(regs[i][ADDR_TORQUE_ENABLE] == 1 for i in range(1, 7))
    assert regs[7][ADDR_TORQUE_ENABLE] == 0 and ADDR_GOAL_POSITION not in regs[7]
    leader.passive()
    assert all(regs[i][ADDR_TORQUE_ENABLE] == 0 for i in range(1, 7))
    leader.follow(target)
    leader.close()
    assert all(regs[i][ADDR_TORQUE_ENABLE] == 0 for i in range(1, 7))
    assert leader.driver.closed


def test_unsupported_leader_servo_refuses_actuation(monkeypatch):
    from dagger_offline.live_robot import Leader

    packet = fake_leader_modules(monkeypatch, {i: 350 for i in range(1, 8)})
    with pytest.raises(RuntimeError, match="passive-leaders"):
        Leader(LEADER_CONFIG, actuated=True)
    assert not packet.writes
    passive = Leader(LEADER_CONFIG)
    passive.follow(np.zeros(7))  # passive leaders never write
    assert not packet.writes


def test_leader_usb_reset_is_reported_and_torque_released_on_fresh_port(monkeypatch):
    from dagger_offline.live_robot import ADDR_TORQUE_ENABLE, Leader

    packet = fake_leader_modules(monkeypatch, {i: 1200 for i in range(1, 8)})
    leader = Leader(LEADER_CONFIG, actuated=True)
    leader.follow(np.zeros(7))
    leader._device = "/dev/ttyUSB-before-reset"
    with pytest.raises(RuntimeError, match="串口断开"):
        leader.sample()
    # The old handle now reports the stuck "port busy" status.
    stale = packet._write

    def busy_once(port, dxl_id, address, value):
        if not hasattr(port, "path") and address == 64:  # old, reset handle
            return -1000, 0
        return stale(port, dxl_id, address, value)

    packet.write1ByteTxRx = busy_once
    leader.close()
    assert all(packet.regs[i][ADDR_TORQUE_ENABLE] == 0 for i in range(1, 7))
    assert leader.driver.closed


@pytest.mark.parametrize("policy", [True, False])
def test_wrist_pitch_gets_its_own_tracking_allowance(policy):
    from dagger_offline.live_robot import WRIST_PITCH_TRACKING_ERROR

    robot = Robot()
    initial = np.zeros(14)
    initial[[6, 13]] = [0.3, 0.6]
    guard = GuardedSender(
        robot, np.tile([-3, 3], (14, 1)), initial, session_id="guard-test"
    )
    target = initial.copy()
    target[[0, 3, 10]] = [0.4, -0.4, 0.4]
    for _ in range(30):
        applied = guard.send(target, initial, generation=1, policy=policy)
    wrist = WRIST_PITCH_TRACKING_ERROR - 0.005
    np.testing.assert_allclose(applied[[3, 10]], [-wrist, wrist])
    assert abs(applied[0]) < wrist  # joints 1-3 keep their tighter bound
