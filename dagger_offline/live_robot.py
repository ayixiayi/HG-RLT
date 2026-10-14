"""RhOSPolicy adapter. Imports and hardware construction are explicit, never on import."""

import os
import re
import time
from types import SimpleNamespace

import numpy as np

ARMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
# Speed limits cover the demonstrations the policy imitates (balance_0922:
# per-frame joint delta p95 0.049 rad, joint speed p90 1.1 rad/s). Travel
# budgets are generous sanity bounds, not tuning parameters.
HUMAN_ARM_VELOCITY = 1.5  # rad/s
HUMAN_TRACKING_ERROR = 0.15
POLICY_ARM_VELOCITY = 1.5  # rad/s
POLICY_TRACKING_ERROR = 0.08  # bounds contact force while the policy pushes
# Joint 4 (wrist pitch, kp 10 vs 80 on joints 1-3) lags far more in normal
# teleoperation (balance_0922: |cmd - measured| p95 0.195 rad left, 0.125
# right; other joints <= 0.05). Contact torque ~ kp * error stays ~2.5 Nm.
WRIST_PITCH = [3, 10]
WRIST_PITCH_TRACKING_ERROR = 0.25
POLICY_STEP = POLICY_ARM_VELOCITY / 30  # rad per 30 Hz frame
POLICY_CHUNK_TRAVEL = 3.0  # rad per joint within one accepted policy chunk
SEGMENT_TRAVEL = 30.0  # rad per joint between operator ownership selections
TASK_TRAVEL = 60.0  # rad per joint over one recording
RAW_TARGET_JUMP = 0.30  # rad between consecutive proposals (chunk switches crossfade).


class PolicyTargetRejected(RuntimeError):
    """A policy proposal failed planning, before any command RPC was attempted."""


class TrackingStalled(PolicyTargetRejected):
    """A joint stopped following (contact or stop): HOLD for the operator."""


JOINT_NAMES = [f"{s}{i}" for s in "LR" for i in (*range(1, 7), "g")]


GRIPPER_PICKUP = 0.05  # trigger within this of the follower gripper engages it


def relative_controller(qpos):
    from deploy.robot.gym.dagger_rollout import RelativeGelloController

    class ContinuousGripper(RelativeGelloController):
        # Pickup instead of ABC's anchor offset and open snap: the follower
        # gripper holds until the trigger reaches it, then tracks absolutely,
        # so the full open/close range stays available after any handover.
        def anchor(self, gello, follower):
            super().anchor(gello, follower)
            self._pickup = {}
            for side in ("left", "right"):
                hold = float(np.clip(self.last[self._slice(side)][6], 0, 1))
                trigger = float(np.clip(gello[side][6], 0, 1))
                self._pickup[side] = (
                    None if abs(trigger - hold) <= GRIPPER_PICKUP
                    else (hold, np.sign(trigger - hold))
                )

        def _gripper(self, side, value):
            value = float(np.clip(value, 0, 1))
            pending = getattr(self, "_pickup", {}).get(side)
            if pending is not None:
                hold, sign = pending
                if abs(value - hold) > GRIPPER_PICKUP and np.sign(value - hold) == sign:
                    return hold
                self._pickup[side] = None
            return value

    config = SimpleNamespace(
        robots={
            side: SimpleNamespace(init_q=np.asarray(qpos)[sl])
            for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14)))
        }
    )
    args = SimpleNamespace(
        intervention_rate=30,
        mink_iterations=4,
        mink_orientation_cost=5,
        mink_gain=0.6,
        mink_max_velocity=HUMAN_ARM_VELOCITY,
        mink_max_joint_delta=HUMAN_ARM_VELOCITY / 30,
        mink_limit_margin=0.05,
        mink_limit_gain=0.5,
        mink_posture_cost=0.005,
        position_scale=1.0,
        rotation_scale=1.0,
    )
    return ContinuousGripper(config, args)


# Leader arm actuation (IDs 1-6 only; the trigger servo always stays passive).
# X-series models that support current-based position control (mode 5).
LEADER_MODELS = {
    1190: "XL330-M077",
    1200: "XL330-M288",
    1210: "XC330-T181",
    1220: "XC330-T288",
    1230: "XC330-M181",
    1240: "XC330-M288",
    1020: "XM430-W350",
    1030: "XM430-W210",
}
LEADER_MODE = 5  # current-based position: position goal with a current ceiling
LEADER_FOLLOW_CURRENT_MA = 1200  # capped by each servo's Current Limit register
# Calibrated trigger, same scale as the follower gripper: 1 open .. 0 closed.
LEADER_TRIGGER_RELEASED = 0.7  # at or above: released (must be seen after h)
LEADER_TRIGGER_SQUEEZED = 0.3  # at or below, after a release: squeezed
LEADER_TRIGGER_HOLD = 0.5  # s both triggers stay squeezed to take over
# Twice the follower's speed so a lagging leader catches up (unit 0.229 rpm).
LEADER_PROFILE_VELOCITY = round(2 * HUMAN_ARM_VELOCITY * 60 / (2 * np.pi) / 0.229)
ADDR_OPERATING_MODE, ADDR_CURRENT_LIMIT = 11, 38
ADDR_GOAL_CURRENT, ADDR_PROFILE_VELOCITY = 102, 112
ADDR_TORQUE_ENABLE, ADDR_GOAL_POSITION = 64, 116
LEADER_COMM_ATTEMPTS = 3  # per transfer, for transient bus errors


class LeaderBus:
    """Serialized register access on a GELLO driver's own port and lock.

    A corrupt or missing status packet (e.g. -3002) happens now and then on a
    shared Dynamixel bus; each transfer is retried, releasing the lock between
    attempts so the driver's reader keeps sampling.
    """

    def __init__(self, driver, ids):
        self.driver, self.ids = driver, list(ids)

    def _transfer(self, what, call):
        from dynamixel_sdk.robotis_def import COMM_SUCCESS

        for _ in range(LEADER_COMM_ATTEMPTS):
            with self.driver._lock:
                *value, result, error = call()
            if result == COMM_SUCCESS and error == 0:
                return value[0] if value else None
        raise RuntimeError(f"Leader servo {what} failed ({result}, {error})")

    def model(self, dxl_id):
        d = self.driver
        return self._transfer(
            f"ping {dxl_id}", lambda: d._packetHandler.ping(d._portHandler, dxl_id)
        )

    def read(self, dxl_id, address, size=1):
        d = self.driver
        reader = getattr(d._packetHandler, {1: "read1ByteTxRx", 2: "read2ByteTxRx"}[size])
        return self._transfer(
            f"read {dxl_id}@{address}", lambda: reader(d._portHandler, dxl_id, address)
        )

    def write(self, address, size, values):
        d = self.driver
        writer = getattr(
            d._packetHandler,
            {1: "write1ByteTxRx", 2: "write2ByteTxRx", 4: "write4ByteTxRx"}[size],
        )
        for dxl_id, value in zip(self.ids, values, strict=True):
            raw = int(value) & (2 ** (8 * size) - 1)
            self._transfer(
                f"write {dxl_id}@{address}",
                lambda dxl_id=dxl_id, raw=raw: writer(
                    d._portHandler, dxl_id, address, raw
                ),
            )

    def goal_positions(self, raw_rad):
        from dynamixel_sdk.group_sync_write import GroupSyncWrite
        from dynamixel_sdk.robotis_def import COMM_SUCCESS

        d = self.driver
        writer = GroupSyncWrite(d._portHandler, d._packetHandler, ADDR_GOAL_POSITION, 4)
        for dxl_id, angle in zip(self.ids, raw_rad, strict=True):
            value = round(float(angle) * 2048 / np.pi) & 0xFFFFFFFF
            if not writer.addParam(dxl_id, list(value.to_bytes(4, "little"))):
                raise RuntimeError(f"Leader goal position for {dxl_id} rejected")
        for _ in range(LEADER_COMM_ATTEMPTS):
            with d._lock:
                result = writer.txPacket()
            if result == COMM_SUCCESS:
                return
        raise RuntimeError(f"Leader goal position sync write failed ({result})")


class Leader:
    """Calibrated GELLO arm. Passive unless `actuated`: then joints 1-6 can
    follow the follower pose (see follow/passive); the trigger stays passive."""

    def __init__(self, config, *, actuated=False):
        from gello.dynamixel.driver import DynamixelDriver

        agent = config["agent"]
        calibration = agent["dynamixel_config"]
        ids = list(calibration["joint_ids"]) + [calibration["gripper_config"][0]]
        self.offsets = np.array([*calibration["joint_offsets"], 0.0], dtype=float)
        self.signs = np.array([*calibration["joint_signs"], 1.0], dtype=float)
        self.gripper = np.deg2rad(calibration["gripper_config"][1:])
        if (
            self.offsets.shape != (7,)
            or self.signs.shape != (7,)
            or not np.isin(self.signs, [-1, 1]).all()
        ):
            raise ValueError("Invalid leader calibration")
        if self.gripper.shape != (2,) or abs(np.diff(self.gripper)[0]) < 1e-6:
            raise ValueError("Invalid gripper calibration")
        if not hasattr(DynamixelDriver, "get_joint_sample"):
            raise RuntimeError("Install the timestamped leader driver patch first")
        self.driver = DynamixelDriver(ids, port=agent["port"], use_fake_fallback=False)
        # A USB reset re-creates the by-id link on a new tty: detect it.
        self.port = agent["port"]
        self._device = os.path.realpath(self.port)
        self.arm = LeaderBus(self.driver, ids[:6]) if actuated else None
        self.following = False
        try:
            self.driver.set_torque_mode(False)
            ready_after = -np.inf
            if self.arm is not None:
                ready_after = self._prepare_actuation()
            # Match existing GelloAgent's startup winding selection once only.
            deadline = time.monotonic() + 2
            while True:
                try:
                    raw, stamp = self.driver.get_joint_sample()
                    if stamp <= ready_after:
                        raise RuntimeError("sample predates operating-mode change")
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            current = (raw - self.offsets) * self.signs
            start = np.asarray(agent["start_joints"], dtype=float)
            self.offsets[:6] += (
                2
                * np.pi
                * np.round((current[:6] - start[:6]) / (2 * np.pi))
                * self.signs[:6]
            )
        except BaseException:
            self.driver.close()
            raise

    def _prepare_actuation(self):
        """Check models; select current-based position mode with torque off.

        A mode change re-bases Present Position, so it precedes the winding
        selection. Returns the time after which samples reflect the final mode.
        """
        changed_at = -np.inf
        currents = []
        for dxl_id in self.arm.ids:
            model = self.arm.model(dxl_id)
            if model not in LEADER_MODELS:
                raise RuntimeError(
                    f"Leader servo {dxl_id} model {model} cannot follow; "
                    "use --passive-leaders"
                )
            if self.arm.read(dxl_id, ADDR_OPERATING_MODE) != LEADER_MODE:
                LeaderBus(self.driver, [dxl_id]).write(
                    ADDR_OPERATING_MODE, 1, [LEADER_MODE]
                )
                changed_at = time.monotonic()
            # Goal Current unit: 1 mA (XL330/XC330), 2.69 mA (XM430).
            unit = 2.69 if LEADER_MODELS[model].startswith("XM430") else 1.0
            limit = self.arm.read(dxl_id, ADDR_CURRENT_LIMIT, 2)
            currents.append(min(round(LEADER_FOLLOW_CURRENT_MA / unit), limit))
        count = len(self.arm.ids)
        self.arm.write(ADDR_GOAL_CURRENT, 2, currents)
        self.arm.write(ADDR_PROFILE_VELOCITY, 4, [LEADER_PROFILE_VELOCITY] * count)
        return changed_at

    def follow(self, qpos7):
        """Drive joints 1-6 toward follower joints (command space, 7 values).

        On the first call after passive, torque turns on holding the leader's
        own present pose, so a set-down leader never snaps; the profile
        velocity then limits travel toward the follower.
        """
        if self.arm is None:
            return
        target = np.asarray(qpos7, dtype=float)[:6]
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Leader follow target must be 6 finite joints")
        if not self.following:
            raw, _ = self.driver.get_joint_sample()
            self.arm.goal_positions(raw[:6])
            self.arm.write(ADDR_TORQUE_ENABLE, 1, [1] * 6)
            self.following = True
        self.arm.goal_positions(target * self.signs[:6] + self.offsets[:6])

    def passive(self):
        if self.arm is not None and self.following:
            self.arm.write(ADDR_TORQUE_ENABLE, 1, [0] * 6)
        self.following = False

    def check_connection(self):
        reader = getattr(self.driver, "_reading_thread", None)
        if (reader is not None and not reader.is_alive()) or os.path.realpath(
            self.port
        ) != self._device:
            raise RuntimeError(
                f"主臂串口断开或已重连（USB 断开？）：{self.port}；"
                "检查 USB 线 / 集线器后重新开始本条"
            )

    def sample(self):
        self.check_connection()
        raw, stamp = self.driver.get_joint_sample()
        q = (raw - self.offsets) * self.signs
        q[6] = np.clip(
            (q[6] - self.gripper[0]) / (self.gripper[1] - self.gripper[0]), 0, 1
        )
        return q, stamp

    def close(self):
        try:
            if self.arm is not None:
                # Unconditional: an interrupted enable must not leave torque on.
                try:
                    self.arm.write(ADDR_TORQUE_ENABLE, 1, [0] * 6)
                except Exception as exc:  # noqa: BLE001 - retried on a fresh port
                    self._release_on_fresh_port(exc)
                self.following = False
        finally:
            self.driver.close()

    def _release_on_fresh_port(self, cause):
        """After a USB reset the old handle is dead but servos keep torque:
        reopen the by-id path (now on the new tty) and switch torque off."""
        from dynamixel_sdk.packet_handler import PacketHandler
        from dynamixel_sdk.port_handler import PortHandler

        port = PortHandler(self.port)
        try:
            if not port.openPort() or not port.setBaudRate(self.driver._baudrate):
                raise RuntimeError("cannot reopen leader port")
            handler = PacketHandler(2.0)
            for dxl_id in self.arm.ids:
                for _attempt in range(3):
                    result, error = handler.write1ByteTxRx(
                        port, dxl_id, ADDR_TORQUE_ENABLE, 0
                    )
                    if result == 0 and error == 0:
                        break
                else:
                    raise RuntimeError(f"servo {dxl_id} torque off failed")
        except Exception as exc:
            exc.add_note(f"original torque-off failure: {cause!r}")
            raise RuntimeError(
                f"主臂 {self.port} 扭矩未能关闭，请给主臂断电后再继续"
            ) from exc
        finally:
            port.closePort()


class LeaderFollower:
    """Background leader writer: the control loop never waits on leader serial I/O.

    follow(qpos14) makes both leaders track the follower; passive() releases
    them for human control. Failures surface through raise_if_failed().
    """

    def __init__(self, leaders, *, continuous=True, period=1 / 30):
        import threading

        # continuous: track the follower whenever the human is not in control.
        # Otherwise only while a takeover is armed (fewer high-current periods).
        self.continuous = continuous
        self.leaders = dict(leaders)
        self._lock = threading.Lock()
        self._targets = None
        self._error = None
        self._stop = threading.Event()
        self._period = period
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def follow(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("Leader follow requires the 14-D follower state")
        with self._lock:
            self._targets = {"left": qpos[:7].copy(), "right": qpos[7:].copy()}

    def passive(self):
        with self._lock:
            self._targets = None

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"Leader follow failed: {self._error}") from self._error

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    targets = self._targets
                for side, leader in self.leaders.items():
                    if targets is None:
                        leader.passive()
                    else:
                        leader.follow(targets[side])
                self._stop.wait(self._period)
        except Exception as exc:  # noqa: BLE001 - reported on the control thread
            self._error = exc
        finally:
            for leader in self.leaders.values():
                try:
                    leader.passive()
                except Exception:  # noqa: BLE001, S110 - Leader.close retries torque off
                    pass

    def close(self):
        self.passive()
        self._stop.set()
        self._thread.join(timeout=1.0)


def leader_name(port):
    """Short operator-facing name of a leader serial port (adapter serial)."""
    match = re.search(r"_([A-Za-z0-9]{6,})-if\d+", str(port))
    return match.group(1) if match else os.path.basename(str(port))


def start_cameras(factory, config, *, attempts=3, wait=1.0, sleep=time.sleep):
    """Open the camera rig, retrying transient start failures.

    RealSense devices closed by the previous episode can still be busy for a
    moment (``set_pu(...) failed ... Device or resource busy``); each attempt
    uses a fresh rig after closing the failed one.
    """
    from dagger_offline.console import message

    for attempt in range(1, attempts + 1):
        rig = factory(config, depth_enabled=False)
        try:
            rig.start()
            rig.begin_episode()
            return rig
        except Exception as exc:
            rig.close()
            if attempt == attempts:
                raise
            message(
                "相机",
                f"启动失败，{wait:g} 秒后重试（{attempt}/{attempts - 1}）：{exc}",
                level="warning",
            )
            sleep(wait)


class GuardedSender:
    """One sender using RhOSPolicy's existing limits/backpressure and daemon ACK.

    Only policy commands and their measured motion consume travel budgets.
    Human corrections retain velocity, joint, tracking and stall guards.
    The policy-chunk budget resets when a new chunk is accepted, not on HOLD.
    """

    def __init__(
        self, robot, limits, initial, *, session_id, planner=None, backpressure=None
    ):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Exclusive session_id required for every command")
        self.session_id = session_id
        if planner is None:
            from rhospolicy.inference.yam_act import (
                apply_tracking_backpressure,
                plan_action,
            )

            planner, backpressure = plan_action, apply_tracking_backpressure
        self.robot, self.limits = robot, np.asarray(limits, dtype=float)
        if (
            self.limits.shape != (14, 2)
            or not np.isfinite(self.limits).all()
            or np.any(self.limits[:, 0] >= self.limits[:, 1])
        ):
            raise ValueError("Expected finite ordered (14,2) joint limits")
        self.plan, self.backpressure = planner, backpressure
        self.reference = np.asarray(initial).copy()
        self.feedback = self.reference.copy()
        self.actual = np.zeros(14)
        self.commanded = np.zeros(14)
        self.chunk_actual = np.zeros(14)
        self.chunk_commanded = np.zeros(14)
        self.generation = None
        self.previous_raw = None
        self.stall = {}
        self.segment_actual = np.zeros(14)
        self.segment_commanded = np.zeros(14)
        self.last_command_policy = False

    def observe(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("Invalid measured joint state")
        delta = np.abs(qpos - self.feedback)
        self.feedback = qpos.copy()
        # Attribute feedback to the last acknowledged command, including its
        # settling motion during HOLD. A requested takeover is not yet an ACK.
        if not self.last_command_policy:
            return
        self.actual += delta
        self.chunk_actual += delta
        self.segment_actual += delta
        if np.any(self.actual[ARMS] >= TASK_TRAVEL):
            raise RuntimeError("Whole-task actual travel budget exhausted")
        if np.any(self.segment_actual[ARMS] >= SEGMENT_TRAVEL):
            raise RuntimeError("Segment actual travel budget exhausted")

    def new_segment(self):
        # Explicit operator selection only, never an automatic inference HOLD.
        self.segment_actual[:] = 0
        self.segment_commanded[:] = 0

    def reanchor(self, qpos):
        # HOLD captures measured pose, but must not erase lifetime travel.
        # A stall is judged per control run: HOLD restarts its tick count.
        self.reference = np.asarray(qpos).copy()
        self.previous_raw = None
        self.stall = {}

    def send(self, target, qpos, *, generation, policy, before_send=lambda: True):
        if generation != self.generation:
            # A continuous chunk switch keeps previous_raw so the raw-jump check
            # spans chunk boundaries; HOLD/ownership changes clear it (reanchor).
            self.generation = generation
            self.chunk_actual[:] = 0
            self.chunk_commanded[:] = 0
        velocity = POLICY_ARM_VELOCITY if policy else HUMAN_ARM_VELOCITY
        step = POLICY_STEP if policy else velocity / 30
        plan = self.plan(
            self.reference,
            target,
            self.limits,
            previous_action=self.previous_raw,
            max_prediction_jump=RAW_TARGET_JUMP,
            max_arm_velocity=velocity,
            max_arm_delta=step,
            dt=1 / 30,
        )
        if plan.violations:
            error = PolicyTargetRejected if policy else RuntimeError
            raise error("Unsafe target: " + "; ".join(plan.violations))
        bounds = np.full(14, POLICY_TRACKING_ERROR if policy else HUMAN_TRACKING_ERROR)
        bounds[WRIST_PITCH] = np.maximum(
            bounds[WRIST_PITCH], WRIST_PITCH_TRACKING_ERROR
        )
        command, indices = self.backpressure(
            plan.applied_action, qpos, max_arm_tracking_error=float(bounds.max())
        )
        command = np.asarray(command, dtype=float)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise RuntimeError("Invalid filtered command")
        # Per-joint ceiling (same 0.005 reserve as RhOS backpressure).
        measured = np.asarray(qpos, dtype=float)
        ceiling = bounds[ARMS] - 0.005
        error = command[ARMS] - measured[ARMS]
        tight = np.abs(error) > ceiling + 1e-12
        command[ARMS] = measured[ARMS] + np.clip(error, -ceiling, ceiling)
        indices = sorted(set(indices or ()) | {int(i) for i in ARMS[tight]})
        if (
            np.any(command[ARMS] < self.limits[ARMS, 0] + 0.05)
            or np.any(command[ARMS] > self.limits[ARMS, 1] - 0.05)
            or np.any(command[[6, 13]] < 0)
            or np.any(command[[6, 13]] > 1)
        ):
            raise RuntimeError("Tracking backpressure exceeded joint limits")
        active = set(indices or ())
        self.stall = {i: value for i, value in self.stall.items() if i in active}
        for i in active:
            anchor, ticks = self.stall.get(i, (qpos[i], 0))
            ticks += 1
            if ticks >= 90:
                if abs(qpos[i] - anchor) < 0.0005:
                    raise TrackingStalled(
                        f"joint {JOINT_NAMES[i]} stalled for 90 ticks "
                        f"(command {command[i]:.3f}, measured {qpos[i]:.3f}); "
                        "check for contact"
                    )
                anchor, ticks = qpos[i], 0
            self.stall[i] = (anchor, ticks)
        delta = np.abs(command - self.reference)
        if np.any(delta[ARMS] > step + 1e-9):
            raise RuntimeError("Tracking backpressure exceeded command slew limit")
        if policy and np.any(self.commanded[ARMS] + delta[ARMS] > TASK_TRAVEL):
            raise RuntimeError("Whole-task commanded travel budget exhausted")
        if policy and np.any(
            self.segment_commanded[ARMS] + delta[ARMS] > SEGMENT_TRAVEL
        ):
            raise RuntimeError("Segment commanded travel budget exhausted")
        if policy and (
            np.any(self.chunk_actual[ARMS] > POLICY_CHUNK_TRAVEL + 1e-9)
            or np.any(
                self.chunk_commanded[ARMS] + delta[ARMS] > POLICY_CHUNK_TRAVEL + 1e-9
            )
        ):
            raise RuntimeError("Policy chunk travel budget exhausted")
        if not before_send():
            return None
        result = self.robot.command_joint_state(command, session_id=self.session_id)
        if not isinstance(result, dict) or "applied_joint_state" not in result:
            raise RuntimeError("Missing daemon command acknowledgement")
        applied = np.asarray(result["applied_joint_state"], dtype=float)
        if (
            applied.shape != (14,)
            or not np.isfinite(applied).all()
            or not np.allclose(applied, command, atol=1e-6, rtol=0)
        ):
            raise RuntimeError("Daemon acknowledgement changed or omitted command")
        if policy:
            self.commanded += delta
            self.chunk_commanded += delta
            self.segment_commanded += delta
        self.last_command_policy = policy
        self.reference = applied.copy()
        self.previous_raw = np.asarray(target).copy()
        return applied
