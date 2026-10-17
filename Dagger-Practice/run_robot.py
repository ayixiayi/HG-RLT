"""HG-DAgger on RhOSPolicy's existing YAM daemon; old Pi0.5 remains the default.

Default is offline preflight. --execute requires a local operator TTY.
Never starts a hardware daemon or enables motors. Recording never auto-homes;
--prepare-start separately authorizes bounded motion to the default zero pose.
New round policies require an explicit content-sealed --deployment manifest.
"""

import argparse
import hashlib
import inspect
import json
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagger_offline.console import (
    IDLE_PROMPT_S,
    UNATTENDED,
    Unattended,
    driver_output,
    message,
    show_controls,
    show_mode,
    timed_input,
)

REVISION = "d0832d12651d1b260a652861a14648dc5f3660c7"


@contextmanager
def keyboard():
    saved = termios.tcgetattr(sys.stdin)
    fd = sys.stdin.fileno()

    def poll():
        if not select.select([fd], [], [], 0)[0]:
            return []
        # TextIOWrapper.read(1) can prefetch the next key into a Python buffer
        # invisible to select(), delaying a stop until another key is pressed.
        data = os.read(fd, 4096)
        if not data:
            raise EOFError("Operator terminal closed")
        # Non-ASCII input is no command but still operator activity: keep it
        # as U+FFFD, which no key handler matches.
        return list(data.decode("ascii", errors="replace").lower())

    try:
        tty.setcbreak(sys.stdin.fileno())
        yield poll
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)


def preflight(args):
    import yaml

    root, abc = args.rhos_root.resolve(), args.abc.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(abc), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(abc), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != REVISION or dirty:
        raise ValueError("ABC must be the clean pinned revision")
    for path in (
        abc,
        root,
        root / "XPolicyLab",
        root / "Client/arms/YAM/gello_software",
        root / "scripts",
    ):
        sys.path.insert(0, str(path))
    deployment = root / "configs/deployments/pi05_yam_heavier.yaml"
    config = yaml.safe_load(deployment.read_text())
    if tuple(
        config.get(key)
        for key in (
            "train_config",
            "repo_id",
            "port",
            "host",
            "checkpoint",
            "action_space",
            "instruction",
        )
    ) != (
        "pi05_yam_heavier",
        "local/balance_0922_train",
        8767,
        "127.0.0.1",
        "checkpoints/pi05_yam_heavier/pi05_yam_20k_20260924_023553/39999",
        "joints",
        "choose_the_heavier_one",
    ) or config.get("direct_policy_targets", False):
        raise ValueError(
            "HG-DAgger round 1 requires the guarded old baseline deployment"
        )
    baseline = dict(config)
    sealed = None
    if getattr(args, "deployment", None) is not None:
        from dagger_offline.deployment import load_deployment

        sealed = load_deployment(args.deployment)
        inference = sealed["run"]["inference"]
        config.update(
            checkpoint=sealed["checkpoint"],
            train_config=inference["config_name"],
            repo_id=inference["asset_id"],
            port=sealed["port"],
        )
    # These imports otherwise happen only in hardware/policy constructors.
    # Check the runtime dependencies without opening a device or connection.
    import pyrealsense2  # noqa: F401
    from client_server.ws.model_client import WsModelClient
    from gello.cameras.realsense_rig import load_camera_rig_config
    from gello.dynamixel.driver import DynamixelDriver
    from gello.profile import load_profile
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot
    from rhospolicy.inference.yam_pi05 import Pi05PolicyClient
    from run_pi05_heavier import require_control_rate

    from dagger_offline.live_recording import AsyncEpisodeRecorder  # noqa: F401
    from dagger_offline.live_robot import (
        HUMAN_ARM_VELOCITY,
        HUMAN_TRACKING_ERROR,
        POLICY_ARM_VELOCITY,
        POLICY_CHUNK_TRAVEL,
        POLICY_STEP,
        POLICY_TRACKING_ERROR,
        RAW_TARGET_JUMP,
        SEGMENT_TRAVEL,
        TASK_TRAVEL,
        relative_controller,
    )

    profile = load_profile(root / config["profile"])
    require_control_rate(profile)
    if profile.network.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "CAN receive timestamps require runner and daemon on the same host"
        )
    if (
        not hasattr(SafetyClient, "acquire_session")
        or not hasattr(SafetyClient, "release_session")
        or not hasattr(ZMQClientRobot, "get_joint_sample")
        or "session_id"
        not in inspect.signature(ZMQClientRobot.command_joint_state).parameters
        or "transport_timeout_s" not in inspect.signature(Pi05PolicyClient).parameters
        or "operation_timeout_s" not in inspect.signature(WsModelClient).parameters
    ):
        raise RuntimeError(
            "HG-DAgger session/feedback/transport protocol patch is missing"
        )
    if not hasattr(DynamixelDriver, "get_joint_sample"):
        raise RuntimeError("Timestamped Dynamixel driver patch is missing")
    if profile.cameras is None or not profile.cameras.required:
        raise ValueError("Required three-camera profile missing")
    camera_config = load_camera_rig_config(profile.cameras.config_path)
    if set(camera_config.cameras) != {"main", "left", "right"}:
        raise ValueError("Expected main/left/right cameras")
    reference = {
        "schema": "yam-heavier-task-start-1",
        "source": "HG-DAgger default: zero arm joints, open grippers",
        "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 2,
    }
    relative_controller(reference["state"])  # Real MuJoCo model/FK, no hardware.
    checkpoint = root / config["checkpoint"]
    stats = checkpoint / "assets" / config["repo_id"] / "norm_stats.json"
    if not (checkpoint / "params").is_dir() or not stats.is_file():
        raise FileNotFoundError("Baseline checkpoint/normalization assets are missing")
    metadata = {
        "abc_revision": REVISION,
        "baseline": baseline,
        "policy": config,
        "model_round": 0 if sealed is None else sealed["model_round"],
        "round": 1 if sealed is None else sealed["model_round"] + 1,
        "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
        "start_reference": reference,
        "reference_sha256": hashlib.sha256(
            json.dumps(reference, sort_keys=True).encode()
        ).hexdigest(),
        "profile_sha256": hashlib.sha256(profile.path.read_bytes()).hexdigest(),
        "control_rate": 30,
        "action_horizon": 30,
        "rtc": False,
        "policy_execution": {
            "mode": "continuous_prefetch",
            "replan_lead": 8,
            "skip": "actions executed after the request observation",
            "max_skip": 12,
            "blend": 8,
        },
        "control_limits": {
            "human_arm_velocity_rad_s": HUMAN_ARM_VELOCITY,
            "human_tracking_error_rad": HUMAN_TRACKING_ERROR,
            "policy_arm_velocity_rad_s": POLICY_ARM_VELOCITY,
            "policy_tracking_error_rad": POLICY_TRACKING_ERROR,
            "policy_step_rad": POLICY_STEP,
            "policy_chunk_travel_rad": POLICY_CHUNK_TRAVEL,
            "segment_travel_rad": SEGMENT_TRAVEL,
            "task_travel_rad": TASK_TRAVEL,
            "raw_target_jump_rad": RAW_TARGET_JUMP,
            "travel_budget_scope": "policy_only",
        },
        "human_mapping": "ABC relative Cartesian; continuous anchored gripper",
        "server_identity": "configured endpoint; operator must launch matching baseline server",
    }
    if sealed is not None:
        metadata.update(
            sealed_policy_id=sealed["policy_id"],
            server_identity="sealed identity required before devices and on each policy reply",
        )
    if getattr(args, "rlt", None) is not None:
        from dagger_offline import rlt

        if sealed is None or not sealed["run"]["inference"].get("rl_token"):
            raise ValueError("--rlt 需要用 --rl-token 训练并封存的模型（--deployment）")
        # One config per RLT run, shared with rlt-learn (RUN/config.json).
        rl_config = rlt.run_config(args.rlt)
        metadata.update(
            rlt={
                "run": str(Path(args.rlt).resolve()),
                "policy_id": sealed["policy_id"],
                "config": asdict(rl_config),
                "config_digest": rl_config.digest(),
                "rl_lead": rl_config.lead,
                "max_deviation": 0.15,
            }
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    return config, profile, camera_config, reference, metadata


@dataclass
class Snapshot:
    qpos: object
    images: dict
    camera_times: dict
    state_requested: float
    state_received: float
    follower_times: object
    leader: dict
    leader_times: dict


def review_outcome(poll, clock=time, idle_s=IDLE_PROMPT_S):
    """Ask for the task result only after confirmed HOLD and lease release.

    A failure also asks for its type; returns ("fail", type) in that case, and
    UNATTENDED when no key comes for ``idle_s`` (the episode stays unlabelled).
    """
    from dagger_offline.live_recording import FAILURE_TYPES

    message("结果标记", "1 成功 · 2 失败 · 3 弃用；直接按键，无需回车")
    choices = {"1": "success", "2": "fail", "3": "discard", "q": "discard"}
    outcome = None
    asked = clock.monotonic()
    while outcome is None:
        keys = poll()
        for key in keys:
            if key in choices:
                outcome = choices[key]
                break
        else:
            if keys:
                asked = clock.monotonic()
            elif clock.monotonic() - asked > idle_s:
                return UNATTENDED
            clock.sleep(0.05)
    if outcome != "fail":
        return outcome
    types = list(FAILURE_TYPES)
    message(
        "失败类型",
        " · ".join(f"{i} {FAILURE_TYPES[t]}" for i, t in enumerate(types, 1)),
    )
    asked = clock.monotonic()
    while True:
        keys = poll()
        for key in keys:
            if key.isdigit() and 1 <= int(key) <= len(types):
                return outcome, types[int(key) - 1]
        if keys:
            asked = clock.monotonic()
        elif clock.monotonic() - asked > idle_s:
            return UNATTENDED
        clock.sleep(0.05)


def run_loop(
    *,
    control,
    sender,
    lease,
    recorder,
    capture,
    poll,
    build_observation,
    clock=time,
    review=None,
    leaders=None,
    idle_s=IDLE_PROMPT_S,
):
    """The real single-sender loop, also exercised with fake I/O and clock.

    No simulator-specific branch. Sensor/transport waits must be bounded by the
    adapters. The daemon watchdog remains responsible while this thread blocks.

    leaders (LeaderFollower or None): when given, the GELLO arms track the
    follower outside human control, and h only arms a takeover: HOLD now,
    then the unpowered triggers confirm (open fully, squeeze fully), return
    powered to the robot gripper, and control passes to the human.
    """
    import numpy as np

    from dagger_offline.arbitration import Ownership
    from dagger_offline.live_robot import (
        LEADER_TRIGGER_CLOSED,
        LEADER_TRIGGER_MARGIN,
        LEADER_TRIGGER_MATCH,
        LEADER_TRIGGER_OPEN,
        PolicyTargetRejected,
        TrackingStalled,
    )

    labels = {
        Ownership.POLICY: "POLICY_RUNNING",
        Ownership.HUMAN: "INTERVENTION",
        Ownership.HOLD: "HOLD",
        Ownership.FAULT: "FAULT",
    }
    last_owner = None
    last_tick = clock.monotonic()
    last_cameras = last_leaders = None
    last_follower = None
    pending_keys = []
    armed = None  # takeover confirmation state while pending (see below)
    last_input = clock.monotonic()  # a key, or a human holding the leaders
    keyboard_poll = poll

    def poll():
        # Every key read anywhere below is operator activity, including keys
        # later dropped as stale (e.g. h / p / r right after a rejection).
        nonlocal last_input
        keys = keyboard_poll()
        if keys:
            last_input = clock.monotonic()
        return keys

    def ns(value=None):
        return int((clock.monotonic() if value is None else value) * 1_000_000_000)

    def event(reason, force=False):
        nonlocal last_owner
        if force or control.owner is not last_owner:
            recorder.event(ns(), labels[control.owner], reason)
            # Synchronous RL chunks HOLD briefly between chunks: record, not show.
            between_rl_chunks = (
                reason == "controller transition"
                and control.autonomous
                and getattr(control, "rl_active", False)
            )
            if (
                control.owner is not last_owner and not between_rl_chunks
            ) or reason == "ownership selected":
                show_mode(labels[control.owner], waiting_for_policy=control.autonomous)
            last_owner = control.owner

    def interrupting(keys):
        return any(
            k in {"s", "x", "k", "q"}
            or k == "h"
            and control.owner is not Ownership.HUMAN
            or k == "p"
            and not (control.autonomous and not control.rl_active)
            or k == "r"
            and getattr(control, "rl", None) is not None
            and not (control.autonomous and control.rl_active)
            for k in keys
        )

    try:
        event("session starts in HOLD")
        recorder.raise_if_failed()
        lease.begin()
        last_tick = clock.monotonic()
        while True:
            tick = clock.monotonic()
            if tick - last_tick > 0.2:
                raise RuntimeError("Control/recording loop stalled; episode aborted")
            last_tick = tick
            recorder.raise_if_failed()
            if leaders is not None:
                leaders.raise_if_failed()
            lease.pulse()
            keys, pending_keys = pending_keys + poll(), []
            if keys or control.owner is Ownership.HUMAN or armed is not None:
                last_input = max(last_input, tick)
            unattended = not keys and tick - last_input > idle_s
            if unattended:
                message(
                    "无人操作",
                    f"{idle_s / 60:.0f} 分钟无人操作：保持并保存本条（不标结果），随后回零并结束采集",
                    level="warning",
                )
            if "?" in keys:
                show_controls()
            if "q" in keys or "k" in keys or unattended:
                # Stop local proposals before waiting for status/HOLD/release.
                # Their latency must not age a stopped human controller into FAULT.
                control.control("s", now=clock.monotonic(), qpos=sender.feedback)
                lease.close()
                event("unattended" if unattended else "finish", force=True)
                # Publication requires confirmed HOLD AND exclusive release.
                outcome = (
                    "discard"
                    if "q" in keys
                    else "unlabelled"
                    if unattended
                    else review()
                    if review is not None
                    else "unlabelled"
                )
                if outcome == UNATTENDED:  # nobody labelled it
                    outcome, unattended = "unlabelled", True
                failure_type = None
                if isinstance(outcome, tuple):
                    outcome, failure_type = outcome
                if outcome not in {"success", "fail", "discard", "unlabelled"}:
                    raise ValueError("Invalid recording outcome")
                keep = outcome != "discard"
                message(
                    "保存" if keep else "丢弃",
                    "已确认 HOLD 并释放控制权；正在等待落盘，请勿中断。",
                )
                if failure_type is not None:
                    recorder.finish(
                        keep=keep, outcome=outcome, failure_type=failure_type
                    )
                elif outcome in {"success", "fail"}:
                    recorder.finish(keep=keep, outcome=outcome)
                else:
                    recorder.finish(keep=keep)
                message(
                    "保存" if keep else "丢弃",
                    "已确认 HOLD 并释放控制权；记录已保存。没有自动归位。"
                    if keep
                    else "已确认 HOLD 并释放控制权；整条记录标记不可用。没有删除原文件。",
                    level="success" if keep else "warning",
                )
                return UNATTENDED if unattended else outcome
            if interrupting(keys):
                control.control("s", now=clock.monotonic(), qpos=sender.feedback)
                lease.hold("operator ownership boundary")
                event("operator boundary", force=True)
            snap = capture()
            now = clock.monotonic()
            if set(snap.camera_times) != {"main", "left", "right"} or set(
                snap.leader_times
            ) != {"left", "right"}:
                raise RuntimeError("Missing camera or leader")
            if any(
                not np.isfinite(t) or not 0 <= now - t <= 0.1
                for t in snap.camera_times.values()
            ):
                raise RuntimeError("Camera acquisition timestamp is stale/future")
            if last_cameras is not None and any(
                snap.camera_times[k] <= last_cameras[k] for k in last_cameras
            ):
                raise RuntimeError("Camera timestamp repeated/regressed")
            stale = [
                side
                for side, t in snap.leader_times.items()
                if not np.isfinite(t) or not 0 <= now - t <= 0.2
            ]
            if stale:
                # A dropped USB adapter shows up here first: name it.
                for side in stale:
                    arm = getattr(leaders, "leaders", {}).get(side)
                    if arm is not None:
                        arm.check_connection()
                names = "、".join({"left": "左", "right": "右"}[s] for s in stale)
                raise RuntimeError(
                    f"A leader is stale: {names}主臂采样超过 0.2 秒未更新"
                    "（检查该主臂的 USB 线、集线器和供电）；takeover must remain available"
                )
            if last_leaders is not None and any(
                snap.leader_times[k] < last_leaders[k] for k in last_leaders
            ):
                raise RuntimeError("A leader timestamp regressed")
            if (
                not max(snap.camera_times.values())
                <= snap.state_requested
                <= snap.state_received
                <= now
                or set(snap.leader) != {"left", "right"}
                or any(
                    np.shape(v) != (7,) or not np.isfinite(v).all()
                    for v in snap.leader.values()
                )
            ):
                raise RuntimeError("Invalid state/leader observation")
            follower_times = np.asarray(snap.follower_times, dtype=float)
            if (
                follower_times.shape != (14,)
                or not np.isfinite(follower_times).all()
                or np.any(follower_times <= 0)
                or np.any(follower_times > snap.state_received)
                or np.any(now - follower_times > 0.1)
                or last_follower is not None
                and np.any(follower_times < last_follower)
            ):
                raise RuntimeError(
                    "Follower CAN receive timestamps stale/future/regressed"
                )
            last_follower = follower_times.copy()
            last_cameras, last_leaders = (
                dict(snap.camera_times),
                dict(snap.leader_times),
            )
            sender.observe(snap.qpos)
            # Stops win over takeover; takeover wins over handback in one poll.
            key = next((k for k in ("x", "s", "h", "p", "r") if k in keys), None)
            if key == "r" and getattr(control, "rl", None) is None:
                message("RL", "未启用 RLT（collect 加 --rlt）；r 无效", level="warning")
                key = None
            if key is not None:
                repeated = (
                    key == "h"
                    and (control.owner is Ownership.HUMAN or armed is not None)
                    or key == "p"
                    and control.autonomous
                    and not control.rl_active
                    or key == "r"
                    and control.autonomous
                    and control.rl_active
                )
                if not repeated and key == "h" and leaders is not None:
                    # Stop now; ownership moves only on the trigger gesture.
                    control.control("s", now=now, qpos=snap.qpos)
                    sender.reanchor(snap.qpos)
                    sides = tuple(snap.leader)
                    armed = {
                        "phase": "confirm",
                        "opened": dict.fromkeys(sides, False),
                        "closed": dict.fromkeys(sides, False),
                    }
                    event("takeover armed", force=True)
                    message(
                        "接管准备",
                        "已保持，扳机已断电。双手握住两只主臂，两个扳机各自完全松开再扣到底确认；s / p 取消",
                        level="warning",
                    )
                elif not repeated:
                    armed = None
                    control.control(
                        "s" if key == "x" else key,
                        now=now,
                        qpos=snap.qpos,
                        leader=snap.leader,
                        sampled_at=min(snap.leader_times.values()),
                    )
                    sender.reanchor(snap.qpos)
                    sender.new_segment()
                    event(
                        "rl selected" if key == "r" else "ownership selected",
                        force=True,
                    )
                    if key == "r":
                        message(
                            "RL", "RL actor 接管策略执行；h 接管 / p 回到 VLA / s 停止"
                        )
            if armed is not None and not any(k in keys for k in ("s", "p", "x")):
                # HOLD; leader joints stay aligned to the robot. Confirmation:
                # with the triggers unpowered, each is opened fully and then
                # squeezed fully (both hands on the leaders); the triggers
                # then return powered to the robot gripper before handover.
                trigger = {side: float(snap.leader[side][6]) for side in armed["opened"]}
                # The re-powered trigger aims inside its span (see Leader.follow).
                gripper = {
                    side: float(
                        np.clip(
                            snap.qpos[index],
                            LEADER_TRIGGER_MARGIN,
                            1 - LEADER_TRIGGER_MARGIN,
                        )
                    )
                    for side, index in (("left", 6), ("right", 13))
                }
                names = {"left": "左", "right": "右"}
                if armed["phase"] == "confirm":
                    for side, value in trigger.items():
                        if value >= LEADER_TRIGGER_OPEN:
                            armed["opened"][side] = True
                        if (
                            armed["opened"][side]
                            and not armed["closed"][side]
                            and value <= LEADER_TRIGGER_CLOSED
                        ):
                            armed["closed"][side] = True
                            message("接管准备", f"{names[side]}扳机已确认")
                    if all(armed["closed"].values()):
                        armed.update(phase="return", since=now, warned=False)
                        message("接管准备", "确认完成，扳机上电回到夹爪开度")
                elif all(
                    abs(trigger[side] - gripper[side]) <= LEADER_TRIGGER_MATCH
                    for side in trigger
                ):
                    armed = None
                    leaders.passive()
                    # Leader was held at the follower pose: the relative anchor
                    # starts the human exactly where the robot stands.
                    control.control(
                        "h",
                        now=now,
                        qpos=snap.qpos,
                        leader=snap.leader,
                        sampled_at=min(snap.leader_times.values()),
                    )
                    sender.reanchor(snap.qpos)
                    sender.new_segment()
                    event("ownership selected", force=True)
                    message("接管", "人工接管：主臂和扳机已放松，按主臂控制")
                elif now - armed["since"] > 2.0 and not armed["warned"]:
                    armed["warned"] = True
                    message(
                        "接管准备",
                        "扳机未回到夹爪开度（"
                        + " / ".join(
                            f"{names[k]} {trigger[k]:.2f}→{gripper[k]:.2f}" for k in trigger
                        )
                        + "）；松开手指或按 s 取消",
                        level="warning",
                    )
            if leaders is not None:
                if control.owner is not Ownership.HUMAN and (
                    leaders.continuous or armed is not None
                ):
                    leaders.follow(
                        snap.qpos,
                        trigger=armed is None or armed["phase"] == "return",
                    )
                else:
                    leaders.passive()
            if "c" in keys:
                recorder.checkpoint(ns())
                message("检查点", "已标记；按 x 可丢弃此后的片段")
            if "x" in keys:
                recorder.discard(ns(), after_checkpoint=True)
                message(
                    "丢弃",
                    "已丢弃检查点后的片段（无检查点则整条无效），保持不动；h / p 重新选择控制",
                    level="warning",
                )
            observed_at = min(snap.camera_times.values())
            target = control.propose(
                now=now,
                observed_at=observed_at,
                observation=build_observation(snap),
                leader=snap.leader,
                sampled_at=min(snap.leader_times.values()),
                qpos=snap.qpos,
            )
            for decision in control.take_decisions():
                recorder.rl_decision(
                    ns(decision["observed_at"]),
                    ns(decision["accepted_at"]),
                    token=decision["token"],
                    proprio=decision["proprio"],
                    reference=decision["reference"],
                    actor=decision["actor"],
                    actor_version=decision["actor_version"],
                    committed=decision["committed"],
                    skip=decision["skip"],
                )
            for chunk in control.take_accepted():
                recorder.policy_chunk(
                    ns(chunk["observed_at"]),
                    ns(chunk["accepted_at"]),
                    chunk["actions"],
                    continuous=chunk["continuous"],
                    skip=chunk["skip"],
                )
            event("controller transition")

            def before_send(
                snap=snap,
                pending_keys=pending_keys,
                tick=tick,
                observed_at=observed_at,
                follower_times=follower_times,
            ):
                # Also polled AFTER activate and AFTER IK/filtering, not only
                # before those potentially blocking calls. c never stops motion.
                recorder.raise_if_failed()
                lease.check()
                incoming = poll()
                pending_keys.extend(incoming)
                if interrupting(incoming):
                    control.control("s", now=clock.monotonic(), qpos=snap.qpos)
                    lease.hold("operator input pending")
                    event("operator preemption", force=True)
                    return False
                if (
                    clock.monotonic() - tick > 0.1
                    or clock.monotonic() - observed_at > 0.1
                    or np.any(clock.monotonic() - follower_times > 0.1)
                ):
                    raise RuntimeError("Command deadline exceeded; no late send")
                return True

            applied = None
            sent_at = ack_at = 0
            if target is not None and before_send():
                lease.activate()

                def commit_send():
                    nonlocal sent_at
                    if not before_send():
                        return False
                    sent_at = ns()
                    return True

                try:
                    applied = sender.send(
                        target,
                        snap.qpos,
                        generation=control.arbiter.generation,
                        policy=control.owner is Ownership.POLICY,
                        before_send=commit_send,
                    )
                except PolicyTargetRejected as exc:
                    # Only a pre-RPC policy planning rejection is recoverable.
                    # A coincident expired lease/deadline or failed HOLD is not.
                    lease.check()
                    if (
                        clock.monotonic() - tick > 0.1
                        or clock.monotonic() - observed_at > 0.1
                        or np.any(clock.monotonic() - follower_times > 0.1)
                    ):
                        raise RuntimeError(
                            "Policy rejection exceeded command deadline"
                        ) from exc
                    reason = f"Policy rejected; explicit h/p required: {exc}"
                    lease.hold(reason)
                    lease.check()
                    control.control("s", now=clock.monotonic(), qpos=snap.qpos)
                    # Do not let a handback queued before rejection restart it.
                    pending_keys.extend(poll())
                    # Any queued start (h / p / r, incl. key repeat) is stale now.
                    pending_keys[:] = [
                        k for k in pending_keys if k not in {"h", "p", "r"}
                    ]
                    event(reason, force=True)
                    message(
                        "跟随卡住" if isinstance(exc, TrackingStalled) else "策略拒绝",
                        f"已保持，不会自动重试。按 h 人工纠偏 / p 重新交还。\n  原因：{exc}",
                        level="warning",
                    )
                if applied is not None:
                    ack_at = ns()
                    lease.check()
                    control.acknowledged(applied)
            if applied is None:
                if lease.active:
                    lease.hold("waiting for fresh observation/inference or handback")
                sender.reanchor(snap.qpos)
            recorder.sample(
                ns(observed_at),
                snap.qpos,
                sender.reference if applied is None else applied,
                {
                    "top": snap.images["main"],
                    "left": snap.images["left"],
                    "right": snap.images["right"],
                },
                raw_action=target,
                timing={
                    "camera_ns": [
                        ns(snap.camera_times[k]) for k in ("main", "left", "right")
                    ],
                    "state_requested_ns": ns(snap.state_requested),
                    "state_received_ns": ns(snap.state_received),
                    "follower_received_ns": [ns(t) for t in follower_times],
                    "command_sent_ns": sent_at if applied is not None else 0,
                    "command_ack_ns": ack_at,
                    "action_valid": applied is not None,
                },
            )
            clock.sleep(max(0, 1 / 30 - (clock.monotonic() - tick)))
    except BaseException as exc:
        errors = [exc]
        try:
            control.fault(now=clock.monotonic())
        except BaseException as fault:  # noqa: BLE001 - re-raised with primary failure
            errors.append(fault)
        try:
            lease.close()
        except BaseException as cleanup:  # noqa: BLE001 - never skip the FAULT journal
            errors.append(cleanup)
        try:
            recorder.abort(ns(), f"{type(exc).__name__}: {exc}")
        except BaseException as journal:  # noqa: BLE001 - preserve every abort failure
            errors.append(journal)
        message(
            "故障",
            "采集异常退出，保存结果未确认。请检查机械臂和记录完整性。\n"
            f"  原因：{type(exc).__name__}: {exc}",
            level="error",
        )
        if len(errors) > 1:
            raise BaseExceptionGroup(
                "HG-DAgger abort/cleanup failures", errors
            ) from None
        raise


def prepare_start(args, profile, reference, *, unattended=False):
    """Home both arms after an on-site confirmation; ``unattended``: without
    one (nobody has operated for IDLE_PROMPT_S). An unanswered confirmation
    raises Unattended before any motion."""
    if not sys.stdin.isatty():
        raise RuntimeError("Preparation requires the onsite operator's terminal")
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot

    from dagger_offline.homing import move_to_start

    def confirm(_legacy_prompt):
        if unattended:
            message(
                "归零确认",
                f"{IDLE_PROMPT_S / 60:.0f} 分钟无人操作：自动回零，不等现场确认",
                level="warning",
            )
            return
        message(
            "归零确认",
            "双臂回零、夹爪张开：沿用数采插值，每帧 ≤0.01 rad、30 Hz。\n"
            "请确认完整运动路径无障碍；关节限位不能检测碰撞。",
            level="warning",
        )
        answer = timed_input("按回车确认；输入其它内容或 Ctrl-C 取消：")
        if answer is None:
            raise Unattended("归零确认无人回应")
        if answer:
            raise RuntimeError("已取消归零，未发送运动指令")

    with ExitStack() as stack:
        safety = SafetyClient(
            host=profile.network.host, port=profile.network.safety_port, timeout_ms=100
        )
        stack.callback(safety.close)
        robot = ZMQClientRobot(
            host=profile.network.host,
            port=profile.network.hardware_port,
            timeout_ms=100,
        )
        stack.callback(robot.close)
        move_to_start(
            robot,
            safety,
            reference,
            confirm=confirm,
            log=args.output.with_suffix(".prepare.jsonl"),
        )
        message("归零完成", "反馈已到位，已确认 HOLD；可启动新记录。", level="success")


# A model server's first inference compiles (often well over the 2 s control
# deadline); every episode first runs one request with this timeout.
POLICY_WARMUP_TIMEOUT_S = 60.0


def warm_up_policy(factory, observe, *, timeout_s=POLICY_WARMUP_TIMEOUT_S, clock=time.monotonic):
    """One inference on the live observation before control starts: the robot
    stays in HOLD, nothing is sent or recorded, the connection is closed."""
    message("模型", "预热推理中；服务刚启动时首次推理需要编译，可能要几十秒")
    client = factory(timeout_s=timeout_s)
    try:
        client.reset()
        start = clock()
        client.infer_chunk(observe())
        elapsed = clock() - start
    finally:
        client.close()
    message("模型", f"推理已就绪（本次 {elapsed:.1f} s）", level="success")
    return elapsed


def execute(args, config, profile, camera_config, reference, metadata):
    if not sys.stdin.isatty():
        raise RuntimeError(
            "--execute requires the onsite operator's interactive terminal"
        )
    message(
        "现场检查",
        "确认起点、场景及运动空间，能随时断电。无需输入口令；启动后先 HOLD，按 h / p 才控制。",
        level="warning",
    )
    import yaml
    from gello.cameras.realsense_rig import CameraRig
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot
    from rhospolicy.inference.yam_act import unpack_action, validate_joint_limits
    from rhospolicy.inference.yam_pi05 import (
        Pi05PolicyClient,
        build_pi05_observation,
        wait_for_camera_bundle,
    )
    from run_pi05_heavier import require_start

    from dagger_offline.live_control import Controller, PolicyWorker, TickLease
    from dagger_offline.live_recording import AsyncEpisodeRecorder
    from dagger_offline.live_robot import (
        GuardedSender,
        Leader,
        LeaderFollower,
        leader_name,
        relative_controller,
        start_cameras,
    )

    actuated = not getattr(args, "passive_leaders", False)
    on_demand = actuated and getattr(args, "leader_on_demand", False)
    metadata = {
        **metadata,
        "leader_takeover": (
            "align_on_h_then_trigger"
            if on_demand
            else "follow_then_trigger"
            if actuated
            else "immediate_relative"
        ),
    }
    url = f"ws://{config['host']}:{config['port']}"
    if "sealed_policy_id" in metadata:
        from dagger_offline.deployment import SealedPolicyClient

        def policy_factory(timeout_s=2.0):
            return SealedPolicyClient(
                url, metadata["sealed_policy_id"], timeout_s=timeout_s
            )

        # Identity-only probe: no reset/inference, no recording/device/session.
        # The worker connects separately and rechecks identity on every reply.
        policy_factory().close()
    else:

        def policy_factory(timeout_s=2):
            return Pi05PolicyClient(
                url, request_timeout_s=timeout_s, transport_timeout_s=timeout_s
            )

    with ExitStack() as stack:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        recorder = stack.enter_context(
            AsyncEpisodeRecorder(
                args.output, task_name=config["instruction"], metadata=metadata
            )
        )
        safety = SafetyClient(
            host=profile.network.host, port=profile.network.safety_port, timeout_ms=100
        )
        stack.callback(safety.close)
        robot = ZMQClientRobot(
            host=profile.network.host,
            port=profile.network.hardware_port,
            timeout_ms=100,
        )
        stack.callback(robot.close)
        TickLease.require_contract(safety.status())
        cap = robot.get_joint_sample_capabilities()
        if (
            cap.get("supported") is not True
            or cap.get("api_version") != 1
            or cap.get("num_dofs") != 14
            or cap.get("time_base") != "host_monotonic_can_receive"
        ):
            raise RuntimeError("Daemon requires per-joint CAN receive timestamps")
        initial = unpack_action(robot.get_joint_sample()["joint_state"])
        require_start(initial, safety.status(), reference)
        limits = validate_joint_limits(safety.status()["joint_limits"])
        stack.enter_context(driver_output())
        leaders = {}
        for side, name in (("left", "左"), ("right", "右")):
            leaders[side] = Leader(
                yaml.safe_load(profile.arms[side].config_path.read_text()),
                actuated=actuated,
            )
            stack.callback(leaders[side].close)
            message("主臂", f"{name}主臂已连接（{leader_name(leaders[side].port)}）")
        follower = None
        if actuated:
            # Closed before the leaders (LIFO): stop writing, then torque off.
            follower = LeaderFollower(leaders, continuous=not on_demand)
            stack.callback(follower.close)
            message(
                "主臂",
                "兼容模式：策略执行时主臂不出力；按 h 后主臂对齐机械臂，扳机断电：完全松开再扣到底确认后接管"
                if on_demand
                else "主臂和扳机自动跟随机械臂；接管：按 h 后两个扳机各自完全松开再扣到底，扳机回位后接管",
                level="warning",
            )
        rig = start_cameras(CameraRig, camera_config)
        stack.callback(rig.close)
        warm_up_policy(
            policy_factory,
            lambda: build_pi05_observation(
                unpack_action(robot.get_joint_sample()["joint_state"]),
                {
                    name: frame.rgb.copy()
                    for name, frame in wait_for_camera_bundle(
                        rig, timeout_s=2.0, poll_interval_s=0.002
                    ).frames.items()
                },
                config["instruction"],
            ),
        )
        worker = PolicyWorker(policy_factory)
        stack.callback(worker.close)
        agent = None
        if "rlt" in metadata:
            from dagger_offline import rlt

            settings = metadata["rlt"]
            rlt.bind_run(settings["run"], settings["policy_id"])
            agent = rlt.RLAgent(
                rlt.RLTConfig(**settings["config"]),
                Path(settings["run"]) / "actor",
                max_deviation=settings["max_deviation"],
                policy_id=settings["policy_id"],
            )
            message(
                "RL", f"RLT 已启用：r 交给 RL actor；actor 目录 {settings['run']}/actor"
            )
        control = Controller(
            relative_controller(initial),
            worker,
            now=time.monotonic(),
            replan_lead=metadata["policy_execution"]["replan_lead"],
            blend=metadata["policy_execution"].get("blend", 0),
            rl=agent,
            rl_lead=metadata.get("rlt", {}).get("rl_lead"),
        )
        lease = TickLease(safety, uuid.uuid4().hex)
        sender = GuardedSender(robot, limits, initial, session_id=lease.session_id)
        poll = stack.enter_context(keyboard())
        show_controls()

        def capture():
            bundle = wait_for_camera_bundle(rig, timeout_s=0.1, poll_interval_s=0.002)
            requested = time.monotonic()
            sample = robot.get_joint_sample()
            qpos = unpack_action(sample["joint_state"])
            received = time.monotonic()
            samples = {side: device.sample() for side, device in leaders.items()}
            return Snapshot(
                qpos,
                {name: frame.rgb.copy() for name, frame in bundle.frames.items()},
                {name: frame.host_timestamp for name, frame in bundle.frames.items()},
                requested,
                received,
                sample["received_monotonic_s"],
                {side: s[0] for side, s in samples.items()},
                {side: s[1] for side, s in samples.items()},
            )

        require_start(
            unpack_action(robot.get_joint_sample()["joint_state"]),
            safety.status(),
            reference,
        )
        return run_loop(
            control=control,
            sender=sender,
            lease=lease,
            recorder=recorder,
            capture=capture,
            poll=poll,
            build_observation=lambda snap: build_pi05_observation(
                snap.qpos, snap.images, config["instruction"]
            ),
            review=lambda: review_outcome(poll),
            leaders=follower,
        )


def main():
    parser = argparse.ArgumentParser(
        description="HG-DAgger：默认离线检查；录制从 HOLD 开始，不自动归位或使能机械臂。",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--rhos-root", type=Path, required=True, help="隔离 RhOS 运行目录"
    )
    parser.add_argument("--abc", type=Path, required=True, help="固定版本 ABC 源码目录")
    parser.add_argument(
        "--output", type=Path, required=True, help="新记录路径，不覆盖已有文件"
    )
    parser.add_argument(
        "--deployment",
        type=Path,
        help="已封存的新轮次策略；省略时使用旧 baseline / 8767",
    )
    parser.add_argument(
        "--passive-leaders",
        action="store_true",
        help="主臂不跟随机械臂；h 立即按当前主臂姿态相对接管",
    )
    parser.add_argument(
        "--rlt",
        type=Path,
        help="RLT 运行目录（与 rlt-learn --run 相同）；启用 r 键交给 RL actor",
    )
    parser.add_argument(
        "--leader-on-demand",
        action="store_true",
        help="兼容模式：主臂只在按 h 后对齐机械臂，策略执行时不出力",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="现场交互录制，先保持并等待 h / p"
    )
    mode.add_argument(
        "--prepare-start",
        action="store_true",
        help="现场确认后，限速移动至双臂零位、夹爪张开",
    )
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("HG-DAgger interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    prepared = preflight(args)
    config, _, _, _, metadata = prepared
    limits = metadata["control_limits"]
    message("HG-DAgger", "配置检查通过", level="success")
    message(
        "模型", f"{config['train_config']} · ws://{config['host']}:{config['port']}"
    )
    message("起点", "双臂零位、夹爪张开；record 不自动归位，需要时先运行 prepare")
    message(
        "人工",
        f"关节上限 {limits['human_arm_velocity_rad_s']:g} rad/s；相对接管，不跳到主臂姿态",
    )
    message(
        "策略",
        f"关节上限 {limits['policy_arm_velocity_rad_s']:g} rad/s；剩 8 步预取下一块，块间连续执行不停顿",
    )
    message("记录", str(args.output.resolve()))
    if args.execute:
        execute(args, *prepared)
    elif args.prepare_start:
        prepare_start(args, prepared[1], prepared[3])
    else:
        message(
            "离线检查",
            "通过。未打开网络连接、相机、串口或电机；这不是实机验收。",
            level="success",
        )


if __name__ == "__main__":
    main()
