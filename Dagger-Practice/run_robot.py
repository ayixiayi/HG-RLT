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
from dagger_offline.console import message, show_controls, show_mode

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
        return list(data.decode("ascii", errors="ignore").lower())

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
                "rl_lead": 3,
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


def review_outcome(poll, clock=time):
    """Ask for the task result only after confirmed HOLD and lease release.

    A failure also asks for its type; returns ("fail", type) in that case.
    """
    from dagger_offline.live_recording import FAILURE_TYPES

    message("结果标记", "1 成功 · 2 失败 · 3 弃用；直接按键，无需回车")
    choices = {"1": "success", "2": "fail", "3": "discard", "q": "discard"}
    outcome = None
    while outcome is None:
        for key in poll():
            if key in choices:
                outcome = choices[key]
                break
        else:
            clock.sleep(0.05)
    if outcome != "fail":
        return outcome
    types = list(FAILURE_TYPES)
    message(
        "失败类型",
        " · ".join(f"{i} {FAILURE_TYPES[t]}" for i, t in enumerate(types, 1)),
    )
    while True:
        for key in poll():
            if key.isdigit() and 1 <= int(key) <= len(types):
                return outcome, types[int(key) - 1]
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
):
    """The real single-sender loop, also exercised with fake I/O and clock.

    No simulator-specific branch. Sensor/transport waits must be bounded by the
    adapters. The daemon watchdog remains responsible while this thread blocks.

    leaders (LeaderFollower or None): when given, the GELLO arms track the
    follower outside human control, and h only arms a takeover: HOLD now,
    then holding both triggers squeezed hands control to the human.
    """
    import numpy as np

    from dagger_offline.arbitration import Ownership
    from dagger_offline.live_robot import (
        LEADER_TRIGGER_HOLD,
        LEADER_TRIGGER_SQUEEZE,
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
    armed = None  # trigger baselines while a takeover gesture is pending
    squeezed_since = None  # both triggers squeezed continuously since

    def ns(value=None):
        return int((clock.monotonic() if value is None else value) * 1_000_000_000)

    def event(reason, force=False):
        nonlocal last_owner
        if force or control.owner is not last_owner:
            recorder.event(ns(), labels[control.owner], reason)
            if control.owner is not last_owner or reason == "ownership selected":
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
            if "?" in keys:
                show_controls()
            if "q" in keys or "k" in keys:
                # Stop local proposals before waiting for status/HOLD/release.
                # Their latency must not age a stopped human controller into FAULT.
                control.control("s", now=clock.monotonic(), qpos=sender.feedback)
                lease.close()
                event("finish", force=True)
                # Publication requires confirmed HOLD AND exclusive release.
                outcome = (
                    "discard"
                    if "q" in keys
                    else review()
                    if review is not None
                    else "unlabelled"
                )
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
                return outcome
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
            if any(
                not np.isfinite(t) or not 0 <= now - t <= 0.2
                for t in snap.leader_times.values()
            ):
                raise RuntimeError("A leader is stale; takeover must remain available")
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
                    armed = {side: float(snap.leader[side][6]) for side in snap.leader}
                    squeezed_since = None
                    event("takeover armed", force=True)
                    message(
                        "接管准备",
                        f"已保持。双手握住两只主臂，两个扳机同时扣住 {LEADER_TRIGGER_HOLD:g} 秒即接管；s / p 取消",
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
                # Both triggers: an unheld leader would sag and command the
                # follower toward a pose it cannot reach (leader/follower DoF).
                travel = {
                    side: abs(float(snap.leader[side][6]) - baseline)
                    for side, baseline in armed.items()
                }
                if all(t >= LEADER_TRIGGER_SQUEEZE for t in travel.values()):
                    squeezed_since = now if squeezed_since is None else squeezed_since
                else:
                    squeezed_since = None
                if (
                    squeezed_since is not None
                    and now - squeezed_since >= LEADER_TRIGGER_HOLD
                ):
                    armed = squeezed_since = None
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
                    message("接管", "人工接管；夹爪在扳机到达机械臂当前开度后接上")
            if leaders is not None:
                if control.owner is not Ownership.HUMAN and (
                    leaders.continuous or armed is not None
                ):
                    leaders.follow(snap.qpos)
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


def prepare_start(args, profile, reference):
    if not sys.stdin.isatty():
        raise RuntimeError("Preparation requires the onsite operator's terminal")
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot

    from dagger_offline.homing import move_to_start

    def confirm(_legacy_prompt):
        message(
            "归零确认",
            "双臂回零、夹爪张开：沿用数采插值，每帧 ≤0.01 rad、30 Hz。\n"
            "请确认完整运动路径无障碍；关节限位不能检测碰撞。",
            level="warning",
        )
        if input("按回车确认；输入其它内容或 Ctrl-C 取消："):
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
        relative_controller,
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

        def policy_factory():
            return SealedPolicyClient(url, metadata["sealed_policy_id"])

        # Identity-only probe: no reset/inference, no recording/device/session.
        # The worker connects separately and rechecks identity on every reply.
        policy_factory().close()
    else:

        def policy_factory():
            return Pi05PolicyClient(url, request_timeout_s=2, transport_timeout_s=2)

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
        leaders = {}
        for side in ("left", "right"):
            leaders[side] = Leader(
                yaml.safe_load(profile.arms[side].config_path.read_text()),
                actuated=actuated,
            )
            stack.callback(leaders[side].close)
        follower = None
        if actuated:
            # Closed before the leaders (LIFO): stop writing, then torque off.
            follower = LeaderFollower(leaders, continuous=not on_demand)
            stack.callback(follower.close)
            message(
                "主臂",
                "兼容模式：策略执行时主臂不出力；按 h 后主臂对齐机械臂，再两个扳机同时扣住 0.5 秒接管"
                if on_demand
                else "主臂会自动跟随机械臂；接管：按 h 后两个扳机同时扣住 0.5 秒",
                level="warning",
            )
        rig = CameraRig(camera_config, depth_enabled=False)
        stack.callback(rig.close)
        rig.start()
        rig.begin_episode()
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
            )
            message(
                "RL", f"RLT 已启用：r 交给 RL actor；actor 目录 {settings['run']}/actor"
            )
        control = Controller(
            relative_controller(initial),
            worker,
            now=time.monotonic(),
            replan_lead=metadata["policy_execution"]["replan_lead"],
            rl=agent,
            rl_lead=metadata.get("rlt", {}).get("rl_lead", 3),
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
