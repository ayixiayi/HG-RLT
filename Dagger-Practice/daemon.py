"""Read-only HG daemon check, or explicitly confirmed onsite stop/start/restart.

Never takes over a live session. Stop disables motors through the daemon's own
off API (the daemon then exits); start enables motors. Both need the onsite TTY.
"""

import argparse
import os
import socket
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagger_offline.console import message


def check(profile):
    from gello.safety.daemon_client import validate_daemon_status
    from gello.safety.rpc import SafetyClient
    from gello.zmq_core.robot_node import ZMQClientRobot

    from dagger_offline.live_control import TickLease

    with ExitStack() as stack:
        safety = SafetyClient(
            host=profile.network.host, port=profile.network.safety_port, timeout_ms=300
        )
        stack.callback(safety.close)
        status = safety.status()
        # Do not expose session or initialization tokens in diagnostics.
        message(
            "硬件进程",
            f"PID {status.get('pid')} · 状态 {status.get('state')} · HOLD 确认 {status.get('hold_confirmed')}",
        )
        message(
            "协议",
            f"会话版本 {status.get('session_api_version')} · 硬件端口 {status.get('hardware_port')} · CAN {status.get('channels')}",
        )
        TickLease.require_contract(status)
        if (
            status.get("state") != "safe_hold"
            or status.get("hold_confirmed") is not True
        ):
            raise RuntimeError("Confirmed SAFE_HOLD required; no state was changed")
        validate_daemon_status(
            status,
            {side: arm.can_channel for side, arm in profile.arms.items()},
            profile.network.hardware_port,
        )
        robot = ZMQClientRobot(
            host=profile.network.host,
            port=profile.network.hardware_port,
            timeout_ms=300,
        )
        stack.callback(robot.close)
        cap = robot.get_joint_sample_capabilities()
        if (
            cap.get("supported") is not True
            or cap.get("api_version") != 1
            or cap.get("num_dofs") != 14
            or cap.get("time_base") != "host_monotonic_can_receive"
        ):
            raise RuntimeError("Daemon lacks per-joint CAN receive timestamps")
    message(
        "协议", "检查通过（CONTRACT OK）。仅只读检查，不是实机验收。", level="success"
    )


def stop(profile, *, confirmed=False):
    """Disable motors through the daemon's own off API; the daemon then exits."""
    from gello.safety.rpc import SafetyClient

    if not sys.stdin.isatty():
        raise RuntimeError("Daemon stop requires the onsite operator's terminal")
    safety = SafetyClient(
        host=profile.network.host, port=profile.network.safety_port, timeout_ms=3000
    )
    try:
        status = safety.status()
        state, pid = status.get("state"), status.get("pid")
        message("硬件进程", f"PID {pid} · 状态 {state} · 原因 {status.get('reason')}")
        if state == "active":
            raise RuntimeError("采集仍在控制机械臂；先在采集终端按 s / k 结束")
        if state not in {"safe_hold", "gravity", "fault", "disarmed"}:
            raise RuntimeError(f"Unexpected daemon state {state}; no state was changed")
        if not confirmed:
            message(
                "下电警告",
                "此操作会关闭双臂电机，机械臂会失去保持力下落。先用手托住双臂。",
                level="warning",
            )
            if input("托住双臂后输入 off 确认下电：") != "off":
                raise RuntimeError("Daemon stop not authorized")
        if state != "disarmed":
            if state == "fault":
                # FAULT accepts the faulted owner, or any id when unowned.
                session = status.get("session_id") or uuid.uuid4().hex
            else:
                # Refuses while an exclusive HG session is still live.
                session = uuid.uuid4().hex
                safety.begin_session(session)
            safety.disarm(session, "off")
            message("下电", "电机已关闭，等待硬件进程退出", level="success")
    finally:
        safety.close()
    deadline = time.monotonic() + 15.0
    while pid and Path(f"/proc/{pid}").exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"Daemon PID {pid} disarmed but did not exit within 15s")
        time.sleep(0.2)
    message("硬件进程", "已退出，端口释放", level="success")


def start(root, profile, *, confirmed=False):
    if not sys.stdin.isatty():
        raise RuntimeError("Daemon startup requires the onsite operator's terminal")
    if not confirmed:
        message(
            "使能警告",
            "此操作会使能双臂，不会归位。旧 daemon 须已停止（./hgrlt daemon-stop）。\n"
            "  请确认运动空间无人、能随时断电。",
            level="warning",
        )
        if (
            input("输入 HG-DAEMON-ON-SITE 确认启动并使能新 daemon：")
            != "HG-DAEMON-ON-SITE"
        ):
            raise RuntimeError("Daemon startup not authorized")

    # A nonresponsive/legacy service still owns its port: never interpret an RPC
    # timeout as permission to initialize another pair of CAN controllers.
    with ExitStack() as stack:
        for port in (profile.network.safety_port, profile.network.hardware_port):
            sock = stack.enter_context(socket.socket())
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(
                    f"Port {port} is occupied; old daemon must be stopped onsite first"
                ) from exc

    from gello.safety.daemon_client import ensure_safety_daemon

    log = (
        Path(__file__).resolve().parents[1]
        / "logs"
        / f"hg-daemon-{uuid.uuid4().hex}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    ensure_safety_daemon(
        repo_root=root / "Client/arms/YAM/gello_software",
        python=sys.executable,
        left_config=profile.arms["left"].config_path,
        right_config=profile.arms["right"].config_path,
        channels={side: arm.can_channel for side, arm in profile.arms.items()},
        hardware_port=profile.network.hardware_port,
        safety_port=profile.network.safety_port,
        log_path=log,
        env=os.environ.copy(),
    )
    message("日志", str(log))
    check(profile)


def main():
    parser = argparse.ArgumentParser(
        description="HG 硬件进程：默认只读检查；--start 会使能，须现场确认。",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--rhos-root", type=Path, required=True, help="隔离 RhOS 运行目录"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--start", action="store_true", help="现场确认后启动硬件进程，不自动停止旧进程"
    )
    action.add_argument(
        "--stop", action="store_true", help="现场确认后关闭电机并退出硬件进程"
    )
    action.add_argument(
        "--restart", action="store_true", help="一次确认：下电退出旧进程，再启动并使能"
    )
    args = parser.parse_args()
    root = args.rhos_root.resolve()
    for path in (
        root,
        root / "XPolicyLab",
        root / "Client/arms/YAM/gello_software",
        root / "scripts",
    ):
        sys.path.insert(0, str(path))

    import yaml
    from gello.profile import load_profile
    from run_pi05_heavier import require_control_rate

    deployment = yaml.safe_load(
        (root / "configs/deployments/pi05_yam_heavier.yaml").read_text()
    )
    profile = load_profile(root / deployment["profile"])
    require_control_rate(profile)
    if profile.network.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Onsite daemon entry requires IPv4 localhost")
    if args.start:
        start(root, profile)
    elif args.stop:
        stop(profile)
    elif args.restart:
        if not sys.stdin.isatty():
            raise RuntimeError("Daemon restart requires the onsite operator's terminal")
        message(
            "重启警告",
            "先关闭双臂电机（机械臂失去保持力），旧进程退出后重新使能，不归位。\n"
            "  全程用手托住双臂；运动空间无人、能随时断电。",
            level="warning",
        )
        if input("托住双臂后输入 HG-DAEMON-ON-SITE 确认重启：") != "HG-DAEMON-ON-SITE":
            raise RuntimeError("Daemon restart not authorized")
        stop(profile, confirmed=True)
        start(root, profile, confirmed=True)
    else:
        check(profile)


if __name__ == "__main__":
    main()
