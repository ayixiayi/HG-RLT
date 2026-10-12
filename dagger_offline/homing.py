"""Operator-confirmed preparation using the collection system's joint trajectory."""

import time
import uuid
from contextlib import ExitStack

import numpy as np

from dagger_offline.live_control import TickLease


def move_to_start(
    robot,
    safety,
    reference,
    *,
    confirm,
    log=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    from gello.safety.return_zero import ARM_INDICES, checked_joints, joint_trajectory
    from rhospolicy.inference.yam_act import (
        JsonlTrajectory,
        plan_action,
        validate_joint_limits,
    )
    from run_pi05_heavier import (
        ARM_TOLERANCE,
        GRIPPER_TOLERANCE,
        require_start,
    )

    lease = TickLease(safety, uuid.uuid4().hex)
    status = safety.status()
    lease.require_contract(status)
    lease.require_hold(status)
    if status.get("strict_session") is not False:
        raise RuntimeError(
            "Preparation requires the previous collection to release control"
        )
    limits = validate_joint_limits(status.get("joint_limits"))
    goal = checked_joints(reference["state"], "preparation target")
    initial = checked_joints(robot.get_joint_state(), "preparation feedback")
    plan = plan_action(initial, goal, limits)
    if plan.violations:
        raise ValueError("Invalid preparation target: " + "; ".join(plan.violations))
    confirm("Confirm the complete return path is clear")
    current = checked_joints(robot.get_joint_state(), "preparation feedback")
    if np.max(np.abs(current - initial)) > 0.005:
        raise RuntimeError(
            "Robot moved during preparation confirmation; no motion sent"
        )

    # Match ordinary collection: <=0.01 rad / frame, 30 Hz. The whole path is
    # confirmed above; its distance is not restricted to the old 0.75-rad start box.
    targets = joint_trajectory(current, target=goal)
    tolerance = np.full(14, ARM_TOLERANCE)
    tolerance[[6, 13]] = GRIPPER_TOLERANCE
    with ExitStack() as stack:
        trajectory = stack.enter_context(JsonlTrajectory(log)) if log is not None else None
        # Confirm HOLD/release before closing the log, including on failure.
        stack.callback(lease.close)
        lease.begin()
        lease.activate()
        for target in targets:
            lease.pulse()
            result = robot.command_joint_state(target, session_id=lease.session_id)
            if not isinstance(result, dict) or "applied_joint_state" not in result:
                raise RuntimeError("Preparation missing command acknowledgement")
            applied = checked_joints(result["applied_joint_state"], "preparation ACK")
            if not np.allclose(applied, target, atol=1e-6, rtol=0):
                raise RuntimeError("Daemon changed preparation command; stopping")
            if trajectory is not None:
                trajectory.write(
                    {
                        "qpos": checked_joints(
                            robot.get_joint_state(), "preparation feedback"
                        ).tolist(),
                        "target": goal.tolist(),
                        "applied_action": applied.tolist(),
                    }
                )
            sleep(1 / 30)

        # The trajectory includes the exact endpoint. Use the same feedback
        # tolerance as HG's existing start check, not a second stricter one.
        deadline = clock() + 10.0
        valid_samples = 0
        while clock() < deadline:
            lease.pulse()
            current = checked_joints(robot.get_joint_state(), "preparation feedback")
            valid = np.all(np.abs(current - goal) <= tolerance)
            valid_samples = valid_samples + 1 if valid else 0
            if valid_samples >= 3:
                break
            sleep(1 / 30)
        else:
            raise RuntimeError(
                "Zero feedback did not settle within 10s; "
                f"maximum arm error {np.max(np.abs(current - goal)[ARM_INDICES]):.4f} rad"
            )
    require_start(robot.get_joint_state(), safety.status(), reference)
