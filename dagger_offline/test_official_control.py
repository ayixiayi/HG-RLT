"""CPU contract checks against the unmodified pinned ABC checkout; no hardware."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from deploy.robot.control import mink_ik
from deploy.robot.control.relative_pose import apply_relative_pose_delta
from deploy.robot.gym import dagger_rollout as official
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_relative_pose_world_frame_noncommuting_rotations(scale):
    anchor = Rotation.from_euler("x", 70, degrees=True)
    delta = Rotation.from_euler("z", 40, degrees=True)
    follower = Rotation.from_euler("y", -35, degrees=True)
    pos, quat = apply_relative_pose_delta(
        tracked_pos=np.array([1.2, -0.6, 0.9]),
        tracked_wxyz=(delta * anchor).as_quat(scalar_first=True),
        anchor_tracked_pos=np.array([1.0, -0.5, 0.6]),
        anchor_tracked_wxyz=anchor.as_quat(scalar_first=True),
        anchor_output_pos=np.array([-0.2, 0.4, 0.1]),
        anchor_output_wxyz=follower.as_quat(scalar_first=True),
        position_scale=scale,
        rotation_scale=scale,
    )
    np.testing.assert_allclose(
        pos, [-0.2 + 0.2 * scale, 0.4 - 0.1 * scale, 0.1 + 0.3 * scale], atol=1e-12
    )
    expected = Rotation.from_euler("z", 40 * scale, degrees=True) * follower
    np.testing.assert_allclose(
        Rotation.from_quat(quat, scalar_first=True).as_matrix(),
        expected.as_matrix(),
        atol=1e-12,
    )


def test_stationary_leader_preserves_different_follower_anchor():
    leader_q = Rotation.from_euler("xyz", [20, 30, -50], degrees=True).as_quat(
        scalar_first=True
    )
    follower_q = Rotation.from_euler("xyz", [-10, 50, 80], degrees=True).as_quat(
        scalar_first=True
    )
    pos, quat = apply_relative_pose_delta(
        tracked_pos=np.array([1, 2, 3]),
        tracked_wxyz=-leader_q,
        anchor_tracked_pos=np.array([1, 2, 3]),
        anchor_tracked_wxyz=leader_q,
        anchor_output_pos=np.array([-1, 0.3, 0.7]),
        anchor_output_wxyz=follower_q,
    )
    np.testing.assert_allclose(pos, [-1, 0.3, 0.7])
    np.testing.assert_allclose(quat, follower_q, atol=1e-12)


def test_real_yam_mink_ik_reaches_known_fk_target_with_step_limits(monkeypatch):
    model = (
        Path(__file__).resolve().parents[1]
        / "vendor/i2rt-model/i2rt/robot_models/yam/yam.xml"
    )
    monkeypatch.setattr(mink_ik, "find_yam_model", lambda: model)
    start = np.array([0.1, 0.7, 1.2, -0.6, 0.15, -0.2])
    solver = mink_ik.MinkArmController(start, posture_cost=0, max_joint_delta=0.02)
    target_pos, target_quat = solver.fk(start + [0.08, -0.05, 0.07, 0.03, -0.04, 0.06])
    q = start.copy()
    initial_error = np.linalg.norm(solver.fk(q)[0] - target_pos)
    assert initial_error > 0.005
    for _ in range(100):
        new = solver.solve(target_pos, target_quat, q)
        assert np.isfinite(new).all()
        assert np.max(np.abs(new - q)) <= 0.02000001
        assert (new >= solver.lower).all() and (new <= solver.upper).all()
        q = new
    pos, quat = solver.fk(q)
    assert np.linalg.norm(pos - target_pos) < 0.001
    rotation_error = Rotation.from_quat(
        quat, scalar_first=True
    ).inv() * Rotation.from_quat(target_quat, scalar_first=True)
    assert rotation_error.magnitude() < 0.01


@pytest.fixture
def loop(monkeypatch):
    # Do not call DaggerLoop/YAMEnv constructors: they open network/hardware paths.
    monkeypatch.setattr(official.time, "monotonic", lambda: 123.0)
    loop = official.DaggerLoop.__new__(official.DaggerLoop)
    loop.args = SimpleNamespace(
        rtc=False,
        prefix_length=4,
        execute_chunk_dim=3,
        gello_stale_timeout=0.2,
        intervention_rate=60,
    )
    loop.config = SimpleNamespace(policy=SimpleNamespace(dt=0))
    loop.current_state = np.arange(14, dtype=np.float32) / 20
    loop.state = official.DaggerState.POLICY
    loop.current_chunk = np.full((3, 14), -999, dtype=np.float32)
    loop.action_index = 2
    loop.rtc_manager = None
    loop.intervention_tail = deque(maxlen=4)
    loop.inference_id = loop.chunk_index = loop.checkpoint = 0
    loop.episode = 1
    loop.inference_pub = None
    loop._next_tail_sample = 0
    loop._gello_stale_warned = False
    loop.latest_gello = {side: np.zeros(7) for side in ("left", "right")}
    loop.gello_update_time = {
        side: official.time.monotonic() for side in ("left", "right")
    }
    loop._update_gellos = lambda: None
    loop.events = []
    loop._publish_event = lambda event: loop.events.append(event)
    loop._publish_recorder = lambda *a, **kw: None
    loop.relative = SimpleNamespace(
        anchor=lambda *a: None, step=lambda _: loop.current_state + 0.01
    )
    loop.commands = []
    loop.observations = []

    def get_obs(**kwargs):
        loop.observations.append(kwargs["state"].copy())
        return {"state": kwargs["state"].copy()}

    def step(command):
        loop.commands.append(command.copy())
        return command.copy()

    loop.env = SimpleNamespace(get_obs=get_obs, step_state=step)
    loop.policy = SimpleNamespace(
        infer=lambda obs: {"actions": np.tile(obs["state"] + 0.02, (3, 1))}
    )
    monkeypatch.setattr(official, "_publish_inference_event", lambda *a, **kw: None)
    monkeypatch.setattr(official.time, "sleep", lambda _: None)
    return loop


def test_real_state_methods_intervene_and_handback_replace_old_chunk(loop):
    loop._handle_key("a")
    assert loop.state == official.DaggerState.INTERVENTION
    loop._intervention_step()
    human_state = loop.current_state.copy()
    loop._handle_key("a")
    assert loop.state == official.DaggerState.POLICY
    np.testing.assert_array_equal(loop.observations[-1], human_state)
    assert loop.action_index == 0
    loop._policy_step()
    np.testing.assert_allclose(loop.commands[-1], human_state + 0.02)
    assert loop.events == ["intervention_start", "intervention_end"]
    assert not any((cmd == -999).any() for cmd in loop.commands)


def test_stale_leader_refuses_entry_and_holds_during_intervention(loop):
    loop.gello_update_time["right"] = float("-inf")
    loop._handle_key("a")
    assert loop.state == official.DaggerState.POLICY
    assert not loop.commands
    loop.state = official.DaggerState.INTERVENTION
    before = loop.current_state.copy()
    loop._intervention_step()
    np.testing.assert_array_equal(loop.commands[-1], before)
    assert loop._gello_stale_warned


def test_checkpoint_and_finish_do_not_issue_motion(loop):
    loop._handle_key("c")
    assert loop.checkpoint == 1
    loop._handle_key("b")
    assert loop.state == official.DaggerState.FINISH_PENDING
    assert not loop.commands
    assert loop.events == ["checkpoint", "finish_pending"]


def test_short_policy_chunk_is_rejected(loop):
    with pytest.raises(ValueError, match="expected"):
        loop._extract_chunk({"actions": np.zeros((2, 14))})
