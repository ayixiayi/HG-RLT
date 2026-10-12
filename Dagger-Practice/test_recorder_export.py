"""Official ABC writer/export integration; synthetic input, no sockets or devices."""

import json
import queue
import subprocess
import sys
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest
from deploy.recording import io as rio
from deploy.robot.recorders.inference_recorder import InferenceRecorderNode


@pytest.fixture
def recorded(tmp_path):
    # Bypass Node.__init__ ONLY to avoid opening ZMQ sockets. Run the real HDF5
    # writer, metadata hooks, event serialization and discard implementation.
    node = object.__new__(InferenceRecorderNode)
    node.current_file_name = str(tmp_path / "synthetic.h5")
    node.recording_id = "synthetic-cpu-only"
    node.task_name = "synthetic_offline_only"
    node.collection_name = "not_training_data"
    node.recording_type = "dagger"
    node.checkpoint_path = "synthetic-parent-not-a-model"
    node.model_size = "synthetic"
    node.diffusion_steps = 5
    node.rtc = False
    node.dagger = True
    node.config = SimpleNamespace(
        robots={name: SimpleNamespace(init_q=[0.0] * 7) for name in ("left", "right")}
    )
    node.worker_queue = queue.Queue()
    times = 1_000_000_000 + np.arange(150, dtype=np.uint64) * 33_333_333
    states = (np.arange(150)[:, None] * 0.001 + np.arange(14) * 0.02).astype(np.float32)
    actions = (states + np.linspace(0.1, 0.3, 14)).astype(np.float32)
    colors = [(210, 30, 50), (40, 180, 60), (50, 70, 230)]
    for i, timestamp in enumerate(times):
        if i in (0, 40, 100):
            node.worker_queue.put(
                (
                    "dagger",
                    None,
                    None,
                    timestamp,
                    {
                        "controller_state": "intervention" if i == 40 else "policy",
                        "event": "transition",
                        "episode": 1,
                        "checkpoint": 0,
                    },
                )
            )
        if i == 80:
            node.worker_queue.put(
                ("checkpoint", None, states[i], timestamp, {"checkpoint": 1})
            )
        for name, color in zip(("top", "left", "right"), colors, strict=True):
            image = np.full((16, 16, 3), color, dtype=np.uint8)
            node.worker_queue.put(("rgb", name, image, timestamp))
        for name, start in (("left", 0), ("right", 7)):
            obs = np.concatenate(
                [
                    states[i, start : start + 7],
                    np.full(7, -2.0),
                    np.full(7, 3.0),
                    actions[i, start : start + 7],
                ]
            )
            node.worker_queue.put(("follower", name, obs, timestamp))
    node.worker_queue.put(("shutdown",))
    node._write_data()
    return node, times, states, actions, colors


def test_real_writer_preserves_streams_and_controller_events(recorded):
    node, times, states, actions, colors = recorded
    with h5py.File(node.current_file_name) as f:
        assert f.attrs["recording_type"] == "dagger"
        assert f.attrs["usable"]
        assert rio.dagger_state_labels(f, times).tolist() == (
            ["POLICY_RUNNING"] * 40 + ["INTERVENTION"] * 60 + ["POLICY_RUNNING"] * 50
        )
        for name, start in (("left", 0), ("right", 7)):
            np.testing.assert_array_equal(
                f[f"data/q_{name}"][:], states[:, start : start + 6]
            )
            np.testing.assert_array_equal(
                f[f"data/q_gripper_{name}"][:], states[:, start + 6 : start + 7]
            )
            np.testing.assert_array_equal(
                f[f"data/q_des_{name}"][:], actions[:, start : start + 7]
            )
            np.testing.assert_array_equal(f[f"timestamps/q_des_{name}"][:], times)
        np.testing.assert_array_equal(f["dagger_checkpoints/state"][0], states[80])
        for name, color in zip(("top", "left", "right"), colors, strict=True):
            rgb = cv2.imdecode(f[f"data/{name}"][0], cv2.IMREAD_COLOR)[:, :, ::-1]
            np.testing.assert_allclose(rgb.mean(axis=(0, 1)), color, atol=3)


def test_checkpoint_discard_to_official_export(recorded, tmp_path):
    node, times, states, actions, colors = recorded
    node._mark_discard(int(times[-1]), after_checkpoint=True)
    with h5py.File(node.current_file_name) as f:
        # The checkpoint sample itself is retained; everything after it is not.
        assert rio.valid_mask(f, times).tolist() == [True] * 81 + [False] * 69
        assert f.attrs["partial_discarded"]
    stats = tmp_path / "parent_norm.json"
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    name: {"mean": [0.0] * 14, "std": [1.0] * 14}
                    for name in ("state", "actions")
                }
            }
        )
    )
    output = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.recording.export",
            "--input-pattern",
            node.current_file_name,
            "--output-dir",
            str(output),
            "--norm-stats-path",
            str(stats),
            "--output-img-height",
            "16",
            "--output-img-width",
            "16",
        ],
        check=True,
        timeout=60,
    )
    summary = json.loads((output / "export_summary.json").read_text())
    assert [e["frames"] for e in summary["exported_episodes"]] == [41]
    episodes = list((output / "train").glob("*/states_actions.bin"))
    assert len(episodes) == 1
    rows = np.fromfile(episodes[0], dtype=np.float64).reshape(-1, 28)
    np.testing.assert_array_equal(rows[:, :14], states[40:81])
    np.testing.assert_array_equal(rows[:, 14:], actions[40:81])
    assert json.loads((output / "norm_stats.json").read_text()) == json.loads(
        stats.read_text()
    )
    # Explicitly expose the upstream default, never call this held-out validation.
    assert (output / "val").resolve() == (output / "train").resolve()
    video = cv2.VideoCapture(str(episodes[0].parent / "combined_camera-images-rgb.mp4"))
    frames = 0
    try:
        while True:
            ok, image = video.read()
            if not ok:
                break
            assert image.shape == (48, 16, 3)
            for camera, color in enumerate(colors):
                rgb = image[camera * 16 : (camera + 1) * 16, :, ::-1]
                np.testing.assert_allclose(rgb.mean(axis=(0, 1)), color, atol=12)
            frames += 1
    finally:
        video.release()
    assert frames == 41


@pytest.mark.parametrize("after_checkpoint", [False, True])
def test_whole_discard_or_missing_checkpoint_excludes_recording(
    recorded, after_checkpoint, tmp_path
):
    from deploy.recording.export import ExportConfig, export_recordings

    node, times, *_ = recorded
    if after_checkpoint:
        with h5py.File(node.current_file_name, "r+") as f:
            del f["dagger_checkpoints"]
    node._mark_discard(int(times[-1]), after_checkpoint=after_checkpoint)
    with h5py.File(node.current_file_name) as f:
        assert not f.attrs["usable"]
        assert rio.is_unusable(f)
    # Whole-file exclusion is separate from the per-frame discard mask.
    output = tmp_path / "excluded"
    with pytest.raises(RuntimeError, match="No episodes were exported"):
        export_recordings(
            ExportConfig(
                input_pattern=node.current_file_name,
                output_dir=str(output),
                compute_norm_stats=True,
            )
        )
    assert not list(output.rglob("states_actions.bin"))
