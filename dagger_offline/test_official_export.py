"""Exercise official HDF5 -> training binary/video export using synthetic data."""

import json
import subprocess
import sys

import cv2
import h5py
import numpy as np
import pytest
from deploy.recording import io as rio


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "synthetic.h5"
    times = 1_000_000_000 + np.arange(120, dtype=np.uint64) * 33_333_333
    states = np.arange(120)[:, None] * 0.001 + np.arange(14)[None, :] * 0.02
    actions = states + np.linspace(0.1, 0.3, 14)
    with h5py.File(path, "w") as f:
        f.attrs.update(
            recording_type="dagger", usable=True, task_name="synthetic_offline_only"
        )
        data = f.create_group("data")
        ts = f.create_group("timestamps")
        for cam, color in zip(("top", "left", "right"), (30, 110, 220), strict=True):
            ok, jpeg = cv2.imencode(".jpg", np.full((16, 16, 3), color, np.uint8))
            assert ok
            stream = data.create_dataset(
                cam, (120,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            for i in range(120):
                stream[i] = jpeg
            ts.create_dataset(cam, data=times)
        for key, values in {
            "q_left": states[:, :6],
            "q_gripper_left": states[:, 6:7],
            "q_right": states[:, 7:13],
            "q_gripper_right": states[:, 13:],
            "q_des_left": actions[:, :7],
            "q_des_right": actions[:, 7:],
        }.items():
            data.create_dataset(key, data=values)
            ts.create_dataset(key, data=times)
        events = f.create_group("dagger_events")
        events.create_dataset("timestamp_ns", data=times[[0, 20, 65, 80]])
        events.create_dataset(
            "controller_state",
            data=["policy", "intervention", "policy", "intervention"],
            dtype=h5py.string_dtype(),
        )
        discarded = f.create_group("discard_segments")
        discarded.create_dataset("start_ns", data=times[110:111])
        discarded.create_dataset("end_ns", data=times[119:120])
    return path, times, states, actions


def test_labels_and_checkpoint_discard_boundary(recording):
    path, times, _, _ = recording
    with h5py.File(path) as f:
        labels = rio.dagger_state_labels(f, times)
        assert (
            labels.tolist()
            == ["POLICY_RUNNING"] * 20
            + ["INTERVENTION"] * 45
            + ["POLICY_RUNNING"] * 15
            + ["INTERVENTION"] * 40
        )
        keep = rio.valid_mask(f, times)
        assert keep.tolist() == [True] * 111 + [False] * 9


def test_official_export_cli_binary_video_and_stats(recording, tmp_path):
    path, _, states, actions = recording
    output = tmp_path / "exported"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.recording.export",
            "--input-pattern",
            str(path),
            "--output-dir",
            str(output),
            "--compute-norm-stats",
            "--output-img-height",
            "16",
            "--output-img-width",
            "16",
        ],
        check=True,
    )
    summary = json.loads((output / "export_summary.json").read_text())
    assert [ep["frames"] for ep in summary["exported_episodes"]] == [45, 31]
    # This upstream default is deliberately asserted, not presented as held-out evaluation.
    assert (output / "val").is_symlink()
    assert (output / "val").resolve() == (output / "train").resolve()
    for i, (start, end) in enumerate(((20, 65), (80, 111))):
        episode = output / "train" / f"synthetic_intervention_{i:02d}"
        rows = np.fromfile(episode / "states_actions.bin", dtype=np.float64).reshape(
            -1, 28
        )
        np.testing.assert_allclose(rows[:, :14], states[start:end])
        np.testing.assert_allclose(rows[:, 14:], actions[start:end])
        cap = cv2.VideoCapture(str(episode / "combined_camera-images-rgb.mp4"))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (48, 16, 3)
            assert image[:16].mean() < image[16:32].mean() < image[32:].mean()
            count += 1
        cap.release()
        assert count == end - start
    stats = json.loads((output / "norm_stats.json").read_text())["norm_stats"]
    expected_actions = np.concatenate([actions[20:65], actions[80:111]])
    np.testing.assert_allclose(stats["actions"]["mean"], expected_actions.mean(axis=0))
