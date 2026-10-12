"""Real CPU LeRobot round-trip tests; optional in the lightweight ABC env."""

import json

import cv2
import h5py
import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest_plugins = ["dagger_offline.test_round_data"]


@pytest.fixture
def bridge_round(round_inputs, tmp_path):
    from dagger_offline.round_data import read_source, sha256

    records, stats = round_inputs
    sources, episodes = [], []
    for record in records:
        with h5py.File(record["path"], "r+") as f:
            # Asymmetric channels distinguish RGB from BGR; temporal changes
            # distinguish the source slice from an accidentally reset index.
            for camera, base in (("top", 20), ("left", 60), ("right", 100)):
                for i in range(120):
                    color = np.full((224, 224, 3), [base, i, 200], np.uint8)
                    ok, jpeg = cv2.imencode(".jpg", color)
                    assert ok
                    f[f"data/{camera}"][i] = jpeg
            _, _, _, _, _, task = read_source(f, record["role"], 100)
        digest = sha256(record["path"])
        sources.append({**record, "sha256": digest})
        ranges = (
            [("previous", 20, 24), ("previous", 80, 83)]
            if record["role"] == "previous"
            else [
                ("intervention", 20, 24),
                ("intervention", 80, 83),
                ("autonomous", 0, 4),
                ("autonomous", 65, 68),
            ]
        )
        for component, start, end in ranges:
            episodes.append(
                {
                    "source": record["path"],
                    "source_sha256": digest,
                    "group": record["group"],
                    "split": record["split"],
                    "component": component,
                    "frame_range": [start, end],
                    "frames": end - start,
                    "task_name": task,
                }
            )
    root = tmp_path / "abc_round"
    root.mkdir()
    (root / "norm_stats.json").write_bytes(stats.read_bytes())
    manifest = {
        "complete": True,
        "horizon": 3,
        "max_age_ms": 100,
        "norm_sha256": sha256(stats),
        "sources": sources,
        "episodes": episodes,
    }
    path = root / "round.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.mark.parametrize("current_only", [False, True])
def test_real_lerobot_round_trip(bridge_round, tmp_path, current_only):
    from deploy.recording import io as rio
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from dagger_offline.lerobot_bridge import convert_round

    path, original = bridge_round
    if current_only:
        original["current_only"] = True
        original["sources"] = [s for s in original["sources"] if s["role"] == "current"]
        original["episodes"] = [
            e for e in original["episodes"] if e["component"] != "previous"
        ]
        path.write_text(json.dumps(original))
    output = tmp_path / "lerobot"
    result = convert_round(path, output)
    assert result == json.loads((output / "lerobot_round.json").read_text())
    assert result["complete"] and result["horizon"] == 3
    assert result["norm_sha256"] == original["norm_sha256"]
    assert (output / "norm_stats.json").read_bytes() == (
        path.parent / "norm_stats.json"
    ).read_bytes()
    assert {d["path"] for d in result["datasets"]} == {
        f"{component}/{split}"
        for component in (
            ("intervention", "autonomous")
            if current_only
            else ("previous", "intervention", "autonomous")
        )
        for split in ("train", "val")
    }
    assert len({d["repo_id"] for d in result["datasets"]}) == (4 if current_only else 6)
    for entry in result["datasets"]:
        dataset = LeRobotDataset(
            entry["repo_id"], root=output / entry["path"], video_backend="pyav"
        )
        assert dataset.meta.info["codebase_version"] == "v2.1"
        assert dataset.fps == 30 and dataset.num_episodes == 2
        assert dataset.num_frames == 7
        assert dataset.episode_data_index["from"].tolist() == [0, 4]
        assert dataset.episode_data_index["to"].tolist() == [4, 7]
        offset = 0
        for index, episode in enumerate(entry["episodes"]):
            assert episode["episode_index"] == index
            expected = next(
                e
                for e in original["episodes"]
                if e["source"] == episode["source"]
                and e["frame_range"] == episode["frame_range"]
                and e["component"] == entry["component"]
            )
            for key in ("frames", "group", "source", "source_sha256", "frame_range"):
                assert episode[key] == expected[key]
            start, end = episode["frame_range"]
            with h5py.File(episode["source"]) as f:
                for relative, source_index in enumerate(range(start, end)):
                    row = dataset[offset + relative]
                    assert row["episode_index"].item() == index
                    assert row["frame_index"].item() == relative
                    assert row["task"] == expected["task_name"]
                    actions = np.concatenate(
                        [f[f"data/{k}"][source_index] for k in rio.ACTION_KEYS]
                    )
                    states = np.concatenate(
                        [f[f"data/{k}"][source_index] for k in rio.STATE_KEYS]
                    )
                    np.testing.assert_allclose(
                        row["action"].numpy(), actions, rtol=1e-6
                    )
                    np.testing.assert_allclose(
                        row["observation.state"].numpy(), states, rtol=1e-6
                    )
                    for camera, source_camera in (
                        ("main", "top"),
                        ("left", "left"),
                        ("right", "right"),
                    ):
                        rgb = rio.decode_frame(f, source_camera, source_index)[
                            ..., ::-1
                        ]
                        actual = (
                            row[f"observation.images.{camera}"]
                            .numpy()
                            .transpose(1, 2, 0)
                            * 255
                        )
                        np.testing.assert_allclose(
                            actual.mean((0, 1)), rgb.mean((0, 1)), atol=5
                        )
            offset += episode["frames"]
    with pytest.raises(FileExistsError):
        convert_round(path, output)


@pytest.mark.parametrize(
    "damage", ["incomplete", "norm", "source", "boundary", "provenance", "missing"]
)
def test_rejects_invalid_round_before_writing(bridge_round, tmp_path, damage):
    from dagger_offline.lerobot_bridge import convert_round

    path, manifest = bridge_round
    if damage == "incomplete":
        manifest["complete"] = False
    elif damage == "norm":
        (path.parent / "norm_stats.json").write_text("{}")
    elif damage == "source":
        with h5py.File(manifest["sources"][0]["path"], "r+") as f:
            f.attrs["changed"] = True
    elif damage == "boundary":
        episode = next(
            e for e in manifest["episodes"] if e["component"] == "intervention"
        )
        episode.update(frame_range=[19, 23])
    elif damage == "provenance":
        manifest["episodes"][0]["group"] = "wrong-parent"
    else:
        manifest["episodes"] = [
            e for e in manifest["episodes"] if e["component"] != "autonomous"
        ]
    path.write_text(json.dumps(manifest))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        convert_round(path, output)
    assert not output.exists()
