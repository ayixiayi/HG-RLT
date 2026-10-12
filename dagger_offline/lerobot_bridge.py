"""Convert a verified ABC round.json to six local LeRobot v2.1 datasets.

Reads original synchronized HDF5 camera JPEGs, never ABC's stacked MP4.
Run with --round path/to/round.json --output new_dir in the LeRobot environment.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from deploy.recording import io as rio

from dagger_offline.round_data import COMPONENTS, read_source, sha256


def _validate_episode(episode, source, data, horizon):
    _, _, _, labels, valid, task = data
    start, end = episode["frame_range"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(valid)
        or end - start < horizon
        or episode["frames"] != end - start
    ):
        raise ValueError("Invalid episode frame_range or horizon")
    component = episode["component"]
    if any(episode[k] != source[k] for k in ("group", "split")) or (
        episode["source_sha256"] != source["sha256"] or episode["task_name"] != task
    ):
        raise ValueError("Episode provenance disagrees with source")
    if component == "previous":
        pure = source["role"] == "previous" and len(set(labels[start:end])) == 1
    else:
        expected = "INTERVENTION" if component == "intervention" else "POLICY_RUNNING"
        pure = source["role"] == "current" and np.all(labels[start:end] == expected)
    if not pure or not valid[start:end].all():
        raise ValueError("Episode crosses an invalid frame or controller boundary")


def _images(f, synced, cameras, index):
    return {
        f"observation.images.{name}": np.ascontiguousarray(
            rio.decode_frame(f, cameras[key], int(synced.indices[cameras[key]][index]))[
                ..., ::-1
            ]
        )
        for name, key in (("main", "top"), ("left", "left"), ("right", "right"))
    }


def convert_round(round_path, output):
    """Validate sources, preserve absolute actions, and return lerobot_round.json."""
    round_path, output = Path(round_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads(round_path.read_text())
    if manifest["complete"] is not True:
        raise ValueError("Round is not complete")
    horizon, max_age_ms = manifest["horizon"], manifest["max_age_ms"]
    if (
        type(horizon) is not int
        or horizon < 1
        or not np.isfinite(max_age_ms)
        or max_age_ms <= 0
    ):
        raise ValueError("Invalid horizon or max_age_ms")
    norm_bytes = (round_path.parent / "norm_stats.json").read_bytes()
    if hashlib.sha256(norm_bytes).hexdigest() != manifest["norm_sha256"]:
        raise ValueError("Normalization hash mismatch")
    sources, groups, hashes = {}, {}, set()
    for source in manifest["sources"]:
        path = source["path"]
        if path in sources or source["sha256"] in hashes:
            raise ValueError("Duplicate source")
        if (
            source["split"] not in ("train", "val")
            or source["role"] not in ("previous", "current")
            or not isinstance(source["group"], str)
            or not source["group"].strip()
        ):
            raise ValueError("Invalid source provenance")
        if groups.setdefault(source["group"], source["split"]) != source["split"]:
            raise ValueError("Source group crosses train/val")
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        hashes.add(source["sha256"])
        sources[path] = source
    components = COMPONENTS[1:] if manifest.get("current_only") else COMPONENTS
    partitions = {
        (component, split): [] for component in components for split in ("train", "val")
    }
    for episode in manifest["episodes"]:
        key = (episode["component"], episode["split"])
        if key not in partitions or episode["source"] not in sources:
            raise ValueError("Unknown episode component, split, or source")
        partitions[key].append(episode)
    if any(not episodes for episodes in partitions.values()):
        raise ValueError("All required component/split datasets must contain episodes")
    # Validate all boundaries before creating any output. No image data is cached.
    for path, source in sources.items():
        with h5py.File(path) as f:
            data = read_source(f, source["role"], max_age_ms)
            ranges = []
            for episode in (e for e in manifest["episodes"] if e["source"] == path):
                _validate_episode(episode, source, data, horizon)
                ranges.append(episode["frame_range"])
            ranges.sort()
            if any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
                raise ValueError("Overlapping source episode ranges")

    # Optional heavyweight dependencies are imported only when converting.
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    from convert_mcap import JOINT_NAMES

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("Install the OpenPI-pinned LeRobot v2.1 revision")
    result = {
        "complete": False,
        "current_only": manifest.get("current_only", False),
        "format": "lerobot_dagger_round_v1",
        "horizon": horizon,
        "norm_sha256": manifest["norm_sha256"],
        "fps": 30,
        "action_semantics": "absolute_joint_position",
        "joint_order": JOINT_NAMES,
        "source_round": str(round_path),
        "source_round_sha256": sha256(round_path),
        "datasets": [],
    }
    output.mkdir(parents=True)
    result_path = output / "lerobot_round.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (output / "norm_stats.json").write_bytes(norm_bytes)
    unique = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    for (component, split), episodes in partitions.items():
        relative = f"{component}/{split}"
        entry = {
            "path": relative,
            "repo_id": f"local/dagger_{unique}_{component}_{split}",
            "component": component,
            "split": split,
            "episodes": [],
        }
        dataset = None
        try:
            for episode_index, episode in enumerate(episodes):
                source = sources[episode["source"]]
                path = source["path"]
                if sha256(path) != source["sha256"]:
                    raise ValueError(f"Source changed during conversion: {path}")
                with h5py.File(path) as f:
                    data = read_source(f, source["role"], max_age_ms)
                    _validate_episode(episode, source, data, horizon)
                    synced, states, actions, _, _, _ = data
                    cameras = rio.camera_streams(f)
                    start, end = episode["frame_range"]
                    images = _images(f, synced, cameras, start)
                    features = {
                        name: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
                        for name in ("observation.state", "action")
                    }
                    features.update(
                        {
                            name: {
                                "dtype": "video",
                                "shape": image.shape,
                                "names": ["height", "width", "channel"],
                            }
                            for name, image in images.items()
                        }
                    )
                    if dataset is None:
                        dataset = LeRobotDataset.create(
                            repo_id=entry["repo_id"],
                            root=output / relative,
                            fps=30,
                            robot_type="yam_dual",
                            features=features,
                            use_videos=True,
                            video_backend="pyav",
                        )
                    for index in range(start, end):
                        images = _images(f, synced, cameras, index)
                        if any(
                            tuple(dataset.features[k]["shape"]) != v.shape
                            for k, v in images.items()
                        ):
                            raise ValueError(
                                "Camera image shape changed within dataset"
                            )
                        dataset.add_frame(
                            {
                                "observation.state": states[index].astype(np.float32),
                                "action": actions[index].astype(np.float32),
                                "task": episode["task_name"],
                                **images,
                            }
                        )
                    dataset.save_episode()
                if sha256(path) != source["sha256"]:
                    raise ValueError(f"Source changed during conversion: {path}")
                entry["episodes"].append(
                    {
                        "episode_index": episode_index,
                        **{
                            k: episode[k]
                            for k in (
                                "frames",
                                "group",
                                "source",
                                "frame_range",
                                "source_sha256",
                            )
                        },
                    }
                )
        finally:
            if dataset is not None:
                dataset.stop_image_writer()
        result["datasets"].append(entry)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    result["complete"] = True
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert_round(args.round, args.output)


if __name__ == "__main__":
    main()
