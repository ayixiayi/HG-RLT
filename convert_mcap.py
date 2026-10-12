#!/usr/bin/env python3
"""Convert YAM JSON/H.264 MCAP episodes to OpenPI-compatible LeRobot v2.1.

Run with --help for options. Source files are never modified or uploaded.
"""

import argparse
import base64
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader

CAMERAS = ("main", "left", "right")
JOINT_NAMES = [
    f"{arm}_{joint}"
    for arm in ("left", "right")
    for joint in (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
]
LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"


@dataclass
class Episode:
    path: Path
    times: dict[str, np.ndarray]
    samples: dict[str, list[dict]]
    action_topic: str


def discover(source: Path, outcome: str) -> list[Path]:
    paths = [source] if source.is_file() else list(source.rglob("*.mcap"))
    paths = sorted(
        (
            p
            for p in paths
            if p.suffix == ".mcap" and (outcome == "all" or p.parent.name == outcome)
        ),
        key=lambda p: (p.name, str(p)),
    )
    if not paths:
        raise ValueError(f"No MCAP files for outcome={outcome!r} in {source}")
    return paths


def read_episode(path: Path, action_topic: str) -> Episode:
    topics = ["/left/state", "/right/state", action_topic]
    topics += [f"/cameras/{camera}/rgb" for camera in CAMERAS]
    samples = {topic: [] for topic in topics}
    timestamps = {topic: [] for topic in topics}
    session = None
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for _, channel, message in reader.iter_messages(topics=["/session", *topics]):
            if channel.message_encoding != "json":
                raise ValueError(f"{path.name}: only YAM JSON MCAP is supported")
            payload = json.loads(message.data)
            data = payload["data"]
            topic = channel.topic
            if topic == "/session":
                if session is not None:
                    raise ValueError(f"{path.name}: multiple sessions in one MCAP")
                session = data
                continue
            timestamp = payload["source_time_ns"]
            if not isinstance(timestamp, int):
                raise ValueError(f"{path.name}: invalid source_time_ns for {topic}")
            if "position" in data:
                position = np.asarray(data["position"], dtype=np.float32)
                expected = 14 if topic == action_topic else 7
                if position.shape != (expected,) or not np.isfinite(position).all():
                    raise ValueError(f"{path.name}: invalid position vector in {topic}")
                data = {"position": position}
            elif topic.startswith("/cameras/"):
                if data.get("encoding") != "h264":
                    raise ValueError(f"{path.name}: expected H.264 packets in {topic}")
                shape = data.get("shape", [])
                if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
                    raise ValueError(f"{path.name}: invalid RGB shape in {topic}")
                if samples[topic] and shape != samples[topic][0]["shape"]:
                    raise ValueError(
                        f"{path.name}: camera resolution changed in {topic}"
                    )
            else:
                raise ValueError(f"{path.name}: missing position in {topic}")
            samples[topic].append(data)
            timestamps[topic].append(timestamp)
    if not session or session.get("arm_names") != ["left", "right"]:
        raise ValueError(f"{path.name}: expected left/right dual-arm session")
    if session.get("action_semantics") != "absolute_joint_position":
        raise ValueError(f"{path.name}: expected absolute_joint_position actions")
    if session.get("video_settings", {}).get("b_frames") != 0:
        raise ValueError(f"{path.name}: only H.264 without B frames is supported")
    for topic in topics:
        if not samples[topic]:
            raise ValueError(f"{path.name}: missing or empty {topic}")
        if any(b < a for a, b in zip(timestamps[topic], timestamps[topic][1:])):
            # Never reorder compressed video packets; their decode order matters.
            raise ValueError(f"{path.name}: source timestamps go backwards in {topic}")
    return Episode(
        path,
        {k: np.asarray(v, dtype=np.int64) for k, v in timestamps.items()},
        samples,
        action_topic,
    )


def align(
    times: dict[str, np.ndarray], fps: int, max_age_ms: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Use the latest sample at or before each grid point, with no future leakage."""
    if (
        fps <= 0
        or fps > 1_000_000_000
        or not np.isfinite(max_age_ms)
        or max_age_ms <= 0
    ):
        raise ValueError("fps and max_age_ms must be positive and finite")
    if not times or any(len(t) == 0 or np.any(np.diff(t) < 0) for t in times.values()):
        raise ValueError("Empty or unsorted source timestamps")
    start = max(int(t[0]) for t in times.values())
    end = min(int(t[-1]) for t in times.values())
    if end < start:
        raise ValueError("Streams have no overlapping time range")
    count = (end - start) * fps // 1_000_000_000 + 1
    # Keep epoch nanoseconds in integers; float64 loses sub-microsecond precision.
    grid = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
    indices = {}
    for topic, timestamp in times.items():
        indices[topic] = np.searchsorted(timestamp, grid, side="right") - 1
        age = grid - timestamp[indices[topic]]
        if np.any(age > max_age_ms * 1_000_000):
            raise ValueError(
                f"{topic}: stale sample, maximum age {age.max() / 1_000_000:.2f} ms"
            )
    return grid, indices


class CameraDecoder:
    """Decode every dependency packet, but retain only the latest RGB frame."""

    def __init__(self, packets: list[dict]):
        self.packets = packets
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.thread_count = 1
        self.index = -1
        self.image = None

    def at(self, index: int) -> np.ndarray:
        if index < self.index:
            raise ValueError("Camera decoder cannot seek backwards")
        while self.index < index:
            self.index += 1
            packet = self.packets[self.index]
            frames = self.codec.decode(
                av.Packet(base64.b64decode(packet["data"], validate=True))
            )
            if len(frames) != 1:
                raise ValueError(
                    "Expected one decoded frame per H.264 packet (no B frames)"
                )
            self.image = frames[0].to_ndarray(format="rgb24")
            if self.image.shape != tuple(packet["shape"]):
                raise ValueError("Decoded image shape disagrees with MCAP metadata")
        return self.image


def iter_frames(
    episode: Episode, fps: int, max_age_ms: float, task: str
) -> Iterator[dict]:
    grid, indices = align(episode.times, fps, max_age_ms)
    decoders = {
        camera: CameraDecoder(episode.samples[f"/cameras/{camera}/rgb"])
        for camera in CAMERAS
    }
    for i in range(len(grid)):
        state = np.concatenate(
            [
                episode.samples[topic][indices[topic][i]]["position"]
                for topic in ("/left/state", "/right/state")
            ]
        )
        frame = {
            "observation.state": state,
            "action": episode.samples[episode.action_topic][
                indices[episode.action_topic][i]
            ]["position"],
            "task": task,
        }
        for camera, decoder in decoders.items():
            frame[f"observation.images.{camera}"] = decoder.at(
                int(indices[f"/cameras/{camera}/rgb"][i])
            )
        yield frame


def features_for(episode: Episode) -> dict:
    features = {
        name: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
        for name in ("observation.state", "action")
    }
    for camera in CAMERAS:
        shape = tuple(episode.samples[f"/cameras/{camera}/rgb"][0]["shape"])
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    return features


def episode_report(episode: Episode, fps: int, max_age_ms: float) -> dict:
    grid, indices = align(episode.times, fps, max_age_ms)
    with episode.path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "source": f"{episode.path.parent.name}/{episode.path.name}",
        "sha256": digest,
        "outcome": episode.path.parent.name
        if episode.path.parent.name in ("success", "fail")
        else "unknown",
        "frames": len(grid),
        "start_source_time_ns": int(grid[0]),
        "end_source_time_ns": int(grid[-1]),
        "message_counts": {topic: len(t) for topic, t in episode.times.items()},
        "max_sample_age_ms": {
            topic: float((grid - t[indices[topic]]).max() / 1_000_000)
            for topic, t in episode.times.items()
        },
    }


def convert(
    paths: list[Path],
    output: Path,
    repo_id: str,
    task: str,
    fps: int = 30,
    action_topic: str = "/action/accepted",
    max_age_ms: float = 100,
) -> None:
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new output directory"
        )
    if not paths or not task.strip():
        raise ValueError(
            "At least one episode and a nonempty task instruction are required"
        )
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(
            "Install the OpenPI-pinned LeRobot revision from requirements.txt"
        )
    first = read_episode(paths[0], action_topic)
    align(first.times, fps, max_age_ms)
    features = features_for(first)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type="yam_dual",
        # The writer adds video metadata to this dict while saving episodes.
        features=features_for(first),
        use_videos=True,
        video_backend="pyav",
    )
    manifest = {
        "complete": False,
        "lerobot_revision": LEROBOT_REVISION,
        "fps": fps,
        "repo_id": repo_id,
        "task": task,
        "action_topic": action_topic,
        "action_semantics": "absolute_joint_position",
        "joint_order": JOINT_NAMES,
        "alignment": "previous sample on source_time_ns within common overlap",
        "max_age_ms": max_age_ms,
        "episodes": [],
    }
    manifest_path = output / "meta" / "conversion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for index, path in enumerate(paths):
            episode = first if index == 0 else read_episode(path, action_topic)
            if features_for(episode) != features:
                raise ValueError(f"{path.name}: feature shapes differ between episodes")
            report = episode_report(episode, fps, max_age_ms)
            print(
                f"[{index + 1}/{len(paths)}] {path.name}: {report['frames']} frames",
                flush=True,
            )
            for frame in iter_frames(episode, fps, max_age_ms, task):
                dataset.add_frame(frame)
            dataset.save_episode()
            report["episode_index"] = index
            manifest["episodes"].append(report)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["complete"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        dataset.stop_image_writer()
    print(
        f"Saved {dataset.num_episodes} episodes, {dataset.num_frames} frames to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("balance-0922/mcap"))
    parser.add_argument("--output", type=Path, help="New dataset root (must not exist)")
    parser.add_argument("--repo-id", default="local/balance_0922")
    parser.add_argument(
        "--task", help="Actual natural-language task instruction, not a session ID"
    )
    parser.add_argument(
        "--outcome", choices=("success", "fail", "all"), default="success"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--action-topic",
        choices=("/action/accepted", "/action/requested"),
        default="/action/accepted",
    )
    parser.add_argument("--max-age-ms", type=float, default=100)
    parser.add_argument(
        "--limit", type=int, help="Convert only the first N selected episodes"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate messages/alignment without decoding video or writing a dataset",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.inspect and (
        args.output is None or not args.task or not args.task.strip()
    ):
        parser.error("conversion requires --output and --task")
    try:
        paths = discover(args.input, args.outcome)
        if args.limit:
            paths = paths[: args.limit]
        if args.inspect:
            total = 0
            for path in paths:
                report = episode_report(
                    read_episode(path, args.action_topic), args.fps, args.max_age_ms
                )
                total += report["frames"]
                print(json.dumps(report))
            print(
                f"Validated {len(paths)} episodes, {total} frames at {args.fps} FPS (video not decoded)"
            )
        else:
            convert(
                paths,
                args.output,
                args.repo_id,
                args.task,
                args.fps,
                args.action_topic,
                args.max_age_ms,
            )
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
