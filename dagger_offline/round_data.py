"""CPU-only, source-grouped DAgger preparation using ABC's episode writer.

Input JSON: records with path, group, split (train/val), role (previous/current).
Paths are relative to the input JSON. A group identifies a whole rollout/scene,
not an intervention fragment. Output is ABC format, not yet a LeRobot adapter.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from deploy.recording import io as rio
from deploy.recording.export import ExportConfig, write_episode

COMPONENTS = ("previous", "intervention", "autonomous")
WEIGHTS = (0.8, 0.1, 0.1)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_source(f, role, max_age_ms):
    """Use official synchronization, rejecting stale/nonfinite/unknown samples."""
    if rio.is_unusable(f):
        raise ValueError("Recording marked unusable; remove it from the input list")
    kind = rio.detect_recording_type(f)
    if role == "current" and kind != "dagger":
        raise ValueError("Current-round data requires DAgger intervention events")
    if kind not in ("dagger", "teleop"):
        raise ValueError(
            "Unlabelled inference recordings are not expert demonstrations"
        )
    cameras = rio.camera_streams(f)
    if set(rio.CAMERA_KEYS) - cameras.keys():
        raise ValueError("All three cameras are required")
    keys = [*rio.STATE_KEYS, *rio.ACTION_KEYS, *(cameras[k] for k in rio.CAMERA_KEYS)]
    for key in keys:
        times = rio.stream_timestamps(f, key)
        if (
            len(times) == 0
            or len(times) != len(f["data"][key])
            or np.any(times[1:] <= times[:-1])
        ):
            raise ValueError(f"Invalid timestamps or sample count: {key}")
    if kind == "dagger":
        times = np.asarray(f["dagger_events/timestamp_ns"][:])
        if len(times) != len(f["dagger_events/controller_state"]) or np.any(
            times[1:] < times[:-1]
        ):
            raise ValueError("Invalid DAgger event timestamps")
    synced = rio.sync_streams(f, keys)
    if (
        len(synced.base_ts_ns) < 2
        or not 30_000_000 <= np.median(np.diff(synced.base_ts_ns)) <= 37_000_000
    ):
        raise ValueError(
            "Expected nominal 30 Hz camera data; resample other rates explicitly"
        )
    states = np.concatenate([synced.gather(f, k) for k in rio.STATE_KEYS], axis=1)
    actions = np.concatenate([synced.gather(f, k) for k in rio.ACTION_KEYS], axis=1)
    if states.shape != (len(synced.base_ts_ns), 14) or actions.shape != states.shape:
        raise ValueError("Expected 14-dimensional YAM states/actions")
    valid = rio.valid_mask(f, synced.base_ts_ns)
    valid &= np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
    for key in keys:
        age = synced.base_ts_ns - rio.stream_timestamps(f, key)[synced.indices[key]]
        valid &= age <= max_age_ms * 1_000_000
    # Do not let a chunk bridge a dropped camera tick (nominal 30 Hz).
    valid[1:] &= np.diff(synced.base_ts_ns) <= 50_000_000
    labels = (
        rio.dagger_state_labels(f, synced.base_ts_ns)
        if kind == "dagger"
        else np.full(len(valid), "DEMO")
    )
    alignment = f.attrs.get("sample_alignment")
    paired = alignment in {"observation_command_pair_v1", "observation_command_pair_v2"}
    if alignment is not None and not paired:
        raise ValueError("Unknown sample alignment contract")
    if paired:
        # Live rows pair a pre-command observation with a subsequent ACK. The
        # official reader joins the shared row index; these are not simultaneous
        # acquisition stamps. Validate the retained raw timing independently.
        timing = f["pairing"]
        count = len(valid)
        names = (
            "state_requested_ns",
            "state_received_ns",
            "command_sent_ns",
            "command_ack_ns",
            "action_valid",
        )
        if (
            timing["camera_ns"].shape != (count, 3)
            or any(timing[name].shape != (count,) for name in names)
            or any(
                not np.array_equal(rio.stream_timestamps(f, key), synced.base_ts_ns)
                for key in keys
            )
        ):
            raise ValueError("Invalid paired sample layout")
        camera = timing["camera_ns"][:]
        requested, received, sent, ack, acknowledged = (
            timing[name][:] for name in names
        )
        valid &= acknowledged.astype(bool)
        valid &= camera.min(axis=1) == synced.base_ts_ns
        valid &= (camera.max(axis=1) <= requested) & (requested <= received)
        valid &= (received <= sent) & (sent <= ack)
        valid &= (ack >= synced.base_ts_ns) & (
            ack - synced.base_ts_ns <= min(max_age_ms, 100) * 1_000_000
        )
        valid[1:] &= np.all(camera[1:] > camera[:-1], axis=1)
        if alignment == "observation_command_pair_v2":
            if "follower_received_ns" not in timing or timing[
                "follower_received_ns"
            ].shape != (count, 14):
                raise ValueError("Missing per-joint CAN receive timestamps")
            follower = timing["follower_received_ns"][:]
            valid &= np.all((follower > 0) & (follower <= received[:, None]), axis=1)
            valid &= np.all(
                (ack[:, None] >= follower)
                & (ack[:, None] - follower <= min(max_age_ms, 100) * 1_000_000),
                axis=1,
            )
            valid[1:] &= np.all(follower[1:] >= follower[:-1], axis=1)
    allowed = ("INTERVENTION", "POLICY_RUNNING") if kind == "dagger" else ("DEMO",)
    valid &= np.isin(labels, allowed)
    if kind == "dagger":
        event_times = np.asarray(f["dagger_events/timestamp_ns"][:])
        intervals = np.searchsorted(event_times, synced.base_ts_ns, side="right")
        for key in rio.ACTION_KEYS:
            action_times = rio.stream_timestamps(f, key)[synced.indices[key]]
            valid &= (
                np.searchsorted(event_times, action_times, side="right") == intervals
            )
        if paired:
            # An ownership change after capture but before ACK invalidates the
            # pair, even if the textual label later returns to the same owner.
            valid &= np.searchsorted(event_times, ack, side="right") == intervals
        # A complete pause can fit between camera ticks with identical labels.
        # Conservatively drop the following tick to separate the two windows.
        valid[1:] &= ~((intervals[1:] != intervals[:-1]) & (labels[1:] == labels[:-1]))
    for start, end in rio.discard_segments(f):
        if end > start:
            valid[1:] &= ~(
                (synced.base_ts_ns[:-1] < end) & (synced.base_ts_ns[1:] > start)
            )
    task = rio.decode_attr(f.attrs.get("task_name", "")).strip()
    if not task:
        raise ValueError("Recording has no task_name")
    return synced, states, actions, labels, valid, task


def prepare_round(
    records,
    output,
    norm_stats_path,
    *,
    horizon=30,
    max_age_ms=100,
    image_size=(168, 224),
    current_only=False,
):
    """Split BEFORE export; preserve parent stats; refuse missing mixture sources."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if horizon < 1 or not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("horizon and max_age_ms must be positive")
    norm_bytes = Path(norm_stats_path).read_bytes()
    norm = json.loads(norm_bytes)["norm_stats"]
    for name in ("state", "actions"):
        for stat in ("mean", "std"):
            values = np.asarray(norm[name][stat])
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError("Expected finite 14-D parent normalization statistics")
        if np.any(np.asarray(norm[name]["std"]) < 0):
            raise ValueError("Negative standard deviation")
    groups, hashes, sources, episodes = {}, set(), [], []
    skipped = []
    for record in records:
        path = Path(record["path"]).resolve()
        group, split, role = record["group"], record["split"], record["role"]
        if current_only and role != "current":
            raise ValueError("current_only accepts only current recordings")
        if (
            not isinstance(group, str)
            or not group.strip()
            or split not in ("train", "val")
            or role not in ("previous", "current")
        ):
            raise ValueError(
                "Each source needs group, train/val split, previous/current role"
            )
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group crosses train/val: {group}")
        groups[group] = split
        digest = sha256(path)
        if digest in hashes:
            raise ValueError(f"duplicate source content: {path}")
        hashes.add(digest)
        source = {
            "path": str(path),
            "sha256": digest,
            "group": group,
            "split": split,
            "role": role,
        }
        sources.append(source)
        with h5py.File(path) as f:
            synced, _, _, labels, valid, task = read_source(f, role, max_age_ms)
            for label in ("DEMO", "INTERVENTION", "POLICY_RUNNING"):
                component = (
                    "previous"
                    if role == "previous"
                    else ("intervention" if label == "INTERVENTION" else "autonomous")
                )
                for start, end in rio.true_ranges(valid & (labels == label)):
                    if end - start < horizon:
                        skipped.append(
                            {
                                "source": str(path),
                                "frame_range": [start, end],
                                "reason": "shorter_than_horizon",
                            }
                        )
                        continue
                    episode = {
                        "source": str(path),
                        "source_sha256": digest,
                        "group": group,
                        "split": split,
                        "component": component,
                        "controller_state": label,
                        "frame_range": [start, end],
                        "frames": end - start,
                        "windows": end - start - horizon + 1,
                        "task_name": task,
                        "path": f"{component}/{split}/{digest[:16]}_{start}_{end}",
                    }
                    episodes.append(episode)
    for component in COMPONENTS[1:] if current_only else COMPONENTS:
        for split in ("train", "val"):
            if not any(
                e["component"] == component and e["split"] == split for e in episodes
            ):
                raise ValueError(
                    f"No usable {component}/{split} windows; cannot use 80:10:10"
                )
    cfg = ExportConfig(
        input_pattern="",
        output_dir=str(output),
        output_img_height=image_size[0],
        output_img_width=image_size[1],
    )
    manifest = {
        "complete": False,
        "current_only": current_only,
        "format": "abc_dagger_round_v1",
        "horizon": horizon,
        "max_age_ms": max_age_ms,
        "sources": sources,
        "episodes": episodes,
        "skipped_segments": skipped,
        "norm_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "mixture": [
            {
                "train_dir": str(output / name / "train"),
                "val_dir": str(output / name / "val"),
                "weight": weight,
                "task_name": "",
            }
            for name, weight in zip(COMPONENTS, WEIGHTS, strict=True)
        ],
    }
    output.mkdir(parents=True)
    manifest_path = output / "round.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Source changed during preparation: {path}")
        with h5py.File(path) as f:
            synced, states, actions, _, _, _ = read_source(
                f, source["role"], max_age_ms
            )
            for episode in (e for e in episodes if e["source"] == str(path)):
                start, end = episode["frame_range"]
                write_episode(
                    f,
                    synced,
                    cfg,
                    output / episode["path"],
                    start,
                    end,
                    states,
                    actions,
                    episode["task_name"],
                    {
                        "source_h5": str(path),
                        "source_sha256": source["sha256"],
                        "source_group": source["group"],
                        "frame_range": [start, end],
                        "controller_state": episode["controller_state"],
                        "component": episode["component"],
                    },
                )
    (output / "norm_stats.json").write_bytes(norm_bytes)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sample_window(manifest, index, *, seed=0):
    """CPU reference for ABC's component-then-uniform-window sampling contract.

    Global index makes samples reproducible across resume; callers of distributed
    training must supply distinct global indices, not each rank's local index.
    This is not yet wired into the Pi0.5 loader.
    """
    if not manifest["complete"]:
        raise ValueError("Incomplete round")
    if manifest.get("current_only"):
        raise ValueError("Attach previous LeRobot data before mixture sampling")
    rng = np.random.default_rng((seed, index))
    name = COMPONENTS[int(rng.choice(3, p=WEIGHTS))]
    episodes = [
        e
        for e in manifest["episodes"]
        if e["split"] == "train" and e["component"] == name
    ]
    sizes = np.array([e["windows"] for e in episodes], dtype=np.int64)
    if not len(sizes) or np.any(sizes <= 0):
        raise ValueError(f"No valid windows for {name}")
    cumulative = sizes.cumsum()
    draw = int(rng.integers(int(cumulative[-1])))
    i = int(np.searchsorted(cumulative, draw, side="right"))
    return {
        "episode": episodes[i],
        "offset": draw - (int(cumulative[i - 1]) if i else 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Prepare new rollouts; attach existing LeRobot BC after bridging",
    )
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    for record in records:
        record["path"] = str(args.records.parent / record["path"])
    manifest = prepare_round(
        records,
        args.output,
        args.norm_stats,
        horizon=args.horizon,
        current_only=args.current_only,
    )
    print(
        json.dumps(
            {
                name: sum(
                    e["windows"]
                    for e in manifest["episodes"]
                    if e["component"] == name and e["split"] == "train"
                )
                for name in COMPONENTS
            }
        )
    )


if __name__ == "__main__":
    main()
