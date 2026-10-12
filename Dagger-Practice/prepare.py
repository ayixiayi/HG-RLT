"""Source-group-isolated ABC exports; no model, robot, or training imports."""

import argparse
import glob
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
from deploy.recording import io as rio
from deploy.recording.export import ExportConfig, export_recordings


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(records, output, norm_stats):
    output, norm_stats = Path(output), Path(norm_stats)
    if output.exists():
        raise FileExistsError(output)
    if not records:
        raise ValueError("records must not be empty")
    groups, hashes, sources = {}, set(), []
    for row in records:
        path = Path(row["path"]).resolve()
        group, split, role = row["group"], row["split"], row["role"]
        if not isinstance(group, str) or not group.strip():
            raise ValueError("group must be a nonempty source rollout/scene ID")
        if split not in ("train", "val") or role not in ("previous", "current"):
            raise ValueError("invalid split or role")
        if group in groups and groups[group] != split:
            raise ValueError("source group crosses train/val")
        groups[group] = split
        sha = digest(path)
        if sha in hashes:
            raise ValueError("duplicate source content")
        hashes.add(sha)
        with h5py.File(path) as f:
            kind = rio.detect_recording_type(f)
            if rio.is_unusable(f):
                raise ValueError(f"unusable recording: {path}")
            if kind not in ("dagger", "teleop") or (
                role == "current" and kind != "dagger"
            ):
                raise ValueError(
                    "current sources require dagger; previous requires teleop/dagger"
                )
        sources.append(
            {
                "path": str(path),
                "group": group,
                "split": split,
                "role": role,
                "sha256": sha,
                "kind": kind,
            }
        )
    norm_sha = digest(norm_stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete round; a failed export leaves no apparent ready dataset.
    with tempfile.TemporaryDirectory(
        prefix="abc-round-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        components = ("previous", "intervention", "autonomous")
        for component in components:
            for split in ("train", "val"):
                (stage / component / split).mkdir(parents=True)
        episodes = []
        exports = []
        for index, source in enumerate(sources):
            labels = (
                ("INTERVENTION", "POLICY_RUNNING")
                if source["kind"] == "dagger"
                else (None,)
            )
            for label in labels:
                component = (
                    "previous"
                    if source["role"] == "previous"
                    else ("intervention" if label == "INTERVENTION" else "autonomous")
                )
                exported = stage / "export"
                cfg = ExportConfig(
                    input_pattern=glob.escape(source["path"]),
                    output_dir=str(exported),
                    mode="interventions" if label else "whole_episodes",
                    state_label=label or "INTERVENTION",
                    norm_stats_path=str(norm_stats),
                )
                try:
                    export_recordings(cfg)
                except RuntimeError as error:
                    if str(error) != "No episodes were exported":
                        raise
                summary = json.loads((exported / "export_summary.json").read_text())
                if summary["skipped_files"]:
                    raise ValueError(
                        f"source export rejected: {summary['skipped_files']}"
                    )
                exports.append(
                    {"source_index": index, "label": label, "summary": summary}
                )
                for episode in sorted((exported / "train").iterdir()):
                    relative = (
                        Path(component)
                        / source["split"]
                        / f"{index:04d}_{label}_{episode.name}"
                    )
                    metadata = json.loads(
                        (episode / "episode_metadata.json").read_text()
                    )
                    episodes.append(
                        dict(path=str(relative), source_index=index, **metadata)
                    )
                    shutil.move(str(episode), stage / relative)
                shutil.rmtree(exported)
        for component in components:
            for split in ("train", "val"):
                if not any((stage / component / split).iterdir()):
                    raise ValueError(f"empty component/split: {component}/{split}")
        if digest(norm_stats) != norm_sha or any(
            digest(s["path"]) != s["sha256"] for s in sources
        ):
            raise ValueError("source or parent statistics changed during export")
        shutil.copyfile(norm_stats, stage / "norm_stats.json")
        manifest = {
            "schema_version": 1,
            "chunk_length": 30,
            "sources": sources,
            "episodes": episodes,
            "exports": exports,
            "norm_sha256": norm_sha,
            "mixture": [
                {
                    "name": c,
                    "train_dir": f"{c}/train",
                    "val_dir": f"{c}/val",
                    "weight": w,
                }
                for c, w in zip(components, (0.8, 0.1, 0.1), strict=True)
            ],
        }
        (stage / "round.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    for row in records:
        row["path"] = str(args.records.resolve().parent / row["path"])
    prepare(records, args.output, args.norm_stats)


if __name__ == "__main__":
    main()
