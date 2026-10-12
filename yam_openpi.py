"""YAM / pi0.5 preparation and training entrypoint for a pinned OpenPI checkout.

Upload this file to the OpenPI repository root. Run with its .venv/bin/python:
  yam_openpi.py --store /path/to/Jinming_Bao prepare
  yam_openpi.py --store /path/to/Jinming_Bao check
  yam_openpi.py --store /path/to/Jinming_Bao norm
  yam_openpi.py --store /path/to/Jinming_Bao check --normalized
  yam_openpi.py --store /path/to/Jinming_Bao train --exp-name smoke --steps 20

Training is explicit, not part of prepare/check/norm. Full fine-tuning is intended
for H100/H200, not the preparation GPU. Heldout data is reserved, not automatically
evaluated by the upstream training loop. Keep the original dataset: split videos
are relative symlinks to it. Transfer the entire datasets directory if moving it.
"""

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
SOURCE_REPO = "local/balance_0922"
TRAIN_REPO = "local/balance_0922_train"
VAL_REPO = "local/balance_0922_val"
CONFIG_NAME = "pi05_yam_heavier"
EXTRA_REPO = None


def select_episodes(
    total: int, val_count: int, seed: int
) -> tuple[list[int], list[int]]:
    if not 0 < val_count < total:
        raise ValueError("Validation count must leave at least one training episode")
    val = sorted(
        np.random.default_rng(seed).choice(total, val_count, replace=False).tolist()
    )
    return [i for i in range(total) if i not in val], val


def write_subset(
    source: Path,
    target: Path,
    repo_id: str,
    episode_ids: list[int],
    extra_sources: list[tuple[Path, list[int]]] | None = None,
) -> None:
    """Write v2.1 metadata/parquet with dense indices; reuse unchanged video bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.common.datasets.compute_stats import get_feature_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    episodes = []
    reference = None
    for root, ids in [(source, episode_ids), *(extra_sources or [])]:
        # Require local data rather than silently downloading an absent repo.
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(root / "meta/info.json")
        src = LeRobotDatasetMetadata(SOURCE_REPO, root=root)
        if src.info["codebase_version"] != "v2.1":
            raise ValueError("Only the OpenPI-pinned LeRobot v2.1 layout is supported")
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(i not in src.episodes for i in ids)
        ):
            raise ValueError("Expected nonempty, distinct, existing episode indices")
        if reference is None:
            reference = src
        elif (src.fps, src.robot_type, src.features, src.tasks) != (
            reference.fps,
            reference.robot_type,
            reference.features,
            reference.tasks,
        ):
            raise ValueError(
                "Source dataset schema, rate, robot or task mapping differs"
            )
        for i in ids:
            for relative in [
                src.get_data_file_path(i),
                *[src.get_video_file_path(i, k) for k in src.video_keys],
            ]:
                if not (root / relative).is_file():
                    raise FileNotFoundError(root / relative)
            episodes.append((root, src, i))
    src = reference
    dst = LeRobotDatasetMetadata.create(
        repo_id,
        root=target,
        fps=src.fps,
        robot_type=src.robot_type,
        features=copy.deepcopy(src.features),
        use_videos=bool(src.video_keys),
    )
    manifest = {
        "complete": False,
        "source_root_relative": os.path.relpath(source, target),
        "source_episode_indices": episode_ids,
    }
    if extra_sources:
        manifest["extra_sources"] = [
            {
                "source_root_relative": os.path.relpath(root, target),
                "source_episode_indices": ids,
            }
            for root, ids in extra_sources
        ]
    manifest_path = target / "meta/split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for task_id in sorted(src.tasks):
        dst.add_task(src.tasks[task_id])
    for new_id, (source_root, src, old_id) in enumerate(episodes):
        table = pq.read_table(source_root / src.get_data_file_path(old_id))
        n = len(table)
        if n != src.episodes[old_id]["length"]:
            raise ValueError(f"Episode {old_id} length disagrees with metadata")
        stats = copy.deepcopy(src.episodes_stats[old_id])
        for key, values in {
            "episode_index": np.full(n, new_id, dtype=np.int64),
            "index": np.arange(dst.total_frames, dst.total_frames + n, dtype=np.int64),
        }.items():
            field = table.schema.field(key)
            table = table.set_column(
                table.schema.get_field_index(key),
                field,
                pa.array(values, type=field.type),
            )
            stats[key] = get_feature_stats(values, axis=0, keepdims=True)
        parquet = target / dst.get_data_file_path(new_id)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        for key in src.video_keys:
            video = target / dst.get_video_file_path(new_id, key)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.symlink_to(
                os.path.relpath(
                    source_root / src.get_video_file_path(old_id, key), video.parent
                )
            )
        dst.save_episode(new_id, n, src.episodes[old_id]["tasks"], stats)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{repo_id}: {dst.total_episodes} episodes, {dst.total_frames} frames",
        flush=True,
    )


def prepare(store: Path) -> None:
    source = store / "datasets" / SOURCE_REPO
    conversion = json.loads((source / "meta/conversion.json").read_text())
    if (
        not conversion["complete"]
        or conversion["action_semantics"] != "absolute_joint_position"
    ):
        raise ValueError("Expected completed conversion with absolute joint targets")
    info = json.loads((source / "meta/info.json").read_text())
    if info["total_episodes"] != 57 or info["fps"] != 30:
        raise ValueError(
            "This preparation targets the confirmed 57-episode, 30 FPS dataset"
        )
    if len(conversion["episodes"]) != 57 or any(
        e["outcome"] != "success" for e in conversion["episodes"]
    ):
        raise ValueError("Expected 57 successful demonstrations")
    extra_sources = []
    if EXTRA_REPO:
        extra = store / "datasets" / EXTRA_REPO
        report = json.loads((extra / "meta/conversion.json").read_text())
        extra_info = json.loads((extra / "meta/info.json").read_text())
        if (
            not report["complete"]
            or report["action_semantics"] != "absolute_joint_position"
            or extra_info["total_episodes"] != 14
            or extra_info["fps"] != 30
            or len(report["episodes"]) != 14
            or any(e["outcome"] != "success" for e in report["episodes"])
        ):
            raise ValueError("Expected completed 14-success, 30 FPS 0928 conversion")
        extra_sources = [(extra, list(range(14)))]
    targets = [store / "datasets" / r for r in (TRAIN_REPO, VAL_REPO)]
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Split already exists: {target}; use check, do not overwrite"
            )
    train_ids, val_ids = select_episodes(57, 5, 42)
    for target, repo_id, ids in zip(
        targets, (TRAIN_REPO, VAL_REPO), (train_ids, val_ids), strict=True
    ):
        write_subset(
            source,
            target,
            repo_id,
            ids,
            extra_sources=extra_sources if repo_id == TRAIN_REPO else None,
        )
    print(f"Heldout SOURCE episode indices (zero-based): {val_ids}")


def make_config(store: Path):
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config, optimizer, weight_loaders

    return config.TrainConfig(
        name=CONFIG_NAME,
        # 30 target positions span one second at the dataset rate. The downstream
        # robot controller must respect that rate, not execute the chunk at 200 Hz.
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=30),
        data=config.LeRobotAlohaDataConfig(
            repo_id=TRAIN_REPO,
            adapt_to_pi=False,  # YAM is NOT an ALOHA: no sign or gripper remapping.
            use_delta_joint_actions=True,  # 12 joint deltas, 2 absolute grippers.
            base_config=config.DataConfig(prompt_from_task=True),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.main",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(store / "models/pi05_base/params")
        ),
        assets_base_dir=str(store / "assets"),
        checkpoint_base_dir=str(store / "checkpoints"),
        batch_size=8,
        num_workers=2,
        num_train_steps=2000,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100, peak_lr=2.5e-5, decay_steps=2000, decay_lr=2.5e-6
        ),
        log_interval=10,
        save_interval=500,
        # Upstream keeps the latest checkpoint; do not retain periodic extras.
        keep_period=None,
        wandb_enabled=False,
        overwrite=False,
    )


def require_splits(store: Path) -> None:
    expected = select_episodes(57, 5, 42)
    for repo_id, ids in zip((TRAIN_REPO, VAL_REPO), expected, strict=True):
        root = store / "datasets" / repo_id
        report = json.loads((root / "meta/split.json").read_text())
        if not report["complete"] or report["source_episode_indices"] != ids:
            raise ValueError(f"Incomplete or unexpected split: {repo_id}")
        extras = []
        if EXTRA_REPO and repo_id == TRAIN_REPO:
            extras = [
                {
                    "source_root_relative": os.path.relpath(
                        store / "datasets" / EXTRA_REPO, root
                    ),
                    "source_episode_indices": list(range(14)),
                }
            ]
        if report.get("extra_sources", []) != extras:
            raise ValueError(f"Unexpected supplemental sources: {repo_id}")
        info = json.loads((root / "meta/info.json").read_text())
        if info["total_episodes"] != len(ids) + (14 if extras else 0):
            raise ValueError(f"Unexpected episode count: {repo_id}")
        if any(not video.is_file() for video in (root / "videos").rglob("*.mp4")):
            raise ValueError(
                f"Broken video link: keep the original dataset alongside {repo_id}"
            )


def check(store: Path, normalized: bool) -> None:
    from openpi import transforms
    from openpi.training import data_loader

    config = make_config(store)
    train_data = config.data.create(config.assets_dirs, config.model)
    if normalized and train_data.norm_stats is None:
        raise ValueError("Run norm before check --normalized")
    for repo_id in (TRAIN_REPO, VAL_REPO):
        # Both splits use TRAINING norm stats; never fit norm on heldout data.
        data = dataclasses.replace(train_data, repo_id=repo_id)
        raw = data_loader.create_torch_dataset(
            data, config.model.action_horizon, config.model
        )
        if normalized:
            transformed = data_loader.transform_dataset(raw, data)
        else:
            transformed = data_loader.TransformedDataset(
                raw, [*data.repack_transforms.inputs, *data.data_transforms.inputs]
            )
        for index in (0, len(raw) // 2, len(raw) - 1):
            row = transformed[index]
            width = 32 if normalized else 14
            if row["state"].shape != (width,) or row["actions"].shape != (30, width):
                raise ValueError(f"Unexpected transformed dimensions: {repo_id}")
            if (
                not np.isfinite(row["state"]).all()
                or not np.isfinite(row["actions"]).all()
            ):
                raise ValueError("Nonfinite transformed state/actions")
            if not all(row["image_mask"].values()):
                raise ValueError("One or more camera views are missing")
            if not normalized and row["prompt"] != "choose_the_heavier_one":
                raise ValueError("Unexpected task instruction")
            if not normalized:
                # Verify inverses against the stored absolute targets on real data.
                back = transforms.compose(data.data_transforms.outputs)(
                    {
                        "state": row["state"].copy(),
                        "actions": row["actions"].copy(),
                    }
                )
                np.testing.assert_allclose(
                    back["actions"], raw[index]["action"], atol=1e-6
                )
        print(f"PASS {repo_id}: {len(raw)} frames, normalized={normalized}", flush=True)


def main() -> None:
    global TRAIN_REPO, VAL_REPO, CONFIG_NAME, EXTRA_REPO
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument(
        "--include-0928",
        action="store_true",
        help="Add 14 new demos to the original 52 training episodes; keep the original 5 heldout",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--normalized", action="store_true")
    commands.add_parser("norm")
    training = commands.add_parser("train")
    training.add_argument("--exp-name", required=True)
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument(
        "--decay-steps",
        type=int,
        default=2000,
        help="Learning-rate decay endpoint, independent of total training steps",
    )
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument(
        "--fsdp-devices",
        type=int,
        default=1,
        help="Number of devices for parameter/optimizer sharding (2 for dual GPU)",
    )
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suffix = "0928" if args.include_0928 else "0922"
    TRAIN_REPO = f"local/balance_{suffix}_train"
    VAL_REPO = f"local/balance_{suffix}_val"
    CONFIG_NAME = "pi05_yam_heavier_0928" if args.include_0928 else "pi05_yam_heavier"
    EXTRA_REPO = "local/balance_2_0928" if args.include_0928 else None
    store = args.store.expanduser().resolve()
    if not store.is_dir():
        parser.error("--store must be an existing storage directory")
    # Set before any LeRobot/OpenPI imports. Override inherited cache defaults.
    for name, path in {
        "HF_HOME": store / "cache/huggingface",
        "HF_LEROBOT_HOME": store / "datasets",
        "OPENPI_DATA_HOME": store / "cache/openpi",
    }.items():
        os.environ[name] = str(path)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    if args.command in ("prepare", "norm"):
        os.environ["JAX_PLATFORMS"] = "cpu"
    if args.command == "prepare":
        prepare(store)
        return
    checkout = store / "code/openpi"
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(
            f"OpenPI revision mismatch: expected {OPENPI_REVISION}, got {revision}"
        )
    sys.path.insert(0, str(checkout))
    require_splits(store)
    if args.command == "check":
        check(store, args.normalized)
        return
    config = make_config(store)
    if args.command == "norm":
        from openpi.training import config as registry
        from scripts import compute_norm_stats

        output = config.assets_dirs / TRAIN_REPO / "norm_stats.json"
        if output.exists():
            raise FileExistsError(
                f"Norm stats already exist: {output}; use check --normalized"
            )
        # Upstream drops incomplete batches. A batch size of one includes every
        # training frame and its future action chunk, without heldout leakage.
        registry._CONFIGS_DICT[CONFIG_NAME] = dataclasses.replace(config, batch_size=1)
        compute_norm_stats.main(CONFIG_NAME)
        print(f"Norm complete: {output}")
    else:
        if (
            args.steps <= 0
            or args.batch_size <= 0
            or args.fsdp_devices <= 0
            or Path(args.exp_name).name != args.exp_name
            or args.exp_name in (".", "..")
        ):
            parser.error(
                "Use positive steps/batch-size/fsdp-devices and a single-component experiment name"
            )
        if args.decay_steps <= config.lr_schedule.warmup_steps:
            parser.error("--decay-steps must exceed the 100 warmup steps")
        if not (config.assets_dirs / TRAIN_REPO / "norm_stats.json").is_file():
            raise FileNotFoundError("Run norm first")
        if not (store / "models/pi05_base/params").is_dir():
            raise FileNotFoundError("Missing local pi05_base/params")
        from scripts import train

        train.main(
            dataclasses.replace(
                config,
                exp_name=args.exp_name,
                num_train_steps=args.steps,
                lr_schedule=dataclasses.replace(
                    config.lr_schedule, decay_steps=args.decay_steps
                ),
                batch_size=args.batch_size,
                fsdp_devices=args.fsdp_devices,
                resume=args.resume,
            )
        )


if __name__ == "__main__":
    main()
