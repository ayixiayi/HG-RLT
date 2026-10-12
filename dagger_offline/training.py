"""LeRobot pure-window mixture for the pinned OpenPI JAX training loop.

No robot, network service, or model initialization occurs on import.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

COMPONENTS = ("previous", "intervention", "autonomous")
WEIGHTS = (0.8, 0.1, 0.1)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_groups(entries):
    groups = {}
    source_splits = {}
    for entry in entries:
        if entry["component"] not in COMPONENTS or entry["split"] not in (
            "train",
            "val",
        ):
            raise ValueError("Unknown dataset component or split")
        for ep in entry["episodes"]:
            group = ep["group"]
            if not group or groups.get(group, entry["split"]) != entry["split"]:
                raise ValueError("Source group crosses train/val")
            groups[group] = entry["split"]
            source_hash = ep.get("source_sha256")
            if source_hash and (
                source_splits.setdefault(source_hash, entry["split"]) != entry["split"]
            ):
                raise ValueError("Source hash crosses train/val across rounds")


class WindowDataset:
    """Expose only full action windows within a single labelled episode."""

    def __init__(self, manifest_path, split):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if not m.get("complete") or split not in ("train", "val"):
            raise ValueError("Incomplete round or invalid split")
        self.horizon = m["horizon"]
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("Invalid action horizon")
        if digest(self.manifest_path.parent / "norm_stats.json") != m["norm_sha256"]:
            raise ValueError("Parent normalization digest mismatch")
        validate_groups(m["datasets"])
        self.entries = sorted(
            (e for e in m["datasets"] if e["split"] == split),
            key=lambda e: COMPONENTS.index(e["component"]),
        )
        self.datasets, self.segments, self.ends = [], [], []
        self.component_ranges = []
        count = 0
        for component in COMPONENTS:
            begin = count
            for entry in (e for e in self.entries if e["component"] == component):
                root = (self.manifest_path.parent / entry["path"]).resolve()
                info_path = root / "meta/info.json"
                if not info_path.is_file():
                    raise FileNotFoundError(info_path)
                info = json.loads(info_path.read_text())
                if info["codebase_version"] != "v2.1" or info["fps"] != 30:
                    raise ValueError("Expected LeRobot v2.1 at 30 FPS")
                ds = LeRobotDataset(
                    entry["repo_id"],
                    root=root,
                    video_backend="pyav",
                    delta_timestamps={"action": [i / 30 for i in range(self.horizon)]},
                )
                if len(entry["episodes"]) != ds.num_episodes:
                    raise ValueError("Episode manifest does not match dataset")
                dataset_id = len(self.datasets)
                self.datasets.append(ds)
                for ep_id, ep in enumerate(entry["episodes"]):
                    frames = ds.meta.episodes[ep_id]["length"]
                    if ep["episode_index"] != ep_id or ep["frames"] != frames:
                        raise ValueError("Episode length/index mismatch")
                    windows = frames - self.horizon + 1
                    if windows <= 0:
                        continue
                    start = int(ds.episode_data_index["from"][ep_id])
                    self.segments.append((dataset_id, start, count))
                    count += windows
                    self.ends.append(count)
            if begin == count:
                raise ValueError(f"No full windows for {component}/{split}")
            self.component_ranges.append((begin, count))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment = int(np.searchsorted(self.ends, index, side="right"))
        dataset_id, start, base = self.segments[segment]
        ds = self.datasets[dataset_id]
        row = ds[start + index - base]
        if bool(row["action_is_pad"].any()):
            raise ValueError("Padded action escaped the pure-window index")
        row["prompt"] = ds.meta.tasks[int(row["task_index"])]
        return row


class MixtureSampler:
    """Stateless sample-index RNG; resume does not replay earlier batches."""

    def __init__(self, dataset, *, seed, start, stop):
        if seed < 0 or not 0 <= start < stop:
            raise ValueError("Invalid sampling seed/range")
        self.ranges = dataset.component_ranges
        self.seed, self.start, self.stop = seed, start, stop

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        for index in range(self.start, self.stop):
            rng = np.random.default_rng((self.seed, index))
            component = int(rng.choice(3, p=WEIGHTS))
            begin, end = self.ranges[component]
            yield int(rng.integers(begin, end))


def record_run(path, contract, *, resume):
    """Keep input identity/hyperparameters fixed across optimizer resume."""
    path = Path(path)
    if resume:
        if json.loads(path.read_text()) != contract:
            raise ValueError("Resume contract differs; use a new experiment")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(contract, stream, indent=2)


def build_config(
    manifest_path,
    store,
    parent_params,
    *,
    name,
    exp_name,
    steps=2000,
    batch_size=8,
    seed=42,
    save_interval=500,
    fsdp_devices=1,
    resume=False,
    rl_token=None,
):
    """rl_token: None, or {"alpha", "width", "depth", "heads", "stride"} to add
    the RLT Stage 1 RL token (alpha 0 trains only the token, VLA frozen)."""
    import dataclasses

    from openpi.training.config import AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    import yam_openpi

    for value in (name, exp_name):
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(
                "Config and experiment names must be single path components"
            )
    if min(steps, batch_size, save_interval, fsdp_devices) < 1 or seed < 0:
        raise ValueError("Invalid training settings")
    root = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    norm = json.loads((root / "norm_stats.json").read_text())["norm_stats"]
    # Pi0.5 uses quantiles of delta-joint actions, NOT ABC absolute-action stats.
    for key in ("state", "actions"):
        for stat in ("mean", "std", "q01", "q99"):
            values = np.asarray(norm[key].get(stat))
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError("Pi0.5 requires finite 14-D parent mean/std/q01/q99")
        if np.any(np.asarray(norm[key]["q99"]) < norm[key]["q01"]):
            raise ValueError("Invalid parent quantiles")
    base = yam_openpi.make_config(Path(store))
    model = dataclasses.replace(base.model, action_horizon=manifest["horizon"])
    loader = CheckpointWeightLoader(str(parent_params))
    freeze = base.freeze_filter
    if rl_token is not None:
        from dagger_offline.rl_token import RLTCheckpointWeightLoader, RLTPi0Config

        fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
        model = RLTPi0Config(
            **fields,
            rl_alpha=float(rl_token["alpha"]),
            rl_width=int(rl_token.get("width", 512)),
            rl_depth=int(rl_token.get("depth", 2)),
            rl_heads=int(rl_token.get("heads", 8)),
            rl_stride=int(rl_token.get("stride", 1)),
        )
        loader = RLTCheckpointWeightLoader(str(parent_params))
        freeze = model.get_freeze_filter()
    return dataclasses.replace(
        base,
        name=name,
        exp_name=exp_name,
        model=model,
        freeze_filter=freeze,
        data=dataclasses.replace(
            base.data,
            repo_id="local/dagger",
            assets=AssetsConfig(assets_dir=str(root.parent), asset_id=root.name),
        ),
        weight_loader=loader,
        num_train_steps=steps,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
        save_interval=save_interval,
        fsdp_devices=fsdp_devices,
        resume=resume,
    )


def create_loader(
    config, manifest_path, *, start_step=0, sharding=None, num_batches=None
):
    """Use upstream transforms, batching, Observation and checkpoint assets."""
    import jax
    from openpi.training import data_loader

    if jax.process_count() != 1:
        raise ValueError("Only single-host JAX training is supported")
    raw = WindowDataset(manifest_path, "train")
    if raw.horizon != config.model.action_horizon:
        raise ValueError("Model and manifest action horizons differ")
    data_config = config.data.create(config.assets_dirs, config.model)
    transformed = data_loader.transform_dataset(raw, data_config)
    # train.py fetches one extra batch before its final checkpoint save.
    batches = (
        config.num_train_steps - start_step + 1 if num_batches is None else num_batches
    )
    sampler = MixtureSampler(
        raw,
        seed=config.seed,
        start=start_step * config.batch_size,
        stop=(start_step + batches) * config.batch_size,
    )
    torch_loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size,
        sampler=sampler,
        sharding=sharding,
        num_batches=batches,
        num_workers=0,
        seed=config.seed,
    )
    return data_loader.DataLoaderImpl(data_config, torch_loader)


def resume_step(checkpoint_dir):
    """Pinned train.py saves directory step N after updating train_state to N+1."""
    import orbax.checkpoint as ocp

    if not Path(checkpoint_dir).is_dir():
        raise ValueError("No committed checkpoint to resume")
    with ocp.CheckpointManager(
        checkpoint_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as manager:
        step = manager.latest_step()
    # Upstream explicitly refuses optimizer resume from step zero.
    if step is None or step <= 0:
        raise ValueError("No resumable checkpoint (step must exceed zero)")
    return step + 1


def attach_previous(manifest_path, output, previous_train, previous_val):
    """Attach existing yam_openpi.py BC splits without rewriting their videos.

    Output must stay beside the bridge manifest so dataset/stat paths remain valid.
    Source directory name + original episode ID preserves split provenance across
    machine moves; do not rename original source directories to hide overlaps.
    """
    import os

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.parent != manifest_path.parent:
        raise ValueError("Write the combined manifest beside the bridge manifest")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or not manifest.get("current_only"):
        raise ValueError("Expected a complete current_only round")
    entries = list(manifest["datasets"])
    for split, root in (("train", previous_train), ("val", previous_val)):
        root = Path(root).resolve()
        provenance = json.loads((root / "meta/split.json").read_text())
        if not provenance.get("complete"):
            raise ValueError("Incomplete BC split")
        groups = []
        for source in (provenance, *provenance.get("extra_sources", [])):
            source_name = (root / source["source_root_relative"]).resolve().name
            groups.extend(
                f"bc:{source_name}:{i}" for i in source["source_episode_indices"]
            )
        meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
        if len(groups) != meta.total_episodes or len(set(groups)) != len(groups):
            raise ValueError("BC provenance count/duplicate mismatch")
        entries.append(
            {
                "path": os.path.relpath(root, output.parent),
                "repo_id": meta.repo_id,
                "component": "previous",
                "split": split,
                "episodes": [
                    {
                        "episode_index": i,
                        "frames": meta.episodes[i]["length"],
                        "group": group,
                    }
                    for i, group in enumerate(groups)
                ],
            }
        )
    validate_groups(entries)
    manifest.update(datasets=entries, current_only=False)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return output


def attach_previous_round(manifest_path, output, previous_manifest):
    """Aggregate every previous component as the next round's 80% pool."""
    import os

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    previous_manifest = Path(previous_manifest).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.parent != manifest_path.parent:
        raise ValueError("Write the combined manifest beside the bridge manifest")
    current = json.loads(manifest_path.read_text())
    previous = json.loads(previous_manifest.read_text())
    if not (
        current.get("complete")
        and current.get("current_only")
        and previous.get("complete")
        and not previous.get("current_only")
    ):
        raise ValueError(
            "Expected current_only new round and complete previous mixture"
        )
    if (
        current["horizon"] != previous["horizon"]
        or current["norm_sha256"] != previous["norm_sha256"]
    ):
        raise ValueError("Round horizon/normalization mismatch")
    for split in ("train", "val"):
        WindowDataset(previous_manifest, split)
    entries = list(current["datasets"])
    for entry in previous["datasets"]:
        entries.append(
            {
                **entry,
                "component": "previous",
                "path": os.path.relpath(
                    previous_manifest.parent / entry["path"], output.parent
                ),
            }
        )
    roots = [(output.parent / e["path"]).resolve() for e in entries]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate dataset across rounds")
    validate_groups(entries)
    current.update(
        datasets=entries,
        current_only=False,
        previous_manifest_sha256=digest(previous_manifest),
    )
    with output.open("x") as stream:
        json.dump(current, stream, indent=2)
    return output


def check_data(config, manifest_path):
    """Read segment endpoints through actual OpenPI transforms, no model weights."""
    from openpi.training import data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    result = {}
    for split in ("train", "val"):
        raw = WindowDataset(manifest_path, split)
        dataset = data_loader.transform_dataset(raw, data_config)
        for _, _, start in raw.segments:
            end = raw.ends[int(np.searchsorted(raw.ends, start, side="right"))] - 1
            for index in {start, end}:
                row = dataset[index]
                if row["actions"].shape != (raw.horizon, 32) or row["state"].shape != (
                    32,
                ):
                    raise ValueError("Unexpected Pi0.5 input dimensions")
                if (
                    not np.isfinite(row["actions"]).all()
                    or not np.isfinite(row["state"]).all()
                ):
                    raise ValueError("Nonfinite transformed sample")
                if not all(row["image_mask"].values()):
                    raise ValueError("Missing camera")
        result[split] = {
            c: end - begin
            for c, (begin, end) in zip(COMPONENTS, raw.component_ranges, strict=True)
        }
    return result


def validate_parent(parent_params, norm_stats):
    """Check the same parent/statistics contract before checking or training."""
    from dagger_offline.deployment import normalization_id

    parent = Path(parent_params).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    norm_id = normalization_id(norm_stats)
    # OpenPI rewrites JSON formatting when saving checkpoint assets.
    if not any(
        normalization_id(path) == norm_id
        for path in (parent.parent / "assets").glob("**/norm_stats.json")
    ):
        raise ValueError("Round normalization does not match parent checkpoint assets")
    return norm_id


def rl_token_contract(model):
    if not hasattr(model, "rl_alpha"):
        return None
    return {
        "alpha": model.rl_alpha,
        "width": model.rl_width,
        "depth": model.rl_depth,
        "heads": model.rl_heads,
        "stride": model.rl_stride,
    }


def check_training(config, manifest_path):
    """CPU readiness check, including the parent; never initialize model weights."""
    validate_parent(
        config.weight_loader.params_path,
        Path(manifest_path).resolve().parent / "norm_stats.json",
    )
    return check_data(config, manifest_path)


def rl_token_shapes(config):
    """Abstract (no-allocation) RL token parameter count and token width."""
    import jax
    from flax import nnx

    if rl_token_contract(config.model) is None:
        return None
    abstract = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
    state = nnx.state(abstract.rl_token, nnx.Param)
    count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(state))
    width = abstract.PaliGemma.llm.module.configs[0].width
    return {"params": count, "token_width": width}


def run_training(config, manifest_path):
    """Scoped injection into the pinned upstream trainer; never patch its files.

    New round = parent params + fresh optimizer. --resume = same run optimizer,
    schedule and sample index. Keep total steps fixed for exact resume semantics.
    """
    from openpi.training import data_loader
    from scripts import train

    import yam_openpi

    manifest_path = Path(manifest_path).resolve()
    parent = Path(config.weight_loader.params_path).resolve()
    norm_hash = digest(manifest_path.parent / "norm_stats.json")
    norm_id = validate_parent(parent, manifest_path.parent / "norm_stats.json")
    check_data(config, manifest_path)
    start = resume_step(config.checkpoint_dir) if config.resume else 0
    if start >= config.num_train_steps:
        raise ValueError("Checkpoint has already reached requested total steps")
    if not config.resume and config.checkpoint_dir.exists():
        raise FileExistsError(config.checkpoint_dir)
    # Content-address metadata, parquet and videos, including BC video symlinks.
    # This costs sequential I/O once on launch, not GPU computation.
    payload = json.loads(manifest_path.read_text())
    files = []
    for entry in payload["datasets"]:
        root = manifest_path.parent / entry["path"]
        for folder in ("meta", "data", "videos"):
            files.extend(
                (
                    str(p.relative_to(manifest_path.parent))
                    if p.is_relative_to(manifest_path.parent)
                    else str(p),
                    digest(p),
                )
                for p in sorted((root / folder).rglob("*"))
                if p.is_file()
            )
    contract = {
        "manifest_sha256": digest(manifest_path),
        "norm_sha256": norm_hash,
        "dataset_files": [list(item) for item in files],
        "parent_params": str(parent),
        "weights": list(WEIGHTS),
        "horizon": config.model.action_horizon,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "fsdp_devices": config.fsdp_devices,
        "lr_schedule": repr(config.lr_schedule),
        "optimizer": repr(config.optimizer),
        "model": repr(config.model),
        "data": repr(config.data),
        "ema_decay": config.ema_decay,
        "freeze_filter": repr(config.freeze_filter),
        "inference": {
            "config_name": config.name,
            "exp_name": config.exp_name,
            "model": "pi05" if config.model.pi05 else "pi0",
            "action_dim": config.model.action_dim,
            "action_horizon": config.model.action_horizon,
            "adapt_to_pi": config.data.adapt_to_pi,
            "delta_joint_actions": config.data.use_delta_joint_actions,
            "asset_id": config.data.assets.asset_id,
            "norm_id": norm_id,
            "openpi_revision": yam_openpi.OPENPI_REVISION,
            "rl_token": rl_token_contract(config.model),
        },
    }
    record_run(
        Path(str(config.checkpoint_dir) + ".dagger.json"),
        contract,
        resume=config.resume,
    )
    original = data_loader.create_data_loader

    def factory(cfg, *, sharding=None, shuffle=False, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported trainer loader options: {kwargs}")
        return create_loader(cfg, manifest_path, start_step=start, sharding=sharding)

    data_loader.create_data_loader = factory
    try:
        train.main(config)
    finally:
        data_loader.create_data_loader = original


def main(argv=None):
    import argparse
    import os
    import subprocess
    import sys

    import yam_openpi

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attach = commands.add_parser("attach-bc")
    attach.add_argument("--manifest", required=True, type=Path)
    attach.add_argument("--output", required=True, type=Path)
    attach.add_argument("--previous-train", required=True, type=Path)
    attach.add_argument("--previous-val", required=True, type=Path)
    promote = commands.add_parser("attach-round")
    promote.add_argument("--manifest", required=True, type=Path)
    promote.add_argument("--output", required=True, type=Path)
    promote.add_argument("--previous-manifest", required=True, type=Path)
    for command in ("check", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", required=True, type=Path)
        sub.add_argument("--store", required=True, type=Path)
        sub.add_argument("--openpi", required=True, type=Path)
        sub.add_argument("--parent-params", required=True, type=Path)
        sub.add_argument("--name", required=True)
        sub.add_argument("--exp-name", required=True)
        sub.add_argument("--steps", type=int, default=2000)
        sub.add_argument("--batch-size", type=int, default=8)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--save-interval", type=int, default=500)
        sub.add_argument("--fsdp-devices", type=int, default=1)
        sub.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "attach-bc":
        print(
            attach_previous(
                args.manifest, args.output, args.previous_train, args.previous_val
            )
        )
        return
    if args.command == "attach-round":
        print(attach_previous_round(args.manifest, args.output, args.previous_manifest))
        return
    checkout = args.openpi.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != yam_openpi.OPENPI_REVISION:
        raise ValueError("OpenPI revision mismatch")
    sys.path[:0] = [
        str(checkout),
        str(checkout / "src"),
        str(checkout / "packages/openpi-client/src"),
    ]
    if args.command == "check":
        os.environ["JAX_PLATFORMS"] = "cpu"
    config = build_config(
        args.manifest,
        args.store.resolve(),
        args.parent_params.resolve(),
        name=args.name,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        save_interval=args.save_interval,
        fsdp_devices=args.fsdp_devices,
        resume=args.resume,
    )
    if args.command == "check":
        print(json.dumps(check_training(config, args.manifest), indent=2))
    else:
        run_training(config, args.manifest)


if __name__ == "__main__":
    main()
