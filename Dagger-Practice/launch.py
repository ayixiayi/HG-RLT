"""Initialize an immutable, relocatable ABC-DiT run, then invoke its trainer.

Only use locally trusted ABC checkpoints: the upstream trainer unpickles them.
init does CPU validation, not model loading. train requires CUDA and local assets.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from abc_minimal.config import MixtureComponent, TrainConfig, validate_train_config
from abc_minimal.preprocess import load_norm_stats, parse_norm_stats
from prepare import digest

ABC_REVISION = "d0832d12651d1b260a652861a14648dc5f3660c7"


@dataclasses.dataclass
class DaggerConfig(TrainConfig):
    # Upstream saves asdict(config) in checkpoints: bind resume to this data run.
    dagger_contract_id: str = ""


def load_safe(path):
    # Native checkpoints contain NumPy norm arrays. Allow these specific data
    # types, never fall back to unrestricted pickle for preflight inspection.
    with torch.serialization.safe_globals(
        [
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.Float32DType,
            np.dtypes.Float64DType,
            np._core.multiarray.scalar,
        ]
    ):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def build(round_dir, parent, clip_dir, output, settings):
    import abc_minimal.config as upstream

    source = Path(upstream.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != ABC_REVISION or dirty:
        raise ValueError("official ABC source differs from pinned version")
    root, parent, clip_dir = (Path(p).resolve() for p in (round_dir, parent, clip_dir))
    manifest = json.loads((root / "round.json").read_text())
    if manifest["schema_version"] != 1 or manifest["chunk_length"] != 30:
        raise ValueError("unsupported round schema/action horizon")
    expected = [("previous", 0.8), ("intervention", 0.1), ("autonomous", 0.1)]
    if [(c["name"], c["weight"]) for c in manifest["mixture"]] != expected:
        raise ValueError("expected previous/intervention/autonomous 80:10:10")
    if digest(root / "norm_stats.json") != manifest["norm_sha256"]:
        raise ValueError("round normalization changed")
    inventory = {}
    for component, _ in expected:
        for split in ("train", "val"):
            directory = root / component / split
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("split must be an independent directory")
            episodes = sorted(directory.iterdir())
            if not episodes:
                raise ValueError("empty data pool")
            for episode in episodes:
                for filename in (
                    "states_actions.bin",
                    "combined_camera-images-rgb.mp4",
                    "episode_metadata.json",
                ):
                    path = episode / filename
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise ValueError(
                            "export must contain local files for portable identity"
                        )
                    inventory[str(path.relative_to(root))] = digest(path)
    parent_hash = digest(parent)
    checkpoint = load_safe(parent)
    if not isinstance(checkpoint, dict) or "norm_stats" not in checkpoint:
        raise ValueError("parent must embed normalization statistics")
    stats = load_norm_stats(root / "norm_stats.json")
    parent_stats = parse_norm_stats(checkpoint["norm_stats"])
    for key in ("state", "actions"):
        for statistic in ("mean", "std"):
            values = stats[key][statistic]
            if (
                values.shape != (14,)
                or not np.isfinite(values).all()
                or (statistic == "std" and (values < 0).any())
            ):
                raise ValueError(
                    "normalization requires 14 finite means and nonnegative stds"
                )
            if not np.array_equal(stats[key][statistic], parent_stats[key][statistic]):
                raise ValueError("parent and round normalization differ")
    del checkpoint
    config = DaggerConfig(
        cache_root=str(root),
        output_dir=str(Path(output).resolve()),
        load_pretrained=True,
        pretrained_ckpt_name=str(parent),
        train_steps=settings["steps"],
        batch_size=settings["batch_size"],
        seed=settings["seed"],
        ckpt_every=settings["save_interval"],
        num_workers=0,
        mixture=[
            MixtureComponent(f"{c}/train", f"{c}/val", w, "") for c, w in expected
        ],
    )
    config.clip.cache_dir = str(clip_dir)
    assets = {
        name: digest(clip_dir / name)
        for name in (config.clip.model_name, config.clip.bpe_name)
    }
    validate_train_config(config, root, parent)
    effective = dataclasses.asdict(config)
    # Replace machine-dependent paths only; all training hyperparameters remain.
    effective.update(
        cache_root="<round>", output_dir="<run>", pretrained_ckpt_name="<parent>"
    )
    effective["clip"]["cache_dir"] = "<clip>"
    contract = {
        "abc_revision": revision,
        "settings": settings,
        "effective_config": effective,
        "round_sha256": digest(root / "round.json"),
        "data_sha256": inventory,
        "parent_sha256": parent_hash,
        "clip_sha256": assets,
    }
    config.dagger_contract_id = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    effective["dagger_contract_id"] = config.dagger_contract_id
    return config, json.loads(json.dumps(contract))


def initialize(round_dir, parent, clip_dir, output, settings):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if min(settings.values()) < 1:
        raise ValueError("settings must be positive")
    _, contract = build(round_dir, parent, clip_dir, output, settings)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def validate_run(run, round_dir, parent, clip_dir, resume=None):
    run = Path(run).resolve()
    saved = json.loads(run.read_text())
    config, current = build(round_dir, parent, clip_dir, run.parent, saved["settings"])
    if current != saved:
        raise ValueError("run contract changed: data, parent, assets or configuration")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != saved["settings"]["world_size"]:
        raise ValueError("WORLD_SIZE differs from initialized run")
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
        raise ValueError("this launcher supports single-host training only")
    if resume is None:
        if any(run.parent.glob("*.pt")):
            raise ValueError("existing checkpoints require explicit resume")
    else:
        resume = Path(resume).resolve()
        if resume.parent != run.parent:
            raise ValueError("resume checkpoint must be inside this run directory")
        ckpt = load_safe(resume)
        required = {
            "model",
            "optimizer",
            "scheduler",
            "global_step",
            "train_config",
            "batch_size",
            "data_world",
            "norm_stats",
        }
        if not required.issubset(ckpt):
            raise ValueError("resume requires a full native training checkpoint")
        if ckpt["batch_size"] != config.batch_size or ckpt["data_world"] != world:
            raise ValueError("resume topology differs")
        if not 0 < ckpt["global_step"] < config.train_steps:
            raise ValueError("resume step must be inside this run's training budget")
        old = ckpt["train_config"]
        expected = saved["effective_config"]
        for key in ("state", "actions"):
            resumed_stats = parse_norm_stats(ckpt["norm_stats"])[key]
            original_stats = load_norm_stats(Path(round_dir) / "norm_stats.json")[key]
            if any(
                not np.array_equal(resumed_stats[s], original_stats[s])
                for s in ("mean", "std")
            ):
                raise ValueError("resume normalization differs")
        old_clip = dict(old["clip"])
        old_clip["cache_dir"] = "<clip>"
        if old_clip != expected["clip"]:
            raise ValueError("resume CLIP config differs")
        for key in expected:
            if key in (
                "cache_root",
                "output_dir",
                "pretrained_ckpt_name",
                "resume_from",
                "load_pretrained",
                "clip",
            ):
                continue
            # torch checkpoint retains tuples; JSON stores lists.
            if json.loads(json.dumps(old.get(key))) != expected[key]:
                raise ValueError(f"resume config differs: {key}")
        config.load_pretrained = False
        config.resume_from = str(resume)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check", "train"))
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run directory (init) or run.json (check/train)",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="per GPU; smoke default, not paper batch size",
    )
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    defaults = {
        "steps": 75000,
        "batch_size": 1,
        "world_size": 1,
        "save_interval": 500,
        "seed": 123,
    }
    if args.command == "init":
        if args.resume_from:
            parser.error("init starts a new round; resume belongs to check/train")
        settings = {
            key: default if getattr(args, key) is None else getattr(args, key)
            for key, default in defaults.items()
        }
        initialize(args.round, args.parent, args.clip_cache, args.run, settings)
        print(f"Initialized (not trained): {args.run / 'run.json'}")
        return
    if any(getattr(args, key) is not None for key in defaults):
        parser.error("check/train use saved run settings; overrides require a new run")
    config = validate_run(
        args.run, args.round, args.parent, args.clip_cache, args.resume_from
    )
    if args.command == "check":
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("training requires available CUDA; CPU preflight is check")
    from abc_minimal.train_loop import main as train

    train(config)


if __name__ == "__main__":
    main()
