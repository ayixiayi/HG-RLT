"""HG-DAgger 数据准备、可迁移打包和显式 Pi0.5 训练；不连接机器人。"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from dagger_offline.console import message

ROOT = Path(__file__).resolve().parents[1]


def round_manifest(path):
    path = Path(path).expanduser().resolve()
    return path if path.is_file() else path / "lerobot/combined.json"


def parent_checkpoint(args):
    """Use the round's selected parent; only initial preparation defaults to BC."""
    if args.parent_checkpoint is not None:
        return args.parent_checkpoint.expanduser().resolve()
    if getattr(args, "previous_round", None) is not None:
        raise ValueError(
            "后续轮次必须用 --parent-checkpoint 明确指定上轮权重，不能默认退回 baseline"
        )
    if hasattr(args, "round"):
        manifest = round_manifest(args.round)
        selected = json.loads(manifest.read_text()).get("parent_checkpoint")
        if not isinstance(selected, str) or not selected:
            raise ValueError(
                "清单未指定父模型或训练包未包含父模型，请使用 --parent-checkpoint"
            )
        return (manifest.parent / selected).resolve()
    if (ROOT / "parent/params").is_dir():
        return ROOT / "parent"
    config_path = args.rhos_root / "configs/deployments/pi05_yam_heavier.yaml"
    if not config_path.is_file():
        raise ValueError(
            "请用 --parent-checkpoint 指定现有 baseline 的 checkpoint 目录"
            "（包含 params/ 和 assets/），不是重新使用 pi05_base"
        )
    import yaml

    config = yaml.safe_load(config_path.read_text())
    return (args.rhos_root / config["checkpoint"]).resolve()


def use_openpi(checkout):
    import yam_openpi

    checkout = checkout.expanduser().resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != yam_openpi.OPENPI_REVISION or dirty:
        raise ValueError("OpenPI 必须是未修改的项目固定版本")
    sys.path[:0] = [
        str(checkout),
        str(checkout / "src"),
        str(checkout / "packages/openpi-client/src"),
    ]


def read_records(args):
    if args.records is not None:
        if args.train or args.val:
            raise ValueError("--records 不能与 --train / --val 混用")
        source = args.records.expanduser().resolve()
        records = json.loads(source.read_text())
        if not isinstance(records, list) or any(
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"].strip()
            for record in records
        ):
            raise ValueError("录制清单必须是对象数组，每项提供非空 path 字符串")
        return [
            {**record, "path": str((source.parent / record["path"]).resolve())}
            for record in records
        ]
    if not args.train or not args.val:
        raise ValueError(
            "必须分别提供独立采集的 --train GROUP=FILE 和 --val GROUP=FILE"
        )
    records = []
    for split in ("train", "val"):
        for item in getattr(args, split):
            group, separator, path = item.partition("=")
            if not separator or not group.strip() or not path:
                raise ValueError(
                    f"--{split} 使用 GROUP=FILE 格式；同一来源组不能跨 split"
                )
            records.append(
                {
                    "path": str(Path(path).expanduser().resolve()),
                    "group": group.strip(),
                    "split": split,
                    "role": "current",
                }
            )
    return records


def assign_splits(records, *, val_fraction=0.2, seed=42):
    """Keep whole source groups together; explicit legacy splits stay unchanged."""
    if not records:
        raise ValueError("录制清单为空；请先保存有效的成功或失败记录")
    for record in records:
        if (
            not isinstance(record.get("group"), str)
            or not record["group"].strip()
            or record.get("role") != "current"
        ):
            raise ValueError("每条本轮记录需要非空 group 和 role=current")
    explicit = ["split" in record for record in records]
    if any(explicit):
        if not all(explicit) or any(
            record["split"] not in ("train", "val") for record in records
        ):
            raise ValueError("split 应全部省略以自动划分，或每项明确填写 train/val")
        return [dict(record) for record in records]
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("--val-fraction 必须大于 0 且小于 1")
    groups = sorted({record["group"] for record in records})
    if len(groups) < 2:
        raise ValueError("自动划分至少需要两个独立 episode / 来源组，请继续采集")
    from yam_openpi import select_episodes

    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * val_fraction)))
    _, indices = select_episodes(len(groups), count, seed)
    validation = {groups[index] for index in indices}
    return [
        {**record, "split": "val" if record["group"] in validation else "train"}
        for record in records
    ]


def prepare_data(args, parent):
    from dagger_offline.lerobot_bridge import convert_round
    from dagger_offline.round_data import prepare_round
    from dagger_offline.training import (
        attach_previous,
        attach_previous_round,
        validate_parent,
    )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"不覆盖已有轮次：{output}")
    if args.previous_round and (args.previous_train or args.previous_val):
        raise ValueError(
            "--previous-round 不能与 --previous-train / --previous-val 混用"
        )
    if bool(args.previous_train) != bool(args.previous_val):
        raise ValueError("--previous-train / --previous-val 必须同时提供")
    previous = round_manifest(args.previous_round) if args.previous_round else None
    if args.norm_stats is not None:
        stats = args.norm_stats.expanduser().resolve()
    elif previous is not None:
        # Preserve original bytes across rounds, not checkpoint-reformatted JSON.
        stats = previous.parent / "norm_stats.json"
    else:
        candidates = sorted((parent / "assets").glob("**/norm_stats.json"))
        if len(candidates) != 1:
            raise ValueError("父模型统计量不唯一或缺失，请指定 --norm-stats")
        stats = candidates[0]
    validate_parent(parent / "params", stats)
    records = read_records(args)
    automatic = all("split" not in record for record in records)
    records = assign_splits(
        records, val_fraction=args.val_fraction, seed=args.split_seed
    )
    split_config = {"method": "explicit"}
    if automatic:
        split_config = {
            "method": "group_random",
            "val_fraction": args.val_fraction,
            "seed": args.split_seed,
        }
        message(
            "自动划分",
            f"按完整来源组随机划分，val={args.val_fraction:g}、seed={args.split_seed}；"
            + " · ".join(
                f"{split}={sum(record['split'] == split for record in records)} 条"
                for split in ("train", "val")
            ),
        )
    if previous is not None:
        from dagger_offline.training import WindowDataset

        for split in ("train", "val"):
            WindowDataset(previous, split)
    else:
        import yam_openpi

        train = args.previous_train or args.baseline_data / yam_openpi.TRAIN_REPO
        val = args.previous_val or args.baseline_data / yam_openpi.VAL_REPO
        for directory in (train, val):
            if not (directory / "meta/split.json").is_file():
                raise FileNotFoundError(f"缺少旧 BC 划分及来源信息：{directory}")
    message("1/3 · 片段", "筛选有效接管/策略片段；固定 30 帧窗口，不跨 HOLD 或时间断点")
    prepare_round(records, output / "abc", stats, current_only=True)
    pre = json.loads((output / "abc/round.json").read_text())["pre_intervention"]
    message(
        "接管前窗口",
        f"每次接管前 {pre['frames']} 帧不进策略池；本轮去掉 {pre['excluded_policy_frames']} 帧",
    )
    message("2/3 · 格式", "转为 LeRobot v2.1；保留三路图像、14 维状态与已确认动作")
    convert_round(output / "abc/round.json", output / "lerobot")
    manifest = output / "lerobot/combined.json"
    message("3/3 · 聚合", "合并旧数据；保留独立 train/val，按 80:10:10 采样")
    if previous is not None:
        attach_previous_round(output / "lerobot/lerobot_round.json", manifest, previous)
    else:
        attach_previous(output / "lerobot/lerobot_round.json", manifest, train, val)
    # Otherwise a later pack-training without flags silently falls back to BC,
    # even when this round was deliberately prepared from an improved policy.
    payload = json.loads(manifest.read_text())
    payload["parent_checkpoint"] = os.path.relpath(parent, manifest.parent)
    payload["split_config"] = split_config
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def training_config(args, manifest, parent):
    from dagger_offline.training import build_config

    return build_config(
        manifest,
        args.store.expanduser().resolve(),
        parent / "params",
        name=args.name,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        save_interval=args.save_interval,
        fsdp_devices=args.fsdp_devices,
        resume=args.resume,
        decay_steps=args.decay_steps,
        keep_period=args.keep_period,
        rl_token=(
            {
                "alpha": args.rl_alpha,
                "width": args.rl_width,
                "stride": args.rl_stride,
            }
            if getattr(args, "rl_token", False)
            else None
        ),
    )


def check(config, manifest):
    import numpy as np

    from dagger_offline.training import check_training, create_loader, rl_token_shapes

    counts = check_training(config, manifest)
    shapes = rl_token_shapes(config)
    if shapes is not None:
        message(
            "RL token",
            f"alpha={config.model.rl_alpha} · 参数 {shapes['params']:,} · "
            f"token 宽度 {shapes['token_width']}"
            + (" · VLA 冻结" if config.model.rl_alpha == 0 else ""),
        )
    for split, components in counts.items():
        message(
            "窗口", f"{split}: " + " · ".join(f"{k}={v}" for k, v in components.items())
        )
    observation, actions = next(iter(create_loader(config, manifest, num_batches=1)))
    if not np.isfinite(actions).all() or not np.isfinite(observation.state).all():
        raise ValueError("训练 batch 含非有限数值")
    message(
        "CPU 检查通过",
        f"动作 batch={tuple(actions.shape)}；未加载模型、未训练、未连接硬件",
        level="success",
    )
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser(
        "pack-model", help="学校端：封存并打包返程模型，不含优化器状态"
    )
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--model-round", type=int, required=True)
    model.add_argument("--port", type=int, default=8769)
    for command in ("prepare-data", "pack-training", "train-check", "train"):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--parent-checkpoint", type=Path, help="父模型目录，包含 params/ 和 assets/"
        )
        sub.add_argument("--rhos-root", type=Path, default=ROOT / "vendor/rhos")
        sub.add_argument(
            "--openpi",
            type=Path,
            default=Path(os.environ.get("HG_OPENPI", ROOT / "vendor/openpi")),
        )
        if command == "prepare-data":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument(
                "--records", type=Path, help="录制清单；省略 split 时按来源组自动划分"
            )
            sub.add_argument("--val-fraction", type=float, default=0.2)
            sub.add_argument("--split-seed", type=int, default=42)
            sub.add_argument(
                "--train", action="append", default=[], metavar="GROUP=FILE"
            )
            sub.add_argument("--val", action="append", default=[], metavar="GROUP=FILE")
            sub.add_argument("--norm-stats", type=Path)
            sub.add_argument("--previous-round", type=Path)
            sub.add_argument("--previous-train", type=Path)
            sub.add_argument("--previous-val", type=Path)
            sub.add_argument(
                "--baseline-data",
                type=Path,
                default=ROOT / "data/baseline",
            )
        else:
            sub.add_argument(
                "--round",
                type=Path,
                default=ROOT / "round",
                help="轮次目录或 combined.json",
            )
        if command == "pack-training":
            sub.add_argument(
                "--output", type=Path, required=True, help=".tar 或 .tar.gz；不覆盖"
            )
            sub.add_argument(
                "--without-parent",
                action="store_true",
                help="仅当云端已有同一父模型时使用",
            )
        else:
            sub.add_argument("--store", type=Path, default=ROOT / "training")
            sub.add_argument("--name", default="pi05_yam_hgdagger")
            sub.add_argument("--exp-name", default="first")
            sub.add_argument("--steps", type=int, default=2000)
            sub.add_argument("--batch-size", type=int, default=8)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--save-interval", type=int, default=500)
            sub.add_argument("--fsdp-devices", type=int, default=1)
            sub.add_argument(
                "--decay-steps",
                type=int,
                help="学习率余弦衰减的步数；默认等于 --steps",
            )
            sub.add_argument(
                "--keep-period",
                type=int,
                help="另外保留步数为该值整数倍的 checkpoint；默认只留最新",
            )
            sub.add_argument("--resume", action="store_true")
            sub.add_argument(
                "--rl-token",
                action="store_true",
                help="RLT 第一阶段：同时训练 RL token（L_ro + alpha·L_vla）",
            )
            sub.add_argument(
                "--rl-alpha",
                type=float,
                default=1.0,
                help="VLA 损失权重；0 表示只训 RL token、VLA 冻结",
            )
            sub.add_argument("--rl-width", type=int, default=512)
            sub.add_argument("--rl-stride", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    if args.command == "pack-model":
        from dagger_offline.training_package import pack_model

        result = pack_model(
            args.checkpoint, args.output, model_round=args.model_round, port=args.port
        )
        message(
            "返程模型包",
            f"{result}；下一步 train.sh upload-model --archive 包路径",
            level="success",
        )
        return result
    parent = parent_checkpoint(args)
    message("父模型", str(parent))
    if args.command == "pack-training":
        from dagger_offline.training_package import pack_training

        result = pack_training(
            round_manifest(args.round),
            parent,
            args.output,
            include_parent=not args.without_parent,
        )
        message(
            "训练包",
            f"{result}；下一步 lab.sh upload-training --archive 包路径，通过 OSS 送到学校服务器",
            level="success",
        )
        return result
    use_openpi(args.openpi)
    manifest = (
        prepare_data(args, parent)
        if args.command == "prepare-data"
        else round_manifest(args.round)
    )
    config = training_config(args, manifest, parent)
    if args.command == "train":
        import jax

        from dagger_offline.training import run_training

        devices = jax.local_devices()
        if not devices or any(device.platform != "gpu" for device in devices):
            raise RuntimeError("训练必须使用 GPU JAX 环境；CPU 验证请用 train-check")
        if len(devices) % config.fsdp_devices or config.batch_size % len(devices):
            raise ValueError(
                "可见 GPU 数必须整除 batch-size，并且能按 fsdp-devices 分组"
            )
        message(
            "开始训练",
            f"全量 Pi0.5 HG-DAgger；输出 {config.checkpoint_dir}；不会启动机器人",
        )
        run_training(config, manifest)
        message(
            "训练结束",
            f"请封存 checkpoint 并回实验室验证；训练结束不代表策略已改善：{config.checkpoint_dir}",
            level="success",
        )
    else:
        check(config, manifest)
        message("数据入口", str(manifest))
    return manifest


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        message("失败", str(exc), level="error", stream=sys.stderr)
        raise SystemExit(1) from exc
