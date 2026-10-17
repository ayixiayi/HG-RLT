"""单入口连续采集：逐条确认归零、接管/交还、结果标记和自动编号。"""

import argparse
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagger_offline.collection import collection_lock, refresh_records, run_collection
from dagger_offline.console import message


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rhos-root", type=Path, required=True)
    parser.add_argument("--abc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="本轮采集目录")
    parser.add_argument(
        "--episodes", type=int, default=0, help="本次最多尝试次数；0 不限"
    )
    parser.add_argument("--group", help="可选：将相关录制绑定同一来源组；默认每条独立")
    parser.add_argument("--deployment", type=Path, help="新轮次模型的 deployment.json")
    parser.add_argument("--index-only", action="store_true", help="只刷新清单和统计")
    parser.add_argument(
        "--passive-leaders",
        action="store_true",
        help="主臂不跟随机械臂；h 立即按当前主臂姿态相对接管",
    )
    parser.add_argument(
        "--rlt",
        type=Path,
        help="RLT 运行目录（与 rlt-learn --run 相同）；启用 r 键交给 RL actor",
    )
    parser.add_argument(
        "--leader-on-demand",
        action="store_true",
        help="兼容模式：主臂只在按 h 后对齐机械臂，策略执行时不出力",
    )
    args = parser.parse_args()
    if args.index_only:
        with collection_lock(args.output_dir) as root:
            refresh_records(root)
        return
    if not sys.stdin.isatty():
        raise RuntimeError("连续采集需要现场操作者的交互终端")
    if args.rlt is not None:
        # rlt-learn trains only on the directory bound to the run; recording
        # anywhere else would never reach the actor.
        from dagger_offline import rlt

        try:
            rlt.bind_recordings(args.rlt, args.output_dir)
        except ValueError:
            bound = rlt.bound_recordings(args.rlt)
            raise ValueError(
                f"RLT 运行目录 {args.rlt} 的 rlt-learn 读取 {bound}；"
                f"--output-dir 请用这个目录，或换新的 --rlt 目录"
            ) from None

    from run_robot import execute, preflight, prepare_start

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("HG-DAgger collection interrupted")

    signal.signal(signal.SIGTERM, interrupted)

    def record_episode(path, group):
        episode_args = SimpleNamespace(
            rhos_root=args.rhos_root,
            abc=args.abc,
            output=path,
            deployment=args.deployment,
            passive_leaders=args.passive_leaders,
            leader_on_demand=args.leader_on_demand,
            rlt=args.rlt,
        )
        config, profile, cameras, reference, metadata = preflight(episode_args)
        prepare_start(episode_args, profile, reference)
        return execute(
            episode_args,
            config,
            profile,
            cameras,
            reference,
            {**metadata, "collection": {"group": group}},
        )

    def home_unattended():
        # Its own log under homing/: not an episode, never renumbers one.
        folder = Path(args.output_dir) / "homing"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        episode_args = SimpleNamespace(
            rhos_root=args.rhos_root,
            abc=args.abc,
            output=folder / f"idle-{stamp}.h5",
            deployment=args.deployment,
            passive_leaders=args.passive_leaders,
            leader_on_demand=args.leader_on_demand,
            rlt=args.rlt,
        )
        _, profile, _, reference, _ = preflight(episode_args)
        prepare_start(episode_args, profile, reference, unattended=True)

    run_collection(
        args.output_dir,
        record_episode,
        episodes=args.episodes,
        group=args.group,
        home_unattended=home_unattended,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        message(
            "采集结束",
            "已中断；请核对本条记录及现场状态，已保存记录保留。",
            level="warning",
        )
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        message("采集停止", str(exc), level="error")
        raise SystemExit(1) from exc
