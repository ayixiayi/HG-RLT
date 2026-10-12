"""RLT replay construction from HG recordings, and the learner loop.

Replay (one shard per labelled episode, built only after its outcome is saved):

* one transition per RL actor decision; ``action`` is what the robot actually
  received (daemon ACK targets) for the k steps until the next decision, with
  the actor's remaining unexecuted steps filling the rest of the chunk;
  ``discount`` = gamma**k;
* decisions of one uninterrupted RL run chain into each other (done=0);
* the last decision of a run is terminal: reward 1 if the episode succeeded
  and no human took over after this run (handing back to the VLA with ``p`` or
  pausing with ``s`` keeps the credit), otherwise 0. The value is the
  probability of finishing without human help once the actor hands off.

Takeover segments carry no RL token, so this version leaves them out of the
replay (they still feed HG training).
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from dagger_offline import rlt
from dagger_offline.console import message


def read_episode(path):
    """Shared HDF5 fields, or None while not complete and labelled."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        if not attrs.get("complete", False) or attrs.get("outcome") not in (
            "success",
            "fail",
        ):
            return None
        episode = {
            "attrs": attrs,
            "metadata": json.loads(attrs["metadata_json"]),
            "sent": f["pairing/command_sent_ns"][()],
            "valid": f["pairing/action_valid"][()].astype(bool),
            "commands": np.hstack(
                [f["data/q_des_left"][()], f["data/q_des_right"][()]]
            ),
            "event_ns": f["dagger_events/timestamp_ns"][()],
            "event_reason": f["dagger_events/reason"].asstr()[()],
            "event_state": f["dagger_events/controller_state"].asstr()[()],
        }
        for group in ("rl_decisions", "policy_chunks"):
            if group in f:
                episode[group] = {k: f[group][k][()] for k in f[group]}
    return episode


def chain_transitions(episode, decisions, config):
    """Transitions for consecutive decisions (accepted_ns, token, proprio,
    reference, planned) of one episode; see the module docstring."""
    success = episode["attrs"]["outcome"] == "success"
    sent, valid, commands = episode["sent"], episode["valid"], episode["commands"]
    event_ns = episode["event_ns"]
    # Any operator ownership event ends a run; "finish" ends the episode.
    boundaries = event_ns[
        np.isin(
            episode["event_reason"],
            ["ownership selected", "rl selected", "operator boundary", "finish"],
        )
    ]
    takeovers = event_ns[episode["event_state"] == "INTERVENTION"]
    d = decisions
    count = len(d["accepted_ns"])
    rows = []
    for i in range(count):
        start = int(d["accepted_ns"][i])
        stop = int(d["accepted_ns"][i + 1]) if i + 1 < count else None
        cut = boundaries[boundaries > start]
        run_end = int(cut[0]) if len(cut) else None
        limits = [x for x in (stop, run_end) if x is not None]
        window = (sent >= start) & valid & (sent > 0)
        if limits:
            window &= sent < min(limits)
        ran = commands[window]
        if len(ran) == 0:
            continue
        executed = ran[: config.chunk]
        qpos = d["proprio"][i][: rlt.JOINTS]
        planned = np.array(d["planned"][i], dtype=np.float64)
        planned[: len(executed)] = executed
        chained = stop is not None and (run_end is None or stop < run_end)
        if chained:
            j = i + 1
            next_ref = rlt.to_residual(
                d["reference"][j], d["proprio"][j][: rlt.JOINTS], config
            )
            next_token, next_proprio = d["token"][j], d["proprio"][j]
            reward, done = 0.0, 0.0
        else:
            helped = bool(np.any(takeovers >= start))
            next_ref = np.zeros(config.action_dim, np.float32)
            next_token = np.zeros_like(d["token"][i])
            next_proprio = np.zeros(rlt.PROPRIO, np.float32)
            reward, done = (1.0 if success and not helped else 0.0), 1.0
        rows.append(
            {
                "token": d["token"][i],
                "proprio": d["proprio"][i],
                "ref": rlt.to_residual(d["reference"][i], qpos, config),
                "action": rlt.to_residual(planned, qpos, config),
                "reward": reward,
                # Elapsed control steps until the next decision (may exceed C
                # when the VLA kept executing its longer chunk).
                "discount": config.gamma ** len(ran),
                "done": done,
                "next_token": next_token,
                "next_proprio": next_proprio,
                "next_ref": next_ref,
                "intervention": 0.0,
            }
        )
    return rows


def episode_transitions(path, config, policy_id=None):
    """RL-actor transitions of a finished recording ([] if it has none);
    None while the episode is not yet complete and labelled."""
    episode = read_episode(path)
    if episode is None:
        return None
    if "rl_decisions" not in episode:
        return []
    policy = episode["metadata"].get("rlt", {}).get("policy_id")
    if policy_id is not None and policy != policy_id:
        raise ValueError(f"{path.name}: RL decisions come from another model")
    d = episode["rl_decisions"]
    if len(d["accepted_ns"]) == 0:
        return []
    if d["reference"].shape[1:] != (config.chunk, rlt.JOINTS):
        raise ValueError(f"{path}: RL chunk differs from the learner config")
    return chain_transitions(episode, {**d, "planned": d["actor"]}, config)


def sync_replay(recordings, replay_dir, config, policy_id=None):
    """Build shards for newly labelled episodes; returns the new transition count."""
    replay_dir = Path(replay_dir)
    added = 0
    for path in sorted(Path(recordings).glob("episodes/episode-*.h5")):
        shard = replay_dir / f"{path.stem}.npz"
        if shard.exists() or shard.with_suffix(".npz.none").exists():
            continue
        try:
            rows = episode_transitions(path, config, policy_id)
        except OSError:
            continue  # still being written or held by the collector
        if rows is None:
            continue  # not finished/labelled yet: look again next cycle
        writer = rlt.ReplayWriter(replay_dir, config, path.stem)
        for row in rows:
            writer.add(**row)
        if writer.close() is None:
            # Labelled but without RL decisions: remember, do not rescan.
            replay_dir.mkdir(parents=True, exist_ok=True)
            shard.with_suffix(".npz.none").touch()
            continue
        added += len(rows)
    return added


def learn(recordings, run, config, *, min_transitions=64, poll_s=10.0, once=False):
    run = Path(run)
    replay_dir, actor_dir = run / "replay", run / "actor"
    learner = rlt.Learner(config)
    version = 0
    message("RLT learner", f"录制 {recordings} → {run}；等待带 RL 决策的已标注录制")
    while True:
        marker = run / "model.json"
        policy_id = (
            json.loads(marker.read_text())["policy_id"] if marker.exists() else None
        )
        added = sync_replay(recordings, replay_dir, config, policy_id)
        replay = rlt.load_replay(replay_dir, config)
        size = 0 if replay is None else len(replay["token"])
        if added and size >= min_transitions:
            log = learner.train(replay, steps=added)
            version += 1
            rlt.publish_actor(actor_dir, config, learner.actor_numpy(), version)
            message(
                "RLT learner",
                f"actor v{version} 已发布 · replay {size} · 新增 {added} · "
                + " · ".join(f"{k}={v:.3f}" for k, v in log.items()),
                level="success",
            )
        elif added:
            message(
                "RLT learner", f"replay {size} 条，攒够 {min_transitions} 条后开始训练"
            )
        if once:
            return version
        time.sleep(poll_s)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RLT 在线 learner：读已标注录制，训练并发布 actor"
    )
    parser.add_argument(
        "--recordings", type=Path, required=True, help="collect 的 --output-dir"
    )
    parser.add_argument(
        "--run", type=Path, required=True, help="RLT 运行目录（replay/ 和 actor/）"
    )
    parser.add_argument("--config", type=Path, help="RLTConfig JSON；默认使用内置配置")
    parser.add_argument("--min-transitions", type=int, default=64)
    parser.add_argument("--once", action="store_true", help="只同步训练一次后退出")
    args = parser.parse_args(argv)
    config = (
        rlt.RLTConfig(**json.loads(args.config.read_text()))
        if args.config
        else rlt.RLTConfig()
    )
    learn(
        args.recordings,
        args.run,
        config,
        min_transitions=args.min_transitions,
        once=args.once,
    )


if __name__ == "__main__":
    main()
