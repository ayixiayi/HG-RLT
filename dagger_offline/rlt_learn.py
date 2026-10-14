"""RLT replay construction from HG recordings, and the learner loop.

Replay (one shard per labelled episode, built only after its outcome is saved):

* RL chunks are synchronous: the next decision observes only after this
  chunk finished (recordings with prefetched RL chunks are refused);
* one transition per RL actor decision that executed; ``action`` is what the
  robot actually received (daemon ACK targets) until the next decision was
  accepted, with the actor's remaining unexecuted steps filling the rest of
  the chunk; ``discount`` = gamma**k for the k control steps between the two
  observations;
* executed decisions of one uninterrupted RL run chain into each other
  (done=0); a decision that executed nothing (e.g. ``k`` right after it) is
  skipped;
* the last executed decision of a run is terminal: reward 1 if the episode succeeded
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
        if not attrs.get("usable", False) or attrs.get("unusable", True):
            return {"attrs": attrs, "unusable": True}
        discards = []
        if "discard_segments" in f:
            g = f["discard_segments"]
            discards = list(
                zip(g["start_ns"][()].tolist(), g["end_ns"][()].tolist(), strict=True)
            )
        episode = {
            "discards": discards,
            "row_ns": f["timestamps/top"][()],
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
    discards = episode.get("discards", [])
    discard_starts = np.array([a for a, _ in discards], dtype=np.int64)
    # An operator-discarded interval (c ... x) ends any run crossing into it.
    boundaries = np.sort(np.concatenate([boundaries, discard_starts]))
    row_ns = episode.get("row_ns")
    d = decisions
    count = len(d["accepted_ns"])
    never = 2**63
    # Pass 1: the commands each decision actually executed. A decision's chunk
    # runs from its acceptance until the next decision is accepted (or the run
    # ends); RL decisions are synchronous (no prefetch), so the next state is
    # observed only after this chunk finished.
    spans = []
    for i in range(count):
        observed, start = int(d["observed_ns"][i]), int(d["accepted_ns"][i])
        cut = boundaries[boundaries > observed]
        run_end = int(cut[0]) if len(cut) else never
        following = int(d["accepted_ns"][i + 1]) if i + 1 < count else never
        end = min(following, run_end)
        if any(a < end and b > observed for a, b in discards):
            spans.append(None)  # touches an operator-discarded interval
            continue
        window = (sent >= start) & valid & (sent > 0) & (sent < end)
        if i + 1 < count and int(d["observed_ns"][i + 1]) < run_end:
            late = window & (sent >= int(d["observed_ns"][i + 1]))
            if np.any(late):
                raise ValueError(
                    "RL decision executed after the next state was observed "
                    "(prefetched RL chunks); this replay needs rl_lead=0"
                )
        spans.append({"run_end": run_end, "end": end, "window": window})
    # Pass 2: chain decisions that executed; a run's last executed decision is
    # terminal and carries the episode result.
    executed = [i for i, span in enumerate(spans) if span and span["window"].any()]
    rows = []
    for n, i in enumerate(executed):
        span = spans[i]
        observed, start = int(d["observed_ns"][i]), int(d["accepted_ns"][i])
        run_end = span["run_end"]
        j = executed[n + 1] if n + 1 < len(executed) else None
        chained = (
            j is not None
            and int(d["observed_ns"][j]) < run_end
            and all(spans[m] is not None for m in range(i + 1, j))
        )
        ran = commands[span["window"]]
        stop = int(d["observed_ns"][j]) if chained else min(span["end"], run_end)
        if row_ns is not None and stop < never:
            elapsed = int(np.sum((row_ns >= observed) & (row_ns < stop)))
        else:
            elapsed = len(ran)
        executed_steps = ran[: config.chunk]
        qpos = d["proprio"][i][: rlt.JOINTS]
        planned = np.array(d["planned"][i], dtype=np.float64)
        planned[: len(executed_steps)] = executed_steps
        if chained:
            next_ref = rlt.to_residual(
                d["reference"][j], d["proprio"][j][: rlt.JOINTS], config
            )
            next_token, next_proprio = d["token"][j], d["proprio"][j]
            reward, done = 0.0, 0.0
        else:
            helped = bool(np.any(takeovers >= start))
            discarded_next = run_end in set(discard_starts.tolist())
            next_ref = np.zeros(config.action_dim, np.float32)
            next_token = np.zeros_like(d["token"][i])
            next_proprio = np.zeros(rlt.PROPRIO, np.float32)
            credited = success and not helped and not discarded_next
            reward, done = (1.0 if credited else 0.0), 1.0
        rows.append(
            {
                "token": d["token"][i],
                "proprio": d["proprio"][i],
                "ref": rlt.to_residual(d["reference"][i], qpos, config),
                "action": rlt.to_residual(planned, qpos, config),
                "reward": reward,
                # Elapsed control steps until the next decision (may exceed C
                # when the VLA kept executing its longer chunk).
                "discount": config.gamma ** max(elapsed, 1),
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
    if episode.get("unusable") or "rl_decisions" not in episode:
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


def learn(recordings, run, config=None, *, min_transitions=64, poll_s=10.0, once=False):
    """``config`` overrides RUN/config.json training parameters only."""
    run = Path(run)
    config = rlt.run_config(run, config)
    replay_dir, actor_dir, state_path = (
        run / "replay",
        run / "actor",
        run / "learner.pkl",
    )
    learner = rlt.Learner(config)
    # Train only for a bound model: the collector binds RUN to its sealed
    # model (model.json) when `collect --rlt RUN` starts.
    policy_id = rlt.bound_policy(run)
    if policy_id is None:
        message(
            "RLT learner",
            f"等待采集端绑定模型：{run}/model.json 在 collect --rlt 启动时写入",
        )
    while policy_id is None:
        if once:
            return 0
        time.sleep(poll_s)
        policy_id = rlt.bound_policy(run)
    if state_path.exists():
        version, processed = learner.load_state(state_path, policy_id)
        message(
            "RLT learner",
            f"恢复训练状态：actor v{version} · 已训练 {len(processed)} 个分片",
        )
        # The state is saved before each publication; an interruption between
        # the two leaves the actor missing or behind: publish it from the state.
        published = None
        if (actor_dir / "actor.npz").exists():
            _, published = rlt.load_actor(actor_dir, config, policy_id)
        if version and published != version:
            if published is not None and published > version:
                version = published + 1  # never step a watched version back
                learner.save_state(
                    state_path, version=version, processed=processed, policy_id=policy_id
                )
            rlt.publish_actor(
                actor_dir, config, learner.actor_numpy(), version, policy_id
            )
            message("RLT learner", f"按训练状态补发 actor v{version}", level="warning")
    elif (actor_dir / "actor.npz").exists():
        raise RuntimeError(
            f"{actor_dir} 已有 actor 但缺少训练状态 {state_path}；拒绝覆盖。换新的 --run 目录重新开始"
        )
    else:
        version, processed = 0, set()
    message("RLT learner", f"录制 {recordings} → {run}；等待带 RL 决策的已标注录制")
    while True:
        sync_replay(recordings, replay_dir, config, policy_id)
        shards = (
            {p.name for p in replay_dir.glob("*.npz")} if replay_dir.exists() else set()
        )
        fresh = sorted(shards - processed)
        replay = rlt.load_replay(replay_dir, config)
        size = 0 if replay is None else len(replay["token"])
        if fresh and size >= min_transitions:
            added = 0
            for name in fresh:
                with np.load(replay_dir / name) as shard:
                    added += len(shard["token"])
            log = learner.train(replay, steps=added)
            version += 1
            processed |= set(fresh)
            # State first: a restart can always re-publish this actor from it.
            learner.save_state(
                state_path, version=version, processed=processed, policy_id=policy_id
            )
            rlt.publish_actor(
                actor_dir, config, learner.actor_numpy(), version, policy_id
            )
            message(
                "RLT learner",
                f"actor v{version} 已发布 · replay {size} · 新增 {added} · "
                + " · ".join(f"{k}={v:.3f}" for k, v in log.items()),
                level="success",
            )
        elif fresh:
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
        "--run", type=Path, required=True, help="RLT 运行目录（与 collect --rlt 相同）"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="RLTConfig JSON；只能修改训练超参数，结构参数以运行目录的 config.json 为准",
    )
    parser.add_argument("--min-transitions", type=int, default=64)
    parser.add_argument("--once", action="store_true", help="只同步训练一次后退出")
    args = parser.parse_args(argv)
    override = (
        rlt.RLTConfig(**json.loads(args.config.read_text())) if args.config else None
    )
    learn(
        args.recordings,
        args.run,
        override,
        min_transitions=args.min_transitions,
        once=args.once,
    )


if __name__ == "__main__":
    main()
