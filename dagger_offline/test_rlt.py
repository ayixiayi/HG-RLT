"""RLT online core: NumPy actor, publication, replay, and the JAX TD3 learner."""

import numpy as np
import pytest

from dagger_offline import rlt

SMALL = rlt.RLTConfig(token_dim=8, chunk=2, lead=1, hidden=64, batch=128)


def test_untrained_actor_follows_the_reference_and_round_trips_residuals():
    actor = rlt.Actor(SMALL, rlt.init_actor(SMALL), seed=1)
    qpos = np.linspace(-0.4, 0.4, 14)
    ref = qpos[None] + np.array([[0.02], [0.05]]) * np.ones((1, 14))
    held = rlt.pad_committed([], qpos, SMALL)
    chunk, residual = actor.act(
        np.zeros(8), rlt.proprio(qpos, qpos), held, ref, qpos, explore=False
    )
    assert chunk.shape == (2, 14) and residual.shape == (28,)
    np.testing.assert_allclose(chunk, ref, atol=5e-3)
    back = rlt.from_residual(rlt.to_residual(ref, qpos, SMALL), qpos, SMALL)
    np.testing.assert_allclose(back, ref, atol=1e-6)
    with pytest.raises(ValueError, match="size"):
        actor.act(np.zeros(9), rlt.proprio(qpos, qpos), held, ref, qpos)
    # Committed steps: padded with the last one, or with qpos when none.
    steps = qpos[None] + 0.1
    np.testing.assert_allclose(rlt.pad_committed(steps, qpos, SMALL), steps)
    two = rlt.RLTConfig(token_dim=8, chunk=4, lead=3)
    padded = rlt.pad_committed(steps, qpos, two)
    assert padded.shape == (3, 14) and np.allclose(padded, steps[0])
    with pytest.raises(ValueError, match="lead"):
        rlt.pad_committed(np.repeat(steps, 2, 0), qpos, SMALL)
    with pytest.raises(ValueError, match="lead"):
        rlt.RLTConfig(chunk=4, lead=4)


def test_publication_is_atomic_versioned_and_config_bound(tmp_path):
    watcher = rlt.ActorWatcher(tmp_path, SMALL)
    assert watcher.poll() is None
    params = rlt.init_actor(SMALL, seed=3)
    rlt.publish_actor(tmp_path, SMALL, params, version=1)
    first = watcher.poll()
    assert first.version == 1 and watcher.poll() is first
    params["b0"] = params["b0"] + 1
    rlt.publish_actor(tmp_path, SMALL, params, version=2)
    second = watcher.poll()
    assert second.version == 2 and np.allclose(second.params["b0"], params["b0"])
    assert [p.name for p in tmp_path.iterdir()] == ["actor.npz"]
    other = rlt.RLTConfig(token_dim=8, chunk=3, lead=1)
    with pytest.raises(ValueError, match="different RLT config"):
        rlt.load_actor(tmp_path, other)


def _row(rng, cfg, **override):
    row = {
        "token": rng.standard_normal(cfg.token_dim),
        "proprio": rng.standard_normal(rlt.PROPRIO),
        "committed": np.zeros(cfg.committed_dim),
        "ref": np.zeros(cfg.action_dim),
        "action": rng.uniform(-0.5, 0.5, cfg.action_dim),
        "reward": 0.0,
        "discount": SMALL.gamma**SMALL.chunk,
        "done": 1.0,
        "next_token": np.zeros(cfg.token_dim),
        "next_proprio": np.zeros(rlt.PROPRIO),
        "next_committed": np.zeros(cfg.committed_dim),
        "next_ref": np.zeros(cfg.action_dim),
        "intervention": 0.0,
    }
    row.update(override)
    return row


def test_replay_shards_round_trip_and_refuse_other_configs(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("episode-1", "episode-2"):
        writer = rlt.ReplayWriter(tmp_path, SMALL, name)
        for _ in range(3):
            writer.add(**_row(rng, SMALL))
        assert writer.close().name == f"{name}.npz"
    replay = rlt.load_replay(tmp_path, SMALL)
    assert replay["token"].shape == (6, 8) and replay["action"].shape == (6, 28)
    with pytest.raises(ValueError, match="another RLT config"):
        rlt.load_replay(tmp_path, rlt.RLTConfig(token_dim=8, chunk=3, lead=1))
    with pytest.raises(ValueError, match="another RLT config"):
        rlt.load_replay(tmp_path, rlt.RLTConfig(token_dim=8, chunk=2, lead=0))
    with pytest.raises(ValueError, match="exactly"):
        rlt.ReplayWriter(tmp_path, SMALL, "x").add(token=np.zeros(8))
    with pytest.raises(FileExistsError):
        writer = rlt.ReplayWriter(tmp_path, SMALL, "episode-1")
        writer.add(**_row(rng, SMALL))
        writer.close()


@pytest.mark.parametrize("reward_scale", [1.0, 0.05])
def test_td3_learner_improves_on_the_reference(reward_scale):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((SMALL.token_dim, SMALL.action_dim)) * 0.15

    def best(token):  # optimal normalized residual depends on the state
        return np.tanh(token @ w)

    rows = []
    for _ in range(3000):
        token = rng.standard_normal(SMALL.token_dim)
        action = rng.uniform(-0.6, 0.6, SMALL.action_dim)
        reward = -reward_scale * np.sum(np.square(action - best(token)))
        rows.append(_row(rng, SMALL, token=token, action=action, reward=reward))
    replay = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    learner = rlt.Learner(SMALL, seed=0)
    log = learner.train(replay, steps=1500, rng=np.random.default_rng(1))
    assert np.isfinite(list(log.values())).all()
    actor = rlt.Actor(SMALL, learner.actor_numpy())
    tokens = rng.standard_normal((200, SMALL.token_dim))
    zero = np.zeros(14)

    def regret(policy):
        return np.mean([np.sum(np.square(policy(t) - best(t))) for t in tokens])

    learned = regret(
        lambda t: actor.act(
            t,
            np.zeros(rlt.PROPRIO),
            rlt.pad_committed([], zero, SMALL),
            np.zeros((2, 14)),
            zero,
            explore=False,
        )[1]
    )
    reference = regret(lambda t: np.zeros(SMALL.action_dim))
    assert learned < 0.5 * reference


def _episode(idle_last=False, **extra):
    """Back-to-back RL timeline at 10 ns per control step: commands every step
    (the next decision observes while the current chunk still runs); command
    r holds the value r/1000 on every joint."""
    rows = np.arange(100)
    observed = np.array([100, 300, 500, 700] + ([900] if idle_last else []))
    # Uneven reply latency: decision 1 is accepted 90 ns after its image.
    accepted = observed + np.array([30, 90, 30, 30] + ([30] if idle_last else []))
    valid = np.ones(100, bool)
    if idle_last:
        valid[93:] = False  # k right after the last decision was accepted
    episode = {
        "attrs": {"outcome": "success"},
        "row_ns": rows * 10,
        "sent": rows * 10 + 5,
        "valid": valid,
        "commands": np.repeat(rows[:, None] / 1000.0, 14, axis=1),
        "event_ns": np.array([950]),
        "event_reason": np.array(["finish"]),
        "event_state": np.array(["HOLD"]),
        "discards": [],
    }
    episode.update(extra)
    n = len(observed)
    decisions = {
        "observed_ns": observed,
        "accepted_ns": accepted,
        "token": np.zeros((n, SMALL.token_dim), np.float32),
        "proprio": np.zeros((n, rlt.PROPRIO), np.float32),
        "committed": np.full((n, SMALL.lead, 14), 0.05),
        "reference": np.zeros((n, SMALL.chunk, 14)),
        "planned": np.zeros((n, SMALL.chunk, 14)),
    }
    return episode, decisions


def test_transitions_tile_observations_and_discount_elapsed_steps():
    from dagger_offline import rlt_learn

    rows = rlt_learn.chain_transitions(*_episode(), SMALL)
    assert [r["done"] for r in rows] == [0.0, 0.0, 0.0, 1.0]
    assert [r["reward"] for r in rows] == [0.0, 0.0, 0.0, 1.0]
    # Decision 0 observed at 100, next state observed at 300: 20 rows elapse.
    assert rows[0]["discount"] == pytest.approx(SMALL.gamma**20)
    # Decision 1 (observed 300, accepted 390): its action starts with what ran
    # from its observation on, i.e. the committed steps at rows 30, 31.
    np.testing.assert_allclose(rows[1]["action"], np.repeat([0.030, 0.031], 14) / 0.5)
    # Committed steps are part of the state and of the next state.
    np.testing.assert_allclose(rows[0]["committed"], 0.05 / 0.5)
    np.testing.assert_allclose(rows[0]["next_committed"], 0.05 / 0.5)
    np.testing.assert_allclose(rows[-1]["next_committed"], 0.0)


def test_synchronous_recordings_without_committed_steps_are_refused():
    from dagger_offline import rlt_learn

    episode, decisions = _episode()
    del decisions["committed"]
    with pytest.raises(ValueError, match="committed"):
        rlt_learn.chain_transitions(episode, decisions, SMALL)


def test_a_decision_ended_before_acting_leaves_the_result_on_the_previous():
    from dagger_offline import rlt_learn

    rows = rlt_learn.chain_transitions(*_episode(idle_last=True), SMALL)
    assert len(rows) == 4
    assert [(r["done"], r["reward"]) for r in rows] == [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
    ]


def test_discarded_interval_breaks_the_chain_and_removes_its_credit():
    from dagger_offline import rlt_learn

    episode, decisions = _episode(discards=[(420, 450)])
    rows = rlt_learn.chain_transitions(episode, decisions, SMALL)
    # 0 -> 1 chained; 1 is cut at the discard start (no credit); 2 -> 3 chained.
    assert [r["done"] for r in rows] == [0.0, 1.0, 0.0, 1.0]
    assert [r["reward"] for r in rows] == [0.0, 0.0, 0.0, 1.0]
    episode, decisions = _episode(discards=[(250, 750)])
    rows = rlt_learn.chain_transitions(episode, decisions, SMALL)
    # Only decision 0 ran entirely before the discard: kept, cut, no credit.
    assert [(r["done"], r["reward"]) for r in rows] == [(1.0, 0.0)]


def test_run_config_is_shared_and_its_layout_locked(tmp_path):
    import dataclasses

    first = rlt.run_config(tmp_path)
    assert first == rlt.RLTConfig() and (tmp_path / "config.json").exists()
    tuned = dataclasses.replace(first, lr=1e-4, beta=2.0)
    assert rlt.run_config(tmp_path, tuned) == tuned
    assert rlt.run_config(tmp_path) == tuned  # the collector sees the same
    rlt.publish_actor(tmp_path / "actor", first, rlt.init_actor(first), 1)
    assert rlt.load_actor(tmp_path / "actor", tuned)[1] == 1  # lr is not layout
    with pytest.raises(ValueError, match="新目录"):
        rlt.run_config(tmp_path, dataclasses.replace(first, chunk=5))


def test_learner_state_round_trip_continues_identically(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    rng = np.random.default_rng(0)
    rows = [_row(rng, SMALL, reward=float(rng.standard_normal())) for _ in range(300)]
    replay = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    a = rlt.Learner(SMALL, seed=0)
    a.train(replay, steps=20, rng=np.random.default_rng(1))
    a.save_state(tmp_path / "learner.pkl", version=3, processed={"x.npz"})
    b = rlt.Learner(SMALL, seed=99)
    assert b.load_state(tmp_path / "learner.pkl") == (3, {"x.npz"})
    a.train(replay, steps=10, rng=np.random.default_rng(2))
    b.train(replay, steps=10, rng=np.random.default_rng(2))
    for key, value in a.actor_numpy().items():
        np.testing.assert_allclose(b.actor_numpy()[key], value, rtol=1e-6, atol=1e-6)
    assert a.updates == b.updates


def test_run_artifacts_are_bound_to_one_model(tmp_path):
    rlt.publish_actor(tmp_path / "actor", SMALL, rlt.init_actor(SMALL), 1, "m1")
    assert rlt.load_actor(tmp_path / "actor", SMALL, "m1")[1] == 1
    with pytest.raises(ValueError, match="not m2"):
        rlt.load_actor(tmp_path / "actor", SMALL, "m2")
    # Training output without a binding cannot be claimed by a model later.
    with pytest.raises(ValueError, match="未绑定"):
        rlt.bind_run(tmp_path, "m2")
    fresh = tmp_path / "fresh"
    rlt.bind_run(fresh, "m1")
    assert rlt.bound_policy(fresh) == "m1"
    with pytest.raises(ValueError, match="另一个模型"):
        rlt.bind_run(fresh, "m2")
    watcher = rlt.ActorWatcher(tmp_path / "actor", SMALL, policy_id="m2")
    with pytest.raises(ValueError, match="not m2"):
        watcher.poll()


def test_learner_waits_for_the_binding_and_states_name_their_model(tmp_path):
    pytest.importorskip("jax")
    from dagger_offline import rlt_learn

    run, recordings = tmp_path / "run", tmp_path / "rec"
    (recordings / "episodes").mkdir(parents=True)
    # Unbound: nothing is trained, published or bound.
    assert rlt_learn.learn(recordings, run, SMALL, once=True) == 0
    assert not (run / "actor").exists() and not (run / "learner.pkl").exists()
    rlt.bind_run(run, "m1")
    learner = rlt.Learner(SMALL)
    learner.save_state(run / "learner.pkl", version=3, processed=set(), policy_id="m1")
    with pytest.raises(ValueError, match="not m2"):
        rlt.Learner(SMALL).load_state(run / "learner.pkl", "m2")
    # A bound run with state but no actor republishes it for its model.
    assert rlt_learn.learn(recordings, run, SMALL, once=True) == 3
    assert rlt.load_actor(run / "actor", SMALL, "m1")[1] == 3


def test_one_learner_per_run_and_one_recording_directory(tmp_path):
    from dagger_offline import rlt_learn

    run = tmp_path / "run"
    first, second = tmp_path / "a", tmp_path / "b"
    for d in (first, second):
        (d / "episodes").mkdir(parents=True)
    # Unbound model: returns after binding the recordings, without training.
    assert rlt_learn.learn(first, run, SMALL, once=True) == 0
    with pytest.raises(ValueError, match="换录制目录"):
        rlt_learn.learn(second, run, SMALL, once=True)
    with rlt.exclusive_learner(run):
        with pytest.raises(RuntimeError, match="已有 learner"):
            rlt_learn.learn(first, run, SMALL, once=True)
    assert rlt_learn.learn(first, run, SMALL, once=True) == 0  # lock released
