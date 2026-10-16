"""RLT online core: NumPy actor, publication, replay, and the JAX TD3 learner."""

import dataclasses

import numpy as np
import pytest

from dagger_offline import rlt

SMALL = rlt.RLTConfig(token_dim=8, chunk=2, lead=1, hidden=64, batch=128)


def test_untrained_actor_follows_the_reference_and_round_trips_residuals():
    actor = rlt.Actor(SMALL, rlt.init_actor(SMALL))
    qpos = np.linspace(-0.4, 0.4, 14)
    ref = qpos[None] + np.array([[0.02], [0.05]]) * np.ones((1, 14))
    held = rlt.pad_committed([], qpos, SMALL)
    chunk, residual = actor.act(np.zeros(8), rlt.proprio(qpos, qpos), held, ref, qpos)
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



def test_exploration_continues_the_executed_command_smoothly(tmp_path):
    # Exploration ramps from the last executed command toward a fresh target:
    # no per-step noise and no jump when an RL chunk takes over.
    cfg = rlt.RLTConfig(token_dim=8)
    agent = rlt.RLAgent(cfg, tmp_path / "none", seed=0)
    qpos = np.linspace(-0.4, 0.4, 14)
    qpos[[6, 13]] = 0.5
    t = np.arange(cfg.chunk)[:, None]
    reference = qpos[None] + 0.004 * t
    committed = reference[: cfg.lead] + 0.05
    for skip in (1, 3, cfg.lead, cfg.lead + 2):
        actions, record = agent.decide(
            np.zeros(8), qpos, qpos, reference, committed, skip=skip
        )
        held = committed[min(skip, cfg.lead) - 1]
        np.testing.assert_allclose(actions[skip - 1], held, atol=1e-6)
        offset = (actions - reference)[skip - 1 :]
        np.testing.assert_allclose(np.diff(offset, 2, axis=0), 0, atol=2e-3)
        assert np.abs(np.diff(offset, axis=0)).max() < 0.05
        np.testing.assert_array_equal(record["actor"], actions)
    # From HOLD (nothing committed) the chunk starts at the reference.
    actions, _ = agent.decide(np.zeros(8), qpos, qpos, reference, (), skip=0)
    offset = actions - reference
    assert np.abs(offset[0]).max() < 0.6 * cfg.actor_std * cfg.action_scale
    np.testing.assert_allclose(np.diff(offset, 2, axis=0), 0, atol=2e-3)
    # Successive targets differ: exploration is not a fixed bias.
    targets = [
        agent.decide(np.zeros(8), qpos, qpos, reference, (), skip=0)[0][-1]
        - reference[-1]
        for _ in range(50)
    ]
    std = np.std(targets, axis=0)[:6].mean()
    expected = cfg.actor_std * cfg.action_scale * cfg.chunk / (cfg.chunk + 1)
    assert 0.6 * expected < std < 1.4 * expected



def test_recordings_binding_is_shared_by_collector_and_learner(tmp_path):
    run = tmp_path / "rlt" / "new"
    assert rlt.bound_recordings(run) is None
    rlt.bind_recordings(run, tmp_path / "recordings" / "a")  # collector first
    assert rlt.bound_recordings(run) == str((tmp_path / "recordings" / "a").resolve())
    rlt.bind_recordings(run, tmp_path / "recordings" / "a")  # learner agrees
    with pytest.raises(ValueError, match="换录制目录"):
        rlt.bind_recordings(run, tmp_path / "eval" / "a")



def test_actor_correction_is_linear_over_the_chunk(tmp_path):
    cfg = rlt.RLTConfig(token_dim=8, hidden=32)
    rng = np.random.default_rng(0)
    params = {k: rng.standard_normal(v.shape).astype(np.float32) * 0.05
              for k, v in rlt.init_actor(cfg).items()}
    assert params[f"w{cfg.layers}"].shape[1] == cfg.knots * 14
    actor = rlt.Actor(cfg, params)
    qpos = np.zeros(14)
    ref = qpos[None] + 0.01 * np.sin(np.arange(cfg.chunk))[:, None]
    held = rlt.pad_committed([], qpos, cfg)
    chunk, _ = actor.act(np.ones(8), rlt.proprio(qpos, qpos), held, ref, qpos)
    correction = chunk - ref
    assert np.abs(correction).max() > 1e-3  # the actor does correct ã
    np.testing.assert_allclose(np.diff(correction, 2, axis=0), 0, atol=1e-6)
    basis = rlt.knot_basis(cfg)
    np.testing.assert_allclose(basis.sum(1), 1, atol=1e-6)
    assert basis[0, 0] == 1 and basis[-1, -1] == 1
    with pytest.raises(ValueError, match="knots"):
        rlt.RLTConfig(chunk=4, lead=1, knots=5)
    # A run made by the per-step actor is refused, not silently reshaped.
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text('{"token_dim": 8, "chunk": 10, "lead": 4}')
    with pytest.raises(ValueError, match="旧版"):
        rlt.run_config(old)



def test_reference_is_never_clipped_and_committed_steps_are_kept():
    cfg = rlt.RLTConfig(token_dim=8, hidden=32)
    params = rlt.init_actor(cfg)  # near-zero correction
    state = np.zeros((1, cfg.state_dim), np.float32)
    # Gripper target 1 from position 0 is 2 normalized units away.
    ref = np.zeros((1, cfg.action_dim), np.float32)
    ref[0, 6::14] = 2.0
    mean = rlt.actor_mean(params, state, ref, cfg)
    np.testing.assert_allclose(mean, ref, atol=5e-3)
    # Reference dropout hides ã from the network only; it stays the offset.
    hidden = rlt.actor_mean(params, state, ref, cfg, ref_input=np.zeros_like(ref))
    np.testing.assert_allclose(hidden, ref, atol=5e-3)
    big = {k: v * 1e4 for k, v in params.items()}
    assert np.abs(rlt.actor_mean(big, state + 1, ref, cfg) - ref).max() <= 1 + 1e-6
    # The first `skip` steps come from the prefix, the rest from the actor.
    action = np.ones((2, cfg.action_dim))
    prefix = np.full((2, cfg.committed_dim), 7.0)
    out = rlt.with_prefix(action, prefix, np.array([0, 3]), cfg)
    np.testing.assert_array_equal(out[0], 1)
    np.testing.assert_array_equal(out[1, : 3 * 14], 7)
    np.testing.assert_array_equal(out[1, 3 * 14 :], 1)


def test_actor_gets_no_gradient_from_committed_steps():
    pytest.importorskip("jax")
    cfg = dataclasses.replace(SMALL, ref_dropout=0.0)
    rng = np.random.default_rng(0)
    rows = [_row(rng, cfg, skip=float(cfg.chunk)) for _ in range(64)]
    batch = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    learner = rlt.Learner(cfg, seed=0)
    import jax

    before = jax.tree.map(np.asarray, learner.actor)
    actor, *_ = learner._actor_step(
        learner.actor, learner.actor_state, learner.critics[0], batch, jax.random.key(0)
    )
    # Every step was committed: nothing the actor outputs reaches Q or BC.
    for k in before:
        np.testing.assert_array_equal(np.asarray(actor[k]), before[k])


def test_first_recordings_binding_is_exclusive(tmp_path, monkeypatch):
    run = tmp_path / "run"
    rlt.bind_recordings(run, tmp_path / "a")
    # A second starter that looked before the first one wrote must still lose.
    real = rlt.bound_recordings
    calls = []

    def stale(path):
        calls.append(path)
        return None if len(calls) == 1 else real(path)

    monkeypatch.setattr(rlt, "bound_recordings", stale)
    with pytest.raises(ValueError, match="换录制目录"):
        rlt.bind_recordings(run, tmp_path / "b")
    monkeypatch.undo()
    assert rlt.bound_recordings(run) == str((tmp_path / "a").resolve())
    assert [p.name for p in run.iterdir()] == ["recordings.json"]


def test_old_synchronous_recordings_are_skipped_not_fatal(tmp_path, monkeypatch):
    from dagger_offline import rlt_learn

    recordings, replay = tmp_path / "rec", tmp_path / "replay"
    (recordings / "episodes").mkdir(parents=True)
    for n in (1, 2):
        (recordings / "episodes" / f"episode-00000{n}.h5").touch()
    rng = np.random.default_rng(0)

    def transitions(path, config, policy_id=None):
        if path.stem.endswith("1"):
            raise rlt_learn.SynchronousRecording("old")
        return [_row(rng, config)]

    monkeypatch.setattr(rlt_learn, "episode_transitions", transitions)
    assert rlt_learn.sync_replay(recordings, replay, SMALL) == 1
    assert sorted(p.name for p in replay.iterdir()) == [
        "episode-000001.npz.none",
        "episode-000002.npz",
    ]



def test_actor_published_under_older_semantics_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(rlt, "ACTOR_FORMAT", rlt.ACTOR_FORMAT - 1)
    rlt.publish_actor(tmp_path, SMALL, rlt.init_actor(SMALL), 1, "m1")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="different RLT config"):
        rlt.load_actor(tmp_path, SMALL, "m1")


def test_synchronous_recording_is_recognised_before_layout_checks(monkeypatch):
    from pathlib import Path

    from dagger_offline import rlt_learn

    old = {  # 20-step synchronous decisions: no committed, no skip
        "metadata": {"rlt": {"policy_id": "m1"}},
        "rl_decisions": {
            "accepted_ns": np.array([1]),
            "reference": np.zeros((1, 20, 14)),
            "actor": np.zeros((1, 20, 14)),
        },
    }
    monkeypatch.setattr(rlt_learn, "read_episode", lambda path: old)
    with pytest.raises(rlt_learn.SynchronousRecording):
        rlt_learn.episode_transitions(Path("episode-000001.h5"), SMALL, "m1")


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
        "skip": 0.0,
        "next_skip": 0.0,
        "next_prefix": np.zeros(cfg.committed_dim),
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
    # A weak anchor: this checks that TD3 finds a better chunk, not how far
    # the production anchor lets it go.
    loose = dataclasses.replace(SMALL, beta=0.05)
    learner = rlt.Learner(loose, seed=0)
    log = learner.train(replay, steps=1500, rng=np.random.default_rng(1))
    assert np.isfinite(list(log.values())).all()
    actor = rlt.Actor(loose, learner.actor_numpy())
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
        "skip": np.array([0] + [SMALL.lead] * (n - 1)),
        "reference": np.zeros((n, SMALL.chunk, 14)),
        "planned": np.zeros((n, SMALL.chunk, 14)),
    }
    return episode, decisions



def test_rl_run_stopped_by_the_safety_layer_is_not_credited():
    from dagger_offline import rlt_learn

    episode, decisions = _episode(
        event_ns=np.array([600, 950]),
        event_reason=np.array(
            ["Policy rejected; explicit h/p required: Unsafe target", "finish"]
        ),
        event_state=np.array(["HOLD", "HOLD"]),
    )
    rows = rlt_learn.chain_transitions(episode, decisions, SMALL)
    # Decisions 0-2 run until the rejection at 600; the episode still succeeds
    # after the operator restarts at 700, but the stopped run earns nothing.
    assert [r["done"] for r in rows] == [0.0, 0.0, 1.0, 1.0]
    assert [r["reward"] for r in rows] == [0.0, 0.0, 0.0, 1.0]


def test_transitions_tile_observations_and_discount_elapsed_steps():
    from dagger_offline import rlt_learn

    rows = rlt_learn.chain_transitions(*_episode(), SMALL)
    assert [r["done"] for r in rows] == [0.0, 0.0, 0.0, 1.0]
    # Committed steps of each decision and of its successor travel along;
    # the successor's prefix is what the robot received (rows 30, 50, 70),
    # not the queued 0.05 the actor saw.
    assert [r["skip"] for r in rows] == [0, 1, 1, 1]
    assert [r["next_skip"] for r in rows] == [1, 1, 1, 0]
    for r, row in zip(rows[:3], (30, 50, 70), strict=True):
        np.testing.assert_allclose(r["next_prefix"], row / 1000 / 0.5, atol=1e-7)
        np.testing.assert_allclose(r["next_committed"], 0.05 / 0.5)
    # It is exactly the head of the successor's own replay action.
    for r, nxt in zip(rows[:3], rows[1:], strict=True):
        np.testing.assert_allclose(r["next_prefix"], nxt["action"][:14])
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
