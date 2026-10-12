"""RLT online core: NumPy actor, publication, replay, and the JAX TD3 learner."""

import numpy as np
import pytest

from dagger_offline import rlt

SMALL = rlt.RLTConfig(token_dim=8, chunk=2, hidden=64, batch=128)


def test_untrained_actor_follows_the_reference_and_round_trips_residuals():
    actor = rlt.Actor(SMALL, rlt.init_actor(SMALL), seed=1)
    qpos = np.linspace(-0.4, 0.4, 14)
    ref = qpos[None] + np.array([[0.02], [0.05]]) * np.ones((1, 14))
    chunk, residual = actor.act(
        np.zeros(8), rlt.proprio(qpos, qpos), ref, qpos, explore=False
    )
    assert chunk.shape == (2, 14) and residual.shape == (28,)
    np.testing.assert_allclose(chunk, ref, atol=5e-3)
    back = rlt.from_residual(rlt.to_residual(ref, qpos, SMALL), qpos, SMALL)
    np.testing.assert_allclose(back, ref, atol=1e-6)
    with pytest.raises(ValueError, match="size"):
        actor.act(np.zeros(9), rlt.proprio(qpos, qpos), ref, qpos)


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
    other = rlt.RLTConfig(token_dim=8, chunk=3)
    with pytest.raises(ValueError, match="different RLT config"):
        rlt.load_actor(tmp_path, other)


def _row(rng, cfg, **override):
    row = {
        "token": rng.standard_normal(cfg.token_dim),
        "proprio": rng.standard_normal(rlt.PROPRIO),
        "ref": np.zeros(cfg.action_dim),
        "action": rng.uniform(-0.5, 0.5, cfg.action_dim),
        "reward": 0.0,
        "discount": SMALL.gamma**SMALL.chunk,
        "done": 1.0,
        "next_token": np.zeros(cfg.token_dim),
        "next_proprio": np.zeros(rlt.PROPRIO),
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
        rlt.load_replay(tmp_path, rlt.RLTConfig(token_dim=8, chunk=3))
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
            t, np.zeros(rlt.PROPRIO), np.zeros((2, 14)), zero, explore=False
        )[1]
    )
    reference = regret(lambda t: np.zeros(SMALL.action_dim))
    assert learned < 0.5 * reference
