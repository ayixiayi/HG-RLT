"""RL token Stage 1 on the tiny `dummy` Pi0.5 (CPU, real OpenPI modules)."""

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from dagger_offline import rl_token  # noqa: E402

CONFIG = rl_token.RLTPi0Config(
    pi05=True,
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    action_horizon=30,
    rl_width=32,
    rl_depth=1,
    rl_heads=4,
)


@pytest.fixture(scope="module")
def model_and_batch():
    model = CONFIG.create(jax.random.key(0))
    observation = CONFIG.fake_obs(batch_size=2)
    actions = CONFIG.fake_act(batch_size=2)
    return model, observation, actions


def test_loss_shape_token_width_and_prefix_equivalence(model_and_batch):
    model, observation, actions = model_and_batch
    loss = model.compute_loss(jax.random.key(1), observation, actions, train=False)
    assert loss.shape == (2, 30) and bool(jnp.all(jnp.isfinite(loss)))
    token = model.compute_rl_token(observation)
    width = model.PaliGemma.llm.module.configs[0].width
    assert token.shape == (2, width) and bool(jnp.all(jnp.isfinite(token)))
    embeddings, mask = model.prefix_embeddings(observation)
    assert embeddings.shape[:2] == mask.shape and embeddings.shape[1] > 0
    # Only camera tokens are encoded: 3 cameras x 256 SigLIP patches.
    assert embeddings.shape[1] == 3 * 256


def test_alpha_zero_freezes_everything_but_the_rl_token():
    frozen = dataclasses.replace(CONFIG, rl_alpha=0.0)
    model = frozen.create(jax.random.key(0))
    trainable = nnx.All(nnx.Param, nnx.Not(frozen.get_freeze_filter()))
    state = nnx.state(model, trainable).to_pure_dict()
    paths = list(rl_token.flax.traverse_util.flatten_dict(state, sep="/"))
    assert paths and all("rl_token" in p for p in paths)


def test_reconstruction_learns_and_token_depends_on_the_image(model_and_batch):
    model, observation, actions = model_and_batch
    graph, params, rest = nnx.split(
        model, nnx.All(nnx.Param, nnx.PathContains("rl_token")), ...
    )
    embeddings, mask = model.prefix_embeddings(observation)

    def loss_fn(p):
        m = nnx.merge(graph, p, rest)
        token = m.rl_token.encode(embeddings, mask)
        return jnp.mean(m.rl_token.reconstruction_loss(token, embeddings, mask))

    optimizer = optax.adam(3e-3)
    opt = optimizer.init(params)
    start = float(loss_fn(params))
    step = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(40):
        value, grads = step(params)
        updates, opt = optimizer.update(grads, opt, params)
        params = optax.apply_updates(params, updates)
    assert float(loss_fn(params)) < 0.5 * start
    trained = nnx.merge(graph, params, rest)

    def scene(seed):
        keys = jax.random.split(jax.random.key(seed), len(observation.images))
        return dataclasses.replace(
            observation,
            images={
                name: jax.random.uniform(key, image.shape, minval=-1, maxval=1)
                for key, (name, image) in zip(
                    keys, observation.images.items(), strict=True
                )
            },
        )

    # A fresh SigLIP has a zero-initialized output head (big_vision default),
    # mapping every image to the same tokens; trained checkpoints do not.
    vision = nnx.state(trained.PaliGemma.img, nnx.Param)
    leaves, tree = jax.tree.flatten(vision)
    keys = jax.random.split(jax.random.key(3), len(leaves))
    noisy = [
        x + 0.05 * jax.random.normal(k, x.shape, x.dtype)
        for k, x in zip(keys, leaves, strict=True)
    ]
    nnx.update(trained.PaliGemma.img, jax.tree.unflatten(tree, noisy))
    a = np.asarray(trained.compute_rl_token(scene(1)))
    b = np.asarray(trained.compute_rl_token(scene(2)))
    assert np.isfinite(a).all() and not np.allclose(a, b, atol=1e-4)


def test_weight_loader_keeps_vla_params_and_initializes_rl_token(monkeypatch):
    base = dataclasses.replace(CONFIG)
    model = base.create(jax.random.key(0))
    reference = nnx.state(model, nnx.Param).to_pure_dict()
    parent = nnx.state(
        rl_token.pi0_config.Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"
        ).create(jax.random.key(7)),
        nnx.Param,
    ).to_pure_dict()
    monkeypatch.setattr(rl_token.download, "maybe_download", lambda p: p)
    monkeypatch.setattr(
        rl_token._model,
        "restore_params",
        lambda *a, **k: jax.tree.map(np.asarray, parent),
    )
    loaded = rl_token.RLTCheckpointWeightLoader("/parent/params").load(reference)
    flat = rl_token.flax.traverse_util.flatten_dict(loaded, sep="/")
    flat_parent = rl_token.flax.traverse_util.flatten_dict(parent, sep="/")
    assert any(k.startswith("rl_token/") for k in flat)
    vla = [k for k in flat if not k.startswith("rl_token/")]
    assert vla and all(np.array_equal(flat[k], flat_parent[k]) for k in vla)


def test_real_openpi_trainer_trains_saves_and_reloads_rl_token(tmp_path):
    """Upstream scripts/train.main with the RLT model: gradient steps on the
    RL token, an Orbax checkpoint, and a reload that yields a token."""
    from openpi.training import config as openpi_config
    from openpi.training import weight_loaders
    from scripts import train

    model = dataclasses.replace(
        CONFIG,
        action_horizon=openpi_config.get_config("debug_pi05").model.action_horizon,
    )
    frozen = dataclasses.replace(model, rl_alpha=0.0)
    config = dataclasses.replace(
        openpi_config.get_config("debug_pi05"),
        name="rlt_debug",
        exp_name="cpu",
        model=frozen,
        freeze_filter=frozen.get_freeze_filter(),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=3,
        save_interval=2,
        log_interval=1,
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        assets_base_dir=str(tmp_path / "assets"),
        wandb_enabled=False,
        overwrite=True,
    )
    train.main(config)
    steps = sorted(
        int(p.name) for p in config.checkpoint_dir.iterdir() if p.name.isdigit()
    )
    assert steps and steps[-1] == 2
    params = rl_token._model.restore_params(
        config.checkpoint_dir / str(steps[-1]) / "params", restore_type=np.ndarray
    )
    flat = rl_token.flax.traverse_util.flatten_dict(params, sep="/")
    assert any(k.startswith("rl_token/") for k in flat)
    reloaded = frozen.load(params)
    token = reloaded.compute_rl_token(frozen.fake_obs(batch_size=1))
    width = reloaded.PaliGemma.llm.module.configs[0].width
    assert token.shape == (1, width) and bool(jnp.all(jnp.isfinite(token)))
    # alpha=0: VLA weights stay at their initialization; the token moved.
    fresh = rl_token.flax.traverse_util.flatten_dict(
        nnx.state(frozen.create(jax.random.key(config.seed)), nnx.Param).to_pure_dict(),
        sep="/",
    )
    vla = [k for k in flat if not k.startswith("rl_token/") and k in fresh]
    moved = [
        k
        for k in flat
        if k.startswith("rl_token/") and not np.allclose(flat[k], fresh[k])
    ]
    assert vla and moved


def test_single_pass_sampling_matches_upstream_actions_and_token(model_and_batch):
    from openpi.policies.policy import Policy

    model, observation, _ = model_and_batch
    rng = jax.random.key(7)
    noise = jax.random.normal(jax.random.key(8), (2, 30, model.action_dim))
    actions, token = model.sample_actions_with_token(rng, observation, noise=noise)
    expected = model.sample_actions(rng, observation, noise=noise)
    np.testing.assert_allclose(np.asarray(actions), np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(token), np.asarray(model.compute_rl_token(observation)), rtol=1e-5, atol=1e-5
    )
    # Without explicit noise both draw it from the same rng.
    a, _ = model.sample_actions_with_token(rng, observation)
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(model.sample_actions(rng, observation)), rtol=1e-5, atol=1e-5
    )
    # Through Policy.infer: the tapped policy returns the actions it would without the tap.
    single = jax.tree.map(lambda x: np.asarray(x[0]), observation.to_dict())
    plain = Policy(model, transforms=[], output_transforms=[])
    inner = Policy(model, transforms=[], output_transforms=[])
    plain._rng = inner._rng = jax.random.key(3)  # Policy(rng=...) truth-tests the key
    tapped = rl_token.TokenTap(inner)
    np.testing.assert_allclose(
        tapped.infer(single)["actions"], plain.infer(single)["actions"], rtol=1e-5, atol=1e-5
    )


def test_token_tap_matches_direct_token_and_sealed_reply(model_and_batch):
    from types import SimpleNamespace

    from openpi.policies.policy import Policy

    from dagger_offline.deployment import SealedModel

    model, observation, _ = model_and_batch
    policy = Policy(model, transforms=[], output_transforms=[])
    single = jax.tree.map(lambda x: np.asarray(x[0]), observation.to_dict())
    tap = rl_token.TokenTap(policy)
    out = tap.infer(single)
    assert out["actions"].shape == (30, model.action_dim)
    direct = np.asarray(model.compute_rl_token(observation)[0])
    np.testing.assert_allclose(tap.last, direct, rtol=1e-5, atol=1e-5)

    class Adapter:  # the XPolicyLab Pi_05 surface SealedModel relies on
        def __init__(self, policy):
            self.policy = self.model = policy

        def update_obs(self, obs):
            self.obs = single

        def get_action(self):
            return np.zeros((30, 14))

        def reset(self):
            pass

    adapter = Adapter(policy)
    sealed = SealedModel(adapter, "0" * 64)
    assert isinstance(adapter.policy, rl_token.TokenTap)
    adapter.policy.infer(single)
    reply = sealed.infer_sealed({"instruction": "choose_the_heavier_one"})
    np.testing.assert_allclose(reply["rl_token"], direct, rtol=1e-5, atol=1e-5)
    baseline = SealedModel(Adapter(SimpleNamespace()), "0" * 64)
    assert "rl_token" not in baseline.infer_sealed(
        {"instruction": "choose_the_heavier_one"}
    )
