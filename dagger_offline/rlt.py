"""RLT online stage (arXiv 2604.23073): small actor-critic on the RL token.

State ``x = (z_rl, proprio)``; the actor also sees the VLA reference chunk
``ã`` (dropped to zero for half the training samples) and outputs a chunk of
``C`` absolute joint targets. Everything is expressed as residuals from the
current joint position, normalized by ``action_scale``.

The actor forward pass is NumPy (the collection environment has no JAX). The
learner (TD3: twin Q, target smoothing, BC anchor ``beta * |a - ã|^2``) needs
JAX and runs in its own process. Actor weights travel through an atomically
replaced ``.npz`` publication that the collector hot-loads at chunk boundaries.
"""

import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

JOINTS = 14
PROPRIO = 2 * JOINTS  # joint position + finite-difference velocity
TRANSITION_FIELDS = (
    "token",
    "proprio",
    "ref",
    "action",
    "reward",
    "discount",  # gamma**k for the k control steps this decision actually ran
    "done",
    "next_token",
    "next_proprio",
    "next_ref",
    "intervention",
)


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    token_dim: int = 2048
    chunk: int = 10  # C: 0.33 s at 30 Hz
    hidden: int = 256
    layers: int = 2
    gamma: float = 0.99  # per control step; a transition discounts gamma**C
    # BC anchor toward the reference / human chunk, on the per-dimension mean
    # squared residual (dimension-independent); Q is normalized by mean |Q|.
    beta: float = 5.0
    actor_std: float = 0.05  # exploration, normalized units
    action_scale: float = 0.5  # rad per normalized unit of residual
    ref_dropout: float = 0.5
    utd: int = 5
    actor_every: int = 2
    tau: float = 0.005
    lr: float = 3e-4
    batch: int = 256
    target_noise: float = 0.1
    noise_clip: float = 0.2

    @property
    def action_dim(self):
        return self.chunk * JOINTS

    @property
    def state_dim(self):
        return self.token_dim + PROPRIO

    # Fields that change what a published actor means / what replay contains.
    # Training hyperparameters (lr, beta, utd, ...) may change between runs.
    ACTOR_FIELDS = ("token_dim", "chunk", "hidden", "layers", "action_scale")
    REPLAY_FIELDS = ("token_dim", "chunk", "action_scale", "gamma")

    def _hash(self, fields):
        values = {k: getattr(self, k) for k in fields}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def actor_digest(self):
        return self._hash(self.ACTOR_FIELDS)

    def replay_digest(self):
        return self._hash(self.REPLAY_FIELDS)

    def digest(self):
        return self._hash(tuple(f.name for f in dataclasses.fields(self)))


# ---------------------------------------------------------------- features


def proprio(qpos, previous_qpos, dt=1 / 30):
    qpos = np.asarray(qpos, dtype=np.float32)
    velocity = (qpos - np.asarray(previous_qpos, dtype=np.float32)) / dt
    return np.concatenate([qpos, velocity]).astype(np.float32)


def to_residual(chunk, qpos, config):
    """(C, 14) absolute targets -> flat normalized residual from qpos."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.shape != (config.chunk, JOINTS):
        raise ValueError(f"Expected a ({config.chunk}, {JOINTS}) chunk")
    residual = (chunk - np.asarray(qpos, dtype=np.float32)[None]) / config.action_scale
    return residual.reshape(-1)


def from_residual(residual, qpos, config):
    residual = np.asarray(residual, dtype=np.float64).reshape(config.chunk, JOINTS)
    return np.asarray(qpos, dtype=np.float64)[None] + residual * config.action_scale


# ---------------------------------------------------------------- networks


def _mlp_init(rng, sizes):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        params[f"w{i}"] = (
            rng.standard_normal((fan_in, fan_out)) / np.sqrt(fan_in)
        ).astype(np.float32)
        params[f"b{i}"] = np.zeros(fan_out, np.float32)
    # Small last layer: an untrained actor starts at the reference.
    last = len(sizes) - 2
    params[f"w{last}"] *= 1e-3
    return params


def init_actor(config, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [config.state_dim + config.action_dim]
    sizes += [config.hidden] * config.layers + [config.action_dim]
    return _mlp_init(rng, sizes)


def init_critic(config, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [config.state_dim + config.action_dim]
    sizes += [config.hidden] * config.layers + [1]
    return _mlp_init(rng, sizes)


def _forward(params, x, xp):
    count = len(params) // 2
    for i in range(count):
        x = x @ params[f"w{i}"] + params[f"b{i}"]
        if i < count - 1:
            x = xp.maximum(x, 0)
    return x


def actor_mean(params, state, ref, xp=np):
    """Normalized residual chunk; the reference is an input AND the offset."""
    x = xp.concatenate([state, ref], axis=-1)
    return xp.clip(ref + _forward(params, x, xp), -1.0, 1.0)


class Actor:
    """NumPy actor for the collector: deterministic mean plus fixed-std noise."""

    def __init__(self, config, params, *, version=0, seed=0):
        self.config, self.params, self.version = config, params, version
        self._rng = np.random.default_rng(seed)

    def act(self, token, proprio_vec, ref_chunk, qpos, *, explore=True):
        cfg = self.config
        state = np.concatenate(
            [np.asarray(token, np.float32), np.asarray(proprio_vec, np.float32)]
        )
        if state.shape != (cfg.state_dim,):
            raise ValueError("RL token / proprio size does not match the actor")
        ref = to_residual(ref_chunk, qpos, cfg)
        mean = actor_mean(self.params, state[None], ref[None])[0]
        if explore and cfg.actor_std > 0:
            mean = np.clip(
                mean + cfg.actor_std * self._rng.standard_normal(mean.shape), -1, 1
            )
        if not np.isfinite(mean).all():
            raise ValueError("Actor produced non-finite actions")
        return from_residual(mean, qpos, cfg), mean.astype(np.float32)


class RLAgent:
    """Collector-side decision maker: latest published actor (or the initial,
    reference-following one), plus a hard bound on deviation from ã."""

    def __init__(
        self, config, actor_directory, *, max_deviation=0.15, seed=0, policy_id=None
    ):
        if max_deviation <= 0:
            raise ValueError("max_deviation must be positive")
        self.config, self.chunk = config, config.chunk
        self.max_deviation = max_deviation
        self.watcher = ActorWatcher(
            actor_directory, config, seed=seed, policy_id=policy_id
        )
        self._initial = Actor(config, init_actor(config), version=0, seed=seed)

    def decide(self, token, qpos, previous_qpos, reference):
        actor = self.watcher.poll() or self._initial
        state = proprio(qpos, previous_qpos)
        reference = np.asarray(reference, dtype=np.float64)
        actions, _ = actor.act(token, state, reference, qpos)
        bound = np.full(JOINTS, self.max_deviation)
        bound[[6, 13]] = 2 * self.max_deviation  # normalized grippers
        actions = reference + np.clip(actions - reference, -bound, bound)
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
        record = {
            "token": np.asarray(token, np.float32),
            "proprio": state,
            "reference": reference,
            "actor": actions,
            "actor_version": int(actor.version),
        }
        return actions, record


def bind_run(run, policy_id):
    """One RLT run = one sealed VLA: RL tokens of another checkpoint differ."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "model.json"
    if marker.exists():
        bound = json.loads(marker.read_text())["policy_id"]
        if bound != policy_id:
            raise ValueError(
                f"RLT 运行目录 {run} 属于另一个模型（{bound[:12]}）；新模型请用新目录"
            )
        return
    trained = (
        (run / "actor" / "actor.npz").exists()
        or (run / "learner.pkl").exists()
        or any((run / "replay").glob("*.npz"))
    )
    if trained:
        raise ValueError(
            f"RLT 运行目录 {run} 已有未绑定模型的训练结果；新模型请用新目录"
        )
    marker.write_text(json.dumps({"policy_id": policy_id}) + "\n")


def bound_policy(run):
    """The model a run directory is bound to (None until a collector binds it)."""
    marker = Path(run) / "model.json"
    return json.loads(marker.read_text())["policy_id"] if marker.exists() else None


def _atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def run_config(run, override=None):
    """The RLT config shared by the collector and the learner (RUN/config.json).

    The first user creates it (defaults or ``override``). Later, training
    hyperparameters may change; the network / action / replay layout may not.
    """
    path = Path(run) / "config.json"
    if path.exists():
        stored = RLTConfig(**json.loads(path.read_text()))
        if override is None or override == stored:
            return stored
        if (
            override.actor_digest() != stored.actor_digest()
            or override.replay_digest() != stored.replay_digest()
        ):
            raise ValueError(
                f"{path}: 网络结构/动作/回放参数与该 RLT 运行目录不一致；请用新目录"
            )
        config = override  # training hyperparameters only
    else:
        config = override or RLTConfig()
    _atomic_write(
        path, (json.dumps(dataclasses.asdict(config), indent=2) + "\n").encode()
    )
    return config


# ---------------------------------------------------------------- publication


def publish_actor(directory, config, params, version, policy_id=None):
    """Atomically replace ``actor.npz`` (never a partially written file)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {f"p_{k}": v for k, v in params.items()}
    payload["meta"] = np.frombuffer(
        json.dumps(
            {
                "version": int(version),
                "config": config.actor_digest(),
                "policy_id": policy_id,
                "time": time.time(),
            }
        ).encode(),
        dtype=np.uint8,
    )
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, directory / "actor.npz")


def load_actor(directory, config, policy_id=None):
    path = Path(directory) / "actor.npz"
    with np.load(path) as data:
        meta = json.loads(bytes(data["meta"]).decode())
        if meta["config"] != config.actor_digest():
            raise ValueError("Published actor was trained with a different RLT config")
        if policy_id is not None and meta.get("policy_id") != policy_id:
            raise ValueError(
                f"Published actor belongs to model {meta.get('policy_id')}, not {policy_id}"
            )
        params = {k[2:]: data[k].copy() for k in data.files if k.startswith("p_")}
    return params, meta["version"]


class ActorWatcher:
    """Collector side: reload when a newer publication appears (block edges)."""

    def __init__(self, directory, config, *, seed=0, policy_id=None):
        self.directory, self.config = Path(directory), config
        self.policy_id = policy_id
        self._mtime = None
        self.actor = None
        self._seed = seed

    def poll(self):
        path = self.directory / "actor.npz"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return self.actor
        if mtime != self._mtime:
            params, version = load_actor(self.directory, self.config, self.policy_id)
            self.actor = Actor(self.config, params, version=version, seed=self._seed)
            self._mtime = mtime
        return self.actor


# ---------------------------------------------------------------- replay


class ReplayWriter:
    """Transitions append to numbered ``.npz`` shards (one per episode)."""

    def __init__(self, directory, config, name):
        self.directory, self.config = Path(directory), config
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.rows = {k: [] for k in TRANSITION_FIELDS}

    def add(self, **row):
        if set(row) != set(TRANSITION_FIELDS):
            raise ValueError(f"Transition needs exactly {TRANSITION_FIELDS}")
        for key, value in row.items():
            self.rows[key].append(np.asarray(value, dtype=np.float32))

    def close(self):
        if not self.rows["token"]:
            return None
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        target = self.directory / f"{self.name}.npz"
        if target.exists():
            raise FileExistsError(target)
        with tempfile.NamedTemporaryFile(
            dir=self.directory, suffix=".tmp", delete=False
        ) as f:
            temporary = Path(f.name)
            np.savez(
                f,
                config=np.frombuffer(self.config.replay_digest().encode(), np.uint8),
                **arrays,
            )
        os.replace(temporary, target)
        return target


def load_replay(directory, config):
    shards = sorted(Path(directory).glob("*.npz"))
    parts = {k: [] for k in TRANSITION_FIELDS}
    for shard in shards:
        with np.load(shard) as data:
            if bytes(data["config"]).decode() != config.replay_digest():
                raise ValueError(
                    f"{shard.name}: replay written with another RLT config"
                )
            for key in TRANSITION_FIELDS:
                parts[key].append(data[key])
    if not parts["token"]:
        return None
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------- learner (JAX)


class Learner:
    """TD3 with BC anchor and reference dropout. Import JAX only here."""

    def __init__(self, config, *, seed=0):
        import jax
        import jax.numpy as jnp
        import optax

        self.config, self.jax, self.jnp = config, jax, jnp
        self.actor = jax.tree.map(jnp.asarray, init_actor(config, seed))
        self.critics = [
            jax.tree.map(jnp.asarray, init_critic(config, seed + 1 + i))
            for i in range(2)
        ]
        self.target_actor = self.actor
        self.target_critics = list(self.critics)
        self.actor_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.actor_state = self.actor_opt.init(self.actor)
        self.critic_state = self.critic_opt.init(self.critics)
        self.key = jax.random.key(seed)
        self.updates = 0
        self._critic_step = jax.jit(self._make_critic_step(optax))
        self._actor_step = jax.jit(self._make_actor_step(optax))

    def _q(self, params, state, action):
        return self.jnp.squeeze(
            _forward(params, self.jnp.concatenate([state, action], -1), self.jnp), -1
        )

    def _make_critic_step(self, optax):
        cfg, jnp, jax = self.config, self.jnp, self.jax

        def loss(critics, target_actor, target_critics, batch, key):
            next_state = jnp.concatenate(
                [batch["next_token"], batch["next_proprio"]], -1
            )
            noise = jnp.clip(
                cfg.target_noise * jax.random.normal(key, batch["action"].shape),
                -cfg.noise_clip,
                cfg.noise_clip,
            )
            next_action = jnp.clip(
                actor_mean(target_actor, next_state, batch["next_ref"], jnp) + noise,
                -1,
                1,
            )
            target_q = jnp.minimum(
                self._q(target_critics[0], next_state, next_action),
                self._q(target_critics[1], next_state, next_action),
            )
            y = batch["reward"] + batch["discount"] * (1 - batch["done"]) * target_q
            y = jax.lax.stop_gradient(y)
            state = jnp.concatenate([batch["token"], batch["proprio"]], -1)
            return sum(
                jnp.mean(jnp.square(self._q(c, state, batch["action"]) - y))
                for c in critics
            )

        def step(critics, opt_state, target_actor, target_critics, batch, key):
            value, grads = jax.value_and_grad(loss)(
                critics, target_actor, target_critics, batch, key
            )
            updates, opt_state = self.critic_opt.update(grads, opt_state, critics)
            return optax.apply_updates(critics, updates), opt_state, value

        return step

    def _make_actor_step(self, optax):
        cfg, jnp, jax = self.config, self.jnp, self.jax

        def loss(actor, critic, batch, key):
            state = jnp.concatenate([batch["token"], batch["proprio"]], -1)
            keep = jax.random.bernoulli(key, 1 - cfg.ref_dropout, (state.shape[0], 1))
            ref_in = batch["ref"] * keep
            action = actor_mean(actor, state, ref_in, jnp)
            q = self._q(critic, state, action)
            bc = jnp.mean(jnp.square(action - batch["ref"]), -1)
            # TD3+BC scaling: beta stays meaningful whatever the reward scale.
            scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(q))) + 1e-6
            return jnp.mean(-q / scale + cfg.beta * bc), (jnp.mean(q), jnp.mean(bc))

        def step(actor, opt_state, critic, batch, key):
            (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(
                actor, critic, batch, key
            )
            updates, opt_state = self.actor_opt.update(grads, opt_state, actor)
            return optax.apply_updates(actor, updates), opt_state, value, aux

        return step

    def _soft(self, target, online):
        tau = self.config.tau
        return self.jax.tree.map(lambda t, o: (1 - tau) * t + tau * o, target, online)

    def train(self, replay, steps, rng=None):
        """``steps`` environment steps' worth of updates (x UTD)."""
        cfg, jnp, jax = self.config, self.jnp, self.jax
        rng = rng if rng is not None else np.random.default_rng(self.updates)
        size = len(replay["token"])
        log = {}
        for _ in range(steps * cfg.utd):
            index = rng.integers(0, size, min(cfg.batch, size))
            batch = {k: jnp.asarray(v[index]) for k, v in replay.items()}
            self.key, k1, k2 = jax.random.split(self.key, 3)
            self.critics, self.critic_state, log["critic"] = self._critic_step(
                self.critics,
                self.critic_state,
                self.target_actor,
                self.target_critics,
                batch,
                k1,
            )
            self.updates += 1
            if self.updates % cfg.actor_every == 0:
                self.actor, self.actor_state, log["actor"], (q, bc) = self._actor_step(
                    self.actor, self.actor_state, self.critics[0], batch, k2
                )
                log["q"], log["bc"] = q, bc
                self.target_actor = self._soft(self.target_actor, self.actor)
                self.target_critics = [
                    self._soft(t, o)
                    for t, o in zip(self.target_critics, self.critics, strict=True)
                ]
        return {k: float(v) for k, v in log.items()}

    def actor_numpy(self):
        return {k: np.asarray(v) for k, v in self.actor.items()}

    def save_state(self, path, *, version, processed, policy_id=None):
        """Everything needed to continue training exactly (atomic)."""
        import pickle

        jax = self.jax
        state = {
            "actor_digest": self.config.actor_digest(),
            "replay_digest": self.config.replay_digest(),
            "version": int(version),
            "policy_id": policy_id,
            "processed": sorted(processed),
            "updates": self.updates,
            "key": np.asarray(jax.random.key_data(self.key)),
            "trees": jax.device_get(
                {
                    "actor": self.actor,
                    "critics": self.critics,
                    "target_actor": self.target_actor,
                    "target_critics": self.target_critics,
                    "actor_state": self.actor_state,
                    "critic_state": self.critic_state,
                }
            ),
        }
        _atomic_write(path, pickle.dumps(state))

    def load_state(self, path, policy_id=None):
        import pickle

        jax, jnp = self.jax, self.jnp
        state = pickle.loads(Path(path).read_bytes())
        if (
            state["actor_digest"] != self.config.actor_digest()
            or state["replay_digest"] != self.config.replay_digest()
        ):
            raise ValueError("Saved learner state belongs to a different RLT config")
        if policy_id is not None and state.get("policy_id") != policy_id:
            raise ValueError(
                f"Saved learner state belongs to model {state.get('policy_id')}, not {policy_id}"
            )
        trees = jax.tree.map(jnp.asarray, state["trees"])
        self.actor, self.critics = trees["actor"], list(trees["critics"])
        self.target_actor = trees["target_actor"]
        self.target_critics = list(trees["target_critics"])
        self.actor_state, self.critic_state = (
            trees["actor_state"],
            trees["critic_state"],
        )
        self.updates = int(state["updates"])
        self.key = jax.random.wrap_key_data(jnp.asarray(state["key"]))
        return state["version"], set(state["processed"])
