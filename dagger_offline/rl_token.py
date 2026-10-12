"""RL Token (RLT, arXiv 2604.23073) Stage 1 for Pi0.5. Import only with OpenPI.

A small encoder reads the VLM's final-layer prefix embeddings of the camera
tokens (stop-gradient) plus a learned ``<rl>`` embedding; the encoder output at
that position is the RL token ``z_rl`` (VLM width, 2048 for Gemma 2B). A small
causal decoder reconstructs the embeddings from ``z_rl`` (loss L_ro). Training
optimizes ``L_ro + alpha * L_vla``; ``alpha == 0`` freezes the whole VLA.

The language/state tokens are not reconstructed: the instruction is fixed and
the online actor receives proprioception separately (as in the paper).
"""

import dataclasses
import re

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import model as _model
from openpi.models import pi0, pi0_config
from openpi.shared import download, nnx_utils
from openpi.training import weight_loaders
from typing_extensions import override

RL_TOKEN_PARAMS = ".*rl_token.*"


class _Block(nnx.Module):
    def __init__(self, width, heads, *, rngs):
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=heads, in_features=width, decode=False, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.mlp_in = nnx.Linear(width, 4 * width, rngs=rngs)
        self.mlp_out = nnx.Linear(4 * width, width, rngs=rngs)

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask, deterministic=True)
        return x + self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x))))


class RLToken(nnx.Module):
    def __init__(self, embed_width, *, width, depth, heads, max_tokens, rngs):
        init = nnx.initializers.normal(0.02)
        self.max_tokens = max_tokens
        self.down = nnx.Linear(embed_width, width, rngs=rngs)
        self.rl = nnx.Param(init(rngs.params(), (width,)))
        self.enc_pos = nnx.Param(init(rngs.params(), (max_tokens + 1, width)))
        # Named attributes, not lists: OpenPI flattens params with "/" keys.
        self.depth = depth
        for i in range(depth):
            setattr(self, f"encoder_{i}", _Block(width, heads, rngs=rngs))
        self.enc_norm = nnx.LayerNorm(width, rngs=rngs)
        self.token_out = nnx.Linear(width, embed_width, rngs=rngs)
        self.dec_in = nnx.Linear(embed_width, width, rngs=rngs)
        self.dec_pos = nnx.Param(init(rngs.params(), (max_tokens, width)))
        for i in range(depth):
            setattr(self, f"decoder_{i}", _Block(width, heads, rngs=rngs))
        self.dec_norm = nnx.LayerNorm(width, rngs=rngs)
        self.recon_out = nnx.Linear(width, embed_width, rngs=rngs)

    def _check(self, embeddings):
        if embeddings.shape[1] > self.max_tokens:
            raise ValueError(
                f"{embeddings.shape[1]} prefix tokens exceed rl_max_tokens={self.max_tokens}"
            )

    def encode(self, embeddings, mask):
        """(b, n, E) embeddings, (b, n) bool mask -> (b, E) RL token."""
        self._check(embeddings)
        x = self.down(jax.lax.stop_gradient(embeddings))
        batch, count, width = x.shape
        rl = jnp.broadcast_to(self.rl.value.astype(x.dtype), (batch, 1, width))
        x = jnp.concatenate([x, rl], axis=1) + self.enc_pos.value[: count + 1]
        valid = jnp.concatenate([mask, jnp.ones((batch, 1), bool)], axis=1)
        attention = valid[:, None, None, :]
        for i in range(self.depth):
            x = getattr(self, f"encoder_{i}")(x, attention)
        return self.token_out(self.enc_norm(x[:, -1]))

    def reconstruction_loss(self, token, embeddings, mask):
        """Mean squared reconstruction error per example, (b,)."""
        self._check(embeddings)
        targets = jax.lax.stop_gradient(embeddings)
        count = targets.shape[1]
        x = jnp.concatenate(
            [self.dec_in(token)[:, None], self.dec_in(targets[:, :-1])], axis=1
        )
        x = x + self.dec_pos.value[:count]
        # Input j is z_rl (j == 0) or target j-1: causal, skipping padded keys.
        keys = jnp.concatenate([jnp.ones_like(mask[:, :1]), mask[:, :-1]], axis=1)
        causal = jnp.tril(jnp.ones((count, count), bool))
        attention = causal[None, None] & keys[:, None, None, :]
        for i in range(self.depth):
            x = getattr(self, f"decoder_{i}")(x, attention)
        error = jnp.mean(jnp.square(self.recon_out(self.dec_norm(x)) - targets), -1)
        weight = mask.astype(error.dtype)
        return jnp.sum(error * weight, -1) / jnp.maximum(jnp.sum(weight, -1), 1.0)


class RLTPi0(pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = self.PaliGemma.llm.module.configs[0].width
        self.rl_alpha = config.rl_alpha
        self.rl_stride = config.rl_stride
        self.rl_token = RLToken(
            width,
            width=config.rl_width,
            depth=config.rl_depth,
            heads=config.rl_heads,
            max_tokens=config.rl_max_tokens,
            rngs=rngs,
        )

    def _image_embeddings(self, prefix_out, prefix_mask, observation):
        # embed_prefix places all camera tokens before the language tokens.
        prompt = observation.tokenized_prompt
        count = prefix_out.shape[1] - (0 if prompt is None else prompt.shape[1])
        stride = self.rl_stride
        return prefix_out[:, :count:stride], prefix_mask[:, :count:stride]

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        vla = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        # Prefix tokens never attend to the suffix, so these equal a
        # prefix-only forward pass (what inference computes).
        embeddings, mask = self._image_embeddings(prefix_out, prefix_mask, observation)
        token = self.rl_token.encode(embeddings, mask)
        reconstruction = self.rl_token.reconstruction_loss(token, embeddings, mask)
        return self.rl_alpha * vla + reconstruction[..., None]

    def prefix_embeddings(self, observation):
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions
        )
        return self._image_embeddings(prefix_out, prefix_mask, observation)

    def compute_rl_token(self, observation):
        """(b, E) RL token for an observation (deterministic, no sampling)."""
        return self.rl_token.encode(*self.prefix_embeddings(observation))


@dataclasses.dataclass(frozen=True)
class RLTPi0Config(pi0_config.Pi0Config):
    rl_width: int = 512
    rl_depth: int = 2
    rl_heads: int = 8
    rl_alpha: float = 1.0
    rl_stride: int = 1
    rl_max_tokens: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("RL token is implemented for Pi0.5")
        if min(self.rl_width, self.rl_depth, self.rl_heads, self.rl_stride) < 1:
            raise ValueError("RL token sizes must be positive")
        if self.rl_width % self.rl_heads or self.rl_alpha < 0:
            raise ValueError("rl_width must divide by rl_heads; rl_alpha >= 0")

    @override
    def create(self, rng):
        return RLTPi0(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self):
        if self.rl_alpha == 0:
            # Train only the RL token; the VLA stays exactly the parent.
            return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(RL_TOKEN_PARAMS)))
        return super().get_freeze_filter()


@dataclasses.dataclass(frozen=True)
class RLTCheckpointWeightLoader(weight_loaders.CheckpointWeightLoader):
    """Parent Pi0.5 checkpoint; RL token weights start from fresh init unless
    the checkpoint already has them (later rounds)."""

    @override
    def load(self, params):
        loaded = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat = flax.traverse_util.flatten_dict(loaded, sep="/")
        unexpected = [k for k in flat if re.fullmatch(RL_TOKEN_PARAMS, k) is None]
        if not unexpected:
            raise ValueError("Parent checkpoint has no VLA parameters")
        return weight_loaders._merge_params(  # noqa: SLF001 - pinned upstream helper
            loaded, params, missing_regex=f"{RL_TOKEN_PARAMS}|.*lora.*"
        )


def config_from_contract(model, contract):
    """Rebuild the trained RLT model config from a run's inference contract."""
    fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
    return RLTPi0Config(
        **fields,
        rl_alpha=float(contract["alpha"]),
        rl_width=int(contract["width"]),
        rl_depth=int(contract["depth"]),
        rl_heads=int(contract["heads"]),
        rl_stride=int(contract["stride"]),
    )


class TokenTap:
    """Wrap an OpenPI Policy: every infer also computes the RL token from the
    exact transformed inputs of that call (``last`` holds it, float32 (E,))."""

    def __init__(self, policy):
        self._policy = policy
        model = policy._model  # noqa: SLF001 - pinned upstream Policy
        if not hasattr(model, "compute_rl_token"):
            raise TypeError("Policy model has no RL token")
        self._token = nnx_utils.module_jit(model.compute_rl_token)
        self.last = None

    def infer(self, obs, **kwargs):
        self.last = None
        outputs = self._policy.infer(obs, **kwargs)
        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        token = np.asarray(self._token(_model.Observation.from_dict(inputs))[0])
        if token.ndim != 1 or not np.isfinite(token).all():
            raise ValueError("Invalid RL token")
        self.last = token.astype(np.float32)
        return outputs

    def __getattr__(self, name):
        return getattr(self._policy, name)
