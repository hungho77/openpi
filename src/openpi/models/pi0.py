import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _stack_cfg_observation(obs: _model.Observation) -> _model.Observation:
    """Stacks the conditional and the unconditional prompt along the batch dimension."""

    def stack(x):
        return jnp.concatenate([x, x], axis=0)

    return _model.Observation(
        images={k: stack(v) for k, v in obs.images.items()},
        image_masks={k: stack(v) for k, v in obs.image_masks.items()},
        state=stack(obs.state),
        tokenized_prompt=jnp.concatenate([obs.tokenized_prompt, obs.tokenized_prompt_uncond], axis=0),
        tokenized_prompt_mask=jnp.concatenate([obs.tokenized_prompt_mask, obs.tokenized_prompt_uncond_mask], axis=0),
        subgoal_images=None if obs.subgoal_images is None else {k: stack(v) for k, v in obs.subgoal_images.items()},
        subgoal_image_masks=None
        if obs.subgoal_image_masks is None
        else {k: stack(v) for k, v in obs.subgoal_image_masks.items()},
        history_images=None if obs.history_images is None else {k: stack(v) for k, v in obs.history_images.items()},
        history_image_masks=None
        if obs.history_image_masks is None
        else {k: stack(v) for k, v in obs.history_image_masks.items()},
    )


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.rtc_max_delay = config.rtc_max_delay
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_max_fast_tokens = config.ki_max_fast_tokens
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                knowledge_insulation=config.knowledge_insulation,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.history_frames = config.history_frames
        if config.history_frames > 0:
            self.history_time_embedding = nnx.Param(jnp.zeros((config.history_frames, paligemma_config.width)))
            self.history_proj = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.history_gate = nnx.Param(jnp.zeros(()))

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "*b s"]]:
        # Called as a plain function so that it can be reused by other models with the same backbone.
        tokens, input_mask, ar_mask, _ = Pi0._embed_prefix(self, obs)
        return tokens, input_mask, ar_mask

    def _embed_prefix(self, obs: _model.Observation):
        """Embeds the prefix. Also returns the (start, length) of the language tokens in the prefix."""
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            if obs.history_images is not None and name in obs.history_images:
                image_tokens = image_tokens + Pi0._embed_history(
                    self, obs.history_images[name], obs.history_image_masks[name]
                ).astype(image_tokens.dtype)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        text_range = (len(ar_mask), 0)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            text_range = (text_range[0], tokenized_inputs.shape[1])

        # add subgoal images as a separate block: they attend to the images, language and each other, but the
        # images and language do not attend to them
        if obs.subgoal_images is not None:
            for i, name in enumerate(obs.subgoal_images):
                subgoal_tokens, _ = self.PaliGemma.img(obs.subgoal_images[name], train=False)
                tokens.append(subgoal_tokens)
                input_mask.append(einops.repeat(obs.subgoal_image_masks[name], "b -> b s", s=subgoal_tokens.shape[1]))
                ar_mask += [i == 0] + [False] * (subgoal_tokens.shape[1] - 1)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        # language tokens with their own autoregressive mask (e.g. the FAST tokens used for knowledge insulation)
        if obs.token_ar_mask is not None and obs.tokenized_prompt is not None:
            start, length = text_range
            ar_mask = jnp.broadcast_to(ar_mask, input_mask.shape)
            ar_mask = ar_mask.at[:, start : start + length].set(obs.token_ar_mask.astype(jnp.bool_))
        return tokens, input_mask, ar_mask, text_range

    def _embed_history(
        self, frames: at.Float[at.Array, "b n h w c"], masks: at.Bool[at.Array, "b n"]
    ) -> at.Float[at.Array, "b s emb"]:
        """Compresses the past frames of a camera into the same number of tokens as a single frame."""
        if getattr(self, "history_frames", 0) == 0:
            raise ValueError("The model was not configured with history frames (history_frames=0).")
        batch_size, num_frames = frames.shape[:2]
        if num_frames != self.history_frames:
            raise ValueError(f"Expected {self.history_frames} history frames, got {num_frames}.")
        frame_tokens, _ = self.PaliGemma.img(frames.reshape(-1, *frames.shape[2:]), train=False)
        frame_tokens = frame_tokens.reshape(batch_size, num_frames, *frame_tokens.shape[1:]).astype(jnp.float32)
        frame_tokens = frame_tokens + self.history_time_embedding.value[None, :, None, :]
        weights = masks[..., None, None].astype(jnp.float32)
        pooled = jnp.sum(frame_tokens * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        has_history = jnp.any(masks, axis=1)[:, None, None]
        # the gate is initialized to zero, so that the model initially behaves like a model without history
        return jnp.tanh(self.history_gate.value) * self.history_proj(pooled) * has_history

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b ah emb"] | None,
    ]:
        """Embeds the noisy actions. The flow matching timestep is either shared by all actions ([b]) or given per
        action ([b, ah]), which is used for real-time chunking."""
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(
            timestep.reshape(-1), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        ).reshape(*timestep.shape, -1)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            if time_emb.ndim == 2:
                time_emb = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # real-time chunking: the first `delay` actions are given (clean, with time 0) and excluded from the loss
        action_time = time
        loss_weights = None
        if self.rtc_max_delay > 0:
            delay = jax.random.randint(jax.random.fold_in(rng, 1), batch_shape, 0, self.rtc_max_delay + 1)
            committed = jnp.arange(self.action_horizon) < delay[..., None]
            x_t = jnp.where(committed[..., None], actions, x_t)
            action_time = jnp.where(committed, 0.0, time[..., None])
            # normalize so that the mean over the chunk is the mean over the predicted actions
            loss_weights = ~committed * (self.action_horizon / jnp.maximum(self.action_horizon - delay, 1))[..., None]

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, (text_start, text_len) = self._embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, action_time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        if prefix_ar_mask.ndim == 2:
            suffix_ar_mask = jnp.broadcast_to(suffix_ar_mask, suffix_mask.shape)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        else:
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # knowledge insulation: the prompt contains the FAST tokens of the action chunk, which are only predicted by the
        # backbone. They are not available at inference time, so no other token attends to them and the positions of
        # the following tokens are the same as without them.
        use_ki = self.knowledge_insulation and observation.token_loss_mask is not None
        if use_ki:
            fast = jnp.zeros(input_mask.shape, jnp.bool_)
            fast = fast.at[:, text_start : text_start + text_len].set(observation.token_ar_mask.astype(jnp.bool_))
            fast = fast & input_mask
            attn_mask = attn_mask & ~(~fast[:, :, None] & fast[:, None, :])
            positions = jnp.where(fast, positions, jnp.cumsum(input_mask & ~fast, axis=1) - 1)

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if loss_weights is not None:
            loss = loss * loss_weights
        if use_ki:
            loss = loss + self._fast_token_loss(prefix_out[:, text_start : text_start + text_len], observation)[:, None]
        return loss

    def _fast_token_loss(
        self, text_out: at.Float[at.Array, "b l emb"], observation: _model.Observation
    ) -> at.Float[at.Array, " b"]:
        """Next token prediction loss on the FAST tokens."""
        loss_mask = observation.token_loss_mask[:, 1:]
        targets = observation.tokenized_prompt[:, 1:]
        # only decode the positions with a loss, since the logits over the vocabulary are large
        order = jnp.argsort(~loss_mask, axis=-1, stable=True)[:, : self.ki_max_fast_tokens]
        selected_mask = jnp.take_along_axis(loss_mask, order, axis=-1)
        selected_targets = jnp.take_along_axis(targets, order, axis=-1)
        selected_out = jnp.take_along_axis(text_out[:, :-1], order[..., None], axis=1)
        logits = self.PaliGemma.llm(selected_out, method="decode")
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, selected_targets[..., None], axis=-1)[..., 0]
        return -jnp.sum(token_log_probs * selected_mask, axis=-1) / jnp.maximum(jnp.sum(selected_mask, axis=-1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cfg_scale: float | at.Float[at.Array, ""] = 1.0,
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix_len: at.Int[at.Array, " b"] | None = None,
    ) -> _model.Actions:
        """Samples an action chunk.

        `action_prefix` (with `action_prefix_len` valid steps) contains already committed actions, e.g. the actions of
        the previous chunk that are executed while this chunk is computed. They are kept fixed and the rest of the
        chunk is generated to continue them. This requires a model trained with `rtc_max_delay` > 0.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # classifier-free guidance: the conditional and unconditional prompts are run as a single batch of size 2b,
        # and the velocities are combined as v_uncond + cfg_scale * (v_cond - v_uncond).
        use_cfg = observation.tokenized_prompt_uncond is not None
        if use_cfg:
            observation = _stack_cfg_observation(observation)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        model_batch_size = observation.state.shape[0]

        committed = None
        if action_prefix is not None:
            prefix_len = (
                action_prefix_len if action_prefix_len is not None else jnp.full(batch_size, self.action_horizon)
            )
            committed = jnp.arange(self.action_horizon) < prefix_len[:, None]
            noise = jnp.where(committed[..., None], action_prefix, noise)

        def step(carry):
            x_t, time = carry
            action_time = jnp.broadcast_to(time, batch_size)
            if committed is not None:
                action_time = jnp.where(committed, 0.0, action_time[:, None])
            x_in = jnp.concatenate([x_t, x_t], axis=0) if use_cfg else x_t
            time_in = jnp.concatenate([action_time, action_time], axis=0) if use_cfg else action_time
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_in, time_in)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                model_batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if use_cfg:
                v_cond, v_uncond = jnp.split(v_t, 2, axis=0)
                v_t = v_uncond + cfg_scale * (v_cond - v_uncond)
            if committed is not None:
                # the committed actions stay fixed
                v_t = jnp.where(committed[..., None], 0.0, v_t)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
