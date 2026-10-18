import logging
import math
import types

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_att_2d_masks_with_fast(pad_masks, att_masks, fast_masks):
    """Like `make_att_2d_masks`, but no token other than the FAST tokens attends to the FAST tokens."""
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    return att_2d_masks & ~(~fast_masks[:, :, None] & fast_masks[:, None, :])


def stack_cfg_observation(obs):
    """Stacks the conditional and the unconditional prompt along the batch dimension (classifier-free guidance)."""

    def stack(x):
        return torch.cat([x, x], dim=0)

    def stack_dict(d):
        return None if d is None else {k: stack(v) for k, v in d.items()}

    return types.SimpleNamespace(
        **{
            **vars(obs),
            "images": stack_dict(obs.images),
            "image_masks": stack_dict(obs.image_masks),
            "state": stack(obs.state),
            "tokenized_prompt": torch.cat([obs.tokenized_prompt, obs.tokenized_prompt_uncond], dim=0),
            "tokenized_prompt_mask": torch.cat([obs.tokenized_prompt_mask, obs.tokenized_prompt_uncond_mask], dim=0),
            "tokenized_prompt_uncond": None,
            "tokenized_prompt_uncond_mask": None,
            "subgoal_images": stack_dict(obs.subgoal_images),
            "subgoal_image_masks": stack_dict(obs.subgoal_image_masks),
            "history_images": stack_dict(obs.history_images),
            "history_image_masks": stack_dict(obs.history_image_masks),
            "token_ar_mask": None,
            "token_loss_mask": None,
        }
    )


class PrefixEmbeddingMixin:
    """Embeds the prefix (images with optional history, language and optional subgoal images).

    Requires `self.paligemma_with_expert` (with `embed_image` / `embed_language_tokens`), `self._apply_checkpoint` and,
    for history frames, `self.history_frames`, `self.history_time_embedding`, `self.history_proj` and
    `self.history_gate`.
    """

    def _embed_history(self, frames, masks):
        """Compresses the past frames of a camera ([B, n, C, H, W]) into the tokens of a single frame."""
        if getattr(self, "history_frames", 0) == 0:
            raise ValueError("The model was not configured with history frames (history_frames=0).")
        batch_size, num_frames = frames.shape[:2]
        if num_frames != self.history_frames:
            raise ValueError(f"Expected {self.history_frames} history frames, got {num_frames}.")
        frame_tokens = self._apply_checkpoint(
            self.paligemma_with_expert.embed_image, frames.reshape(batch_size * num_frames, *frames.shape[2:])
        )
        frame_tokens = frame_tokens.reshape(batch_size, num_frames, *frame_tokens.shape[1:]).to(torch.float32)
        frame_tokens = frame_tokens + self.history_time_embedding[None, :, None, :].to(torch.float32)
        weights = masks[..., None, None].to(torch.float32)
        pooled = (frame_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        has_history = masks.any(dim=1)[:, None, None].to(torch.float32)
        # the gate is initialized to zero, so that the model initially behaves like a model without history
        projected = self.history_proj(pooled.to(self.history_proj.weight.dtype)).to(torch.float32)
        return torch.tanh(self.history_gate.to(torch.float32)) * projected * has_history

    def _embed_prefix(self, obs):
        """Returns the embeddings, padding masks, attention masks and the (start, length) of the language tokens."""
        embs = []
        pad_masks = []
        att_masks = []
        bsize = obs.state.shape[0]

        history_images = getattr(obs, "history_images", None)
        for name, img in obs.images.items():
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            if history_images is not None and name in history_images:
                history = self._embed_history(history_images[name], obs.history_image_masks[name])
                img_emb = img_emb + history.to(img_emb.dtype)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(obs.image_masks[name][:, None].expand(bsize, num_img_embs))
            # image tokens attend to each other
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, obs.tokenized_prompt)
        text_range = (len(att_masks), lang_emb.shape[1])
        embs.append(lang_emb)
        pad_masks.append(obs.tokenized_prompt_mask)
        # full attention between image and language inputs
        att_masks += [0] * lang_emb.shape[1]

        # subgoal images are a separate block: they attend to the images, language and each other, but the images
        # and language do not attend to them
        subgoal_images = getattr(obs, "subgoal_images", None)
        if subgoal_images is not None:
            for i, (name, img) in enumerate(subgoal_images.items()):
                subgoal_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
                embs.append(subgoal_emb)
                pad_masks.append(obs.subgoal_image_masks[name][:, None].expand(bsize, subgoal_emb.shape[1]))
                att_masks += [int(i == 0)] + [0] * (subgoal_emb.shape[1] - 1)

        embs = torch.cat([e.to(embs[0].dtype) for e in embs], dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks)).clone()

        # language tokens with their own autoregressive mask (e.g. FAST tokens or the subtask of the high-level policy)
        if getattr(obs, "token_ar_mask", None) is not None:
            start, length = text_range
            att_masks[:, start : start + length] = obs.token_ar_mask.to(torch.bool)
        return embs, pad_masks, att_masks, text_range


def text_token_loss(decode, text_out, tokens, loss_mask, max_tokens):
    """Next token prediction loss on the language tokens with a loss mask, decoding at most `max_tokens` positions."""
    loss_mask = loss_mask[:, 1:].to(torch.bool)
    targets = tokens[:, 1:]
    order = torch.argsort((~loss_mask).to(torch.int32), dim=-1, stable=True)[:, :max_tokens]
    selected_mask = torch.gather(loss_mask, 1, order).to(torch.float32)
    selected_targets = torch.gather(targets, 1, order)
    selected_out = torch.gather(text_out[:, :-1], 1, order[..., None].expand(-1, -1, text_out.shape[-1]))
    log_probs = F.log_softmax(decode(selected_out), dim=-1)
    token_log_probs = torch.gather(log_probs, 2, selected_targets[..., None].long())[..., 0]
    return -(token_log_probs * selected_mask).sum(dim=-1) / selected_mask.sum(dim=-1).clamp(min=1.0)


class PI0Pytorch(PrefixEmbeddingMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.rtc_max_delay = getattr(config, "rtc_max_delay", 0)
        self.history_frames = getattr(config, "history_frames", 0)
        self.knowledge_insulation = getattr(config, "knowledge_insulation", False)
        self.ki_max_fast_tokens = getattr(config, "ki_max_fast_tokens", 128)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            knowledge_insulation=self.knowledge_insulation,
        )

        if self.history_frames > 0:
            self.history_time_embedding = nn.Parameter(torch.zeros(self.history_frames, paligemma_config.width))
            self.history_proj = nn.Linear(paligemma_config.width, paligemma_config.width)
            self.history_gate = nn.Parameter(torch.zeros(()))

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        return _preprocessing.preprocess_observation_pytorch(observation, train=train)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        obs = types.SimpleNamespace(
            images=dict(enumerate(images)),
            image_masks=dict(enumerate(img_masks)),
            state=lang_tokens,
            tokenized_prompt=lang_tokens,
            tokenized_prompt_mask=lang_masks,
        )
        embs, pad_masks, att_masks, _ = self._embed_prefix(obs)
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        The flow matching timestep is either shared by all actions ([B]) or given per action ([B, action_horizon]),
        which is used for real-time chunking.
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep.reshape(-1),
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).reshape(*timestep.shape, -1)
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            if time_emb.ndim == 2:
                time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb.to(action_emb.dtype)], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=noisy_actions.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        obs = self._preprocess_observation(observation, train=True)
        bsize, action_horizon = actions.shape[:2]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(bsize, actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # real-time chunking: the first `delay` actions are given (clean, with time 0) and excluded from the loss
        action_time = time
        loss_weights = None
        if self.rtc_max_delay > 0:
            delay = torch.randint(0, self.rtc_max_delay + 1, (bsize,), device=actions.device)
            committed = torch.arange(action_horizon, device=actions.device)[None, :] < delay[:, None]
            x_t = torch.where(committed[..., None], actions, x_t)
            action_time = torch.where(committed, torch.zeros_like(x_t[..., 0]), time[:, None])
            # normalize so that the mean over the chunk is the mean over the predicted actions
            loss_weights = (~committed).to(torch.float32) * (action_horizon / (action_horizon - delay).clamp(min=1))[
                :, None
            ].to(torch.float32)

        prefix_embs, prefix_pad_masks, prefix_att_masks, (text_start, text_len) = self._embed_prefix(obs)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(obs.state, x_t, action_time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks.to(torch.bool)], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # knowledge insulation: the prompt contains the FAST tokens of the action chunk, which are only predicted by the
        # backbone. They are not available at inference time, so no other token attends to them and the positions of
        # the following tokens are the same as without them.
        use_ki = self.knowledge_insulation and obs.token_loss_mask is not None
        if use_ki:
            fast = torch.zeros_like(pad_masks)
            fast[:, text_start : text_start + text_len] = obs.token_ar_mask.to(torch.bool)
            fast = fast & pad_masks
            att_2d_masks = make_att_2d_masks_with_fast(pad_masks, att_masks, fast)
            position_ids = torch.where(fast, position_ids, torch.cumsum(pad_masks & ~fast, dim=1) - 1)

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")
        if loss_weights is not None:
            loss = loss * loss_weights[..., None]
        if use_ki:
            text_out = prefix_out[:, text_start : text_start + text_len]
            fast_loss = text_token_loss(
                self.paligemma_with_expert.decode,
                text_out,
                obs.tokenized_prompt,
                obs.token_loss_mask,
                self.ki_max_fast_tokens,
            )
            loss = loss + fast_loss[:, None, None]
        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        cfg_scale=1.0,
        action_prefix=None,
        action_prefix_len=None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)

        If the observation contains an unconditional prompt, classifier-free guidance is applied with
        v = v_uncond + cfg_scale * (v_cond - v_uncond). `action_prefix` (with `action_prefix_len` valid steps) contains
        already committed actions (real-time chunking), which are kept fixed.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        obs = self._preprocess_observation(observation, train=False)
        use_cfg = obs.tokenized_prompt_uncond is not None
        if use_cfg:
            obs = stack_cfg_observation(obs)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(obs)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        committed = None
        if action_prefix is not None:
            if action_prefix_len is None:
                action_prefix_len = torch.full((bsize,), self.config.action_horizon, device=device)
            committed = torch.arange(self.config.action_horizon, device=device)[None, :] < action_prefix_len[:, None]
            noise = torch.where(committed[..., None], action_prefix.to(noise.dtype), noise)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            action_time = time.expand(bsize)
            if committed is not None:
                action_time = torch.where(committed, torch.zeros_like(x_t[..., 0]), action_time[:, None])
            x_in = torch.cat([x_t, x_t], dim=0) if use_cfg else x_t
            time_in = torch.cat([action_time, action_time], dim=0) if use_cfg else action_time
            v_t = self.denoise_step(
                obs.state,
                prefix_pad_masks,
                past_key_values,
                x_in,
                time_in,
            )
            if use_cfg:
                v_cond, v_uncond = torch.chunk(v_t, 2, dim=0)
                v_t = v_uncond + cfg_scale * (v_cond - v_uncond)
            if committed is not None:
                # the committed actions stay fixed
                v_t = torch.where(committed[..., None], torch.zeros_like(v_t), v_t)

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
