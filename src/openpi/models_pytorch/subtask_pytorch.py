"""PyTorch version of the high-level policy that predicts the next subtask (see `openpi.models.subtask_policy`).

The PaliGemma backbone reads the images, the task and the previous subtasks, and generates the subtask text
autoregressively. It can be initialized from a pi05 checkpoint.
"""

import types

import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaBackboneModel
from openpi.models_pytorch.pi0_pytorch import PrefixEmbeddingMixin
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import text_token_loss
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

PALIGEMMA_EOS_TOKEN = 1


class SubtaskPytorch(PrefixEmbeddingMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.paligemma_with_expert = PaliGemmaBackboneModel(paligemma_config, precision=config.dtype)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _forward_text(self, obs):
        """Returns the outputs at the language tokens."""
        embs, pad_masks, att_masks, (text_start, text_len) = self._embed_prefix(obs)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        attention_mask = torch.where(att_2d_masks[:, None], 0.0, -2.3819763e38).to(embs.dtype)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        out = self._apply_checkpoint(self.paligemma_with_expert.forward, embs, attention_mask, position_ids)
        return out[:, text_start : text_start + text_len]

    def forward(self, observation, actions=None) -> torch.Tensor:
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=True)
        text_out = self._forward_text(obs)
        return text_token_loss(
            self.paligemma_with_expert.decode,
            text_out,
            obs.tokenized_prompt,
            obs.token_loss_mask,
            obs.tokenized_prompt.shape[1],
        )

    @torch.no_grad()
    def sample_actions(self, device, observation, max_decoding_steps=48, **kwargs) -> torch.Tensor:
        """Greedily generates the subtask tokens. Returns the generated tokens ([B, max_decoding_steps])."""
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        tokens, mask = obs.tokenized_prompt, obs.tokenized_prompt_mask.to(torch.bool)
        bsize = tokens.shape[0]
        lengths = mask.sum(dim=1)
        # Room for the generated tokens after the prompt.
        tokens = torch.cat([tokens, torch.zeros(bsize, max_decoding_steps, dtype=tokens.dtype, device=device)], 1)
        mask = torch.cat([mask, torch.zeros(bsize, max_decoding_steps, dtype=torch.bool, device=device)], 1)
        ar_mask = torch.zeros_like(mask, dtype=torch.int32)
        output = torch.zeros(bsize, max_decoding_steps, dtype=torch.long, device=device)
        done = torch.zeros(bsize, dtype=torch.bool, device=device)
        batch = torch.arange(bsize, device=device)
        for step in range(max_decoding_steps):
            step_obs = types.SimpleNamespace(
                **{**vars(obs), "tokenized_prompt": tokens, "tokenized_prompt_mask": mask, "token_ar_mask": ar_mask}
            )
            text_out = self._forward_text(step_obs)
            last = text_out[batch, lengths + step - 1]
            next_token = self.paligemma_with_expert.decode(last[:, None])[:, 0].argmax(dim=-1)
            next_token = torch.where(done, torch.full_like(next_token, PALIGEMMA_EOS_TOKEN), next_token)
            output[:, step] = next_token
            tokens[batch, lengths + step] = next_token.to(tokens.dtype)
            mask[batch, lengths + step] = True
            ar_mask[batch, lengths + step] = 1
            done = done | (next_token == PALIGEMMA_EOS_TOKEN)
            if bool(done.all()):
                break
        return output
