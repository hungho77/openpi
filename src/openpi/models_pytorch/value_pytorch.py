"""PyTorch version of the RECAP value function (see `openpi.models.value_model`)."""

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaBackboneModel
from openpi.models_pytorch.pi0_pytorch import PrefixEmbeddingMixin
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


class ValuePytorch(PrefixEmbeddingMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bins = config.num_bins
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Same parameter names as pi0, so that the backbone can be loaded from a pi0 / pi05 checkpoint.
        self.paligemma_with_expert = PaliGemmaBackboneModel(paligemma_config, precision=config.dtype)
        self.value_head = nn.Linear(paligemma_config.width, config.num_bins)
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

    @property
    def bin_values(self) -> torch.Tensor:
        return torch.linspace(-1.0, 0.0, self.num_bins, device=self.value_head.weight.device)

    def value_logits(self, obs) -> torch.Tensor:
        embs, pad_masks, att_masks, _ = self._embed_prefix(obs)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        attention_mask = torch.where(att_2d_masks[:, None], 0.0, -2.3819763e38).to(embs.dtype)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        out = self._apply_checkpoint(self.paligemma_with_expert.forward, embs, attention_mask, position_ids)
        # Mean pooling over the valid tokens.
        weights = pad_masks[..., None].to(torch.float32)
        pooled = (out.to(torch.float32) * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.value_head(pooled.to(self.value_head.weight.dtype)).to(torch.float32)

    def forward(self, observation, actions=None) -> torch.Tensor:
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=True)
        if obs.value_target is None:
            raise ValueError("The value function requires `value_target` in the observation.")
        logits = self.value_logits(obs)
        target_bins = torch.round((obs.value_target.clamp(-1.0, 0.0) + 1.0) * (self.num_bins - 1)).long()
        return F.cross_entropy(logits, target_bins, reduction="none")

    @torch.no_grad()
    def predict_value(self, observation) -> torch.Tensor:
        """Returns the expected value under the predicted distribution."""
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        probs = F.softmax(self.value_logits(obs), dim=-1)
        return (probs * self.bin_values).sum(dim=-1)

    def sample_actions(self, device, observation, **kwargs):
        raise NotImplementedError("The value function does not predict actions, use `predict_value` instead.")
