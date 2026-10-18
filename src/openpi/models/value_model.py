"""Distributional value function used by RECAP.

The value function shares the PaliGemma backbone (SigLIP + Gemma) with pi0 / pi05 so that it can be initialized from a
pi05 checkpoint. It reads the same inputs as the policy prefix (images, prompt and discretized state), mean-pools the
final backbone activations and predicts a categorical distribution over `num_bins` values evenly spaced in [-1, 0].
The value is trained with a cross-entropy loss against the discretized normalized return (see
`openpi.transforms.ValueTarget`).
"""

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


@dataclasses.dataclass(frozen=True)
class ValueConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Number of bins of the categorical value distribution.
    num_bins: int = 201

    # These are only used to build the inputs. The value function does not predict actions, so a single action step
    # is enough to keep the data pipeline cheap.
    action_dim: int = 32
    action_horizon: int = 1
    max_token_len: int = 200
    # Read by the ModelTransformFactory, same as for pi05.
    discrete_state_input: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.VALUE

    @override
    def create(self, rng: at.KeyArrayLike) -> "ValueModel":
        return ValueModel(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(_model.IMAGE_KEYS, image_spec),
                image_masks=dict.fromkeys(_model.IMAGE_KEYS, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                value_target=jax.ShapeDtypeStruct([batch_size], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freezes the backbone (except for the LoRA weights) when a LoRA variant is used."""
        if "lora" not in self.paligemma_variant:
            return nnx.Nothing
        return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))


class ValueModel(_model.BaseModel):
    def __init__(self, config: ValueConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.num_bins = config.num_bins
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        llm = nnx_bridge.ToNNX(_gemma.Module(configs=[paligemma_config], embed_dtype=config.dtype))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
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
        # Same parameter names as pi0, so that the backbone can be loaded from a pi0 / pi05 checkpoint.
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.value_head = nnx.Linear(paligemma_config.width, config.num_bins, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @property
    def bin_values(self) -> at.Float[at.Array, " n"]:
        return jnp.linspace(-1.0, 0.0, self.num_bins)

    def value_logits(self, observation: _model.Observation) -> at.Float[at.Array, "b n"]:
        # The prefix is embedded exactly like in pi0.
        prefix_tokens, prefix_mask, prefix_ar_mask = _pi0.Pi0.embed_prefix(self, observation)  # type: ignore[arg-type]
        attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out,), _ = self.PaliGemma.llm([prefix_tokens], mask=attn_mask, positions=positions)

        # Mean pooling over the valid tokens.
        weights = prefix_mask[..., None].astype(jnp.float32)
        pooled = jnp.sum(prefix_out.astype(jnp.float32) * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        return self.value_head(pooled)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        if observation.value_target is None:
            raise ValueError("The value function requires `value_target` in the observation.")
        observation = _model.preprocess_observation(rng, observation, train=train)
        logits = self.value_logits(observation)
        target_bins = jnp.round((jnp.clip(observation.value_target, -1.0, 0.0) + 1.0) * (self.num_bins - 1))
        target_bins = target_bins.astype(jnp.int32)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(log_probs, target_bins[..., None], axis=-1)[..., 0]

    def predict_value(self, observation: _model.Observation) -> at.Float[at.Array, " b"]:
        """Returns the expected value under the predicted distribution."""
        observation = _model.preprocess_observation(None, observation, train=False)
        probs = jax.nn.softmax(self.value_logits(observation).astype(jnp.float32), axis=-1)
        return jnp.sum(probs * self.bin_values, axis=-1)

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        raise NotImplementedError("The value function does not predict actions, use `predict_value` instead.")
