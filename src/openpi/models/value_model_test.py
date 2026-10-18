import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.models import value_model
from openpi.training import weight_loaders


def test_value_model():
    key = jax.random.key(0)
    config = value_model.ValueConfig(paligemma_variant="dummy")
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    obs = obs.replace(value_target=jnp.array([-0.25, 0.0]))
    loss = model.compute_loss(key, obs, config.fake_act(batch_size), train=True)
    assert loss.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(loss))

    value = model.predict_value(obs)
    assert value.shape == (batch_size,)
    assert jnp.all((value >= -1.0) & (value <= 0.0))


def test_value_model_loads_pi05_backbone():
    pi05_params = nnx.state(
        nnx.eval_shape(
            pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy").create,
            jax.random.key(0),
        )
    ).to_pure_dict()
    value_params = nnx.state(
        nnx.eval_shape(value_model.ValueConfig(paligemma_variant="dummy").create, jax.random.key(0))
    ).to_pure_dict()

    merged = weight_loaders._merge_params(pi05_params, value_params, missing_regex=".*value_head.*")  # noqa: SLF001
    # All backbone weights come from the pi05 checkpoint and only the value head is missing.
    assert jax.tree.structure(merged) == jax.tree.structure(value_params)
    assert set(merged) == {"PaliGemma", "value_head"}
