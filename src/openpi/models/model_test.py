import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_sample_actions_cfg():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)
    noise = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    actions = sample_actions(key, obs, num_steps=2, noise=noise)

    uncond_obs = dataclasses.replace(
        obs, tokenized_prompt_uncond=obs.tokenized_prompt, tokenized_prompt_uncond_mask=obs.tokenized_prompt_mask
    )
    cfg_actions = sample_actions(key, uncond_obs, num_steps=2, noise=noise, cfg_scale=2.0)
    assert cfg_actions.shape == (batch_size, model.action_horizon, model.action_dim)
    # With identical prompts the guidance has no effect.
    assert jnp.allclose(actions, cfg_actions, atol=1e-2)

    # With cfg_scale=0 only the unconditional prompt is used.
    uncond_tokens = jnp.zeros_like(obs.tokenized_prompt)
    uncond_actions = sample_actions(
        key, dataclasses.replace(obs, tokenized_prompt=uncond_tokens), num_steps=2, noise=noise
    )
    other_obs = dataclasses.replace(
        obs, tokenized_prompt_uncond=uncond_tokens, tokenized_prompt_uncond_mask=obs.tokenized_prompt_mask
    )
    cfg_actions = sample_actions(key, other_obs, num_steps=2, noise=noise, cfg_scale=0.0)
    assert jnp.allclose(uncond_actions, cfg_actions, atol=1e-2)


def test_pi0_subgoal_images():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    subgoal_obs = dataclasses.replace(
        obs,
        subgoal_images=obs.images,
        subgoal_image_masks={k: jnp.array([True, False]) for k in obs.images},
    )
    loss = model.compute_loss(key, subgoal_obs, act, train=True)
    assert loss.shape == (batch_size, config.action_horizon)

    sample_actions = nnx_utils.module_jit(model.sample_actions)
    noise = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    actions = sample_actions(key, obs, num_steps=2, noise=noise)
    subgoal_actions = sample_actions(key, subgoal_obs, num_steps=2, noise=noise)
    assert subgoal_actions.shape == actions.shape
    # A fully masked subgoal is equivalent to no subgoal.
    assert jnp.allclose(actions[1], subgoal_actions[1], atol=1e-2)
