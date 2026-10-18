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


def _dummy_pi05(**kwargs):
    return pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", **kwargs)


def test_pi0_rtc():
    key = jax.random.key(0)
    config = _dummy_pi05(rtc_max_delay=3)
    model = config.create(key)
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))

    prefix = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    prefix_len = jnp.array([2, 0])
    actions = nnx_utils.module_jit(model.sample_actions)(
        key, obs, num_steps=2, action_prefix=prefix, action_prefix_len=prefix_len
    )
    # The committed actions are kept.
    assert jnp.allclose(actions[0, :2], prefix[0, :2])
    assert not jnp.allclose(actions[1, :2], prefix[1, :2])


def test_pi0_history():
    key = jax.random.key(0)
    config = _dummy_pi05(history_frames=2)
    model = config.create(key)
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    history_obs = dataclasses.replace(
        obs,
        history_images={k: jnp.stack([v, -v], axis=1) for k, v in obs.images.items()},
        history_image_masks={k: jnp.ones((batch_size, 2), dtype=bool) for k in obs.images},
    )
    loss = model.compute_loss(key, history_obs, act, train=True)
    assert loss.shape == (batch_size, config.action_horizon)

    # The history gate is initialized to zero, so the model initially ignores the history.
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    noise = jax.random.normal(key, (batch_size, config.action_horizon, config.action_dim))
    assert jnp.allclose(
        sample_actions(key, obs, num_steps=2, noise=noise),
        sample_actions(key, history_obs, num_steps=2, noise=noise),
        atol=1e-2,
    )


def _ki_observation(obs, num_fast_tokens):
    length = obs.tokenized_prompt.shape[1]
    is_fast = jnp.broadcast_to(jnp.arange(length) >= length - num_fast_tokens, obs.tokenized_prompt.shape)
    return dataclasses.replace(
        obs,
        tokenized_prompt=jnp.where(is_fast, 7, obs.tokenized_prompt),
        token_ar_mask=is_fast.astype(jnp.int32),
        token_loss_mask=jnp.zeros_like(is_fast),
    )


def _grad_norms(model, obs, act):
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)

    def loss_fn(state):
        return jnp.mean(nnx.merge(graphdef, state, rest).compute_loss(jax.random.key(1), obs, act))

    grads = jax.grad(loss_fn)(state)
    flat = {"/".join(map(str, k)): float(jnp.sum(jnp.abs(v.value))) for k, v in grads.flat_state().items()}
    backbone = sum(v for k, v in flat.items() if "img" in k or ("llm" in k and "_1" not in k))
    expert = sum(v for k, v in flat.items() if "llm" in k and "_1" in k)
    return backbone, expert


def _perturbed(model, key):
    # At initialization, the adaRMS gates of the action expert are zero and it ignores the backbone.
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(key, len(leaves))
    leaves = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys, strict=True)]
    return nnx.merge(graphdef, jax.tree.unflatten(treedef, leaves), rest)


def test_pi0_knowledge_insulation():
    key = jax.random.key(0)
    batch_size = 2
    config = _dummy_pi05(knowledge_insulation=True)
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    obs = _ki_observation(obs, num_fast_tokens=5)

    # Without a FAST token loss, the action loss does not update the backbone.
    backbone, expert = _grad_norms(_perturbed(config.create(key), key), obs, act)
    assert backbone == 0.0
    assert expert > 0.0

    # Without knowledge insulation, it does.
    backbone, _ = _grad_norms(_perturbed(_dummy_pi05().create(key), key), obs, act)
    assert backbone > 0.0

    # With the FAST token loss, the backbone is trained by the cross-entropy loss.
    obs = dataclasses.replace(obs, token_loss_mask=obs.token_ar_mask.astype(bool))
    model = _perturbed(config.create(key), key)
    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))
    backbone, _ = _grad_norms(model, obs, act)
    assert backbone > 0.0


def test_subtask_policy():
    from openpi.models import subtask_policy

    key = jax.random.key(0)
    config = subtask_policy.SubtaskPolicyConfig(paligemma_variant="dummy")
    model = config.create(key)
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size,)
    tokens = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert tokens.shape[0] == batch_size
