import types

import pytest
import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models import subtask_policy
from openpi.models import value_model
from openpi.models_pytorch import factory

BATCH_SIZE = 2
PROMPT_LEN = 12


def _config(**kwargs):
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_horizon=4,
        max_token_len=PROMPT_LEN,
        pytorch_compile_mode=None,
        **kwargs,
    )


def _observation(**kwargs):
    torch.manual_seed(0)
    images = {
        k: torch.rand(BATCH_SIZE, 3, 224, 224) * 2 - 1 for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    fields = {
        "images": images,
        "image_masks": {k: torch.ones(BATCH_SIZE, dtype=torch.bool) for k in images},
        "state": torch.zeros(BATCH_SIZE, 32),
        "tokenized_prompt": torch.randint(0, 1000, (BATCH_SIZE, PROMPT_LEN)),
        "tokenized_prompt_mask": torch.ones(BATCH_SIZE, PROMPT_LEN, dtype=torch.bool),
        "tokenized_prompt_uncond": None,
        "tokenized_prompt_uncond_mask": None,
        "value_target": None,
        "subgoal_images": None,
        "subgoal_image_masks": None,
        "history_images": None,
        "history_image_masks": None,
        "token_ar_mask": None,
        "token_loss_mask": None,
    }
    fields.update(kwargs)
    return types.SimpleNamespace(**fields)


def _with_fast_tokens(obs, num_fast_tokens=4, *, loss=True):
    is_fast = torch.arange(PROMPT_LEN)[None].expand(BATCH_SIZE, -1) >= PROMPT_LEN - num_fast_tokens
    return types.SimpleNamespace(
        **{**vars(obs), "token_ar_mask": is_fast.to(torch.int32), "token_loss_mask": is_fast & loss}
    )


def _perturb(model):
    # At initialization, the adaRMS gates of the action expert are zero and it ignores the backbone.
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


def _grad_sums(model, obs, actions):
    model.zero_grad()
    model(obs, actions).mean().backward()
    backbone = sum(
        p.grad.abs().sum().item() for n, p in model.named_parameters() if ".paligemma." in n and p.grad is not None
    )
    expert = sum(
        p.grad.abs().sum().item() for n, p in model.named_parameters() if "gemma_expert" in n and p.grad is not None
    )
    return backbone, expert


def test_forward_with_all_features():
    config = _config(rtc_max_delay=2, history_frames=2, knowledge_insulation=True)
    model = factory.create_model(config)
    obs = _observation()
    base = obs.images["base_0_rgb"]
    obs.history_images = {"base_0_rgb": torch.stack([base, -base], dim=1)}
    obs.history_image_masks = {"base_0_rgb": torch.tensor([[True, True], [False, True]])}
    obs.subgoal_images = {"base_0_rgb": base}
    obs.subgoal_image_masks = {"base_0_rgb": torch.tensor([True, False])}
    obs = _with_fast_tokens(obs)
    actions = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim)
    with torch.no_grad():
        loss = model(obs, actions)
    assert loss.shape == (BATCH_SIZE, config.action_horizon, config.action_dim)
    assert torch.isfinite(loss).all()


def test_knowledge_insulation():
    obs = _with_fast_tokens(_observation(), loss=False)
    actions = torch.randn(BATCH_SIZE, 4, 32)

    # Without a FAST token loss, the action loss does not update the backbone.
    backbone, expert = _grad_sums(_perturb(factory.create_model(_config(knowledge_insulation=True))), obs, actions)
    assert backbone == 0.0
    assert expert > 0.0

    # Without knowledge insulation, it does.
    backbone, _ = _grad_sums(_perturb(factory.create_model(_config())), obs, actions)
    assert backbone > 0.0

    # With the FAST token loss, the backbone is trained by the cross-entropy loss.
    obs = _with_fast_tokens(_observation(), loss=True)
    backbone, _ = _grad_sums(_perturb(factory.create_model(_config(knowledge_insulation=True))), obs, actions)
    assert backbone > 0.0


def test_sample_actions():
    config = _config(rtc_max_delay=2, history_frames=2)
    model = factory.create_model(config).eval()
    obs = _observation()
    noise = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim)
    actions = model.sample_actions("cpu", obs, noise=noise, num_steps=2)
    assert actions.shape == noise.shape

    # Real-time chunking: the committed actions are kept.
    prefix = torch.randn_like(noise)
    rtc_actions = model.sample_actions(
        "cpu", obs, noise=noise, num_steps=2, action_prefix=prefix, action_prefix_len=torch.tensor([2, 0])
    )
    assert torch.allclose(rtc_actions[0, :2], prefix[0, :2])
    assert not torch.allclose(rtc_actions[1, :2], prefix[1, :2])

    # Classifier-free guidance with identical prompts has no effect.
    cfg_obs = _observation(
        tokenized_prompt_uncond=obs.tokenized_prompt, tokenized_prompt_uncond_mask=obs.tokenized_prompt_mask
    )
    cfg_actions = model.sample_actions("cpu", cfg_obs, noise=noise, num_steps=2, cfg_scale=2.0)
    assert torch.allclose(actions, cfg_actions, atol=1e-4)

    # The history gate is initialized to zero, so the model initially ignores the history.
    base = obs.images["base_0_rgb"]
    history_obs = _observation(
        history_images={"base_0_rgb": torch.stack([base, -base], dim=1)},
        history_image_masks={"base_0_rgb": torch.ones(BATCH_SIZE, 2, dtype=torch.bool)},
        subgoal_images={"base_0_rgb": obs.images["base_0_rgb"]},
        subgoal_image_masks={"base_0_rgb": torch.zeros(BATCH_SIZE, dtype=torch.bool)},
    )
    # A fully masked subgoal is also equivalent to no subgoal.
    assert torch.allclose(actions, model.sample_actions("cpu", history_obs, noise=noise, num_steps=2), atol=1e-4)


def test_value_model(tmp_path):
    # The value function is initialized from the backbone of a pi05 checkpoint.
    pi05 = factory.create_model(_config())
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(pi05, str(path))

    config = value_model.ValueConfig(paligemma_variant="dummy", dtype="float32", max_token_len=PROMPT_LEN)
    model = factory.create_model(config)
    factory.load_pretrained_weights(model, str(path))
    name = "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(model.state_dict()[name], pi05.state_dict()[name])

    obs = _observation(value_target=torch.tensor([-0.5, 0.0]))
    loss = model(obs)
    assert loss.shape == (BATCH_SIZE,)
    value = model.predict_value(obs)
    assert ((value >= -1) & (value <= 0)).all()

    # Missing weights that are not allowed raise an error.
    with pytest.raises(ValueError, match="missing"):
        factory.load_pretrained_weights(model, str(path), allowed_missing=())


def test_subtask_policy():
    config = subtask_policy.SubtaskPolicyConfig(paligemma_variant="dummy", dtype="float32", max_token_len=PROMPT_LEN)
    model = factory.create_model(config)
    obs = _with_fast_tokens(_observation())
    loss = model(obs)
    assert loss.shape == (BATCH_SIZE,)
    assert torch.isfinite(loss).all()

    obs = _observation(token_ar_mask=torch.zeros(BATCH_SIZE, PROMPT_LEN, dtype=torch.int32))
    tokens = model.eval().sample_actions("cpu", obs, max_decoding_steps=3)
    assert tokens.shape == (BATCH_SIZE, 3)
