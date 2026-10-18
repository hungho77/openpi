# PyTorch Training of the π0.7-Style Features

All features described in [pi07.md](pi07.md) are also available in PyTorch (`scripts/train_pytorch.py`, DDP):

- episode metadata prompting and classifier-free guidance;
- RECAP: value function, advantages and the advantage-conditioned policy;
- subgoal images;
- training-time real-time chunking;
- memory (past frames);
- knowledge insulation;
- the high-level policy that predicts the next subtask.

The configs are the same as for JAX, but some config fields work differently in PyTorch:

| Config field | In PyTorch |
|---|---|
| `pytorch_weight_path` | The initial weights, a converted PyTorch checkpoint. This replaces `weight_loader`. |
| `pytorch_training_precision` | Use `"bfloat16"` (the default) or `"float32"`. |
| LoRA, `freeze_filter`, EMA | Not supported. PyTorch always does a full fine-tune. |
| `fsdp_devices` | Not supported. Each GPU holds a full copy of the model (DDP). |

## 0. Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

The patched Gemma file supports per-token adaRMS conditioning, which real-time chunking needs. **Copy it again after
every update of the repository**, even if transformers was already patched for an older version of openpi.

## 1. Convert pi05_base to PyTorch

```bash
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
uv run examples/convert_jax_model_to_pytorch.py --config_name pi05_libero \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

Use a plain π0.5 config for the conversion, such as `pi05_libero`. The checkpoint path must contain "pi05".

The same converted checkpoint initializes all three models:

- **Policy:** the history encoder starts from its initialization. Its gate is zero, so the model initially behaves like
  π0.5.
- **Value function:** only the backbone is loaded.
- **High-level policy:** only the backbone is loaded.

Weights that are missing from the checkpoint are reported in the log, and unexpected missing weights raise an error.

## 2. Configs

Set `pytorch_weight_path` in each training config:

```python
TrainConfig(
    name="pi05_myrobot",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=50, max_token_len=300,
                               rtc_max_delay=4, history_frames=2, knowledge_insulation=True),
    data=...,  # episode_metadata / subgoal as for JAX
    pytorch_weight_path="/home/<user>/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
    pytorch_training_precision="bfloat16",
    batch_size=32,  # total over all GPUs
    num_train_steps=30_000,
)
```

The value function (`value_model.ValueConfig()`) and the high-level policy (`subtask_policy.SubtaskPolicyConfig()`) use
the same `pytorch_weight_path`.

Run `scripts/compute_norm_stats.py --config-name <config>` for each config, as for JAX.

## 3. Multi-GPU training

On regular GPUs:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py <config> --exp_name <run>
```

On **MIG instances** (for example 2× H100 MIG 3g.40gb in one Slurm job), every process may only see its own instance.
`scripts/per_device.sh` sets `CUDA_VISIBLE_DEVICES` per rank:

```bash
ALL_DEVICES=$CUDA_VISIBLE_DEVICES uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 --no-python \
    scripts/per_device.sh python scripts/train_pytorch.py <config> --exp_name <run>
```

The MIG instances must be on different physical GPUs. Check this with `nvidia-smi -L`, and verify NCCL with a short
all-reduce before a long run.

Always start with a short run, such as `--num_train_steps 20`, to check the memory usage. The memory usage of the first
steps is logged. If the GPUs run out of memory, try these in order:

1. Reduce `batch_size`.
2. Reduce `ki_max_fast_tokens`.
3. Disable knowledge insulation.
4. Remove `history_frames` or `subgoal`.

## 4. RECAP and the high-level policy

The steps are the same as in [recap.md](recap.md), with `scripts/train_pytorch.py` instead of `scripts/train.py`.
`scripts/compute_advantages.py` detects PyTorch checkpoints (`model.safetensors`) automatically:

```bash
<torchrun ...> scripts/train_pytorch.py value_myrobot --exp_name v0
uv run scripts/compute_advantages.py --config-name value_myrobot \
    --checkpoint-dir checkpoints/value_myrobot/v0/<step> --output assets/value_myrobot/advantages.npz
<torchrun ...> scripts/train_pytorch.py pi05_myrobot_recap --exp_name r1
<torchrun ...> scripts/train_pytorch.py subtask_myrobot --exp_name h1
```

The PyTorch high-level policy decodes greedily without a KV cache, so each new token runs a full forward pass. This is
fine for a policy that is queried only when the subtask may change. Pass `max_decoding_steps` in `sample_kwargs` to
limit the length of the subtask.

## 5. Inference

`scripts/serve_policy.py` and `create_trained_policy` detect PyTorch checkpoints automatically. The inputs are the
same as for JAX: `subtask`, `history`, `subgoal`, `actions` (the action prefix for real-time chunking) and
`previous_subtasks`.

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_myrobot \
    --policy.dir=checkpoints/pi05_myrobot/<run>/<step>
```
