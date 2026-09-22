# π₀.₅ finetune — SO101 single-arm multitask

Full recipe for the checkpoint published at
[`twanghcmut/pi05-SO101-Multitask`](https://huggingface.co/twanghcmut/pi05-SO101-Multitask),
trained on [`hungho77/so101-multitask`](https://huggingface.co/datasets/hungho77/so101-multitask)
with **this repo's own PyTorch trainer** (`scripts/train_pytorch.py`), not a third-party port.

One checkpoint covers all three tasks; the language prompt selects the behaviour.

| | |
|---|---|
| Base | `gs://openpi-assets/checkpoints/pi05_base` → converted to PyTorch |
| Config | `pi05_so101` (`src/openpi/training/config.py`) |
| Batch / steps | 128 / 6,000 = 11.4 epochs over 67,496 frames |
| Precision | bfloat16 (`pytorch_training_precision` default) |
| Hardware | 1× H100 80GB, peak 71 GB, 100% util |
| Runtime | 13.8 h at 8.28 s/step |
| Final loss | 0.00302 (mean of steps 5,751–6,000) |

## What this branch adds

| file | why |
|---|---|
| `src/openpi/policies/so101_policy.py` | `SO101Inputs`/`SO101Outputs` — camera mapping and the 6-dim action trim |
| `src/openpi/training/config.py` | `LeRobotSO101DataConfig` + the `pi05_so101` `TrainConfig` |
| `scripts/eval_so101_open_loop.py` | open-loop eval, mirroring `gr00t/eval/open_loop_eval.py` so the numbers are comparable |
| `run_so101.sh` | the exact launcher used for the published run |

## Three decisions that are not obvious

**1. `action_sequence_keys = ("action",)`.** This dataset's column is `action`, the LeRobot
convention. `DataConfig` defaults to `"actions"` (what `physical-intelligence/libero` happens
to use). Getting it wrong fails late, inside the dataset query, with
`KeyError: "Column actions not in the dataset"`.

**2. Delta actions, gripper absolute.** The dataset records *absolute* joint targets, unlike
LIBERO whose actions are already deltas. pi0/pi0.5 are trained on deltas from the chunk's
first state, so `DeltaActions(make_bool_mask(5, -1))` is enabled — 5 arm joints as deltas, the
gripper left absolute, because an open/close command is a target and not a displacement.
`AbsoluteActions` on the output side converts back, so `infer()` returns absolute targets.

**3. LeRobot v2.1, not v3.0.** The Hub copy is v3.0 but openpi pins lerobot at
`0cf8648` (2025-05-28), which only reads v2.1. Convert once and point `HF_LEROBOT_HOME` at
the result — the `repo_id` then resolves locally and nothing is re-downloaded.

## Setup

```bash
cd openpi
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_multitask_v21
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export HF_TOKEN=$(python -c "from huggingface_hub import get_token;print(get_token())")

# 1. dataset v3.0 -> v2.1  (run from the Isaac-GR00T checkout, which ships the converter)
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id hungho77/so101-multitask --root $HF_LEROBOT_HOME

# 2. base weights JAX -> PyTorch.  --config_name matters: it fixes action_horizon (50),
#    and a checkpoint converted at the wrong horizon loads but trains on the wrong shape.
python examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_so101 \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
  --output_path /mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/pi05_base_pytorch

# 3. normalization statistics -> assets/pi05_so101/hungho77/so101-multitask/norm_stats.json
python scripts/compute_norm_stats.py --config-name pi05_so101
```

## Train

```bash
./run_so101.sh            # nohup-able; see the file for the exact env
```

which is:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_pytorch.py pi05_so101 --exp_name so101_bs128 --overwrite
```

Every hyperparameter not named in the config is left at openpi's default on purpose —
cosine LR (warmup 1,000 → peak 2.5e-5, `decay_steps=30_000`), AdamW (b1 0.9, b2 0.95,
wd 1e-10, clip 1.0), `action_horizon=50`. The single forced deviation is `ema_decay=None`:
openpi's default is 0.99 but the PyTorch trainer does not support EMA.

**The LR barely decayed.** `decay_steps=30_000` against a 6,000-step run ends at 2.35e-5,
94% of peak, and the loss was still falling when training stopped. Set `decay_steps` equal
to `num_train_steps` for a clean decay, or train the full 30,000.

## Evaluate

```bash
python scripts/eval_so101_open_loop.py \
  --checkpoint-dir checkpoints/pi05_so101/so101_bs128/6000 \
  --traj-ids 0 25 60 80 110 130 --steps 400 \
  --eval-horizon 16 --norm-std 22.33
```

`--eval-horizon 16` scores only the first 16 of the 50 predicted steps. Error accumulates
along a chunk, so this is required to compare against a 16-horizon model — the same
checkpoint reads 4.50% at horizon 16 and 6.90% at horizon 50.

`--norm-std` must be the **absolute** action std (22.33 here). The checkpoint's own
`norm_stats.json` holds *delta* std for the arm joints (mean 9.71); dividing absolute-unit
MAE by that gives a meaningless number, which is why the flag has no default.

| model, same data / bs / steps | MAE (h=16) | % of action scale |
|---|---|---|
| GR00T N1.6 | 0.957 | **4.29%** |
| π₀.₅ (this) | 1.005 | **4.50%** |
| GR00T N1.7 | 1.280 | **5.73%** |

**No held-out split.** All 143 episodes were used for training, so these are measured on
training data. They show the policy fits its data and is not degenerate; they are not a
success rate and do not measure generalization.

## Serve

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101 \
  --policy.dir=checkpoints/pi05_so101/so101_bs128/6000
```

The data is **30 fps**. A client interpolating between model steps must derive its sub-step
count as `smooth_step = control_hz / 30`; a value borrowed from a 15 fps robot stretches
every trajectory by 2× and the arm creeps without finishing the task.
