# π₀.₅ finetune — SO101 left arm, 11-task multitask

Companion to `README.md` (the right-arm, 3-task run). Same robot and same code path; what
differs is the data and, because of that, the hardware envelope.

Source: the 11 datasets in the
[`hungho77/SO101_LeftArmOnly_Task`](https://huggingface.co/collections/hungho77/so101-leftarmonly-task)
collection, merged into one multitask set. The language prompt selects the task.

| | |
|---|---|
| Episodes / frames | 551 / 253,745 — **3.8×** the right-arm set |
| Tasks | 11, one per source dataset, `task_index` 0–10 |
| fps | 30 |
| Cameras | `top` **360×640**, `wrist` 480×640 |
| State / action | 6 — 5 arm joints + gripper |
| Absolute action std | **28.67** (mean over dims) — the `--norm-std` for eval |

The 11 sources are uniform field-for-field: all v3.0, all 30 fps, all `so_follower`, all
6-dim, all two cameras at the same resolutions, each with exactly one task. Nothing had to
be reconciled.

Note `wrist_roll` has std 19.63 here against 0.95 in the right-arm set, where it barely
moved. This data actually exercises that joint.

## Building the dataset

```bash
Z=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_src

# 1. v3.0 -> v2.1 for each source (openpi pins lerobot at 0cf8648, which reads v2.1 only).
#    NOTE: --root is a LeRobot HOME, not the dataset directory — the converter writes to
#    <root>/<repo_id>. Point it at a parent and the tree lands one level deeper than the
#    path you passed.
for n in <the 11 names>; do
  uv run --project scripts/lerobot_conversion \
    python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id "hungho77/$n" --root "$Z"
done

# 2. merge into one multitask tree
python examples/so101/merge_leftarm_collection.py

# 3. normalization statistics
HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_multitask_v21 \
  python scripts/compute_norm_stats.py --config-name pi05_so101_leftarm_multitask
```

The merge rewrites three global parquet columns — `episode_index`, `index` (the global
frame counter) and `task_index`. Renaming episode files without rewriting these yields a
dataset that loads and trains on silently wrong episode boundaries. Task order is fixed in
the script's `SRC` list, so `task_index` is reproducible; reordering that list renumbers
every task.

Verified after merging: 551 parquet and 551 mp4 per camera, `episode_index` matching every
filename, `index` strictly contiguous 0…253,744, `frame_index` restarting at 0 in every
episode, 50 episodes per task (51 for task 4). Zero discrepancies.

`meta/stats.json` is written with lerobot's own `aggregate_stats`, so it holds exactly what
the loader would compute. openpi does not read it — for v2.1 `LeRobotDatasetMetadata`
aggregates from `episodes_stats.jsonl` itself — but GR00T's loader does, so the file keeps
the dataset usable by both.

## Train

```bash
./run_so101_leftarm.sh
```

Two MIG 3g.40gb slices under DDP. `CUDA_VISIBLE_DEVICES` must name the MIG instance
**UUIDs**: with MIG enabled, a plain ordinal addresses the parent H100, which has no
compute instance assigned.

### Why batch 32 and not 128

Measured, not chosen. DDP replicates the optimizer on every rank — openpi's PyTorch
trainer has no FSDP, and no gradient accumulation either, so 32 is the *effective* batch:

| global batch | per GPU | reserved | result |
|---|---|---|---|
| 32 | 16 | 35.3 GB | runs, **2.50 s/step** |
| 48 | 24 | 38.4 GB | runs, 1.1 GB headroom |
| 64 | 32 | 41.3 GB | **OOM** |

The fixed cost of weights + gradients + AdamW state is ~29 GB of the 39.5 GB slice, and
each sample adds ~0.36 GB. The second GPU buys throughput, not memory. 48 is unusable
whenever anything else shares the slice: the bs-64 OOM message named the co-tenant process
holding 1.2 GB.

Throughput is 12.8 samples/s, about 83% of one undivided H100 (which ran 8.28 s/step at
batch 128) — in line with 120 of 132 SMs.

`batch_size=32` is also what openpi's own `pi05_droid_finetune` uses.

### Why 24,000 steps

24,000 × 32 = **768,000 samples**, the same sample budget as the right-arm run
(6,000 × 128), which is what makes the two checkpoints comparable. In epochs it is 3.0
against that run's 11.4, because the dataset is 3.8× larger. ~16.7 h at the measured rate.

`decay_steps` is tied to `num_train_steps`. The right-arm run left it at the default
30,000 against 6,000 steps, so the LR ended at 94% of peak and the cosine never decayed.

### One unexplained failure

A first launch sat 44 minutes past the data loader without completing step 0, with no
error, no OOM and no traceback — both ranks had loaded weights and DDP was constructed.
It has not reproduced since `TORCH_NCCL_BLOCKING_WAIT=1` was set, which is why the
launcher sets it. **The cause was never identified.** If this recurs, that env var is the
first thing to interrogate, not to trust.

## Evaluate

`scripts/eval_so101_open_loop.py` needs no changes — point it at this config and dataset:

```bash
python scripts/eval_so101_open_loop.py \
  --config-name pi05_so101_leftarm_multitask \
  --repo-id hungho77/so101-left-arm-multitask \
  --checkpoint-dir checkpoints/pi05_so101_leftarm_multitask/leftarm_bs32/24000 \
  --traj-ids 0 50 100 150 200 251 301 351 401 451 501 \
  --steps 400 --eval-horizon 16 --norm-std 28.67
```

One trajectory per task. `--norm-std 28.67` is this dataset's absolute action std; the
right-arm figure (22.33) does not apply, and the checkpoint's own `norm_stats.json` holds
delta stats for the arm joints, which are a different quantity again.

**No held-out split.** All 551 episodes are used for training, so open-loop numbers are
measured on training data. They show the policy fits its data; they are not a success rate.
