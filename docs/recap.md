# RECAP: Advantage-Conditioned Policies

RECAP (from π\*0.6) improves a policy using its own experience: autonomous rollouts, failures and human corrections.
Each iteration has three steps:

1. Train a **value function** on all the data collected so far. It predicts the normalized number of steps left until
   the task succeeds, and failed episodes get a penalty.
2. Use the value function to compute an **advantage** for every frame, then binarize it into an indicator.
3. Train the **policy** with `Advantage: positive` or `Advantage: negative` in its prompt. At inference time, prompt it
   with `Advantage: positive`.

RECAP builds on [episode metadata prompting](episode_metadata.md): the advantage indicator is just one more field in the
prompt, and it shares the same dropout, inference-time injection and classifier-free guidance.

## 0. Collect and annotate data

Add every episode to a single LeRobot dataset: demonstrations, autonomous rollouts and rollouts with human
interventions. Then label the episodes in the [annotations file](episode_metadata.md):

```json
{
  "12": {"success": true, "quality": 4},
  "13": {"success": false, "quality": 1},
  "14": {
    "success": true,
    "segments": [
      {"start": 0, "end": 300},
      {"start": 300, "end": 420, "intervention": true}
    ]
  }
}
```

- `success`: whether the episode completed the task. Episodes without a label use `ValueTargetConfig.default_success`,
  which defaults to `true`.
- `intervention`: segments where a human took over, e.g. via teleoperation. These frames are always labeled with a
  positive advantage.

## 1. Train the value function

The value function uses the PaliGemma backbone (SigLIP and Gemma 2B) with a 201-bin categorical head, and it is
initialized from `pi05_base`. See `value_libero` in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="value_my_robot",
    model=value_model.ValueConfig(),  # or ValueConfig(paligemma_variant="gemma_2b_lora") with a freeze filter
    data=MyDataConfig(
        repo_id="<user>/<dataset>",
        base_config=DataConfig(
            prompt_from_task=True,
            value_target=ValueTargetConfig(annotations_path="/path/to/annotations.json"),
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params", missing_regex=".*value_head.*"
    ),
    ...
)
```

The return of frame `t` in an episode of length `T` is `-(T - 1 - t) / max_length`. Here `max_length` is the longest
episode of the same task. Failed episodes get an extra `-failure_penalty`, and the result is clipped to `[-1, 0]`.

```bash
uv run scripts/compute_norm_stats.py --config-name value_my_robot
uv run scripts/train.py value_my_robot --exp-name=iter0
```

## 2. Compute the advantages

```bash
uv run scripts/compute_advantages.py --config-name value_my_robot \
    --checkpoint-dir checkpoints/value_my_robot/iter0/<step> \
    --output assets/value_my_robot/advantages.npz \
    --n-step 50 --positive-fraction 0.3
```

The script computes an n-step advantage, `A_t = sum of the next N rewards + V(o_{t+N}) - V(o_t)`. Use `--n-step 0` for
the Monte Carlo advantage `R_t - V(o_t)` instead.

Within each task, the `positive-fraction` of frames with the highest advantage are labeled positive. The paper uses 30%
for pretraining and about 40% for fine-tuning on rollouts.

For each task, the script also prints `corr(value, return)`. If this correlation is low, the value function has not
learned much, so train it longer or on more data before trusting the advantages.

The output file is indexed by the dataset `index`. Compute it again whenever the dataset changes.

## 3. Train the policy

Set `advantages_path` in the policy's `EpisodeMetadataConfig`, as in `pi05_libero_recap`:

```python
episode_metadata=EpisodeMetadataConfig(
    annotations_path="/path/to/annotations.json",
    advantages_path="assets/value_my_robot/advantages.npz",
    advantage_dropout=0.3,
    target_advantage=True,
    cfg_scale=1.0,
),
```

During training, the indicator is dropped 30% of the time, and it is also dropped together with the rest of the
metadata. At inference time, `Advantage: positive` is added to the prompt automatically.

Set `cfg_scale` to a value between 1.5 and 2.5 to apply classifier-free guidance on the metadata and the advantage.

## 4. Iterate

1. Deploy the policy and collect autonomous rollouts. Optionally, let a human take over when the robot is about to
   fail, and record those corrections as `intervention` segments.
2. Label the new episodes with `success` (and `quality` / `mistake` if available), then add them to the dataset.
3. Retrain the value function and then the policy, both starting from `pi05_base` rather than the previous iteration,
   to avoid drift.

The paper reports large gains after only one or two iterations, each with a few hundred episodes per task.
