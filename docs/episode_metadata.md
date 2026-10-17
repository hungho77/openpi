# Episode Metadata Prompting

Episode metadata prompting adds information about *how* an episode was performed to the prompt of every training
example. With it, you can train on mixed-quality data (slow demonstrations, failures, autonomous rollouts) and still
prompt the model for fast, high-quality, mistake-free behavior at inference time.

A training prompt looks like this:

```
peel vegetables. Subtask: pick up the peeler. Speed: 8000. Quality: 5. Mistake: false
```

| Field | Source | Meaning |
|---|---|---|
| `Speed` | Episode length in the dataset | Episode length in steps, rounded to the nearest multiple of `speed_bin_size`. |
| `Quality` | Annotations file | Quality score of the episode, e.g. 1 to 5. |
| `Mistake` | Annotations file | Whether the robot made a mistake in the current segment (or the episode). |
| `Subtask` | Annotations file | The current subtask of the segment. |

During training, all of the metadata is dropped with probability `metadata_dropout` (default 0.15). Each field is also
dropped on its own with probability `field_dropout` (default 0.05). Because of this, the model can be prompted with any
subset of the fields, or with none at all. The dropout is also what makes classifier-free guidance possible.

## 1. Inspect the dataset and create an annotations template

```bash
uv run scripts/compute_episode_metadata.py --repo-id <repo_id> --output annotations.json
```

The script prints episode length percentiles for the whole dataset and for each task. Use a low percentile, such as the
15th, as the `target_speed` at inference time.

The template has one entry per episode. Fill in the labels you have and remove the ones you don't. The `length` and
`tasks` fields are only there for reference and are ignored when the file is loaded.

```json
{
  "0": {"quality": 5},
  "1": {"quality": 2, "mistake": true},
  "2": {
    "quality": 4,
    "segments": [
      {"start": 0, "end": 150, "subtask": "pick up the cup"},
      {"start": 150, "end": 260, "subtask": "place the cup in the sink", "mistake": true}
    ]
  }
}
```

How the fields are read:

- Every field is optional. An episode without an entry only gets the `Speed` field.
- Segment ranges are `[start, end)` in terms of the dataset `frame_index`.
- If an episode has segments, any frame without a segment-level mistake label uses the episode-level label. If the
  episode has no label either, the frame is treated as `Mistake: false`.

## 2. Train

Set `episode_metadata` in the `DataConfig` of your training config. See `pi05_libero_metadata` in
`src/openpi/training/config.py` for an example:

```python
data=LeRobotLiberoDataConfig(
    repo_id="physical-intelligence/libero",
    base_config=DataConfig(
        prompt_from_task=True,
        episode_metadata=EpisodeMetadataConfig(
            annotations_path="/path/to/annotations.json",
            speed_bin_size=500,
            target_speed=2000,
            cfg_scale=1.0,
        ),
    ),
),
```

Requirements:

- The dataset must provide a prompt, either from `prompt_from_task=True` or from a `prompt` key.
- The repack transforms must keep the `prompt` key.
- Only LeRobot datasets are supported, since the metadata is looked up through `episode_index` and `frame_index`.

## 3. Inference

When a policy is created from a config with `episode_metadata`, the desired metadata is added to the prompt
automatically: `target_speed`, `target_quality` (default 5) and `target_mistake` (default false).

You can override these values per request by adding any of the keys `speed`, `quality`, `mistake` or `subtask` to the
observation that is sent to the policy server. Use the `subtask` key, for example, to pass the output of a high-level
policy.

```python
action = client.infer({**observation, "prompt": "clean the kitchen", "subtask": "open the fridge"})["actions"]
```

### Classifier-free guidance

Set `cfg_scale` to a value larger than 1.0 to turn on classifier-free guidance on the metadata. Each denoising step
then combines the velocity from the full prompt with the velocity from the prompt without metadata:

```
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

Values around 1.5 to 2.5 push the model further toward the requested speed and quality. Higher values can produce
aggressive motions. Guidance runs the model on a batch of twice the size, so it adds inference time. It is supported
for JAX pi0 and pi05 models.
