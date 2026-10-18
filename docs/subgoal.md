# Subgoal Images

Besides a text instruction, the policy can be prompted with **subgoal images**: one image per camera showing what the
scene should look like a few seconds from now. A subgoal often specifies the objective more clearly than text, for
example how to grasp a handle or what a folded shirt should look like. It also makes it possible to bring in
knowledge from an image generation model, as done in π0.7.

## How it works

**Training.** A fraction of the examples (`probability`, default 25%) get a subgoal: a future frame of the same episode.

- With probability `end_of_segment_probability` (default 0.25), the subgoal is the last frame of the current
  annotated segment. Segments come from the [annotations file](episode_metadata.md).
- Otherwise, the subgoal is a frame between 0 and `max_horizon_s` seconds ahead (default 4 s).
- When a subgoal is given, the subtask is dropped from the prompt with probability `subtask_dropout` (default 0.3),
  since the image often carries the same information.

The other examples get empty subgoal images that are fully masked out. Because of this, the model also learns to act
without a subgoal, and the result is identical to not passing a subgoal at all.

**Model.** The subgoal images go through the same SigLIP encoder as the observations. They are added to the prefix as a
separate block after the prompt:

- The subgoal tokens attend to the observation images, the prompt and each other.
- The observation and prompt tokens do not attend to the subgoal.
- The action expert attends to everything.

**Cost.** Each training example processes one extra image per subgoal view, so the prefix gets longer. Restrict
`views` to reduce the cost, e.g. to the base camera and one wrist camera.

## Training

Set `subgoal` in the `DataConfig`. See `pi05_libero_subgoal` in `src/openpi/training/config.py`:

```python
base_config=DataConfig(
    prompt_from_task=True,
    episode_metadata=EpisodeMetadataConfig(annotations_path="/path/to/annotations.json", ...),
    subgoal=SubgoalConfig(views=("base_0_rgb", "left_wrist_0_rgb")),
),
```

Subgoals can be combined with episode metadata and RECAP. Only LeRobot datasets are supported.

## Inference

Pass the subgoal in the `subgoal` key of the observation. Use the same format as the observation itself: the same keys
that your robot runtime sends for the camera images.

```python
obs = {
    "observation/image": base_image,
    "observation/wrist_image": wrist_image,
    "observation/state": state,
    "prompt": "put the bowl on the plate",
}
goal = {"observation/image": goal_base_image, "observation/wrist_image": goal_wrist_image}

actions = client.infer({**obs, "subgoal": goal})["actions"]  # with a subgoal
actions = client.infer(obs)["actions"]  # without a subgoal
```

The subgoal images can come from:

- a person, e.g. a photo of the desired final arrangement;
- frames of a demonstration of the task;
- an image generation or editing model that predicts the next subgoal from the current observation and the subtask.
  π0.7 uses a 14B BAGEL model fine-tuned on robot and human video, and regenerates the subgoal whenever the subtask
  changes or every 4 seconds.

Subgoal images are supported for the JAX pi0 / pi05 models.
