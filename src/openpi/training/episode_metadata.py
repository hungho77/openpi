"""Episode metadata prompting.

The prompt of every training example is extended with information about how the episode was performed: its speed
(episode length), quality score, whether the robot made a mistake, and optionally the current subtask. This makes it
possible to train on mixed quality data (failures, autonomous rollouts, slow demonstrations) and then prompt the model
for fast, high quality, mistake free behavior at inference time.

Annotations are stored in a JSON file, keyed by episode index:

```
{
    "0": {"quality": 5},
    "1": {"quality": 2, "mistake": true, "success": false},
    "2": {
        "quality": 4,
        "segments": [
            {"start": 0, "end": 150, "subtask": "pick up the cup"},
            {"start": 150, "end": 260, "subtask": "place the cup in the sink", "mistake": true},
            {"start": 260, "end": 400, "intervention": true}
        ]
    }
}
```

All fields are optional. The episode speed is always computed from the episode length in the dataset. Segment frame
ranges are `[start, end)` in terms of the dataset "frame_index". See `scripts/compute_episode_metadata.py` to create a
template for a dataset.

The "success" and "intervention" labels are used by RECAP: the value function is trained on the returns of successful
and failed episodes (`ValueTargetConfig`), and frames inside intervention segments (human corrections) are always
labeled with a positive advantage (`scripts/compute_advantages.py`).
"""

from collections.abc import Mapping
import dataclasses
import json
import logging
import pathlib

import etils.epath as epath
import numpy as np

import openpi.transforms as _transforms

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class EpisodeMetadataConfig:
    # Path to the JSON annotations file. If None, only the episode speed is used.
    annotations_path: str | None = None
    # Episode lengths are rounded to the nearest multiple of this value (in steps).
    speed_bin_size: int = 500

    # Training: probability of dropping all of the metadata from the prompt.
    metadata_dropout: float = 0.15
    # Training: probability of dropping each metadata field individually.
    field_dropout: float = 0.05
    # Training: probability of dropping the subtask from the prompt.
    subtask_dropout: float = 0.0

    # Inference: desired episode length in steps. A good choice is the 15th percentile of the episode lengths of the
    # task (see `scripts/compute_episode_metadata.py`). If None, the speed is left out of the prompt.
    target_speed: int | None = None
    # Inference: desired quality score.
    target_quality: int | None = 5
    # Inference: desired mistake label.
    target_mistake: bool | None = False
    # Inference: classifier-free guidance scale on the metadata. Each denoising step uses
    # v = v_uncond + cfg_scale * (v_cond - v_uncond), where the unconditional prompt has no metadata. 1.0 disables
    # guidance; values around 1.5-2.5 work well in practice. Requires the model to be trained with metadata dropout.
    cfg_scale: float = 1.0

    # RECAP: path to the advantages file created by `scripts/compute_advantages.py`. If set, the prompt also contains
    # the advantage indicator ("Advantage: positive" / "Advantage: negative").
    advantages_path: str | None = None
    # Training: probability of dropping the advantage indicator.
    advantage_dropout: float = 0.3
    # Inference: desired advantage indicator. Only used if `advantages_path` is set.
    target_advantage: bool | None = True

    def inference_transforms(self) -> list[_transforms.DataTransformFn]:
        """Transforms that add the desired metadata to the prompt. Applied before the data transforms."""
        return [
            _transforms.InjectMetadataPrompt(
                speed=self.target_speed,
                quality=self.target_quality,
                mistake=self.target_mistake,
                advantage=self.target_advantage if self.advantages_path is not None else None,
                speed_bin_size=self.speed_bin_size,
            )
        ]


@dataclasses.dataclass(frozen=True)
class ValueTargetConfig:
    """Configures the returns used to train the RECAP value function."""

    # Path to the JSON annotations file, which provides the "success" labels.
    annotations_path: str | None = None
    # Success label of episodes without a "success" annotation, e.g. True if most of the data are demonstrations.
    default_success: bool = True
    # Penalty added to the return of failed episodes, in units of the normalized return. The return of every frame is
    # normalized by the maximum episode length of the task to [-1, 0], so with the default value all frames of failed
    # episodes have the lowest return.
    failure_penalty: float = 1.0


@dataclasses.dataclass(frozen=True)
class SubgoalConfig:
    """Configures subgoal image conditioning.

    During training, a future frame of the same episode is added as the subgoal to a fraction of the examples. At
    inference time, the subgoal is passed in the "subgoal" key of the observation, using the same format as the
    observation (e.g. {"observation/image": ..., "observation/wrist_image": ...}).
    """

    # Training: fraction of the examples that get a subgoal.
    probability: float = 0.25
    # Training: probability of using the last frame of the current annotated segment as the subgoal. Otherwise (or if
    # the frame is not inside a segment), a frame between 0 and `max_horizon_s` seconds in the future is used.
    end_of_segment_probability: float = 0.25
    max_horizon_s: float = 4.0
    # Training: probability of dropping the subtask from the prompt when a subgoal is given.
    subtask_dropout: float = 0.3
    # Annotations file with the segments. If None, the annotations of the episode metadata config are used.
    annotations_path: str | None = None
    # Model image keys used as subgoals, e.g. ("base_0_rgb",). If None, all views are used.
    views: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class SubtaskTargetConfig:
    """Configures the training targets of the high-level policy, which predicts the next subtask.

    Only frames inside annotated segments with a subtask are used. The previous subtasks of the episode are given as
    context.
    """

    # Annotations file with the subtask segments.
    annotations_path: str = ""
    # Maximum number of previous subtasks given as context.
    max_previous_subtasks: int = 5


def load_annotations(path: str | pathlib.Path) -> dict[int, dict]:
    """Loads the annotations file and converts it to the format expected by `EpisodeMetadataPrompt`."""
    with epath.Path(path).open() as f:
        raw = json.load(f)

    episodes = {}
    for key, value in raw.items():
        episode = {}
        if (quality := value.get("quality")) is not None:
            episode["quality"] = int(quality)
        if (mistake := value.get("mistake")) is not None:
            episode["mistake"] = bool(mistake)
        if (success := value.get("success")) is not None:
            episode["success"] = bool(success)
        if (segments := value.get("segments")) is not None:
            episode["segments"] = tuple(
                (
                    int(segment["start"]),
                    int(segment["end"]),
                    segment.get("subtask"),
                    None if segment.get("mistake") is None else bool(segment["mistake"]),
                )
                for segment in segments
            )
            episode["interventions"] = tuple(
                (int(segment["start"]), int(segment["end"])) for segment in segments if segment.get("intervention")
            )
        episodes[int(key)] = episode
    return episodes


def get_episode_lengths(dataset_meta) -> dict[int, int]:
    """Returns the episode lengths from a `LeRobotDatasetMetadata`."""
    return {int(index): int(episode["length"]) for index, episode in dataset_meta.episodes.items()}


def get_episode_tasks(dataset_meta) -> dict[int, str]:
    """Returns the (first) task of each episode from a `LeRobotDatasetMetadata`."""
    return {int(index): (episode.get("tasks") or [""])[0] for index, episode in dataset_meta.episodes.items()}


def build_episode_table(
    config: EpisodeMetadataConfig | ValueTargetConfig, episode_lengths: Mapping[int, int]
) -> dict[int, dict]:
    """Merges the episode lengths with the annotations."""
    annotations = load_annotations(config.annotations_path) if config.annotations_path is not None else {}
    unknown = set(annotations) - set(episode_lengths)
    if unknown:
        logger.warning(f"Annotations found for {len(unknown)} episodes that are not in the dataset, ignoring them.")

    table = {index: {"length": length, **annotations.get(index, {})} for index, length in episode_lengths.items()}
    num_annotated = sum(index in annotations for index in episode_lengths)
    logger.info(f"Episode metadata: {num_annotated}/{len(table)} episodes have annotations.")
    return table


def load_advantages(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Loads the advantages file created by `scripts/compute_advantages.py`."""
    with epath.Path(path).open("rb") as f:
        return dict(np.load(f))


def create_training_transform(config: EpisodeMetadataConfig, dataset_meta) -> _transforms.EpisodeMetadataPrompt:
    advantages = None
    if config.advantages_path is not None:
        advantages = load_advantages(config.advantages_path)["indicator"].astype(bool)
        if len(advantages) != dataset_meta.total_frames:
            raise ValueError(
                f"The advantages file has {len(advantages)} frames but the dataset has {dataset_meta.total_frames}. "
                "Recompute the advantages for the current dataset."
            )
        logger.info(f"Advantage indicator: {advantages.mean():.1%} of the frames are positive.")
    return _transforms.EpisodeMetadataPrompt(
        episodes=build_episode_table(config, get_episode_lengths(dataset_meta)),
        speed_bin_size=config.speed_bin_size,
        metadata_dropout=config.metadata_dropout,
        field_dropout=config.field_dropout,
        subtask_dropout=config.subtask_dropout,
        advantages=advantages,
        advantage_dropout=config.advantage_dropout,
    )


def build_value_table(config: ValueTargetConfig, dataset_meta) -> dict[int, dict]:
    """Returns {"length", "max_length", "success", ...} for every episode. "max_length" is per task."""
    table = build_episode_table(config, get_episode_lengths(dataset_meta))
    tasks = get_episode_tasks(dataset_meta)
    max_lengths: dict[str, int] = {}
    for index, episode in table.items():
        max_lengths[tasks[index]] = max(max_lengths.get(tasks[index], 0), episode["length"])
    for index, episode in table.items():
        episode["max_length"] = max_lengths[tasks[index]]
        episode["task"] = tasks[index]
        episode.setdefault("success", config.default_success)
    num_failed = sum(not episode["success"] for episode in table.values())
    logger.info(f"Value targets: {num_failed}/{len(table)} episodes are labeled as failed.")
    return table


def create_value_transform(config: ValueTargetConfig, dataset_meta) -> _transforms.ValueTarget:
    table = build_value_table(config, dataset_meta)
    episodes = {
        index: {"length": e["length"], "max_length": e["max_length"], "success": e["success"]}
        for index, e in table.items()
    }
    return _transforms.ValueTarget(episodes=episodes, failure_penalty=config.failure_penalty)


def speed_percentiles(episode_lengths: Mapping[int, int], percentiles=(10, 15, 25, 50, 75, 90)) -> dict[int, int]:
    lengths = np.asarray(list(episode_lengths.values()))
    return {p: int(np.percentile(lengths, p)) for p in percentiles}
