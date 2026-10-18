from collections.abc import Callable, Mapping, Sequence
import dataclasses
import random
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        data = {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}

        # Optional unconditional prompt, used for classifier-free guidance at inference time.
        if (prompt_uncond := data.pop("prompt_uncond", None)) is not None:
            if not isinstance(prompt_uncond, str):
                prompt_uncond = prompt_uncond.item()
            tokens, token_masks = self.tokenizer.tokenize(prompt_uncond, state)
            data = {**data, "tokenized_prompt_uncond": tokens, "tokenized_prompt_uncond_mask": token_masks}
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


# Fields that make up the episode metadata part of the prompt, in the order they appear.
_METADATA_FIELDS = ("Speed", "Quality", "Mistake", "Advantage")
_METADATA_PATTERN = re.compile(r"\.\s*(?:" + "|".join(_METADATA_FIELDS) + r"): [^.,]*")


def bin_episode_speed(length: int, bin_size: int) -> int:
    """Rounds an episode length (in steps) to the nearest multiple of `bin_size`, e.g. 1750-2249 -> 2000."""
    return max(bin_size, round(length / bin_size) * bin_size)


def format_metadata_prompt(
    task: str,
    *,
    subtask: str | None = None,
    speed: int | None = None,
    quality: int | None = None,
    mistake: bool | None = None,
    advantage: bool | None = None,
) -> str:
    """Builds a prompt that contains the task, the current subtask and the episode metadata.

    Example: "peel vegetables. Subtask: pick up the peeler. Speed: 8000. Quality: 5. Mistake: false"

    `advantage` is the RECAP advantage indicator ("Advantage: positive" / "Advantage: negative").

    Fields that are None are left out of the prompt.
    """
    parts = [task.strip().rstrip(".")]
    if subtask:
        parts.append(f"Subtask: {subtask.strip().rstrip('.')}")
    if speed is not None:
        parts.append(f"Speed: {int(speed)}")
    if quality is not None:
        parts.append(f"Quality: {int(quality)}")
    if mistake is not None:
        parts.append(f"Mistake: {'true' if mistake else 'false'}")
    if advantage is not None:
        parts.append(f"Advantage: {'positive' if advantage else 'negative'}")
    return ". ".join(parts)


def strip_metadata_prompt(prompt: str) -> str:
    """Removes the episode metadata from a prompt built by `format_metadata_prompt`, keeping task and subtask."""
    return _METADATA_PATTERN.sub("", prompt)


def _to_python(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


@dataclasses.dataclass(frozen=True)
class EpisodeMetadataPrompt(DataTransformFn):
    """Adds episode metadata (speed, quality, mistake) and the current subtask to the prompt during training.

    This must be applied to the raw dataset samples since it relies on "episode_index", "frame_index" and, for the
    advantage indicator, "index". Each part of the metadata is randomly dropped so that the model can be prompted with
    any subset of it at inference time.
    """

    # Maps episode index to its annotations. Supported keys:
    #   "length": int, the episode length in steps (used for the speed).
    #   "quality": int, the quality score of the episode (e.g. 1-5).
    #   "mistake": bool, episode level mistake label.
    #   "segments": list of (start_frame, end_frame, subtask, mistake) tuples. `subtask` and `mistake` may be None.
    #     Frames inside a segment use the segment labels. If segments are provided, frames that are not covered by a
    #     segment with a mistake label fall back to the episode level label, or to "no mistake".
    episodes: Mapping[int, Mapping]
    speed_bin_size: int = 500
    # Probability of dropping all of the metadata.
    metadata_dropout: float = 0.15
    # Probability of dropping each metadata field individually.
    field_dropout: float = 0.05
    # Probability of dropping the subtask.
    subtask_dropout: float = 0.0
    # RECAP advantage indicator for each frame, indexed by the dataset "index". See `scripts/compute_advantages.py`.
    advantages: np.ndarray | None = None
    # Probability of dropping the advantage indicator. It is also dropped together with the rest of the metadata.
    advantage_dropout: float = 0.3
    # If None, the global `random` module is used, which is seeded separately in each data loader worker.
    seed: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "_rng", random.Random(self.seed) if self.seed is not None else None)

    def __call__(self, data: DataDict) -> DataDict:
        if "prompt" not in data:
            raise ValueError("A prompt is required to add the episode metadata.")
        if "episode_index" not in data or "frame_index" not in data:
            raise ValueError('Cannot add the episode metadata without "episode_index" and "frame_index".')

        episode = self.episodes.get(int(data["episode_index"]), {})
        frame = int(data["frame_index"])

        speed = episode.get("length")
        if speed is not None:
            speed = bin_episode_speed(speed, self.speed_bin_size)
        quality = episode.get("quality")
        mistake = episode.get("mistake")
        subtask = None
        if (segments := episode.get("segments")) is not None:
            if mistake is None:
                mistake = False
            for start, end, segment_subtask, segment_mistake in segments:
                if start <= frame < end:
                    subtask = segment_subtask
                    if segment_mistake is not None:
                        mistake = segment_mistake
                    break
        advantage = None
        if self.advantages is not None:
            if "index" not in data:
                raise ValueError('Cannot add the advantage indicator without "index".')
            advantage = bool(self.advantages[int(data["index"])])

        rand = self._rng.random if self._rng is not None else random.random  # type: ignore[attr-defined]
        if rand() < self.metadata_dropout:
            speed = quality = mistake = advantage = None
        else:
            speed = None if rand() < self.field_dropout else speed
            quality = None if rand() < self.field_dropout else quality
            mistake = None if rand() < self.field_dropout else mistake
            advantage = None if rand() < self.advantage_dropout else advantage
        if rand() < self.subtask_dropout:
            subtask = None

        prompt = format_metadata_prompt(
            str(_to_python(data["prompt"])),
            subtask=subtask,
            speed=speed,
            quality=quality,
            mistake=mistake,
            advantage=advantage,
        )
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class InjectMetadataPrompt(DataTransformFn):
    """Adds the desired episode metadata to the prompt at inference time.

    The defaults can be overridden per request by passing "subtask", "speed", "quality", "mistake" or "advantage" in
    the observation.
    """

    # Desired episode length in steps. A good choice is a low percentile (e.g. 15th) of the episode lengths of the
    # task in the training data. If None, the speed is left out of the prompt.
    speed: int | None = None
    quality: int | None = 5
    mistake: bool | None = False
    # RECAP advantage indicator. Only set this if the model was trained with advantages.
    advantage: bool | None = None
    speed_bin_size: int = 500

    def __call__(self, data: DataDict) -> DataDict:
        subtask = _to_python(data.pop("subtask", None))
        speed = _to_python(data.pop("speed", self.speed))
        quality = _to_python(data.pop("quality", self.quality))
        mistake = _to_python(data.pop("mistake", self.mistake))
        advantage = _to_python(data.pop("advantage", self.advantage))
        if "prompt" not in data:
            return data

        if speed is not None:
            speed = bin_episode_speed(int(speed), self.speed_bin_size)
        if isinstance(mistake, str):
            mistake = mistake.lower() == "true"
        if isinstance(advantage, str):
            advantage = advantage.lower() in ("true", "positive")
        prompt = format_metadata_prompt(
            str(_to_python(data["prompt"])),
            subtask=subtask,
            speed=speed,
            quality=quality,
            mistake=mistake,
            advantage=advantage,
        )
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class UnconditionalPrompt(DataTransformFn):
    """Adds a copy of the prompt without the episode metadata, used for classifier-free guidance."""

    def __call__(self, data: DataDict) -> DataDict:
        if "prompt" not in data:
            return data
        return {**data, "prompt_uncond": strip_metadata_prompt(str(_to_python(data["prompt"])))}


def normalized_return(
    frame: int, length: int, max_length: int, *, success: bool, failure_penalty: float, clip: bool = True
) -> float:
    """RECAP return: -1 per remaining step until the end of the episode, plus a penalty if the episode failed.

    The return is normalized by the maximum episode length of the task so that it lies in [-1, 0].
    """
    value = -(length - 1 - frame) / max_length
    if not success:
        value -= failure_penalty
    return max(-1.0, min(0.0, value)) if clip else value


@dataclasses.dataclass(frozen=True)
class ValueTarget(DataTransformFn):
    """Adds the normalized return ("value_target") used to train the value function.

    This must be applied to the raw dataset samples since it relies on "episode_index" and "frame_index".
    """

    # Maps episode index to {"length": int, "max_length": int, "success": bool}, where "max_length" is the maximum
    # episode length of the episode's task.
    episodes: Mapping[int, Mapping]
    # Penalty added to the return of failed episodes, in units of the normalized return.
    failure_penalty: float = 1.0

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data or "frame_index" not in data:
            raise ValueError('Cannot compute the value target without "episode_index" and "frame_index".')
        episode = self.episodes[int(data["episode_index"])]
        value = normalized_return(
            int(data["frame_index"]),
            episode["length"],
            episode["max_length"],
            success=episode["success"],
            failure_penalty=self.failure_penalty,
        )
        return {**data, "value_target": np.asarray(value, dtype=np.float32)}


@dataclasses.dataclass(frozen=True)
class PassthroughKeys(DataTransformFn):
    """Applies `transform` and copies `keys` from the input to the output, e.g. to keep training targets that would
    otherwise be removed by the repack or data transforms."""

    transform: DataTransformFn
    keys: Sequence[str]

    def __call__(self, data: DataDict) -> DataDict:
        kept = {key: data[key] for key in self.keys if key in data}
        return {**self.transform(data), **kept}


_SUBTASK_PATTERN = re.compile(r"\.\s*Subtask: [^.,]*")


def strip_subtask_prompt(prompt) -> str:
    """Removes the subtask from a prompt built by `format_metadata_prompt`."""
    return _SUBTASK_PATTERN.sub("", str(_to_python(prompt)))


@dataclasses.dataclass(frozen=True)
class SubgoalInputs(DataTransformFn):
    """Adds subgoal images ("subgoal_image" / "subgoal_image_mask") to the model inputs.

    The subgoal is passed in the "subgoal" key, using the same format as the observation (e.g. the same image keys as
    the dataset or the robot runtime). It is converted with `subgoal_transform`, which should contain the same input
    transforms as the observation images (e.g. the robot specific data transforms and the image resizing), and only
    its images are kept. All other inputs are processed with `transform`.
    """

    transform: DataTransformFn
    subgoal_transform: DataTransformFn
    # Restricts the subgoal to these model image keys, e.g. ("base_0_rgb",). The other views are masked out.
    views: Sequence[str] | None = None
    # If true, empty (fully masked) subgoal images are added when there is no subgoal, so that all examples have the
    # same structure. Required for training.
    always_add: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        subgoal = data.pop("subgoal", None)
        if subgoal is not None:
            # The rest of the observation is included since the data transforms may rely on it (e.g. the state).
            subgoal_inputs = self.subgoal_transform({**data, **subgoal})
        outputs = self.transform(data)

        if subgoal is not None:
            images = subgoal_inputs["image"]
            masks = {key: np.asarray(subgoal_inputs["image_mask"][key]) for key in images}
            if self.views is not None:
                masks = {key: mask & (key in self.views) for key, mask in masks.items()}
        elif self.always_add:
            images = {key: np.zeros_like(image) for key, image in outputs["image"].items()}
            masks = dict.fromkeys(images, np.False_)
        else:
            return outputs
        return {**outputs, "subgoal_image": images, "subgoal_image_mask": masks}


@dataclasses.dataclass(frozen=True)
class HistoryInputs(DataTransformFn):
    """Adds the past frames of each camera ("history_image" / "history_image_mask") to the model inputs.

    The past frames are passed in the "history" key as a list (oldest first) of dicts with the same format as the
    observation, and optionally a "history_valid" array that marks padded frames. They are converted with
    `frame_transform`, which should contain the same input transforms as the observation images. All other inputs are
    processed with `transform`.
    """

    transform: DataTransformFn
    frame_transform: DataTransformFn
    num_frames: int
    # Probability of dropping the whole history (training only).
    dropout: float = 0.0
    # If true, empty (fully masked) frames are added when there is no history, so that all examples have the same
    # structure. Required for training.
    always_add: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        history = data.pop("history", None)
        valid = data.pop("history_valid", None)
        converted = []
        if history is not None:
            history = list(history)[-self.num_frames :]
            valid = [True] * len(history) if valid is None else [bool(v) for v in np.asarray(valid)][-len(history) :]
            # The rest of the observation is included since the data transforms may rely on it (e.g. the state).
            converted = [(self.frame_transform({**data, **frame}), v) for frame, v in zip(history, valid, strict=True)]
        outputs = self.transform(data)
        if history is None and not self.always_add:
            return outputs

        images = {
            key: np.zeros((self.num_frames, *np.shape(image)), dtype=np.asarray(image).dtype)
            for key, image in outputs["image"].items()
        }
        masks = {key: np.zeros(self.num_frames, dtype=bool) for key in images}
        if converted and not (self.dropout > 0 and random.random() < self.dropout):
            # Missing frames are padded at the beginning.
            offset = self.num_frames - len(converted)
            for i, (frame, is_valid) in enumerate(converted):
                for key, frames in images.items():
                    if key in frame["image"]:
                        frames[offset + i] = frame["image"][key]
                        masks[key][offset + i] = is_valid and bool(frame["image_mask"][key])
        return {**outputs, "history_image": images, "history_image_mask": masks}


@dataclasses.dataclass(frozen=True)
class TokenizeKIInputs(DataTransformFn):
    """Tokenizes the prompt and appends the FAST tokens of the action chunk (knowledge insulation, training only).

    Without actions, this is the same as `TokenizePrompt`. At inference time, use `TokenizePrompt` instead (see
    `for_inference`), since the actions passed for real-time chunking must not be tokenized.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    fast_tokenizer: _tokenizer.FASTActionTokenizer
    discrete_state_input: bool = False

    def for_inference(self) -> TokenizePrompt:
        return TokenizePrompt(self.tokenizer, self.discrete_state_input)

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return self.for_inference()(data)
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        state = data["state"] if self.discrete_state_input else None
        action_tokens = self.fast_tokenizer.encode(np.asarray(data["actions"]))
        tokens, mask, ar_mask, loss_mask = self.tokenizer.tokenize_with_action_tokens(
            str(_to_python(prompt)), state, action_tokens
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeSubtaskInputs(DataTransformFn):
    """Tokenizes the inputs of the high-level policy: task, previous subtasks and (during training) the subtask."""

    tokenizer: _tokenizer.SubtaskTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        previous = _to_python(data.pop("previous_subtasks", ""))
        if not isinstance(previous, str):
            previous = "; ".join(str(_to_python(p)) for p in previous)
        target = data.pop("subtask_target", None)
        tokens, mask, ar_mask, loss_mask = self.tokenizer.tokenize(
            str(_to_python(prompt)), previous, None if target is None else str(_to_python(target))
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractSubtask(DataTransformFn):
    """Decodes the subtask predicted by the high-level policy."""

    tokenizer: _tokenizer.SubtaskTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        tokens = data.pop("actions")
        data.pop("state", None)
        return {**data, "subtask": self.tokenizer.decode(np.asarray(tokens).astype(np.int32))}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
