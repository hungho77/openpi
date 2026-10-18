import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_format_and_strip_metadata_prompt():
    prompt = _transforms.format_metadata_prompt(
        "peel vegetables.", subtask="pick up the peeler", speed=8000, quality=5, mistake=False
    )
    assert prompt == "peel vegetables. Subtask: pick up the peeler. Speed: 8000. Quality: 5. Mistake: false"
    assert _transforms.strip_metadata_prompt(prompt) == "peel vegetables. Subtask: pick up the peeler"

    prompt = _transforms.format_metadata_prompt("fold the shirt", quality=3, mistake=True)
    assert prompt == "fold the shirt. Quality: 3. Mistake: true"
    assert _transforms.strip_metadata_prompt(prompt) == "fold the shirt"
    assert _transforms.format_metadata_prompt("fold the shirt") == "fold the shirt"


def test_bin_episode_speed():
    assert _transforms.bin_episode_speed(1750, 500) == 2000
    assert _transforms.bin_episode_speed(2249, 500) == 2000
    assert _transforms.bin_episode_speed(100, 500) == 500


def test_episode_metadata_prompt():
    episodes = {
        0: {"length": 1900, "quality": 4, "segments": ((0, 10, "pick up the cup", None), (10, 20, "place it", True))},
        1: {"length": 600},
    }
    transform = _transforms.EpisodeMetadataPrompt(episodes, metadata_dropout=0.0, field_dropout=0.0)

    data = transform({"prompt": "clean up", "episode_index": 0, "frame_index": 5})
    assert data["prompt"] == "clean up. Subtask: pick up the cup. Speed: 2000. Quality: 4. Mistake: false"
    data = transform({"prompt": "clean up", "episode_index": 0, "frame_index": 15})
    assert data["prompt"] == "clean up. Subtask: place it. Speed: 2000. Quality: 4. Mistake: true"
    data = transform({"prompt": np.asarray("clean up"), "episode_index": np.asarray(1), "frame_index": 0})
    assert data["prompt"] == "clean up. Speed: 500"

    transform = _transforms.EpisodeMetadataPrompt(episodes, metadata_dropout=1.0, subtask_dropout=1.0)
    data = transform({"prompt": "clean up", "episode_index": 0, "frame_index": 5})
    assert data["prompt"] == "clean up"


def test_episode_metadata_prompt_dropout_rate():
    episodes = {0: {"length": 1000, "quality": 5, "mistake": False}}
    transform = _transforms.EpisodeMetadataPrompt(episodes, metadata_dropout=0.3, field_dropout=0.0, seed=0)
    prompts = [transform({"prompt": "task", "episode_index": 0, "frame_index": 0})["prompt"] for _ in range(2000)]
    assert 0.25 < np.mean([p == "task" for p in prompts]) < 0.35


def test_inject_metadata_prompt():
    transform = _transforms.InjectMetadataPrompt(speed=1900)
    assert transform({"prompt": "clean up"})["prompt"] == "clean up. Speed: 2000. Quality: 5. Mistake: false"

    data = transform({"prompt": "clean up", "subtask": np.asarray("open the fridge"), "quality": 3})
    assert data == {"prompt": "clean up. Subtask: open the fridge. Speed: 2000. Quality: 3. Mistake: false"}

    # The override keys are always removed.
    assert transform({"speed": 100}) == {}


def test_tokenize_prompt_uncond():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=24)
    data = {"prompt": "clean up. Quality: 5. Mistake: false"}
    data = _transforms.UnconditionalPrompt()(data)
    assert data["prompt_uncond"] == "clean up"

    data = _transforms.TokenizePrompt(tokenizer)(data)
    tok_prompt, _ = tokenizer.tokenize("clean up")
    assert "prompt_uncond" not in data
    assert np.array_equal(tok_prompt, data["tokenized_prompt_uncond"])


def test_advantage_prompt():
    prompt = _transforms.format_metadata_prompt("fold the shirt", quality=5, advantage=True)
    assert prompt == "fold the shirt. Quality: 5. Advantage: positive"
    assert _transforms.strip_metadata_prompt(prompt) == "fold the shirt"
    assert _transforms.format_metadata_prompt("fold", advantage=False) == "fold. Advantage: negative"

    advantages = np.array([True, False])
    transform = _transforms.EpisodeMetadataPrompt(
        {0: {"length": 2}}, metadata_dropout=0.0, field_dropout=0.0, advantages=advantages, advantage_dropout=0.0
    )
    data = transform({"prompt": "fold", "episode_index": 0, "frame_index": 1, "index": 1})
    assert data["prompt"] == "fold. Speed: 500. Advantage: negative"

    transform = _transforms.InjectMetadataPrompt(quality=None, mistake=None, advantage=True)
    assert transform({"prompt": "fold"})["prompt"] == "fold. Advantage: positive"
    assert transform({"prompt": "fold", "advantage": "negative"})["prompt"] == "fold. Advantage: negative"


def test_value_target():
    episodes = {
        0: {"length": 11, "max_length": 20, "success": True},
        1: {"length": 5, "max_length": 20, "success": False},
    }
    transform = _transforms.ValueTarget(episodes, failure_penalty=0.5)
    assert transform({"episode_index": 0, "frame_index": 10})["value_target"] == 0.0
    assert np.isclose(transform({"episode_index": 0, "frame_index": 0})["value_target"], -0.5)
    assert np.isclose(transform({"episode_index": 1, "frame_index": 4})["value_target"], -0.5)
    assert np.isclose(transform({"episode_index": 1, "frame_index": 0})["value_target"], -0.7)


def test_passthrough_keys():
    transform = _transforms.PassthroughKeys(_transforms.RepackTransform({"a": "a"}), keys=("value_target",))
    assert transform({"a": 1, "b": 2, "value_target": 3}) == {"a": 1, "value_target": 3}


def test_strip_subtask_prompt():
    prompt = "clean up. Subtask: open the fridge. Quality: 5"
    assert _transforms.strip_subtask_prompt(prompt) == "clean up. Quality: 5"
    assert _transforms.strip_subtask_prompt(np.asarray("clean up")) == "clean up"


def _to_model_images(data):
    return {
        "image": {"base_0_rgb": data["cam"], "left_wrist_0_rgb": data["wrist"]},
        "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
        "state": data["state"],
    }


def test_subgoal_inputs():
    obs = {"cam": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8), "state": np.zeros(3)}
    goal = {"cam": np.ones((2, 2, 3), np.uint8), "wrist": np.ones((2, 2, 3), np.uint8)}
    transform = _transforms.SubgoalInputs(_to_model_images, _to_model_images, views=("base_0_rgb",))

    data = transform({**obs, "subgoal": goal})
    assert np.all(data["subgoal_image"]["base_0_rgb"] == 1)
    assert data["subgoal_image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": False}
    assert np.all(data["image"]["base_0_rgb"] == 0)

    # Without a subgoal, nothing is added at inference time and empty subgoals are added for training.
    assert "subgoal_image" not in transform(dict(obs))
    transform = _transforms.SubgoalInputs(_to_model_images, _to_model_images, always_add=True)
    data = transform(dict(obs))
    assert not any(data["subgoal_image_mask"].values())
    assert data["subgoal_image"]["base_0_rgb"].shape == (2, 2, 3)


class _FakeFAST:
    def __call__(self, actions):
        return [[0, 1, 2]]


def test_tokenize_ki_inputs():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=64)
    fast = _tokenizer.FASTActionTokenizer(fast_tokenizer_path=None, fast_tokenizer=_FakeFAST())
    transform = _transforms.TokenizeKIInputs(tokenizer, fast, discrete_state_input=True)
    state = np.zeros(4)

    data = transform({"prompt": "fold", "state": state, "actions": np.zeros((10, 4))})
    prompt_tokens, _ = tokenizer.tokenize("fold", state)
    num_prompt = int(_tokenizer.PaligemmaTokenizer(64).tokenize("fold", state)[1].sum())
    # The prompt is unchanged and followed by the FAST tokens, "|" and the end of sequence token.
    assert np.array_equal(data["tokenized_prompt"][:num_prompt], prompt_tokens[:num_prompt])
    fast_tokens = data["tokenized_prompt"][num_prompt : num_prompt + 3]
    assert fast_tokens.tolist() == [_tokenizer.PALIGEMMA_VOCAB_SIZE - 129 - i for i in range(3)]
    assert not data["token_ar_mask"][:num_prompt].any()
    assert data["token_ar_mask"][num_prompt : num_prompt + 3].all()
    assert np.array_equal(data["token_ar_mask"].astype(bool), data["token_loss_mask"])

    # Without actions, this is the regular prompt.
    data = transform({"prompt": "fold", "state": state})
    assert np.array_equal(data["tokenized_prompt"], prompt_tokens)
    assert "token_ar_mask" not in data


def test_subtask_tokenizer():
    tokenizer = _tokenizer.SubtaskTokenizer(max_len=48)
    transform = _transforms.TokenizeSubtaskInputs(tokenizer)
    data = transform({"prompt": "clean up", "previous_subtasks": "open the fridge", "subtask_target": "grab the milk"})
    tokens = data["tokenized_prompt"]
    target = tokens[data["token_loss_mask"]]
    assert tokenizer.decode(target) == "grab the milk"

    data = transform({"prompt": "clean up"})
    assert not data["token_loss_mask"].any()

    extract = _transforms.ExtractSubtask(tokenizer)
    assert extract({"actions": target.astype(np.float32), "state": np.zeros(2)}) == {"subtask": "grab the milk"}


def test_history_inputs():
    obs = {"cam": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8), "state": np.zeros(3)}
    history = [{"cam": np.full((2, 2, 3), i, np.uint8), "wrist": np.full((2, 2, 3), i, np.uint8)} for i in (1, 2)]
    transform = _transforms.HistoryInputs(_to_model_images, _to_model_images, num_frames=3)

    data = transform({**obs, "history": history, "history_valid": np.array([False, True])})
    frames = data["history_image"]["base_0_rgb"]
    assert frames.shape == (3, 2, 2, 3)
    # Missing frames are padded at the beginning and invalid frames are masked.
    assert frames[1].max() == 1
    assert frames[2].max() == 2
    assert data["history_image_mask"]["base_0_rgb"].tolist() == [False, False, True]

    assert "history_image" not in transform(dict(obs))
    transform = _transforms.HistoryInputs(_to_model_images, _to_model_images, num_frames=3, dropout=1.0)
    data = transform({**obs, "history": history})
    assert not data["history_image_mask"]["base_0_rgb"].any()
