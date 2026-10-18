import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_subgoal_dataset():
    from openpi.training import episode_metadata as _episode_metadata

    # Two episodes of length 5 and 10, stored contiguously. Each item is its global index.
    episode_index = np.array([0] * 5 + [1] * 10)
    frame_index = np.array(list(range(5)) + list(range(10)))
    items = [{"index": i, "prompt": "task. Subtask: grasp"} for i in range(15)]
    config = _episode_metadata.SubgoalConfig(
        probability=1.0, end_of_segment_probability=0.5, max_horizon_s=1.0, subtask_dropout=0.0
    )
    dataset = _data_loader.SubgoalDataset(
        items,
        config,
        episode_index=episode_index,
        frame_index=frame_index,
        episode_lengths={0: 5, 1: 10},
        segments={1: ((0, 4, "grasp", None),)},
        fps=3,
    )
    for _ in range(50):
        for i in range(15):
            item = dataset[i]
            goal = item["subgoal"]["index"]
            # The subgoal is in the same episode, in the future, and at most 3 frames (1 second) ahead.
            assert episode_index[goal] == episode_index[i]
            assert i <= goal <= i + 3 or (i in (5, 6, 7, 8) and goal == 8)


def test_history_dataset():
    items = [{"cam": np.arange(3), "cam_is_pad": np.array([True, False, False]), "state": i} for i in range(2)]
    item = _data_loader.HistoryDataset(items, ["cam"], num_frames=2)[0]
    assert item["cam"] == 2
    assert [frame["cam"] for frame in item["history"]] == [0, 1]
    assert item["history_valid"].tolist() == [False, True]
    assert "cam_is_pad" not in item


def test_subtask_target_dataset():
    from openpi.training import episode_metadata as _episode_metadata

    items = [{"index": i} for i in range(8)]
    dataset = _data_loader.SubtaskTargetDataset(
        items,
        _episode_metadata.SubtaskTargetConfig(max_previous_subtasks=1),
        episode_index=np.array([0] * 4 + [1] * 4),
        frame_index=np.array([0, 1, 2, 3, 0, 1, 2, 3]),
        episode_lengths={0: 4, 1: 4},
        segments={1: ((0, 2, "a", None), (2, 4, "b", None)), 0: ((1, 2, None, True),)},
    )
    assert len(dataset) == 4
    assert [(x["index"], x["subtask_target"], x["previous_subtasks"]) for x in dataset] == [
        (4, "a", ""),
        (5, "a", ""),
        (6, "b", "a"),
        (7, "b", "a"),
    ]
