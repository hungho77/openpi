"""Compute RECAP advantages with a trained value function.

Runs the value function on every frame of the dataset, computes the advantage of each frame and binarizes it into the
advantage indicator that is used to train the policy (see `EpisodeMetadataConfig.advantages_path`):

- n-step advantage (default, N=50): A_t = sum_{k=t}^{t+N-1} r_k + V(o_{t+N}) - V(o_t).
- Monte Carlo advantage (--n-step 0): A_t = R_t - V(o_t).

Rewards are -1 per step (normalized by the maximum episode length of the task) and failed episodes receive an extra
penalty at the end, consistent with the returns used to train the value function. The indicator is positive for the
`positive_fraction` of frames with the largest advantage within each task, and always positive for frames inside
intervention segments (human corrections).

Usage:
    uv run scripts/compute_advantages.py --config-name <value_config> \
        --checkpoint-dir checkpoints/<value_config>/<exp_name>/<step> --output advantages.npz
"""

import dataclasses
import logging
import pathlib

import etils.epath as epath
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
from openpi.shared import nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.episode_metadata as _episode_metadata


def compute_values(
    config: _config.TrainConfig, checkpoint_dir: epath.Path, *, batch_size: int, num_workers: int
) -> np.ndarray:
    """Returns the predicted value of every frame of the dataset, indexed by the dataset "index"."""
    data_config = config.data.create(config.assets_dirs, config.model)
    # Use the norm stats of the checkpoint, same as during training.
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    predict_value = nnx_utils.module_jit(model.predict_value)

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
    )

    values = []
    for batch in tqdm.tqdm(loader, desc="Computing values"):
        observation = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        values.append(np.asarray(predict_value(observation), dtype=np.float32))
    return np.concatenate(values)


def compute_advantages(
    values: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    table: dict[int, dict],
    *,
    n_step: int,
    failure_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the advantage and the normalized return of every frame."""
    advantages = np.zeros_like(values)
    returns = np.zeros_like(values)
    for episode, info in table.items():
        (indices,) = np.nonzero(episode_index == episode)
        indices = indices[np.argsort(frame_index[indices])]
        length, max_length, success = info["length"], info["max_length"], info["success"]
        if len(indices) != length:
            raise ValueError(f"Episode {episode} has {len(indices)} frames but its length is {length}.")

        value = values[indices]
        frames = np.arange(length)
        # Same as `transforms.normalized_return`, vectorized over the frames of the episode.
        unclipped = -(length - 1 - frames) / max_length - (0.0 if success else failure_penalty)
        ret = np.clip(unclipped, -1.0, 0.0)
        # Monte Carlo advantage, also used when fewer than n_step frames are left in the episode.
        advantage = ret - value
        if n_step > 0:
            bootstrap = frames + n_step < length
            t = frames[bootstrap]
            advantage[bootstrap] = unclipped[t] - unclipped[t + n_step] + value[t + n_step] - value[t]
        advantages[indices] = advantage
        returns[indices] = ret
    return advantages, returns


def run(
    config: _config.TrainConfig,
    checkpoint_dir: str,
    output: str,
    *,
    n_step: int = 50,
    positive_fraction: float = 0.3,
    batch_size: int = 32,
    num_workers: int = 4,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.value_target is None:
        raise ValueError(f"Config {config.name} is not a value function config (data.value_target is not set).")

    values = compute_values(config, epath.Path(checkpoint_dir), batch_size=batch_size, num_workers=num_workers)

    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    columns = dataset.hf_dataset.select_columns(["index", "episode_index", "frame_index"]).with_format("numpy")
    index = np.asarray(columns["index"])
    if len(values) != len(index) or not np.array_equal(index, np.arange(len(index))):
        raise ValueError("The dataset frames are expected to be ordered by their global index.")
    episode_index = np.asarray(columns["episode_index"])
    frame_index = np.asarray(columns["frame_index"])

    table = _episode_metadata.build_value_table(data_config.value_target, dataset.meta)
    advantages, returns = compute_advantages(
        values,
        episode_index,
        frame_index,
        table,
        n_step=n_step,
        failure_penalty=data_config.value_target.failure_penalty,
    )

    # Binarize per task.
    indicator = np.zeros(len(values), dtype=bool)
    frame_tasks = np.asarray([table[int(e)]["task"] for e in episode_index])
    for task in np.unique(frame_tasks):
        mask = frame_tasks == task
        threshold = np.quantile(advantages[mask], 1.0 - positive_fraction)
        indicator[mask] = advantages[mask] > threshold
        corr = np.corrcoef(values[mask], returns[mask])[0, 1] if mask.sum() > 1 else float("nan")
        print(
            f"[{task}] frames={mask.sum()} threshold={threshold:.4f} positive={indicator[mask].mean():.1%} "
            f"corr(value, return)={corr:.3f}"
        )

    # Human corrections are always labeled as positive.
    num_interventions = 0
    for episode, info in table.items():
        for start, end in info.get("interventions", ()):
            mask = (episode_index == episode) & (frame_index >= start) & (frame_index < end)
            indicator[mask] = True
            num_interventions += int(mask.sum())
    print(f"{num_interventions} intervention frames labeled as positive.")
    print(f"Overall: {len(values)} frames, {indicator.mean():.1%} positive.")

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, indicator=indicator, advantage=advantages, value=values, value_target=returns)
    print(f"Wrote advantages to: {output_path}")


def main(
    config_name: str,
    checkpoint_dir: str,
    output: str,
    *,
    n_step: int = 50,
    positive_fraction: float = 0.3,
    batch_size: int = 32,
    num_workers: int = 4,
):
    logging.basicConfig(level=logging.INFO)
    run(
        _config.get_config(config_name),
        checkpoint_dir,
        output,
        n_step=n_step,
        positive_fraction=positive_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    tyro.cli(main)
