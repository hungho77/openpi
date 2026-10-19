"""Open-loop evaluation of a trained policy on episodes of its LeRobot dataset.

Every `action_horizon` steps, the policy predicts an action chunk from the recorded observation, and the first
`action_horizon` actions are compared with the recorded actions. The errors are computed on the unnormalized actions.

Usage:
    uv run scripts/eval_open_loop.py --config-name pi05_so101_left \
        --checkpoint-dir checkpoints/pi05_so101_left/so101_v1/30000 \
        --episodes 0 50 100 --output-dir eval/so101_v1
"""

import json
import logging
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

from openpi import transforms as _transforms
from openpi.policies import policy_config
from openpi.training import config as _config


def _plot(path: pathlib.Path, pred: np.ndarray, gt: np.ndarray, action_horizon: int, title: str) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    num_dims = gt.shape[-1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(10, 2 * num_dims), sharex=True)
    for dim, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(gt[:, dim], label="recorded")
        ax.plot(pred[:, dim], label="predicted")
        for start in range(0, len(gt), action_horizon):
            ax.axvline(start, color="gray", alpha=0.1)
        ax.set_ylabel(f"dim {dim}")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_episode(
    policy, dataset, prepare, episode: int, action_horizon: int, steps: int, speed: str = "episode"
) -> dict[str, np.ndarray]:
    start = int(dataset.episode_data_index["from"][episode])
    end = int(dataset.episode_data_index["to"][episode])
    length = min(end - start, steps)
    gt = np.stack([np.asarray(a) for a in dataset.hf_dataset.select(range(start, start + length))["action"]])

    pred = []
    for step in range(0, length, action_horizon):
        obs = prepare(dataset[start + step])
        # The recorded actions would be used as the real-time chunking prefix.
        obs.pop("actions", None)
        if speed == "episode":
            # Prompt with the speed of the recorded episode, so that the predictions can match its timing.
            obs["speed"] = end - start
        elif speed == "none":
            obs["speed"] = None
        chunk = np.asarray(policy.infer(obs)["actions"])
        pred.append(chunk[: min(action_horizon, length - step)])
    pred = np.concatenate(pred)[:, : gt.shape[-1]]
    # Baseline: repeat the recorded action at the start of each chunk.
    hold = np.repeat(gt[::action_horizon], action_horizon, axis=0)[:length]
    return {"pred": pred, "gt": gt, "hold": hold}


def main(
    config_name: str,
    checkpoint_dir: str,
    *,
    episodes: list[int],
    output_dir: str,
    action_horizon: int = 16,
    steps: int = 400,
    device: str | None = None,
    plot: bool = True,
    speed: str = "episode",
) -> None:
    """Runs the open-loop evaluation.

    Args:
        speed: Speed in the prompt of models trained with episode metadata. "episode" uses the length of the
            evaluated episode, "target" the `target_speed` of the config, and "none" leaves the speed out.
    """
    if speed not in ("episode", "target", "none"):
        raise ValueError(f"Unknown speed mode: {speed}")
    logging.basicConfig(level=logging.INFO, force=True)
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)

    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    prepare = _transforms.compose(
        [_transforms.PromptFromLeRobotTask(meta.tasks), *data_config.repack_transforms.inputs]
    )

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for episode in tqdm.tqdm(episodes, desc="Episodes"):
        r = evaluate_episode(policy, dataset, prepare, episode, action_horizon, steps, speed)
        np.savez(out / f"episode_{episode}.npz", **r)
        results[episode] = {
            "steps": len(r["gt"]),
            "mse": float(np.mean((r["pred"] - r["gt"]) ** 2)),
            "mae": float(np.mean(np.abs(r["pred"] - r["gt"]))),
            "hold_mse": float(np.mean((r["hold"] - r["gt"]) ** 2)),
            "hold_mae": float(np.mean(np.abs(r["hold"] - r["gt"]))),
        }
        logging.info(f"Episode {episode}: {results[episode]}")
        if plot:
            _plot(
                out / f"episode_{episode}.png", r["pred"], r["gt"], action_horizon, f"{config_name} episode {episode}"
            )

    summary = {
        key: float(np.mean([r[key] for r in results.values()])) for key in ("mse", "mae", "hold_mse", "hold_mae")
    }
    report = {
        "config": config_name,
        "checkpoint": str(checkpoint_dir),
        "action_horizon": action_horizon,
        "steps": steps,
        "speed": speed,
        "episodes": results,
        "mean": summary,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    logging.info(f"Mean over {len(results)} episodes: {summary}")


if __name__ == "__main__":
    tyro.cli(main)
