"""Print episode length statistics for a LeRobot dataset and optionally write an annotations template.

The statistics help to pick `EpisodeMetadataConfig.target_speed` (e.g. the 15th percentile of the episode lengths of
the task). The template contains one entry per episode that can be filled with "quality", "mistake" and "segments"
labels, see `openpi.training.episode_metadata` for the format.

Usage:
    uv run scripts/compute_episode_metadata.py --repo-id <repo_id> [--output annotations.json]
"""

import collections
import json
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import tyro

import openpi.training.episode_metadata as _episode_metadata


def main(repo_id: str, *, output: str | None = None, overwrite: bool = False, speed_bin_size: int = 500):
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    lengths = _episode_metadata.get_episode_lengths(dataset_meta)

    lengths_per_task = collections.defaultdict(dict)
    for index, episode in dataset_meta.episodes.items():
        for task in episode.get("tasks", []):
            lengths_per_task[task][int(index)] = lengths[int(index)]

    print(f"{len(lengths)} episodes, fps={dataset_meta.fps}, speed bin size={speed_bin_size}")
    print(f"All episodes: {_episode_metadata.speed_percentiles(lengths)}")
    for task, task_lengths in sorted(lengths_per_task.items()):
        print(f"[{len(task_lengths)} episodes] {task}: {_episode_metadata.speed_percentiles(task_lengths)}")

    if output is not None:
        output_path = pathlib.Path(output)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists, use --overwrite to replace it.")
        # "length" and "tasks" are only informative and ignored when loading the annotations.
        template = {
            str(index): {
                "length": lengths[int(index)],
                "tasks": episode.get("tasks", []),
                "quality": None,
                "success": None,
            }
            for index, episode in sorted(dataset_meta.episodes.items())
        }
        output_path.write_text(json.dumps(template, indent=2))
        print(f"Wrote annotations template to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
