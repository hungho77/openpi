"""scripts/compute_norm_stats.py over ALL frames, with video decoding stubbed out (a 3x8x8 zero frame per camera).

Norm stats only use "state" and "actions", which come from the parquet columns; the images are dropped before the
statistics, so the result is identical to the stock script, without decoding 63k x 2 AV1 frames.
Usage: HF_LEROBOT_HOME=... .venv/bin/python examples/aloha_left_hand/compute_norm_stats_novideo.py <config-name>
"""

import sys

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import torch


def _fake_videos(self, query_timestamps, ep_idx):
    return {
        k: torch.zeros(len(ts), 3, 8, 8) if len(ts) > 1 else torch.zeros(3, 8, 8) for k, ts in query_timestamps.items()
    }


lerobot_dataset.LeRobotDataset._query_videos = _fake_videos  # noqa: SLF001

if __name__ == "__main__":  # the loader workers are spawned and re-import this file (which re-applies the stub)
    sys.path.insert(0, "scripts")
    import compute_norm_stats

    compute_norm_stats.main(sys.argv[1])
