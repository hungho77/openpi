"""Merge the 11 SO101 left-arm task datasets into one multitask LeRobot v2.1 dataset.

Sources: every dataset in the hungho77/SO101_LeftArmOnly_Task collection, each a single
task, 50 episodes (one has 51), 30 fps, cameras `top` (360x640) and `wrist` (480x640).

Renaming episode files is not enough. Three columns inside each parquet are GLOBAL and
have to be rewritten, or the merged dataset loads and trains on silently wrong episode
boundaries:

    episode_index   shifted by the running episode count
    index           the global frame counter, shifted by the running frame count
    task_index      set to this source's ordinal (0..10)

Task order is fixed by SRC below, not by directory listing order, so the mapping from
task_index to instruction is reproducible across runs.

Videos are hardlinked where possible — same filesystem, so the merged copy costs no extra
space and still looks like a normal file to anything reading it.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ZOO = Path("/mnt/data/sftp/data/hunght23/DATA_ZOO")
SRC_ROOT = ZOO / "so101_leftarm_src"

# Fixed order => fixed task_index. Keep this list stable; reordering it silently
# renumbers every task in the published dataset.
SRC = [
    "so101_left_arm_pick_red_cube_into_pot",
    "so101_left_arm_pick_blue_cube_into_pot",
    "so101_left_arm_pick_banana_close_lid",
    "so101_left_arm_blue_cube_next_to_blue_cube",
    "so101_left_arm_blue_cube_out_of_pot_onto_blue_cube",
    "so101_left_arm_red_cube_out_of_pot_onto_red_cube",
    "so101_left_arm_stack_yellow_on_orange_cube",
    "so101_left_arm_stack_orange_blocks",
    "so101_left_arm_balance_red_blue_on_wooden_board",
    "so101_left_arm_red_cube_into_orange_hollow_cube",
    "so101_left_arm_blue_cubes_shortest_to_tallest",
]

# openpi resolves repo_id under HF_LEROBOT_HOME, so the tree has to live at
# <HF_LEROBOT_HOME>/<owner>/<name>.
DST = ZOO / "so101_leftarm_multitask_v21" / "hungho77" / "so101-left-arm-multitask"

VIDEO_KEYS = ["observation.images.top", "observation.images.wrist"]

# Compared field-for-field across sources. Video shapes are allowed to differ between the
# two cameras but not between datasets, which this check also covers.
MUST_MATCH = ("features", "fps", "robot_type", "codebase_version", "chunks_size",
              "data_path", "video_path")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    if DST.exists():
        print(f"refusing to overwrite existing {DST}")
        return 1

    # convert_v3_to_v2.py resolves its --root as a LeRobot HOME, not as the dataset
    # directory: it writes to <root>/<repo_id>. So each converted v2.1 tree sits one
    # level deeper than the directory the raw snapshot was downloaded into.
    roots = {n: SRC_ROOT / n / "hungho77" / n for n in SRC}
    missing = [n for n, p in roots.items() if not (p / "meta" / "episodes.jsonl").exists()]
    if missing:
        print("not converted to v2.1 yet (no meta/episodes.jsonl):")
        for n in missing:
            print("   ", n)
        return 1

    infos = {n: json.loads((p / "meta" / "info.json").read_text()) for n, p in roots.items()}
    ref = infos[SRC[0]]
    for n in SRC[1:]:
        for key in MUST_MATCH:
            if infos[n][key] != ref[key]:
                print(f"{n}: mismatch in {key}; refusing to merge")
                return 1

    ep_offset = 0
    frame_offset = 0
    episodes: list[dict] = []
    ep_stats: list[dict] = []
    tasks: list[dict] = []

    for task_index, name in enumerate(SRC):
        root = roots[name]
        src_tasks = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").open()]
        if len(src_tasks) != 1:
            print(f"{name}: expected exactly 1 task, found {len(src_tasks)}")
            return 1
        tasks.append({"task_index": task_index, "task": src_tasks[0]["task"]})

        src_eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").open()]
        src_stats = [json.loads(line) for line in (root / "meta" / "episodes_stats.jsonl").open()]
        if not len(src_eps) == len(src_stats) == infos[name]["total_episodes"]:
            print(f"{name}: episodes/stats/info disagree on episode count")
            return 1

        frames_here = 0
        for ep in src_eps:
            old = ep["episode_index"]
            new = old + ep_offset

            df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{old:06d}.parquet")
            df["episode_index"] = df["episode_index"] + ep_offset
            df["index"] = df["index"] + frame_offset
            df["task_index"] = task_index
            out = DST / "data" / "chunk-000" / f"episode_{new:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            frames_here += len(df)

            for vk in VIDEO_KEYS:
                link_or_copy(
                    root / "videos" / "chunk-000" / vk / f"episode_{old:06d}.mp4",
                    DST / "videos" / "chunk-000" / vk / f"episode_{new:06d}.mp4",
                )

            episodes.append({
                "episode_index": new,
                "tasks": [tasks[task_index]["task"]],
                "length": ep["length"],
            })

        for st in src_stats:
            st = dict(st)
            st["episode_index"] = st["episode_index"] + ep_offset
            ep_stats.append(st)

        print(f"  [{task_index:2d}] {name:52s} {len(src_eps):3d} ep -> "
              f"{ep_offset}..{ep_offset + len(src_eps) - 1}, {frames_here:6d} frames")
        ep_offset += len(src_eps)
        frame_offset += frames_here

    meta = DST / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "episodes.jsonl").open("w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as f:
        for s in ep_stats:
            f.write(json.dumps(s) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    info = dict(ref)
    info["total_episodes"] = ep_offset
    info["total_frames"] = frame_offset
    info["total_tasks"] = len(tasks)
    info["total_videos"] = ep_offset * len(VIDEO_KEYS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{ep_offset}"}
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    print(f"\nmerged -> {DST}")
    print(f"  {ep_offset} episodes, {frame_offset} frames, {len(tasks)} tasks")
    print("  meta/stats.json still needs aggregating from episodes_stats.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
