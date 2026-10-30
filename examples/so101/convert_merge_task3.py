"""Convert three SO101 left-arm LeRobot v3.0 datasets to v2.1 and merge them into one multitask set.

Sources (task_index in this order, fixed):
    0 hungho77/so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube
    1 hungho77/so101-left_arm_task_drawer_placing_red_squre_block
    2 hungho77/so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube

openpi pins lerobot at a version that reads v2.1 only, and the converter used for the 11-task set
(scripts/lerobot_conversion/convert_v3_to_v2.py) is no longer in the tree, so this does both steps:

* data: v3.0 packs many episodes per parquet file; v2.1 wants one file per episode. The three global
  columns (episode_index, index, task_index) are rewritten, as in merge_leftarm_collection.py.
* videos: v3.0 concatenates episodes per mp4. Every episode starts on a keyframe (each was encoded
  separately and stream-concatenated), so each one is cut back out with `ffmpeg -c copy`: no
  re-encode, the frames are bit-identical. The packet count of every cut is checked against the
  episode length.
* `observation.images.top_depth` is dropped: π0.5 uses only `top` and `wrist`, and the v2.1 loader
  would decode every video key it finds in the features.
* per-episode stats: action/state/image stats come from the v3.0 episode metadata unchanged; the
  stats of the rewritten index columns are recomputed.
"""

import glob
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ZOO = Path("/mnt/data/sftp/data/hunght23/DATA_ZOO")
SRC = [
    ZOO / "so101_leftarm_task3_src/so101-left_arm_task_red_cube_out_of_pot_into_the_orange_hollow_cube",
    ZOO / "so101_drawer_red_block_v30/hungho77/so101-left_arm_task_drawer_placing_red_squre_block",
    ZOO / "so101_leftarm_task3_src/so101_left_arm_open_drawer_pick_red_cube_stack_on_blue_cube",
]
DST = ZOO / "so101_leftarm_task3_v21" / "hungho77" / "so101-left-arm-task3"
VIDEO_KEYS = ["observation.images.top", "observation.images.wrist"]
DROP = ["observation.images.top_depth"]
STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
CHUNK = 1000


def tolist(x):
    if isinstance(x, np.ndarray):
        return [tolist(v) for v in x] if x.dtype == object else x.tolist()
    return x


def column_stats(v: np.ndarray) -> dict:
    v = v.astype(np.float64)
    q = np.quantile(v, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {"min": [v.min()], "max": [v.max()], "mean": [v.mean()], "std": [v.std()], "count": [len(v)],
            **{k: [float(x)] for k, x in zip(["q01", "q10", "q50", "q90", "q99"], q)}}


def cut(job):
    src, start, dur, dst, length = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-i", str(src), "-t", f"{dur:.6f}",
                    "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst)], check=True)
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(dst)], check=True, capture_output=True,
                       text=True).stdout.strip()
    return str(dst), int(n), length


def main() -> int:
    if DST.exists():
        print(f"refusing to overwrite existing {DST}")
        return 1
    infos = [json.loads((s / "meta/info.json").read_text()) for s in SRC]
    feats = [{k: v for k, v in i["features"].items() if k not in DROP} for i in infos]
    assert all(f == feats[0] for f in feats), "features differ between sources"
    assert all(i["fps"] == 30 and i["robot_type"] == infos[0]["robot_type"] for i in infos)
    fps = infos[0]["fps"]

    episodes, ep_stats, tasks, jobs = [], [], [], []
    ep_off = frame_off = 0
    for task_index, s in enumerate(SRC):
        meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{s}/meta/episodes/chunk-*/*.parquet"))])
        meta = meta.sort_values("episode_index")
        task = {t for ts in meta["tasks"] for t in ts}
        assert len(task) == 1, f"{s.name}: expected one task, got {task}"
        tasks.append({"task_index": task_index, "task": task.pop()})
        data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{s}/data/chunk-*/*.parquet"))])
        for _, m in meta.iterrows():
            e_src, e = int(m["episode_index"]), ep_off
            d = data[data.episode_index == e_src].sort_values("frame_index").reset_index(drop=True)
            n = int(m["length"])
            assert len(d) == n and (d.frame_index.to_numpy() == np.arange(n)).all(), f"{s.name} ep {e_src}"
            assert abs(float(d.timestamp.iloc[0])) < 1e-6, f"{s.name} ep {e_src}: timestamps not episode-relative"
            d["episode_index"] = e
            d["index"] = np.arange(frame_off, frame_off + n)
            d["task_index"] = task_index
            out = DST / f"data/chunk-{e // CHUNK:03d}/episode_{e:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(out, index=False)
            for k in VIDEO_KEYS:
                src = s / f"videos/{k}/chunk-{int(m[f'videos/{k}/chunk_index']):03d}/file-{int(m[f'videos/{k}/file_index']):03d}.mp4"
                t0, t1 = float(m[f"videos/{k}/from_timestamp"]), float(m[f"videos/{k}/to_timestamp"])
                jobs.append((src, t0, t1 - t0, DST / f"videos/chunk-{e // CHUNK:03d}/{k}/episode_{e:06d}.mp4", n))
            st = {}
            for k in ["action", "observation.state", *VIDEO_KEYS]:
                st[k] = {q: tolist(m[f"stats/{k}/{q}"]) for q in STAT_KEYS}
            for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                st[k] = column_stats(d[k].to_numpy())
            episodes.append({"episode_index": e, "tasks": [tasks[-1]["task"]], "length": n})
            ep_stats.append({"episode_index": e, "stats": st})
            ep_off += 1
            frame_off += n
        print(f"{task_index} {s.name}: {len(meta)} episodes, {len(data)} frames", flush=True)

    with ProcessPoolExecutor(8) as ex:
        bad = [(p, n, L) for p, n, L in ex.map(cut, jobs, chunksize=4) if n != L]
    if bad:
        print("frame count mismatch:", bad[:10])
        return 1

    (DST / "meta").mkdir(parents=True, exist_ok=True)
    info = {k: v for k, v in infos[0].items() if k != "features"}
    info.update(codebase_version="v2.1", total_episodes=ep_off, total_frames=frame_off, total_tasks=len(tasks),
                total_videos=ep_off * len(VIDEO_KEYS), total_chunks=(ep_off - 1) // CHUNK + 1, chunks_size=CHUNK,
                splits={"train": f"0:{ep_off}"},
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                features=feats[0])
    for k in ("data_files_size_in_mb", "video_files_size_in_mb"):
        info.pop(k, None)
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4))
    for name, rows in (("episodes", episodes), ("episodes_stats", ep_stats), ("tasks", tasks)):
        (DST / f"meta/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    from lerobot.common.datasets.compute_stats import aggregate_stats

    agg = aggregate_stats([{k: {q: np.asarray(v) for q, v in s.items()} for k, s in r["stats"].items()} for r in ep_stats])
    (DST / "meta/stats.json").write_text(json.dumps({k: {q: np.asarray(v).tolist() for q, v in s.items()}
                                                      for k, s in agg.items()}, indent=4))
    print(f"done: {ep_off} episodes, {frame_off} frames, {len(jobs)} videos -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
