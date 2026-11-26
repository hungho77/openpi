"""Merge the three ALOHA left-hand LeRobot v2.1 datasets into one multitask v2.1 set.

Sources (task_index in this order, fixed):
    0 cover_and_stack (100 eps)   1 sort_blocks (100 eps)   2 store_block (49 eps)

Episodes are renumbered 0..248 in that order; episode_index / index / task_index are rewritten in every parquet;
videos are hard-linked (symlink fallback), never re-encoded; meta is regenerated. Source trees are only read.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ZOO = Path("/mnt/data/sftp/data/hunght23/DATA_ZOO")
SRC = [ZOO / "aloha_left_hand" / n for n in ("cover_and_stack", "sort_blocks", "store_block")]
DST = ZOO / "aloha_left_hand_merged_v21" / "hungho77" / "aloha-left-hand-3task"
CHUNK = 1000


def column_stats(v: np.ndarray) -> dict:
    v = v.astype(np.float64)
    return {"min": [v.min()], "max": [v.max()], "mean": [v.mean()], "std": [v.std()], "count": [len(v)]}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def main() -> int:
    if DST.exists():
        print(f"refusing to overwrite existing {DST}")
        return 1
    infos = [json.loads((s / "meta/info.json").read_text()) for s in SRC]
    assert all(i["features"] == infos[0]["features"] for i in infos), "features differ"
    assert all(i["fps"] == 15 and i["robot_type"] == "ALOHA" and i["codebase_version"] == "v2.1" for i in infos)
    video_keys = [k for k, f in infos[0]["features"].items() if f["dtype"] == "video"]

    episodes, ep_stats, tasks = [], [], []
    e = frame_off = 0
    for task_index, (s, info) in enumerate(zip(SRC, infos, strict=True)):
        t = [json.loads(x) for x in (s / "meta/tasks.jsonl").read_text().splitlines() if x.strip()]
        assert len(t) == 1, t
        assert t[0]["task_index"] == 0, t
        tasks.append({"task_index": task_index, "task": t[0]["task"]})
        src_eps = [json.loads(x) for x in (s / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
        src_stats = {
            r["episode_index"]: r["stats"]
            for r in map(json.loads, (s / "meta/episodes_stats.jsonl").read_text().splitlines())
            if r
        }
        n_frames = 0
        for m in src_eps:
            e_src, n = m["episode_index"], m["length"]
            c_src = e_src // info["chunks_size"]
            d = pd.read_parquet(s / info["data_path"].format(episode_chunk=c_src, episode_index=e_src))
            assert len(d) == n, f"{s.name} ep {e_src}"
            assert (d.frame_index.to_numpy() == np.arange(n)).all(), f"{s.name} ep {e_src}"
            assert (d.episode_index == e_src).all()
            assert (d.task_index == 0).all()
            d["episode_index"] = e
            d["index"] = np.arange(frame_off, frame_off + n)
            d["task_index"] = task_index
            out = DST / info["data_path"].format(episode_chunk=e // CHUNK, episode_index=e)
            out.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(out, index=False)
            for k in video_keys:
                link(
                    s / info["video_path"].format(episode_chunk=c_src, video_key=k, episode_index=e_src),
                    DST / info["video_path"].format(episode_chunk=e // CHUNK, video_key=k, episode_index=e),
                )
            st = dict(src_stats[e_src])
            for k in ["episode_index", "index", "task_index"]:
                st[k] = column_stats(d[k].to_numpy())
            episodes.append({"episode_index": e, "tasks": [tasks[-1]["task"]], "length": n})
            ep_stats.append({"episode_index": e, "stats": st})
            e += 1
            frame_off += n
            n_frames += n
        assert n_frames == info["total_frames"], s.name
        assert len(src_eps) == info["total_episodes"], s.name
        print(f"{task_index} {s.name}: {len(src_eps)} episodes, {n_frames} frames", flush=True)

    (DST / "meta").mkdir(parents=True, exist_ok=True)
    info = dict(infos[0])
    info.update(
        total_episodes=e,
        total_frames=frame_off,
        total_tasks=len(tasks),
        total_videos=e * len(video_keys),
        total_chunks=(e - 1) // CHUNK + 1,
        chunks_size=CHUNK,
        splits={"train": f"0:{e}"},
    )
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4))
    for name, rows in (("episodes", episodes), ("episodes_stats", ep_stats), ("tasks", tasks)):
        (DST / f"meta/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    from lerobot.common.datasets.compute_stats import aggregate_stats

    agg = aggregate_stats(
        [{k: {q: np.asarray(v) for q, v in s.items()} for k, s in r["stats"].items()} for r in ep_stats]
    )
    (DST / "meta/stats.json").write_text(
        json.dumps({k: {q: np.asarray(v).tolist() for q, v in s.items()} for k, s in agg.items()}, indent=4)
    )
    print(f"done: {e} episodes, {frame_off} frames -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
