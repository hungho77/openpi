"""Summarize eval_open_loop.py outputs (episode_*.npz) for the ALOHA left-hand set: MAE in radians over all 14 dims,
the left 7 (arm + gripper), the left arm 6, the left gripper and the right 7, per episode / task and the hold
baseline (recorded action at each chunk start repeated over the chunk).
For the left-arm-only (7-dim) set the groups are all7 / arm6 / gripper instead."""

import json
import pathlib
import sys

import numpy as np

TASK_OF = lambda e: 0 if e < 100 else (1 if e < 200 else 2)  # noqa: E731  (merged order 100 / 100 / 49)
GROUPS = {
    "all14": slice(0, 14),
    "left7": slice(0, 7),
    "left_arm6": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right7": slice(7, 14),
}
GROUPS_LEFT = {"all7": slice(0, 7), "arm6": slice(0, 6), "gripper": slice(6, 7)}


def main(eval_dir: str) -> None:
    d = pathlib.Path(eval_dir)
    rows = {}
    for f in sorted(d.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        e = int(f.stem.split("_")[1])
        r = np.load(f)
        groups = {14: GROUPS, 7: GROUPS_LEFT}[r["gt"].shape[1]]
        rows[e] = {
            f"{src}_{g}": float(np.mean(np.abs(r[src][:, s] - r["gt"][:, s])))
            for src in ("pred", "hold")
            for g, s in groups.items()
        }
        rows[e]["steps"] = len(r["gt"])
    keys = [k for k in next(iter(rows.values())) if k != "steps"]
    mean = {k: float(np.mean([v[k] for v in rows.values()])) for k in keys}
    per_task = {
        t: {k: float(np.mean([v[k] for e, v in rows.items() if TASK_OF(e) == t])) for k in keys}
        for t in sorted({TASK_OF(e) for e in rows})
    }
    out = {"episodes": rows, "per_task": per_task, "mean": mean}
    (d / "metrics_aloha.json").write_text(json.dumps(out, indent=2))
    print(f"{'ep':>4} {'steps':>5} " + " ".join(f"{k:>18}" for k in keys))
    for e, v in rows.items():
        print(f"{e:>4} {v['steps']:>5} " + " ".join(f"{v[k]:>18.4f}" for k in keys))
    for t, v in per_task.items():
        print(f"task{t:<1} {'':>5} " + " ".join(f"{v[k]:>18.4f}" for k in keys))
    print(f"mean {'':>5} " + " ".join(f"{mean[k]:>18.4f}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1])
