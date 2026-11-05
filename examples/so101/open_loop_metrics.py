"""Metrics of `scripts/eval_open_loop.py` runs on SO101 data, per task.

Besides the MAE over all predicted steps, it reports where the error comes from:

- `step1` / `step16`: MAE of the first and the last action of each 16-step chunk.
- `gain`: how much of the recorded motion over a chunk the prediction covers (1 = all), over the chunks in which an
  arm joint moves by at least 2 units.
- `jump`: mean change between the last action of a chunk and the first action of the next one; the recorded
  trajectory gives the reference value (`jump_recorded`).
- `grasp` / `place`: MAE of the 5 arm joints over the `--phase-frames` frames before the gripper closes (grasp) or
  opens again (release). These are the phases that decide whether a precise placement succeeds.

Usage:
    uv run examples/so101/open_loop_metrics.py --repo-id hungho77/so101-left-arm-task3 --norm-std 29.97 \
        eval/run_a eval/run_b
"""

import json
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tyro


def gripper_events(gripper: np.ndarray) -> tuple[list[int], list[int]]:
    """Frames at which the recorded gripper closes and opens again, with a hysteresis around the midpoint between
    its extreme values."""
    low, high = np.percentile(gripper, 2), np.percentile(gripper, 98)
    if high - low < 5:
        return [], []
    close_level, open_level = low + 0.35 * (high - low), low + 0.65 * (high - low)
    closes, opens, closed = [], [], False
    for t, g in enumerate(gripper):
        if not closed and g < close_level:
            closes.append(t)
            closed = True
        elif closed and g > open_level:
            opens.append(t)
            closed = False
    return closes, opens


def phase_mae(err: np.ndarray, events: list[int], frames: int) -> list[float]:
    return [float(np.abs(err[max(0, t - frames) : t, :5]).mean()) for t in events if t > 0]


def episode_metrics(pred: np.ndarray, gt: np.ndarray, horizon: int, phase_frames: int) -> dict:
    n = len(gt) // horizon * horizon
    err = pred - gt
    chunks_p, chunks_g = pred[:n].reshape(-1, horizon, pred.shape[-1]), gt[:n].reshape(-1, horizon, gt.shape[-1])
    by_step = np.abs(chunks_p - chunks_g).mean(axis=(0, 2))
    disp_g = chunks_g[:, -1, :5] - chunks_g[:, 0, :5]
    disp_p = chunks_p[:, -1, :5] - chunks_p[:, 0, :5]
    moving = np.abs(disp_g).max(axis=1) >= 2
    boundaries = np.arange(horizon, len(gt), horizon)
    closes, opens = gripper_events(gt[:, 5])
    return {
        "steps": len(gt),
        "mae": float(np.abs(err).mean()),
        "mse": float((err**2).mean()),
        "step1": float(by_step[0]),
        "step16": float(by_step[-1]),
        "gain_num": float((disp_p[moving] * disp_g[moving]).sum()),
        "gain_den": float((disp_g[moving] ** 2).sum()),
        "jump": float(np.abs(pred[boundaries] - pred[boundaries - 1]).mean()),
        "jump_recorded": float(np.abs(gt[boundaries] - gt[boundaries - 1]).mean()),
        "grasp": phase_mae(err, closes, phase_frames),
        "place": phase_mae(err, opens, phase_frames),
    }


def summarize(rows: list[dict], norm_std: float) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    grasp = [v for r in rows for v in r["grasp"]]
    place = [v for r in rows for v in r["place"]]
    return {
        "episodes": len(rows),
        "mae": mean("mae"),
        "mae_pct": mean("mae") / norm_std * 100,
        "mse": mean("mse"),
        "step1": mean("step1"),
        "step16": mean("step16"),
        "gain": sum(r["gain_num"] for r in rows) / max(sum(r["gain_den"] for r in rows), 1e-9),
        "jump": mean("jump"),
        "jump_recorded": mean("jump_recorded"),
        "grasp": float(np.mean(grasp)) if grasp else float("nan"),
        "place": float(np.mean(place)) if place else float("nan"),
        "num_grasp": len(grasp),
        "num_place": len(place),
    }


def main(
    runs: tyro.conf.Positional[list[pathlib.Path]],
    *,
    repo_id: str,
    norm_std: float,
    horizon: int = 16,
    phase_frames: int = 45,
    output: pathlib.Path | None = None,
) -> None:
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    report = {}
    for run in runs:
        per_task: dict[str, list[dict]] = {}
        for path in sorted(run.glob("episode_*.npz")):
            episode = int(path.stem.split("_")[1])
            data = np.load(path)
            task = meta.episodes[episode]["tasks"][0]
            per_task.setdefault(task, []).append(episode_metrics(data["pred"], data["gt"], horizon, phase_frames))
        rows = [r for task_rows in per_task.values() for r in task_rows]
        report[str(run)] = {
            "all": summarize(rows, norm_std),
            "tasks": {task: summarize(task_rows, norm_std) for task, task_rows in per_task.items()},
        }

    keys = ("mae", "mae_pct", "step1", "step16", "gain", "jump", "grasp", "place")
    print(f"{'run / task':58s} {'eps':>4s} " + " ".join(f"{k:>8s}" for k in keys))
    for run, r in report.items():
        for name, s in [("all", r["all"]), *r["tasks"].items()]:
            label = run if name == "all" else f"  {name[:54]}"
            print(f"{label[-58:]:58s} {s['episodes']:4d} " + " ".join(f"{s[k]:8.3f}" for k in keys))
    if output is not None:
        output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
