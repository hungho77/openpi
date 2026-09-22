"""Open-loop evaluation of a trained SO101 pi0.5 checkpoint against the dataset.

openpi has no open_loop_eval.py, so this mirrors what gr00t/eval/open_loop_eval.py does,
to make the two numbers comparable:

  * walk a trajectory in strides of `action_horizon`
  * at each anchor, feed the observation and take the predicted chunk
  * compare against the ground-truth action chunk starting at that anchor
  * report MSE/MAE in the dataset's own action units

Raw MAE is meaningless across robots, so a normalized figure is also printed: MAE divided
by the mean per-dimension action std, i.e. error as a fraction of how much the actions
actually move. That is the number to compare against another model.

Usage:
  HF_LEROBOT_HOME=... python scripts/eval_so101_open_loop.py \
      --checkpoint-dir checkpoints/pi05_so101/so101_bs128/6000 \
      --traj-ids 0 25 60 80 110 130 --steps 400
"""

import argparse
import json
import pathlib

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_so101")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--repo-id", default="hungho77/so101-multitask")
    p.add_argument("--traj-ids", type=int, nargs="+", default=[0, 25, 60, 80, 110, 130])
    p.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Max frames of each episode to walk. Episodes shorter than this are walked in full.",
    )
    p.add_argument(
        "--eval-horizon",
        type=int,
        default=None,
        help="Score only the first N steps of each predicted chunk (default: the whole chunk). "
        "Set this to another model's horizon to compare fairly — error accumulates along a "
        "chunk, so a 50-step prediction is not comparable to a 16-step one.",
    )
    p.add_argument(
        "--norm-std",
        type=float,
        default=None,
        help="Divide the reported MAE by this, to express error as a fraction of action "
        "scale. Pass the ABSOLUTE action std (the checkpoint's own norm_stats holds DELTA "
        "std for the arm joints, which is a different quantity and must not be mixed in).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    train_config = _config.get_config(args.config_name)
    horizon = train_config.model.action_horizon
    # How many steps of each chunk to score. Striding still uses the scored length so the
    # anchors tile the trajectory without gaps or overlap.
    score = min(args.eval_horizon or horizon, horizon)

    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    dataset = LeRobotDataset(args.repo_id)
    meta = dataset.meta

    # Episode boundaries come straight from the dataset; don't recompute them from lengths.
    edi = dataset.episode_data_index
    ep_start = {i: int(edi["from"][i]) for i in range(meta.total_episodes)}
    ep_len = {i: int(edi["to"][i]) - int(edi["from"][i]) for i in range(meta.total_episodes)}

    per_traj = {}
    for tid in args.traj_ids:
        start, length = ep_start[tid], ep_len[tid]
        n = min(args.steps, length - score)
        if n <= 0:
            print(f"  traj {tid}: too short ({length} frames) for horizon {score}, skipped")
            continue

        errs = []
        for t in range(0, n, score):
            frame = dataset[start + t]
            obs = {
                "observation/image": frame["observation.images.top"].numpy(),
                "observation/wrist_image": frame["observation.images.wrist"].numpy(),
                "observation/state": frame["observation.state"].numpy(),
                "prompt": frame["task"],
            }
            pred = np.asarray(policy.infer(obs)["actions"])[:score]  # (score, 6)

            gt = np.stack(
                [dataset[start + t + k]["action"].numpy() for k in range(score)], axis=0
            )
            errs.append(pred[: len(gt)] - gt)

        e = np.concatenate(errs, axis=0)
        mse, mae = float((e**2).mean()), float(np.abs(e).mean())
        per_traj[tid] = (mse, mae)
        print(f"  traj {tid:3d}: MSE {mse:.6f}  MAE {mae:.6f}   ({len(e)} steps compared)")

    if not per_traj:
        print("nothing evaluated")
        return

    mses = [v[0] for v in per_traj.values()]
    maes = [v[1] for v in per_traj.values()]
    print(f"\nScored horizon: {score} of {horizon} predicted steps")
    print(f"Average MSE: {np.mean(mses):.6f}")
    print(f"Average MAE: {np.mean(maes):.6f}")

    if args.norm_std:
        print(f"Normalized MAE: {np.mean(maes) / args.norm_std * 100:.2f}% of action scale "
              f"(divided by absolute action std {args.norm_std})")
    else:
        # Deliberately not defaulting to the checkpoint's own norm_stats: for this config
        # those are DELTA stats for the arm joints, while the errors above are in absolute
        # units. Dividing one by the other produces a meaningless number.
        ns_path = pathlib.Path(args.checkpoint_dir) / "assets" / args.repo_id / "norm_stats.json"
        if ns_path.exists():
            ns = json.loads(ns_path.read_text())
            ns = ns.get("norm_stats", ns)
            std = np.asarray(ns["actions"]["std"])[:6]
            print(f"(checkpoint norm_stats action std = {np.round(std, 2)} — these are DELTA "
                  f"stats for dims 0-4, not comparable to the absolute-unit MAE above. "
                  f"Pass --norm-std with the absolute action std to get a normalized figure.)")


if __name__ == "__main__":
    main()
