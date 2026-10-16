#!/usr/bin/env python3
"""Held-out probe trajectories for the delayed-action experiment (Appendix F.5, Table 19).

For each delay k, rolls out NoisyShape under ActionDelay(k) with a uniform
random policy (300 episodes x 200 steps, env and policy seeded with 42) and
saves images, the ISSUED one-hot actions and the colour / shape / size labels
of what is rendered in each frame. k=0 uses the plain environment (no
ActionDelay wrapper). Model-free, CPU only. One file per k is shared by the
adjacent- and shifted-window runs of that k.

Example:
  python analysis/delay/collect_delay_trajs.py --k 0 1 2 4 --out_dir runs/traj_cache
-> runs/traj_cache/delay_k{k}_traj.npz
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # StarWM root
import numpy as np  # noqa: E402

import envs.wrappers as wrappers  # noqa: E402
from shapeenv import NoisyShapeEnv  # noqa: E402


def collect(k, davis_path, n_eps=300, T=200, seed=42):
    base = NoisyShapeEnv(size=(64, 64), seed=seed, davis_path=davis_path)
    env = wrappers.OneHotAction(base)
    if k > 0:
        env = wrappers.ActionDelay(env, k)
    rng = np.random.RandomState(seed)
    IMG, ACT, ISF, COL, SHP, SZ = [], [], [], [], [], []
    for _ in range(n_eps):
        r = env.reset()
        o = r[0] if isinstance(r, tuple) else r
        im, act_ep, isf = [o["image"]], [], [1.0]
        col, shp, sz = [base._color_id], [base._shape_id], [base._size_id]
        for _ in range(T):
            a = np.zeros(4, np.float32); a[rng.randint(4)] = 1.0
            o = env.step(a)[0]
            im.append(o["image"]); act_ep.append(a); isf.append(0.0)
            col.append(base._color_id); shp.append(base._shape_id); sz.append(base._size_id)
        IMG.append(np.stack(im)); ACT.append(np.stack(act_ep)); ISF.append(np.array(isf, np.float32))
        COL.append(np.array(col)); SHP.append(np.array(shp)); SZ.append(np.array(sz))
    return dict(imgs=np.stack(IMG), acts=np.stack(ACT), isfirst=np.stack(ISF),
                color=np.stack(COL), shape=np.stack(SHP), size=np.stack(SZ))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 4], help="action delays")
    ap.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR", ""),
                    help="DAVIS JPEGImages/480p (default: $DAVIS_DIR)")
    ap.add_argument("--out_dir", default="runs/traj_cache")
    ap.add_argument("--n_eps", type=int, default=300)
    ap.add_argument("--T", type=int, default=200, help="steps per episode")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.davis_path or not os.path.isdir(args.davis_path):
        ap.error("DAVIS JPEGImages/480p directory not found: pass --davis_path or set $DAVIS_DIR")
    os.makedirs(args.out_dir, exist_ok=True)
    for k in args.k:
        d = collect(k, args.davis_path, args.n_eps, args.T, args.seed)
        p = os.path.join(args.out_dir, f"delay_k{k}_traj.npz")
        np.savez_compressed(p, **d)
        print(f"saved {p}: {args.n_eps}ep x {args.T + 1}f  color/shape/size labels", flush=True)


if __name__ == "__main__":
    main()
