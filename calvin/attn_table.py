#!/usr/bin/env python3
"""CALVIN spatial attention concentration, Table 15 (Appendix F.3 (iii)).

All rows are measured on the SAME frames, with a measured (not derived)
attention share and a per-row shuffled null:
  gripper   - disk r=5 px around the projected end-effector;
  active    - disk r=3 px around the block that moves most between t and t+1
              (only defined when some block moves by >= 1e-3);
  static    - disks around the other blocks;
  allblocks - all three blocks;
  combined  - gripper + all blocks ("Gripper + all blocks" in Table 15).
Columns: area share, attention share, concentration = mean attention share /
mean area share, and the row's own null (mean over frames of the per-frame
concentration when the attention map is scored against another frame's mask
of the same region; 5 permutations, +- SE).
Attention per frame = max over queries, nearest-upsampled, normalised.
Frames: 40 random held-out 500-frame episodes, t = 20, 45, ..., 495.

  --moving : only frames in which some block moves (Table 15, n = 204);
  default  : all frames (gripper row quoted in the text: 9.71x vs 2.84x null).

Example:
  python calvin/attn_table.py --ckpt runs/calvin/starwm/checkpoints/wm_step180000.pt \
      --data_dir <conv>/D_val --state_dir <conv>/D_val_state --device cuda:0 --moving
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import calvin_attn_metric_proj as M  # noqa: E402
from train_calvin_offline import build_config  # noqa: E402

ROWS = ["gripper", "active", "static", "allblocks", "combined"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--state_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--moving", action="store_true", help="only frames with a moving block (Table 15)")
    ap.add_argument("--n_eps", type=int, default=40)
    a = ap.parse_args()

    cfg = build_config("starwm", {"device": a.device, "seed": 0})
    wm = M.load_wm(a.ckpt, cfg)
    files = M.val_files(a.data_dir, a.n_eps)
    rows = {k: {"a": [], "s": []} for k in ROWS}
    masks = {k: [] for k in rows}
    attn = []
    for f in files:
        imgs = np.load(f)["image"]
        st = np.load(os.path.join(a.state_dir, os.path.basename(f)))
        sc, rb = st["scene_obs"], st["robot_obs"]
        idx = np.arange(20, 499, 25)
        A = M.get_attn(wm, cfg, imgs[idx])
        C = A.max(axis=1); C = C / (C.sum(axis=(-2, -1), keepdims=True) + 1e-8)
        for jj, t in enumerate(idx):
            pos = np.stack([sc[t, b] for b in M.BLK])
            disp = np.linalg.norm(np.stack([sc[t + 1, b] for b in M.BLK]) - pos, axis=1)
            moving = disp.max() >= 1e-3
            if a.moving and not moving:
                continue
            ai = int(disp.argmax()) if moving else -1
            ee = M.project(rb[t:t + 1, :3])[0]
            px = M.project(pos)
            mg = M.disk(ee[0], ee[1], 5) if (0 <= ee[0] < 64 and 0 <= ee[1] < 64) else np.zeros((64, 64), bool)
            ma = np.zeros((64, 64), bool); ms = np.zeros((64, 64), bool); mall = np.zeros((64, 64), bool)
            for k, (bx, by) in enumerate(px):
                if 0 <= bx < 64 and 0 <= by < 64:
                    d = M.disk(bx, by, 3); mall |= d
                    if k == ai:
                        ma |= d
                    else:
                        ms |= d
            cur = {"gripper": mg, "active": ma, "static": ms, "allblocks": mall, "combined": mg | mall}
            if not cur["combined"].any():
                continue
            attn.append(C[jj])
            for k, m in cur.items():
                masks[k].append(m)
                if m.any():
                    rows[k]["a"].append(float(m.mean())); rows[k]["s"].append(float(C[jj][m].sum()))

    def null_of(ms_, n=5):
        v = []
        for r in range(n):
            pm = np.random.RandomState(300 + r).permutation(len(ms_))
            for i in range(len(ms_)):
                m = ms_[pm[i]]
                if m.any():
                    v.append(float(attn[i][m].sum()) / (m.mean() + 1e-8))
        return (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) if v else (float("nan"), float("nan"))

    print(f"frames used: {len(attn)}   (subset = "
          f"{'only frames with a moving block' if a.moving else 'ALL frames'})")
    print(f"{'region':22s} {'area':>7} {'attn':>7} {'conc=attn/area':>15} {'own null':>12}")
    for k in ROWS:
        ar = np.mean(rows[k]["a"]); s = np.mean(rows[k]["s"])
        nm, ns = null_of(masks[k])
        print(f"{k:22s} {ar * 100:6.1f}% {s * 100:6.1f}% {s / ar:14.2f}x {nm:10.2f}+-{ns:.2f}   n={len(rows[k]['a'])}")


if __name__ == "__main__":
    main()
