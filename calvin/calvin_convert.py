#!/usr/bin/env python3
"""Convert CALVIN play data to replay episodes for offline world-model training
(Appendix F.3 (iii); input to train_calvin_offline.py, Table 15, Fig. 26).

CALVIN stores ONE timestep per file ``episode_XXXXXXX.npz`` with keys
  rgb_static (200,200,3 uint8), rel_actions (7,), actions (7,), scene_obs (24,), robot_obs (15,)
and contiguous play segments in ``ep_start_end_ids.npy`` (rows [start, end], inclusive).

Emits replay episodes (chunks of --chunk_len frames) with keys
  image  (L,64,64,3 uint8) -- rgb_static resized 200 -> 64 (cv2.INTER_AREA)
  action (L,7 float32)     -- action[t] = rel_actions[c0+t-1], action[0] = 0
  reward (L,) = 0, discount (L,) = 1, is_first / is_terminal / is_last (L,)
action[t] is the action that produced frame t (transition t-1 -> t), matching
the replay convention of tools.add_to_cache (index-0 action = zeros), so the
inverse-dynamics target action[j+1] is the action between frames j and j+1.

Also writes a parallel state cache ``<out>_state/<name>.npz`` with scene_obs
(24,) and robot_obs (15,) per emitted frame (used by calvin_attn_metric_proj.py
and attn_table.py to place the gripper / block masks).

Example:
  python calvin/calvin_convert.py --src <calvin>/task_D_D/training   --out <conv>/D_train
  python calvin/calvin_convert.py --src <calvin>/task_D_D/validation --out <conv>/D_val
"""
import argparse
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

cv2.setNumThreads(1)


def load_frame(args):
    """-> (image64 uint8 HWC, rel_action f32(7), scene_obs f32(24), robot_obs f32(15))."""
    src, gid = args
    d = np.load(os.path.join(src, f"episode_{gid:07d}.npz"))
    img64 = cv2.resize(d["rgb_static"], (64, 64), interpolation=cv2.INTER_AREA)
    return (img64.astype(np.uint8), d["rel_actions"].astype(np.float32),
            d["scene_obs"].astype(np.float32), d["robot_obs"].astype(np.float32))


def convert(src, out, chunk_len, workers, max_frames):
    src = pathlib.Path(src)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    out_state = pathlib.Path(str(out) + "_state"); out_state.mkdir(parents=True, exist_ok=True)
    se = np.load(src / "ep_start_end_ids.npy")             # (n_seg, 2) inclusive
    print(f"[convert] {len(se)} play segment(s) in {src}")
    total_emit, n_ep = 0, 0
    for si, (s, e) in enumerate(se):
        gids = list(range(int(s), int(e) + 1))
        if max_frames:
            gids = gids[:max_frames]
        Ns = len(gids)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(load_frame, [(str(src), g) for g in gids]))
        imgs = np.stack([r[0] for r in res], 0)            # (Ns,64,64,3)
        rels = np.stack([r[1] for r in res], 0)            # (Ns,7)
        scns = np.stack([r[2] for r in res], 0)            # (Ns,24)
        robs = np.stack([r[3] for r in res], 0)            # (Ns,15)
        n_chunks = (Ns + chunk_len - 1) // chunk_len
        for ci in range(n_chunks):
            c0 = ci * chunk_len
            c1 = min(c0 + chunk_len, Ns)
            L = c1 - c0
            if L < 2:
                continue
            action = np.zeros((L, 7), np.float32)
            action[1:] = rels[c0: c0 + L - 1]               # action[t] = rel[c0+t-1]; action[0] = 0
            is_first = np.zeros((L,), bool); is_first[0] = True
            is_last = np.zeros((L,), bool); is_last[-1] = True
            name = f"calvin_{si:03d}_{ci:04d}-{L}"
            np.savez_compressed(
                out / f"{name}.npz",
                image=imgs[c0:c1], action=action, reward=np.zeros((L,), np.float32),
                discount=np.ones((L,), np.float32), is_first=is_first,
                is_terminal=np.zeros((L,), bool), is_last=is_last)
            np.savez_compressed(out_state / f"{name}.npz",
                                scene_obs=scns[c0:c1], robot_obs=robs[c0:c1], is_first=is_first)
            total_emit += L; n_ep += 1
        print(f"[convert] seg {si}: {Ns} frames -> {n_chunks} episodes  ({time.time() - t0:.1f}s)")
    print(f"[convert] DONE: {n_ep} episodes, {total_emit} frames -> {out}")
    print(f"[convert] state cache -> {out_state}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="CALVIN split dir with episode_*.npz + ep_start_end_ids.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_len", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per segment (0 = all)")
    a = ap.parse_args()
    convert(a.src, a.out, a.chunk_len, a.workers, a.max_frames)
