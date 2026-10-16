#!/usr/bin/env python3
"""Render the Reacher observations shown in Fig 29 (left).

For VH (sequential DAVIS video) and VHR (random DAVIS frame every step) this
resets ``reacher_easy`` with the DAVIS pixel background (seed 5, 64x64, action
repeat 2) and stores the reset frame followed by 3 frames after random actions:
``{'vh': (4,64,64,3), 'vhr': (4,64,64,3)}`` uint8. Fig 29 shows frame 0, which
is fully determined by the seed; the later frames depend on unseeded random
actions unless ``--action_seed`` is given.

Needs MuJoCo rendering (``MUJOCO_GL=egl``) and the DAVIS 2017 JPEG frames:

    python analysis/dmc/render_reacher_frames.py \
        --davis_path /path/to/DAVIS/JPEGImages/480p --out figures/reacher_frames.npz
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # StarWM/

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--davis_path", required=True,
                    help="DAVIS JPEGImages/480p directory (same as --distracting_background_path)")
    ap.add_argument("--out", default=os.path.join("figures", "reacher_frames.npz"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3, help="frames after the reset frame")
    ap.add_argument("--action_seed", type=int, default=None,
                    help="seed the random actions (unseeded for the paper figure)")
    args = ap.parse_args()

    import envs.dmc as dmc

    out = {}
    for tag, rand in (("vh", False), ("vhr", True)):
        env = dmc.DeepMindControl("reacher_easy", 2, (64, 64), seed=args.seed,
                                  pixel_video_bg=True, video_random_frames=rand,
                                  davis_path=args.davis_path)
        obs = env.reset()
        img = obs[0]["image"] if isinstance(obs, tuple) else obs["image"]
        frames = [np.asarray(img, np.uint8)]
        sp = env.action_space
        if args.action_seed is not None:
            sp.seed(args.action_seed)
        for _ in range(args.steps):
            o = env.step(sp.sample())    # bare DeepMindControl takes the raw action array
            im = o[0]["image"] if isinstance(o, tuple) else o["image"]
            frames.append(np.asarray(im, np.uint8))
        out[tag] = np.stack(frames)
        print(tag, out[tag].shape, "range", out[tag].min(), out[tag].max(), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
