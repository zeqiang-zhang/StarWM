#!/usr/bin/env python3
"""Render example DMC observations under Clean, VH and VHR (Fig 2, left).

For each requested (setting, task) the environment (64x64, action repeat 2,
seed 42) takes 40 random actions; every 5th observation is saved as a PNG and
the 8 frames are tiled into a 2x4 grid upscaled 4x (nearest neighbour). VH and
VHR use the DMC-GB2 ``video_hard`` backgrounds (``envs/video_hard_wrapper.py``),
VHR drawing a random frame of the episode's video at every step. The paper's
Fig 2 is a hand-made composite of these frames and the NoisyShape renderings.
Random actions are unseeded unless ``--action_seed`` is given.

Needs MuJoCo rendering (``MUJOCO_GL=egl``) and the video_hard clips
(``$DMCGB_DATA_DIR/video_hard/video{0..99}.mp4``):

    python analysis/dmc/render_env_frames.py --out_dir figures/env_frames
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # StarWM/

import numpy as np
from PIL import Image

# setting:task pairs rendered for the paper
DEFAULT_JOBS = ["clean:walker_walk", "vh:walker_walk", "vhr:walker_walk",
                "vh:cheetah_run", "vhr:cheetah_run"]


def render(setting, task, out_dir, n_frames=8, every=5, seed=42, action_seed=None):
    """``task`` is the release task key, e.g. ``walker_walk`` or ``cup_catch``."""
    import envs.dmc as dmc
    env = dmc.DeepMindControl(task, action_repeat=2, size=(64, 64), seed=seed)
    domain = task.split("_", 1)[0]
    if domain == "cup":
        domain = "ball_in_cup"
    if setting != "clean":
        from envs.video_hard_wrapper import VideoHardWrapper
        env._env = VideoHardWrapper(env._env, domain, seed=seed,
                                    random_frames=(setting == "vhr"))
    if action_seed is not None:
        env.action_space.seed(action_seed)
    env.reset()
    frames = []
    for i in range(n_frames * every):
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        if i % every == 0:
            frames.append(obs["image"])
            if setting != "clean":
                Image.fromarray(obs["image"]).save(
                    os.path.join(out_dir, f"{setting}_{task}_frame{i // every:02d}.png"))
        if done:
            env.reset()
    rows, cols = 2, 4
    grid = np.zeros((rows * 64, cols * 64, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames[:rows * cols]):
        r, c = idx // cols, idx % cols
        grid[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = frame
    path = os.path.join(out_dir, f"{setting}_{task}_grid.png")
    Image.fromarray(grid).resize((cols * 64 * 4, rows * 64 * 4), Image.NEAREST).save(path)
    print(f"Saved {len(frames)} frames and grid for {setting} {task} -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS,
                    help="setting:task pairs, setting in clean/vh/vhr (default: the paper's set)")
    ap.add_argument("--out_dir", default=os.path.join("figures", "env_frames"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--action_seed", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for job in args.jobs:
        setting, task = job.split(":")
        render(setting, task, args.out_dir, seed=args.seed, action_seed=args.action_seed)


if __name__ == "__main__":
    main()
