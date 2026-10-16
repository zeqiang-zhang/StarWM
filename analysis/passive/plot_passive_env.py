#!/usr/bin/env python3
"""PassiveShapeEnv illustration (Appendix F.4 (i), Fig. 27).

Five consecutive steps of one episode per background regime (top: random
frame per step, bottom: sequential video). The action-driven object A changes
attributes and position every step; the passive object B (ringed in cyan)
keeps its colour and position for the whole episode.

Example:
  python analysis/passive/plot_passive_env.py --davis_path $DAVIS_DIR --out figures/passive_env
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402

import probe_common as pc  # noqa: E402
from shapeenv import PassiveShapeEnv, PassiveShapeSeqEnv  # noqa: E402

NSTEP, SEED = 5, 3
ROWS = [("random-frame", PassiveShapeEnv), ("sequential video", PassiveShapeSeqEnv)]


def rollout(EnvCls, n, davis_path):
    env = EnvCls(size=(64, 64), seed=SEED, davis_path=davis_path)
    o = env.reset(seed=SEED)
    frames, bpos, bcol = [o["image"]], [env._b_pos], [env._b_color_id]
    rng = np.random.RandomState(SEED)
    for _ in range(n - 1):
        o, *_ = env.step(rng.randint(4))
        frames.append(o["image"]); bpos.append(env._b_pos); bcol.append(env._b_color_id)
    env.close()
    return [np.asarray(f, np.uint8) for f in frames], bpos, bcol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--davis_path", default=pc.default_davis())
    ap.add_argument("--out", default="figures/passive_env", help="path without extension")
    args = ap.parse_args()
    pc.require_davis(ap, args.davis_path)

    plt = pc.setup_mpl()
    TEXTWIDTH_IN = 5.5
    cell = (TEXTWIDTH_IN - 1.0) / NSTEP
    fig, axes = plt.subplots(len(ROWS), NSTEP, squeeze=False,
                             figsize=(TEXTWIDTH_IN, cell * len(ROWS) + 0.34))
    for r, (label, EnvCls) in enumerate(ROWS):
        frames, bpos, bcol = rollout(EnvCls, NSTEP, args.davis_path)
        assert len(set(bcol)) == 1 and len(set(bpos)) == 1, "B moved within an episode"
        bx, by = bpos[0]
        for c, img in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.add_patch(plt.Circle((bx * 64, by * 64), 9.5, fill=False,
                                    edgecolor="#00E5FF", lw=1.4))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("#999999")
            if r == 0:
                ax.set_title(f"$t{{=}}{c}$", fontsize=9, fontweight="normal", pad=2)
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right",
                              va="center", labelpad=7)
    fig.suptitle("PassiveShapeEnv: passive object $B$ (cyan ring) vs action-driven object $A$",
                 fontsize=10, fontweight="normal", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.25)
    fig.subplots_adjust(hspace=0.06, wspace=0.06)
    pc.savefig(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
