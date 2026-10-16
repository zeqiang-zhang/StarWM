#!/usr/bin/env python3
"""Textured foreground ladder illustration (Appendix F.3 (i), Fig. 23).

Renders the same state (same background frame, object position, shape and
size) at each level L1/L2/L3 for each of the 4 texture IDs, so only the
foreground appearance changes across the grid: L1 procedural textures, L2
DAVIS crops (same domain as the background), L3 the L2 crops alpha-blended
into the background.

Example:
  python analysis/textured/plot_textured_envs.py --davis_path $DAVIS_DIR \
      --out figures/textured_ladder_envs
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import probe_common as pc  # noqa: E402
from shapeenv import TexturedNoisyShapeEnv  # noqa: E402
from shapeenv.textured_shape import NUM_TEXTURES  # noqa: E402

LEVELS = [1, 2, 3]
ROW_LABEL = {1: "L1\nprocedural", 2: "L2\nsame-domain", 3: "L3\nblended"}
SEED = 7          # identical across rows so only appearance differs
SHAPE_COUNT = 0   # shape+size held fixed so the texture is the only varying attribute


def frame(level, tex_id, davis_path):
    env = TexturedNoisyShapeEnv(size=(64, 64), seed=SEED, davis_path=davis_path,
                                level=level, alpha=0.5, edge_blur=1.0)
    env.reset(seed=SEED)                  # same RNG -> same background frame and position
    env._color_count = int(tex_id)        # texture ID is the probed attribute
    env._shape_count = SHAPE_COUNT
    img = env._make_obs()["image"]
    env.close()
    return np.asarray(img, np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--davis_path", default=pc.default_davis())
    ap.add_argument("--out", default="figures/textured_ladder_envs", help="path without extension")
    args = ap.parse_args()
    pc.require_davis(ap, args.davis_path)

    plt = pc.setup_mpl()
    TEXTWIDTH_IN = 5.5   # natural width == paper \textwidth
    cell = (TEXTWIDTH_IN - 1.05) / NUM_TEXTURES
    fig, axes = plt.subplots(len(LEVELS), NUM_TEXTURES,
                             figsize=(TEXTWIDTH_IN, cell * len(LEVELS) + 0.22))
    for r, L in enumerate(LEVELS):
        for c in range(NUM_TEXTURES):
            ax = axes[r, c]
            ax.imshow(frame(L, c, args.davis_path), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("#999999")
            if r == 0:
                ax.set_title(f"Texture {c + 1}", fontsize=9, fontweight="normal", pad=2)
        axes[r, 0].set_ylabel(ROW_LABEL[L], fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=7)
    fig.suptitle("Textured NoisyShape: foreground-complexity ladder", fontsize=10,
                 fontweight="normal", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.25)
    fig.subplots_adjust(hspace=0.06, wspace=0.06)
    pc.savefig(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
