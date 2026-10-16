#!/usr/bin/env python3
"""Fig 29: DMC Reacher (easy) with DAVIS video backgrounds. Left: one observation
for VH and VHR (``render_reacher_frames.py``); right: return curves of StarWM,
StarWM-R, DreamerV3 and R2-Dreamer.

Evaluation return, EMA-0.3 per seed, 200-point grid over the steps every seed
covers; line = mean, band = +- standard error over seeds. Geometry and style
match ``plot_lowcontrast.py`` so the two appendix figures look alike. The table
numbers (Table 17) come from ``stress_tables.py``.

    python analysis/dmc/render_reacher_frames.py --out figures/reacher_frames.npz
    python analysis/dmc/plot_reacher.py --frames figures/reacher_frames.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import agg_seeds_se, comparable
from layouts import add_layout_args, layout_from_args, dirs
from plot_lowcontrast import (setup_mpl, savefig, plot_curve, METHOD_COLOR, METHOD_LABEL,
                              TEXTWIDTH_IN, ASPECT)

METHODS = ["starwm", "starwm_r", "dreamerv3", "r2dreamer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--frames", default=os.path.join("figures", "reacher_frames.npz"),
                    help="npz with 'vh' and 'vhr' frame stacks (render_reacher_frames.py)")
    ap.add_argument("--out", default=os.path.join("figures", "reacher_env_returns"),
                    help="output path without extension")
    args = ap.parse_args()
    layout = layout_from_args(args)

    plt = setup_mpl()
    z = np.load(args.frames)
    # Explicit figure-fraction placement: tight/constrained layout mishandle the
    # fixed-aspect image axes next to the curve panels.
    W = TEXTWIDTH_IN
    H = (TEXTWIDTH_IN / 2) * ASPECT + 0.62      # same panel height as plot_lowcontrast.py
    fig = plt.figure(figsize=(W, H))
    TOP, BOT_P = 0.805, 0.315                   # curve panels need room for "Steps (M)"
    BOT_I = 0.115                               # the image strip may run further down
    gs = fig.add_gridspec(1, 2, left=0.375, right=0.985, bottom=BOT_P, top=TOP, wspace=0.30)
    gap_in = 0.05
    side_in = ((TOP - BOT_I) * H - gap_in) / 2
    wf, hf = side_in / W, side_in / H
    x0 = 0.088
    for i, (tag, label) in enumerate((("vh", "VH"), ("vhr", "VHR"))):
        y0 = TOP - hf if i == 0 else BOT_I
        ax = fig.add_axes([x0, y0, wf, hf])
        ax.imshow(z[tag][0], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("#999999")
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=4)
        if i == 0:
            ax.set_title("Observation", loc="left")

    axes = []
    for c, (setting, title) in enumerate((("vh", "Reacher · VH"),
                                          ("vhr", "Reacher · VHR"))):
        ax = fig.add_subplot(gs[0, c]); axes.append(ax)
        for m in METHODS:
            rs = comparable(dirs(layout.reacher(setting, m)))
            if not rs:
                continue
            g, mean, se = agg_seeds_se(rs, "return", smooth=0.3)
            if len(g):
                plot_curve(ax, g, mean, se, METHOD_COLOR[m], METHOD_LABEL[m])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Steps (M)")
        if c == 0:
            ax.set_ylabel("return")
        ax.margins(x=0.05)

    fig.suptitle("Reacher (passive fixed target) — episode return",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
               ncol=len(seen), frameon=False, bbox_to_anchor=(0.5, -0.03))
    savefig(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
