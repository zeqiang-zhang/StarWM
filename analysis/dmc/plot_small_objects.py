#!/usr/bin/env python3
"""Fig 20: small-agent variant (``--task dmc_<task>_small``: wider camera field of
view, agent appears half-sized) on Cheetah Run and Walker Walk under Clean, VH
and VHR, for StarWM, DreamerV3 and R2-Dreamer.

Training return (StarWM / DreamerV3: ``train_return``, R2-Dreamer:
``episode/score``), 25-point moving average per seed, 4K-step grid; line = mean,
band = +- population std over seeds. Printed per panel: mean of the last 10 grid
points ending at the last step where ALL seeds have data.

    python analysis/dmc/plot_small_objects.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import collect_curves, interp_to_grid, final_window, savefig_all
from layouts import add_layout_args, layout_from_args, dirs, TRAIN_METRIC

SMOOTH_WINDOW = 25
INTERP_STEP = 4000
TASKS = [("cheetah_run", "Cheetah Run"), ("walker_walk", "Walker Walk")]
MODES = [("clean", "Clean", 1000), ("vh", "VH", 3000), ("vhr", "VHR", 1000)]
# (legend label, variant)
METHODS = [("StarWM", "starwm"), ("Orig Dreamer", "dreamerv3"), ("R2Dreamer", "r2dreamer")]
COLOR = {"StarWM": "#E53935", "Orig Dreamer": "#1565C0", "R2Dreamer": "#2E7D32"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--shade", choices=["std", "none"], default="std")
    ap.add_argument("--out", default=os.path.join("figures", "small_results"),
                    help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()
    layout = layout_from_args(args)

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "font.size": 10,
    })
    fig, axes = plt.subplots(len(TASKS), len(MODES), figsize=(7, 4.5))
    for r, (task, task_label) in enumerate(TASKS):
        for c, (mode, mode_label, max_k) in enumerate(MODES):
            ax = axes[r, c]
            for method, variant in METHODS:
                steps_list, returns_list = collect_curves(
                    dirs(layout.small(mode, task, variant)), TRAIN_METRIC[variant],
                    SMOOTH_WINDOW)
                if not steps_list:
                    print(f"[{task} {mode}] {method}: no seeds")
                    continue
                got = interp_to_grid(steps_list, returns_list, INTERP_STEP)
                if got is None:
                    continue
                common, mean, std, _, _, mask = got
                x = common / 1000
                ax.plot(x, mean, color=COLOR[method], linewidth=1.8, label=method)
                if args.shade == "std":
                    ax.fill_between(x, mean - std, mean + std, color=COLOR[method], alpha=0.12)
                fw = final_window(common, mean, std, mask, rule="all")
                if fw is not None:
                    print(f"[{task} {mode}] {method}: step={fw[0]}  mean={fw[1]:.2f}  "
                          f"std={fw[2]:.2f}  (n={len(steps_list)})")
            ax.set_xlim(left=0, right=max_k)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(mode_label, fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(f"{task_label}\nReturn", fontsize=10, fontweight="bold")
            if r == len(TASKS) - 1:
                ax.set_xlabel("Steps (K)", fontsize=9)
    seen = {}
    for ax in axes.flatten():
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="upper center",
                   ncol=len(seen), fontsize=11, bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout()
    savefig_all(plt, args.out, args.formats, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
