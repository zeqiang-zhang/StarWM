#!/usr/bin/env python3
"""Fig 17 and Table 10: number of cross-attention queries N in {1, 2, 4, 8} on
DMC VH (Walker Walk, Cheetah Run, Cartpole Swingup; 3M steps).

N=4 is the default StarWM run (variant ``starwm``); the others are trained with
``--crossattn_num_queries N`` (variants ``N1``, ``N2``, ``N8``). Training return,
25-point moving average per seed, 4K-step grid; line = mean, band = +- population
std over seeds. Table 10 (printed): mean of the last 10 grid points ending at the
last step where ALL seeds have data, +- the seed std averaged over the same
points.

    python analysis/dmc/plot_queries.py
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
MAX_STEPS_K = 3000
TASKS = ["walker_walk", "cheetah_run", "cartpole_swingup"]
TASK_LABELS = ["Walker Walk", "Cheetah Run", "Cartpole Swingup"]
N_VALUES = [1, 2, 4, 8]
COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#E53935", 8: "#2ca02c"}


def variant_of(n):
    return "starwm" if n == 4 else f"N{n}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--shade", choices=["std", "none"], default="std")
    ap.add_argument("--out", default=os.path.join("figures", "ablation_N_curves"),
                    help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=["pdf"])
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
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    table = {}
    for col_idx, (task, task_label) in enumerate(zip(TASKS, TASK_LABELS)):
        ax = axes[col_idx]
        for n in N_VALUES:
            all_steps, all_returns = collect_curves(
                dirs(layout.dmc("vh", task, variant_of(n))), TRAIN_METRIC["starwm"],
                SMOOTH_WINDOW)
            if not all_steps:
                print(f"[{task}] N={n}: no seeds found")
                continue
            got = interp_to_grid(all_steps, all_returns, INTERP_STEP)
            if got is None:
                continue
            common, mean, std, _, _, mask = got
            steps_k = common / 1000
            label = f"N={n}" + (" [main]" if n == 4 else "")
            ax.plot(steps_k, mean, color=COLORS[n], linewidth=2.0, label=label)
            if args.shade == "std":
                ax.fill_between(steps_k, mean - std, mean + std, color=COLORS[n], alpha=0.12)
            fw = final_window(common, mean, std, mask, rule="all")
            if fw is not None:
                print(f"[{task}] N={n}: step={fw[0]}  mean={fw[1]:.2f}  std={fw[2]:.2f}"
                      f"  (n={len(all_steps)})")
                table[(task, n)] = fw[1:] + (len(all_steps),)
        ax.set_title(task_label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(left=0, right=MAX_STEPS_K)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Steps (K)", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Train Return", fontsize=10)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=12, bbox_to_anchor=(0.5, 1.06), frameon=False)
    plt.tight_layout()
    savefig_all(plt, args.out, args.formats, bbox_inches="tight", facecolor="white")
    plt.close()

    print("\nTable 10: VH return at 3M steps, mean ± std over seeds")
    print(f"{'N':>3s}" + "".join(f"{t:>22s}" for t in TASKS))
    for n in N_VALUES:
        cells = []
        for task in TASKS:
            r = table.get((task, n))
            cells.append("--" if r is None else
                         f"{r[0]:.2f} ± {r[1]:.2f}" + ("" if r[2] == 3 else f" (n={r[2]})"))
        print(f"{n:>3d}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
