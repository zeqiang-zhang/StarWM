#!/usr/bin/env python3
"""Fig 16: full StarWM (inverse dynamics + contrastive) vs the inverse-only
variant (``--infonce_scale 0``, run variant ``inv_only``) on Cartpole Swingup,
Cheetah Run and Walker Run under Clean, VH and VHR.

Training return (``train_return``), 25-point moving average per seed, 4K-step
grid; line = mean, band = +- population std over seeds. The printed value per
panel is the mean of the last 10 grid points ending at the last step where any
seed has data.

    python analysis/dmc/plot_contrastive.py
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
TYPES = ["clean", "vh", "vhr"]
TASKS = ["cartpole_swingup", "cheetah_run", "walker_run"]
TASK_LABELS = ["Cartpole\nSwingup", "Cheetah\nRun", "Walker\nRun"]
TYPE_LABELS = {"clean": "Clean", "vh": "Video Hard (VH)", "vhr": "Video Hard Random (VHR)"}
TYPE_STEPS = {"clean": 1000, "vh": 3000, "vhr": 1000}  # x-axis limit in K steps
# (legend label, variant, color); "noNCE" = no InfoNCE (contrastive) loss
METHODS = [("StarWM", "starwm", "#E53935"), ("StarWM noNCE", "inv_only", "#0d47a1")]


def plot_method(ax, layout, setting, task, variant, color, label, shade):
    all_steps, all_returns = collect_curves(dirs(layout.dmc(setting, task, variant)),
                                            TRAIN_METRIC["starwm"], SMOOTH_WINDOW)
    if not all_steps:
        return False
    got = interp_to_grid(all_steps, all_returns)
    if got is None:
        return False
    common, mean, std, min_vals, max_vals, mask = got
    steps_k = common / 1000
    ax.plot(steps_k, mean, color=color, linewidth=2.0, label=f"{label}")
    fw = final_window(common, mean, std, mask, rule="any")
    if fw is not None:
        print(f"Environment: {task}, Type: {setting}, Method: {label}, Steps: {fw[0]}, "
              f"Mean: {fw[1]:.2f}, Std: {fw[2]:.2f}  (n={len(all_steps)})")
    if shade == "std":
        ax.fill_between(steps_k, mean - std, mean + std, color=color, alpha=0.1)
    elif shade == "minmax":
        ax.fill_between(steps_k, min_vals, max_vals, color=color, alpha=0.1)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--shade", choices=["std", "minmax"], default="std")
    ap.add_argument("--out", default=os.path.join("figures", "contrastive_results"),
                    help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
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
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for row_idx, setting in enumerate(TYPES):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx, col_idx]
            has_data = False
            for label, variant, color in METHODS:
                if plot_method(ax, layout, setting, task, variant, color, label, args.shade):
                    has_data = True
            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
            ax.set_xlim(left=0, right=TYPE_STEPS[setting])
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
            if row_idx == 0:
                ax.set_title(TASK_LABELS[col_idx], fontsize=12, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{TYPE_LABELS[setting]}\nReturn", fontsize=10, fontweight="bold")
            if row_idx == 1:   # (sic) the paper figure labels the middle row's x axis
                ax.set_xlabel("Steps (K)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_all(plt, args.out + ("" if args.shade == "std" else "_minmax"), args.formats,
                bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
