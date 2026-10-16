#!/usr/bin/env python3
"""Fig 5 (DMC learning curves, VH/VHR x 6 tasks) and, with ``--table``, Table 5.

Curves: training-episode return (StarWM, StarWM-R, DreamerV3, Iso-Dream++:
``train_return``; R2-Dreamer: ``episode/score``; DenoisedMDP: TensorBoard
``train_reward``), each seed smoothed with a 25-point moving average,
interpolated on a 4K-step grid; line = mean over seeds, band = +- population
std (``--shade minmax`` for min/max).

Table 5 (``--table``):
  VH / VHR rows: mean of the last 10 grid points ending at the last step where
  ANY seed has data, +- the mean of the seed-std curve over the same points.
  Clean rows: same, but ending at the last step where ALL seeds have data.
  "DreamerV3 (Original Paper)" is the value reported by Hafner et al. (2025).

    python analysis/dmc/plot_main_results.py --table
    python analysis/dmc/plot_main_results.py --layout paper --root /path/to/code --table
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (TASKS, TASK_LABELS, collect_curves, interp_to_grid, final_window,
                    savefig_all)
from layouts import add_layout_args, layout_from_args, dirs, TRAIN_METRIC

SMOOTH_WINDOW = 25
TYPES = ["vh", "vhr"]
TYPE_LABELS = {"vh": "VH", "vhr": "VHR"}
TYPE_STEPS = {"vh": 3.0, "vhr": 1.0}  # x-axis limit in M steps
TYPE_TICKS = {"vh": [0, 1, 2, 3], "vhr": [0, 0.5, 1.0]}

COLORS = {
    "StarWM":           "#E53935",   # red
    "StarWM-R":         "#e5a435",   # amber (StarWM + reward prediction)
    "Orig DreamerV3":   "#1565C0",   # blue
    "R2Dreamer":        "#2E7D32",   # green
    "DenoisedMDP":      "#6A1B9A",   # purple
    "IsoDreamer":       "#00838F",   # teal
}
# (legend label, source, variant/baseline name, linestyle) in drawing order
METHODS = [
    ("Orig DreamerV3", "dmc", "dreamerv3", "-"),
    ("R2Dreamer", "baseline", "r2dreamer", "-"),
    ("DenoisedMDP", "baseline", "denoised_mdp", "-"),
    ("IsoDreamer", "baseline", "isodream", "-"),
    ("StarWM-R", "dmc", "starwm_r", "--"),
    ("StarWM", "dmc", "starwm", "-"),
]
# Table 5 row names / Clean reference values reported in the DreamerV3 paper
TABLE_NAME = {"Orig DreamerV3": "DreamerV3", "R2Dreamer": "R2-Dreamer",
              "DenoisedMDP": "DenoisedMDP", "IsoDreamer": "Iso-Dream++",
              "StarWM": "StarWM", "StarWM-R": "StarWM-R"}
TABLE_ROWS = ["Orig DreamerV3", "DenoisedMDP", "IsoDreamer", "R2Dreamer", "StarWM", "StarWM-R"]
TABLE_COLS = ["walker_walk", "walker_run", "cheetah_run", "cartpole_swingup",
              "finger_spin", "cup_catch"]
DREAMERV3_REPORTED_CLEAN = {"walker_walk": 966, "walker_run": 797, "cheetah_run": 846,
                            "cartpole_swingup": 843, "finger_spin": 886, "cup_catch": 975}


def method_runs(layout, source, name, setting, task):
    if source == "dmc":
        return dirs(layout.dmc(setting, task, name))
    return dirs(layout.baseline(name, setting, task))


def metric_of(source, name):
    return TRAIN_METRIC["starwm" if source == "dmc" else name]


def curve(layout, source, name, setting, task):
    runs = method_runs(layout, source, name, setting, task)
    all_steps, all_returns = collect_curves(runs, metric_of(source, name), SMOOTH_WINDOW)
    if not all_steps:
        return None
    return interp_to_grid(all_steps, all_returns), len(all_steps)


def plot_method(ax, layout, source, name, setting, task, color, label, shade, linestyle, results):
    got = curve(layout, source, name, setting, task)
    if got is None or got[0] is None:
        return False
    (common, mean, std, min_vals, max_vals, mask), n = got
    steps_m = common / 1_000_000
    ax.plot(steps_m, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)
    fw = final_window(common, mean, std, mask, rule="any")
    if fw is not None:
        print(f"Environment: {task}, Type: {setting}, Method: {label}, Steps: {fw[0]}, "
              f"Mean: {fw[1]:.2f}, Std: {fw[2]:.2f}  (n={n})")
        results[(setting, task, label)] = fw[1:] + (n,)
    if shade == "std":
        ax.fill_between(steps_m, mean - std, mean + std, color=color, alpha=0.1)
    elif shade == "minmax":
        ax.fill_between(steps_m, min_vals, max_vals, color=color, alpha=0.1)
    return True


def make_figure(layout, shade, out, formats, results):
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "font.size": 10,
    })
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    for row_idx, setting in enumerate(TYPES):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx, col_idx]
            has_data = False
            for label, source, name, ls in METHODS:
                if plot_method(ax, layout, source, name, setting, task, COLORS[label], label,
                               shade, ls, results):
                    has_data = True
            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
            ax.set_xlim(left=0, right=TYPE_STEPS[setting])
            ax.set_ylim(bottom=0)
            ax.set_xticks(TYPE_TICKS[setting])
            ax.set_xticklabels([f"{t:g}" for t in TYPE_TICKS[setting]])
            ax.tick_params(labelsize=9.5)
            if row_idx == 0:
                ax.set_title(TASK_LABELS[task], fontsize=12, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{TYPE_LABELS[setting]}\nReturn", fontsize=10, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Steps (M)", fontsize=9.5)
    # shared legend on top, one entry per method
    seen = {}
    for ax in axes.flatten():
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l not in seen:
                seen[l] = h
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()),
                   loc="upper center", ncol=len(seen),
                   fontsize=12, bbox_to_anchor=(0.5, 1.06), frameon=False)
    savefig_all(plt, out, formats, bbox_inches="tight", facecolor="white")
    plt.close()


def clean_rows(layout):
    """Clean-setting cells of Table 5 (last step where all seeds have data)."""
    res = {}
    for task in TASKS:
        for label, name in (("StarWM", "starwm"), ("Orig DreamerV3", "dreamerv3")):
            got = curve(layout, "dmc", name, "clean", task)
            if got is None or got[0] is None:
                continue
            (common, mean, std, _, _, mask), n = got
            fw = final_window(common, mean, std, mask, rule="all")
            if fw is not None:
                print(f"Environment: {task}, Type: clean, Method: {label}, Steps: {fw[0]}, "
                      f"Mean: {fw[1]:.2f}, Std: {fw[2]:.2f}  (n={n})")
                res[("clean", task, label)] = fw[1:] + (n,)
    return res


def print_table(results):
    def cell(key):
        if key not in results:
            return "--"
        m, s, n = results[key]
        return f"{m:.2f} ± {s:.2f}" + ("" if n == 3 else f" (n={n})")
    w = 20
    print("\nTable 5: DMC returns, mean ± std over seeds "
          "(VH: 3M steps, VHR: 1M steps)")
    print(f"{'':6s} {'':26s}" + "".join(f"{t:>{w}s}" for t in TABLE_COLS))
    print(f"{'Clean':6s} {'DreamerV3 (orig. paper)':26s}"
          + "".join(f"{DREAMERV3_REPORTED_CLEAN[t]:>{w}d}" for t in TABLE_COLS))
    for label, name in (("Orig DreamerV3", "DreamerV3 (PyTorch)"), ("StarWM", "StarWM")):
        print(f"{'Clean':6s} {name:26s}" + "".join(f"{cell(('clean', t, label)):>{w}s}"
                                                  for t in TABLE_COLS))
    for setting in TYPES:
        for label in TABLE_ROWS:
            print(f"{setting.upper():6s} {TABLE_NAME[label]:26s}"
                  + "".join(f"{cell((setting, t, label)):>{w}s}" for t in TABLE_COLS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--shade", choices=["std", "minmax"], default="std",
                    help="shaded region: +- std (default) or min/max over seeds")
    ap.add_argument("--table", action="store_true", help="also print Table 5")
    ap.add_argument("--no_figure", action="store_true", help="only compute the numbers")
    ap.add_argument("--out", default=os.path.join("figures", "main_results"),
                    help="output path without extension")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    args = ap.parse_args()
    layout = layout_from_args(args)

    results = {}
    if args.no_figure:
        for setting in TYPES:
            for task in TASKS:
                for label, source, name, _ in METHODS:
                    got = curve(layout, source, name, setting, task)
                    if got is None or got[0] is None:
                        continue
                    (common, mean, std, _, _, mask), n = got
                    fw = final_window(common, mean, std, mask, rule="any")
                    if fw is not None:
                        print(f"Environment: {task}, Type: {setting}, Method: {label}, "
                              f"Steps: {fw[0]}, Mean: {fw[1]:.2f}, Std: {fw[2]:.2f}  (n={n})")
                        results[(setting, task, label)] = fw[1:] + (n,)
    else:
        out = args.out + ("" if args.shade == "std" else "_minmax")
        make_figure(layout, args.shade, out, args.formats, results)
    if args.table:
        results.update(clean_rows(layout))
        print_table(results)


if __name__ == "__main__":
    main()
