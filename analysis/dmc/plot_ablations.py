#!/usr/bin/env python3
"""Gradient-isolation ablation: Fig 7 (main text), Fig 13 and Table 7 (appendix).

Evaluation return (``eval_return``) of three configurations, each seed smoothed
with a 5-point moving average and interpolated on a 4K-step grid; line = mean,
band = +- population std over seeds:

    ★  (SS)                 starwm     full StarWM (detach A and B, SS losses)
    ✩1 (SS+RSSM+Recon.)     iso_star1  no detach A/B, with SS  (``--configs iso_star1``)
    ✩3 (RSSM)               iso_star3  only detach B, no SS    (``--configs iso_star3``)

Fig 7: Clean / VHR Walker Walk. Fig 13: Clean / VHR x Walker Walk / Cheetah Run.
Table 7 (``--table``): mean of the last 10 grid points ending at the last step
where any seed has data, +- the seed std averaged over the same points.

    python analysis/dmc/plot_ablations.py --table
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_eval_rows, moving_average, interp_to_grid, final_window
from layouts import add_layout_args, layout_from_args, dirs

SMOOTH_WINDOW = 5
INTERP_STEP = 4000

ABLATIONS = [
    # (label, variant, color)
    ("✩1 (SS+RSSM+Recon.)", "iso_star1", "#2E7D32"),
    ("✩3 (RSSM)",           "iso_star3", "#1565C0"),
    ("★ (SS)",              "starwm",    "#E53935"),
]
FIGURES = {
    # name: (panels [(setting, task, title)], figsize, default output name)
    "main": ([("clean", "walker_walk", "Clean Walker Walk"),
              ("vhr", "walker_walk", "VHR Walker Walk")],
             (5.5, 2.7), "ablation_reconRSSM_eval_main"),
    "full": ([("clean", "walker_walk", "Clean Walker Walk"),
              ("clean", "cheetah_run", "Clean Cheetah Run"),
              ("vhr", "walker_walk", "VHR Walker Walk"),
              ("vhr", "cheetah_run", "VHR Cheetah Run")],
             (11, 2.7), "ablation_reconRSSM_eval"),
}


def collect(run_dirs):
    all_steps, all_returns = [], []
    for d in run_dirs:
        steps, returns = load_eval_rows(d, "eval_return")
        if steps is None or len(steps) < 2:
            continue
        all_steps.append(steps)
        all_returns.append(moving_average(returns, SMOOTH_WINDOW))
    return all_steps, all_returns


def draw_panel(ax, layout, setting, task, title, results):
    for ab_label, variant, color in ABLATIONS:
        all_steps, all_returns = collect(dirs(layout.dmc(setting, task, variant)))
        if not all_steps:
            print(f"[{title}] [{ab_label}] no seeds found")
            continue
        got = interp_to_grid(all_steps, all_returns, INTERP_STEP)
        if got is None:
            continue
        common, mean, std = got[:3]
        steps_k = common / 1000.0
        ax.plot(steps_k, mean, color=color, linewidth=2.0, label=ab_label)
        ax.fill_between(steps_k, mean - std, mean + std, color=color, alpha=0.1)
        fw = final_window(common, mean, std, None, rule="any")
        if fw is not None:
            print(f"[{title}] [{ab_label}]  step={fw[0]}  mean={fw[1]:.2f}  std={fw[2]:.2f}"
                  f"  (n={len(all_steps)})")
            results[(setting, task, variant)] = fw[1:] + (len(all_steps),)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Eval Return")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)


def make_figure(layout, which, out_dir, results):
    panels, figsize, name = FIGURES[which]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (setting, task, title) in zip(axes.flatten(), panels):
        draw_panel(ax, layout, setting, task, title, results)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8,
               fontsize=12, bbox_to_anchor=(0.5, 1.09), frameon=False)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def print_table(results):
    cols = [("walker_walk", "clean"), ("walker_walk", "vhr"),
            ("cheetah_run", "clean"), ("cheetah_run", "vhr")]
    print("\nTable 7: eval return, mean ± std over seeds")
    print(f"{'':24s}" + "".join(f"{t + ' ' + s:>24s}" for t, s in cols))
    for label, variant, _ in (ABLATIONS[2], ABLATIONS[0], ABLATIONS[1]):  # paper row order
        row = []
        for task, setting in cols:
            r = results.get((setting, task, variant))
            row.append("--" if r is None else
                       f"{r[0]:.2f} ± {r[1]:.2f}" + ("" if r[2] == 3 else f" (n={r[2]})"))
        print(f"{label:24s}" + "".join(f"{c:>24s}" for c in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--figure", choices=["main", "full", "both"], default="both",
                    help="main = Fig 7, full = Fig 13 (whose numbers are Table 7)")
    ap.add_argument("--table", action="store_true", help="print Table 7")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    layout = layout_from_args(args)
    results = {}
    for which in (["main", "full"] if args.figure == "both" else [args.figure]):
        make_figure(layout, which, args.out_dir, results)
    if args.table:
        if args.figure == "main":
            for setting, task, title in FIGURES["full"][0]:
                draw_panel(plt.figure().gca(), layout, setting, task, title, results)
                plt.close("all")
        print_table(results)


if __name__ == "__main__":
    main()
