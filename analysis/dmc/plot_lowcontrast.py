#!/usr/bin/env python3
"""Fig 25: return curves of the low-contrast agent (``--recolor_agent True``) on
Walker Walk, VH and VHR, for StarWM, DreamerV3 and R2-Dreamer.

Evaluation return, EMA-0.3 per seed, interpolated on a 200-point grid spanning
the steps every seed covers (seeds trained less than half as long as the longest
are dropped); line = mean, band = +- standard error over seeds. The table numbers
(Table 14) come from ``stress_tables.py``.

Also defines the shared style of the stress-test figures (used by plot_reacher.py).

    python analysis/dmc/plot_lowcontrast.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import agg_seeds_se, comparable
from layouts import add_layout_args, layout_from_args, dirs

# paper palette: StarWM red, StarWM-R amber, DreamerV3 blue, R2-Dreamer green
METHOD_COLOR = {"starwm": "#E53935", "starwm_r": "#e5a435", "dreamerv3": "#1565C0",
                "r2dreamer": "#2E7D32"}
METHOD_LABEL = {"starwm": "StarWM", "starwm_r": "StarWM-R", "dreamerv3": "DreamerV3",
                "r2dreamer": "R2-Dreamer"}
# natural figure width = the paper's \textwidth, so 10pt here is 10pt on the page
TEXTWIDTH_IN, ASPECT = 5.5, 3.1 / 4.6
DOMAIN_LABEL = {"walker_walk": "Walker", "cheetah_run": "Cheetah"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "grid.linewidth": 0.6, "axes.linewidth": 0.8,
        "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
        "axes.titleweight": "600",
    })
    return plt


def plot_curve(ax, grid, mean, se, color, label):
    if len(grid) == 0:
        return
    ax.plot(grid / 1e6, mean, color=color, label=label, solid_capstyle="round")
    if se is not None and len(se):
        ax.fill_between(grid / 1e6, mean - se, mean + se, color=color, alpha=0.10, linewidth=0)


def savefig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    print("saved", path + ".{pdf,png}")


def panel(ax, run_sets, title):
    """``run_sets``: [(method, [run dirs])]."""
    any_data = False
    for m, runs in run_sets:
        runs = comparable(runs)
        if not runs:
            continue
        g, mean, se = agg_seeds_se(runs, "return", smooth=0.3)
        if len(g):
            any_data = True
            plot_curve(ax, g, mean, se, METHOD_COLOR[m], METHOD_LABEL[m])
    ax.set_title(title, loc="left")
    ax.set_xlabel("Steps (M)"); ax.set_ylabel("return")
    ax.margins(x=0.05)
    return any_data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--task", default="walker_walk", choices=sorted(DOMAIN_LABEL))
    ap.add_argument("--out", default=None, help="output path without extension "
                    "(default: figures/lowcontrast_<domain>_returns)")
    args = ap.parse_args()
    layout = layout_from_args(args)
    dom = DOMAIN_LABEL[args.task]
    out = args.out or os.path.join("figures", f"lowcontrast_{dom.lower()}_returns")
    methods = ["starwm", "dreamerv3", "r2dreamer"]

    plt = setup_mpl()
    specs = [("vh", f"{dom} · VH"), ("vhr", f"{dom} · VHR")]
    ncol = 2
    nrow = (len(specs) + ncol - 1) // ncol
    panel_w = TEXTWIDTH_IN / ncol
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(TEXTWIDTH_IN, panel_w * ASPECT * nrow + 0.45))
    axes = axes.ravel()
    any_data = False
    for ax, (setting, title) in zip(axes, specs):
        any_data |= panel(ax, [(m, dirs(layout.lowcontrast(setting, args.task, m)))
                               for m in methods], title)
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"DMC low-contrast — {dom} return", x=0.02, ha="left", fontsize=10,
                 fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="lower center",
                   ncol=len(seen), frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, out)
    plt.close(fig)
    print(f"  {os.path.basename(out)}: {'plotted' if any_data else 'NO DATA'}")


if __name__ == "__main__":
    main()
