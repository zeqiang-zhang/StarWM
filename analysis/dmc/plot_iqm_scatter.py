#!/usr/bin/env python3
"""Fig 1 (right): IQM under VH (x) vs VHR (y) per method, with 95% CIs.

Reads ``scores_table.csv`` from ``rliable_analysis.py``, recomputes the IQM with
stratified-bootstrap 95% CIs (10,000 reps, ``RandomState(0)``, VH first) on the
normalised scores, separately for VH and VHR, and writes
``iqm_scatter.{png,pdf,svg}`` next to the table. The CIs vary slightly between
runs unless ``--np_seed`` is given (see ``rliable_analysis.py``).

    python analysis/dmc/plot_iqm_scatter.py --root figures/rliable
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rliable import library as rly
from rliable import metrics

RNG = np.random.RandomState(0)

METHODS = ["StarWM", "StarWM-R", "Orig DreamerV3", "R2Dreamer", "DenoisedMDP", "IsoDreamer"]
COLOR = {
    "StarWM":         "#E53935",
    "StarWM-R":       "#FF8A80",
    "Orig DreamerV3": "#1565C0",
    "R2Dreamer":      "#2E7D32",
    "DenoisedMDP":    "#6A1B9A",
    "IsoDreamer":     "#00838F",
}
MARKER = {"StarWM": "o", "StarWM-R": "*", "Orig DreamerV3": "s",
          "R2Dreamer": "D", "DenoisedMDP": "P", "IsoDreamer": "^"}
LABEL = {"StarWM": "StarWM (ours)", "StarWM-R": "StarWM-R (ours)",
         "Orig DreamerV3": "Orig DreamerV3", "R2Dreamer": "R2Dreamer",
         "DenoisedMDP": "DenoisedMDP", "IsoDreamer": "Iso-Dreamer++"}


def to_score_array(df, mode):
    """(n_seeds, n_tasks) per method, using the seeds every method has."""
    sub = df[df["mode"] == mode].copy()
    tasks = sorted(sub["env"].unique())
    common_seeds_by_task = {}
    for t in tasks:
        seeds_ok = None
        for m in METHODS:
            sw = set(sub[(sub.env == t) & (sub.method == m) &
                         np.isfinite(sub.norm_score)]["seed"].tolist())
            seeds_ok = sw if seeds_ok is None else (seeds_ok & sw)
        common_seeds_by_task[t] = sorted(seeds_ok) if seeds_ok else []
    min_seeds = min((len(v) for v in common_seeds_by_task.values()), default=0)
    if min_seeds == 0:
        return {}
    arr = {}
    for m in METHODS:
        mat = np.full((min_seeds, len(tasks)), np.nan)
        for j, t in enumerate(tasks):
            for i, sd in enumerate(common_seeds_by_task[t][:min_seeds]):
                row = sub[(sub.env == t) & (sub.method == m) & (sub.seed == sd)]
                if len(row) == 1 and np.isfinite(row.norm_score.iloc[0]):
                    mat[i, j] = row.norm_score.iloc[0]
        arr[m] = mat
    return arr


def iqm_with_ci(score_arrays, reps=10000):
    """{method: (point, lo, hi)} of the IQM."""
    aggr_func = lambda x: np.array([metrics.aggregate_iqm(x)])
    pts, cis = rly.get_interval_estimates(score_arrays, aggr_func, reps=reps,
                                          random_state=RNG)
    return {m: (float(pts[m][0]), float(cis[m][0, 0]), float(cis[m][1, 0]))
            for m in score_arrays}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.join("figures", "rliable"),
                    help="output directory of rliable_analysis.py")
    ap.add_argument("--out_dir", default=None, help="where to write the figure (default: --root)")
    ap.add_argument("--np_seed", type=int, default=None,
                    help="seed NumPy's global RNG (repeatable CIs; unset for the paper)")
    args = ap.parse_args()
    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.root, "scores_table.csv"))
    iqm_vh = iqm_with_ci(to_score_array(df, "vh"))
    iqm_vhr = iqm_with_ci(to_score_array(df, "vhr"))

    print(f"{'method':<18} {'VH IQM':>20} {'VHR IQM':>20}")
    print("-" * 62)
    for m in METHODS:
        if m not in iqm_vh or m not in iqm_vhr:
            continue
        x, xlo, xhi = iqm_vh[m]
        y, ylo, yhi = iqm_vhr[m]
        print(f"{m:<18}  {x:.3f} [{xlo:.3f},{xhi:.3f}]   {y:.3f} [{ylo:.3f},{yhi:.3f}]")

    fig, ax = plt.subplots(figsize=(4.25, 4))
    for m in METHODS:
        if m not in iqm_vh or m not in iqm_vhr:
            continue
        x, xlo, xhi = iqm_vh[m]
        y, ylo, yhi = iqm_vhr[m]
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt=MARKER[m], markersize=14 if m == "StarWM-R" else 11,
                    color=COLOR[m], ecolor=COLOR[m], elinewidth=1.2, capsize=4,
                    markeredgecolor="black", markeredgewidth=0.7, label=LABEL[m])
    # y = x reference
    lo = 0
    hi = max(max(iqm_vh[m][2] for m in iqm_vh), max(iqm_vhr[m][2] for m in iqm_vhr)) * 1.05
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_xlabel("VH (Sequential) IQM", fontsize=10)
    ax.set_ylabel("VHR (Random) IQM", fontsize=10)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Method IQM: VH vs VHR", fontsize=11)
    ax.set_title("norm. by Orig DreamerV3 clean", fontsize=9, style="italic",
                 color="#444444", pad=4)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=1)
    # axes drawn as two arrows from the origin; a single "0" label at the origin
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    nice_ticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(nice_ticks)
    ax.set_yticks(nice_ticks)
    ax.set_xticklabels([f"{t:g}" for t in nice_ticks])
    ax.set_yticklabels([f"{t:g}" for t in nice_ticks])
    for xy in ((hi, 0), (0, hi)):
        ax.annotate("", xy=xy, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"),
                    annotation_clip=False)
    ax.text(-0.012, -0.012, "0", fontsize=9, ha="right", va="top", transform=ax.transData)
    plt.tight_layout(rect=[0, 0, 1, 1.02])
    for ext in ("png", "pdf", "svg"):
        out = os.path.join(out_dir, f"iqm_scatter.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
