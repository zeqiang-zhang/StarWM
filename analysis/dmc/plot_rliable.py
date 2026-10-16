#!/usr/bin/env python3
"""Fig 10 (a/b/c): rliable aggregate metrics, performance profiles and
probability-of-improvement heatmaps for VH, VHR and VH+VHR.

Reads ``{vh,vhr,combined}/{aggregate,profile,poi}.npz`` written by
``rliable_analysis.py`` and writes, in the same directory::

    all_aggregate.{png,pdf}    (a) 3 rows (VH, VHR, VH+VHR) x 4 metrics
    all_profile.{png,pdf}      (b) one performance profile per block
    all_poi_heatmap.{png,pdf}  (c) one P(A > B) heatmap per block

    python analysis/dmc/plot_rliable.py --root figures/rliable
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rliable import plot_utils

METHODS = ["StarWM", "StarWM-R", "Orig DreamerV3", "R2Dreamer", "DenoisedMDP", "IsoDreamer"]
COLORS = {
    "StarWM":         "#E53935",
    "StarWM-R":       "#e5a435",
    "Orig DreamerV3": "#1565C0",
    "R2Dreamer":      "#2E7D32",
    "DenoisedMDP":    "#6A1B9A",
    "IsoDreamer":     "#00838F",
}
BLOCKS = [("vh", "VH"), ("vhr", "VHR"), ("combined", "VH + VHR")]


def _load_npz_dict(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _save(out_path):
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_path}.{ext}", dpi=150, bbox_inches="tight")
        print(f"  saved {out_path}.{ext}")


def plot_aggregate_combined(root, out_path):
    """Forest plot: per metric, a CI bar per method and the point estimate."""
    metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
    n_blocks, n_metrics = len(BLOCKS), len(metric_names)
    fig, axes = plt.subplots(n_blocks, n_metrics,
                             figsize=(2.6 * n_metrics, 1.6 * n_blocks), sharey=False)
    for r, (sub, label) in enumerate(BLOCKS):
        data = _load_npz_dict(os.path.join(root, sub, "aggregate.npz"))
        algos = [m for m in METHODS if f"{m}__point" in data]
        for c, mname in enumerate(metric_names):
            ax = axes[r, c]
            # reversed so that the first method is at the top
            for i, m in enumerate(reversed(algos)):
                pt = float(data[f"{m}__point"][c])
                lo, hi = data[f"{m}__ci"][:, c]
                ax.barh(i, hi - lo, left=lo, height=0.55,
                        color=COLORS[m], alpha=0.6, edgecolor="none")
                ax.plot([pt, pt], [i - 0.32, i + 0.32], color="black", linewidth=1.4)
            ax.set_yticks(range(len(algos)))
            ax.set_yticklabels(list(reversed(algos)) if c == 0 else [], fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == 0:
                ax.set_title(mname, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if r == n_blocks - 1:
                ax.set_xlabel("Norm. score", fontsize=9)
    plt.tight_layout()
    _save(out_path)
    plt.close(fig)


def plot_profile_combined(root, out_path):
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(4.2 * len(BLOCKS), 3.5), sharey=True)
    handles_by_label = {}
    for ax, (sub, label) in zip(axes, BLOCKS):
        data = _load_npz_dict(os.path.join(root, sub, "profile.npz"))
        thresholds = data["__thresholds__"]
        prof_pts = {m: data[f"{m}__point"] for m in METHODS if f"{m}__point" in data}
        prof_cis = {m: data[f"{m}__ci"] for m in METHODS if f"{m}__ci" in data}
        plot_utils.plot_performance_profiles(
            prof_pts, thresholds,
            performance_profile_cis=prof_cis,
            colors={m: COLORS[m] for m in prof_pts if m in COLORS},
            xlabel=r"Normalized score $(\tau)$",
            ax=ax,
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0.0, 0.5, 1.0])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in handles_by_label:
                handles_by_label[lbl] = h
    axes[0].set_ylabel("Fraction of runs\n" r"with score $> \tau$", fontsize=11)
    ordered = [(lbl, handles_by_label[lbl]) for lbl in METHODS if lbl in handles_by_label]
    if ordered:
        labels, handles = zip(*ordered)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(labels), frameon=False, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(out_path)
    plt.close(fig)


def plot_poi_combined(root, out_path):
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(5.2 * len(BLOCKS), 4.5))
    last_im = None
    for col, (ax, (sub, label)) in enumerate(zip(axes, BLOCKS)):
        data = _load_npz_dict(os.path.join(root, sub, "poi.npz"))
        M = data["matrix"]
        methods_present = [str(s) for s in data["methods"]]
        n = len(methods_present)
        last_im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods_present, rotation=30, ha="right", fontsize=8)
        if col == 0:
            ax.set_yticklabels(methods_present, fontsize=8)
            ax.set_ylabel("Method (A)", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("vs (B)", fontsize=9)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v - 0.5) > 0.25 else "black")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout()
    plt.subplots_adjust(right=0.93, wspace=0.10)
    cbar_ax = fig.add_axes([0.95, 0.18, 0.012, 0.65])
    fig.colorbar(last_im, cax=cbar_ax, label="P(A > B)")
    _save(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.join("figures", "rliable"),
                    help="output directory of rliable_analysis.py")
    ap.add_argument("--out_dir", default=None, help="where to write the figures (default: --root)")
    args = ap.parse_args()
    out = args.out_dir or args.root
    os.makedirs(out, exist_ok=True)
    plot_aggregate_combined(args.root, os.path.join(out, "all_aggregate"))
    plot_profile_combined(args.root, os.path.join(out, "all_profile"))
    plot_poi_combined(args.root, os.path.join(out, "all_poi_heatmap"))
    print("\nDone.")


if __name__ == "__main__":
    main()
