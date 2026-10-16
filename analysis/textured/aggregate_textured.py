#!/usr/bin/env python3
"""Textured ladder: Table 13 and Fig. 24 (Appendix F.3 (i)) from per-run probe results.

Optionally runs probe_textured.py on every run first (--run). Then aggregates
<root>/L{1,2,3}/{starwm,dreamerv3}_s<seed>/probe_textured_results.json into
mean +- SE over seeds, prints the table, writes textured_probe_summary.json and
draws the ladder figure (accuracy vs level at H0 solid / H15 dashed for
texture-ID, shape and size; position MSE at H0, where higher = position
filtered out). Runs whose metrics.jsonl ends before --min_step are excluded
(and listed), as in the paper.

Example:
  python analysis/textured/aggregate_textured.py --root runs/texnoisyshape \
      --run --device cuda:0 --cache_dir runs/traj_cache --out_dir figures
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import probe_common as pc  # noqa: E402

LEVELS = [1, 2, 3]
METHODS = ["starwm", "dreamerv3"]
ATTRS = ["texture", "shape", "size"]
PROBE = str(pathlib.Path(__file__).resolve().parent / "probe_textured.py")
# Only aggregate seeds trained to a comparable level; an under-trained seed
# probes near chance and would pollute the mean.
MIN_STEP = 150000


def runs_of(root, level, method):
    return pc.seed_dirs(os.path.join(root, f"L{level}", f"{method}_s"))


def run_probes(args):
    # Trajectory sets are model-free and identical for every run of a level -> one cache per level.
    os.makedirs(args.cache_dir, exist_ok=True)
    for L in LEVELS:
        cache = os.path.join(args.cache_dir, f"tex_L{L}.npz")
        for m in METHODS:
            for run in runs_of(args.root, L, m):
                if not os.path.exists(os.path.join(run, args.checkpoint)):
                    continue
                st = pc.last_logged_step(run)
                if st < args.min_step:
                    print(f"SKIP {run} (step {st} < {args.min_step}, under-trained)")
                    continue
                cmd = [sys.executable, PROBE, "--logdir", run, "--checkpoint", args.checkpoint,
                       "--tex_level", str(L), "--davis_path", args.davis_path, "--model", m,
                       "--n_curr", str(args.n_curr), "--n_imag", str(args.n_imag),
                       "--horizon", str(args.horizon), "--device", args.device,
                       "--traj_cache", cache]
                print("PROBE", run)
                subprocess.run(cmd, check=True)


def aggregate(root, min_step):
    """-> data[level][method][stage][attr] = list of per-seed values."""
    data, excluded = {}, []
    for L in LEVELS:
        data[L] = {}
        for m in METHODS:
            acc = {"H0": {a: [] for a in ATTRS + ["position_mse"]},
                   "H15": {a: [] for a in ATTRS}}
            for run in runs_of(root, L, m):
                rp = os.path.join(run, "probe_textured_results.json")
                if not os.path.exists(rp):
                    continue
                st = pc.last_logged_step(run)
                if st < min_step:
                    excluded.append(f"{run}(step {st})")
                    continue
                r = json.load(open(rp))
                for a in ATTRS + ["position_mse"]:
                    if a in r.get("H0", {}):
                        acc["H0"][a].append(r["H0"][a])
                for a in ATTRS:
                    if a in r.get("H15", {}):
                        acc["H15"][a].append(r["H15"][a])
            data[L][m] = acc
    if excluded:
        print(f"[excluded {len(excluded)} under-trained seed(s) (step < {min_step}): "
              + ", ".join(excluded) + "]")
    return data


def print_table(data):
    cols = [("H0", "texture"), ("H15", "texture"), ("H0", "shape"), ("H15", "shape"),
            ("H0", "size"), ("H15", "size"), ("H0", "position_mse")]
    print("\nTable 13 (mean over seeds; +-SE in the summary json)")
    print(f"{'level':5s} {'model':10s} {'n':>2s} " + " ".join(f"{a[:5]}{s:>4s}" for s, a in cols))
    for L in LEVELS:
        for m in METHODS:
            n = len(data[L][m]["H0"]["texture"])
            vals = []
            for s, a in cols:
                mean, _ = pc.mean_se(data[L][m][s][a])
                vals.append("   -    " if mean is None else f"{mean:9.3f}")
            print(f"L{L:<4d} {pc.METHOD_LABEL[m]:10s} {n:2d} " + " ".join(vals))


def plot(data, out):
    """Ladder figure: three accuracy panels sharing a y-axis + position-MSE bar panel."""
    plt = pc.setup_mpl()
    FS_TITLE, FS_LAB, FS_TICK, FS_LEG = 9, 8.5, 8, 8
    plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK})
    fig, axes = plt.subplots(1, 4, figsize=(5.45, 1.85),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
    for k in (1, 2):
        axes[k].sharey(axes[0])
    C = {m: pc.METHOD_COLOR[m] for m in METHODS}
    MK = {"starwm": "o", "dreamerv3": "s"}
    CHANCE = {"texture": 0.25, "shape": 0.25, "size": 1.0 / 3.0}
    TITLE = {"texture": "Texture-ID", "shape": "Shape", "size": "Size"}
    x = np.arange(len(LEVELS))

    def boxify(ax):
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("black")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    for ax, attr in zip(axes[:3], ATTRS):
        for m in METHODS:
            for stage, ls, fill in (("H0", "-", True), ("H15", "--", False)):
                ys, es = [], []
                for L in LEVELS:
                    mean, se = pc.mean_se(data[L][m][stage].get(attr, []))
                    ys.append(np.nan if mean is None else mean)
                    es.append(0.0 if se is None else se)
                ys, es = np.array(ys, float), np.array(es, float)
                ax.plot(x, ys, ls=ls, marker=MK[m], ms=4, lw=2, color=C[m],
                        markerfacecolor=(C[m] if fill else "white"),
                        markeredgecolor=C[m], markeredgewidth=1.3,
                        label=f"{pc.METHOD_LABEL[m]} {stage}")
                ax.fill_between(x, ys - es, ys + es, color=C[m], alpha=0.2, linewidth=0)
        ax.axhline(CHANCE[attr], color="gray", linestyle="--", alpha=0.6, lw=1.4)
        ax.set_title(TITLE[attr], fontsize=FS_TITLE, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in LEVELS])
        ax.set_xlim(-0.28, len(LEVELS) - 0.72); ax.set_ylim(0.15, 1.06)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Complexity", fontsize=FS_LAB)
        boxify(ax)
    axes[0].set_ylabel("Accuracy", fontsize=FS_LAB, fontweight="bold")
    for k in (1, 2):
        axes[k].tick_params(labelleft=False)

    # position MSE @H0 -- higher is better here (position is objective-irrelevant)
    axp = axes[3]
    width = 0.32
    for i, m in enumerate(METHODS):
        ys, es = [], []
        for L in LEVELS:
            mean, se = pc.mean_se(data[L][m]["H0"].get("position_mse", []))
            ys.append(np.nan if mean is None else mean)
            es.append(0.0 if se is None else se)
        pos = x + (i - 0.5) * width
        axp.bar(pos, ys, width, yerr=es, color=C[m], capsize=2,
                error_kw={"linewidth": 1.0}, label=pc.METHOD_LABEL[m])
    axp.axhline(0.071, color="gray", linestyle="--", alpha=0.6, lw=1.4)   # chance
    axp.set_title("Position (H0)", fontsize=FS_TITLE, fontweight="bold")
    axp.set_xticks(x); axp.set_xticklabels([f"L{L}" for L in LEVELS])
    axp.set_xlim(-0.62, len(LEVELS) - 0.38); axp.set_ylim(0, 0.088)
    axp.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08])
    axp.set_ylabel(r"MSE $\uparrow$", fontsize=FS_LAB, fontweight="bold")
    axp.set_xlabel("Complexity", fontsize=FS_LAB)
    boxify(axp)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=FS_LEG,
               frameon=False, handlelength=1.5, columnspacing=0.9,
               handletextpad=0.4, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout(pad=0.3, w_pad=0.35)
    pc.savefig(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="runs/texnoisyshape",
                    help="contains L{1,2,3}/{starwm,dreamerv3}_s<seed>/")
    ap.add_argument("--run", action="store_true", help="run probe_textured.py on all runs first")
    ap.add_argument("--checkpoint", default="latest.pt")
    ap.add_argument("--davis_path", default=pc.default_davis())
    ap.add_argument("--cache_dir", default="runs/traj_cache")
    ap.add_argument("--n_curr", type=int, default=500)
    ap.add_argument("--n_imag", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min_step", type=int, default=MIN_STEP)
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    if args.run:
        run_probes(args)   # DAVIS is only needed where a level's cache does not exist yet
    data = aggregate(args.root, args.min_step)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "textured_probe_summary.json"), "w") as f:
        json.dump({str(L): {m: data[L][m] for m in METHODS} for L in LEVELS}, f, indent=2)
    n = sum(len(data[L][m]["H0"]["texture"]) for L in LEVELS for m in METHODS)
    print(f"aggregated {n} probe results")
    print_table(data)
    plot(data, os.path.join(args.out_dir, "textured_probe_ladder"))


if __name__ == "__main__":
    main()
