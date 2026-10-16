"""Fig. 15: which self-supervised signal guides the attention (NoisyShape).

StarWM (inverse dynamics + InfoNCE) vs. ``inv_only`` vs. ``ctr_only`` (model
presets in configs.yaml; detach A/B on in all three). Curves: mean ± std over
seeds of color / shape / size accuracy H0..H15; bars: H0 position MSE (x + y).

Example:
    python analysis/noisyshape/plot_ss_ablation.py --runs runs --out fig15_ss_ablation.pdf
    python analysis/noisyshape/plot_ss_ablation.py --runs logdir_paper --name_map paper
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

CONFIGS = [  # (variant, label, color, marker)
    ("starwm", "StarWM (inv + InfoNCE)", "#C62828", "o"),
    ("inv_only", "inv only", "#1565C0", "D"),
    ("ctr_only", "InfoNCE only", "#2E7D32", "^"),
]


def plot_ablation(curves, poses, horizon, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    H = np.arange(horizon + 1)
    for idx, attr in enumerate(["color", "shape", "size"]):
        ax = axes[idx]
        for name, label, color, marker in CONFIGS:
            if name not in curves:
                continue
            m, s = curves[name][attr]["mean"], curves[name][attr]["std"]
            ax.plot(H, m, "-", marker=marker, color=color, markersize=3.5, linewidth=1.8, label=label)
            ax.fill_between(H, m - s, m + s, color=color, alpha=0.15)
        ax.axhline(y=C.CHANCE[attr], color="gray", linestyle="--", alpha=0.5,
                   label="chance" if idx == 0 else None)
        ax.set_title(attr.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Imagination Horizon", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_ylim([0.15, 1.05])
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

    # Position MSE bars (configs without results are drawn as 0)
    ax = axes[3]
    vals, errs, colors = [], [], []
    for name, _, color, _ in CONFIGS:
        vals.append(poses[name]["position"]["mean"] if name in poses else 0)
        errs.append(poses[name]["position"]["std"] if name in poses else 0)
        colors.append(color)
    ax.bar(np.arange(len(CONFIGS)), vals, yerr=errs, color=colors, alpha=0.85,
           capsize=3, error_kw={"linewidth": 1.2})
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.0005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold",
                color=colors[i])
    ax.set_xticks([])
    ax.axhline(y=C.CHANCE_POS, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("Position (H0)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    lo = min(vals) - 0.01
    hi = max(v + e for v, e in zip(vals, errs)) + 0.005
    ax.set_ylim(max(0, lo), hi)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(CONFIGS) + 1, fontsize=12,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    C.ensure_parent(save_path)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--benchmark", default="noisyshape")
    p.add_argument("--out", default="fig15_ss_ablation.pdf")
    args = p.parse_args()

    curves, poses = {}, {}
    for name, label, _, _ in CONFIGS:
        res = C.load_results(args, args.benchmark, name)
        if len(res) == 0:
            print(f"WARNING: no seeds found for {name}")
            continue
        curves[name] = C.extract_imagined_curves(res, args.horizon)
        poses[name] = C.extract_position_mse(res)
        c = curves[name]
        print(f"  {label:<24} n={len(res)}  H15 color/shape/size = "
              + " / ".join(f"{c[a]['mean'][-1]:.4f}±{c[a]['std'][-1]:.4f}" for a in ["color", "shape", "size"])
              + f"  pos H0 = {poses[name]['position']['mean']:.4f}±{poses[name]['position']['std']:.4f}")
    plot_ablation(curves, poses, args.horizon, args.out)


if __name__ == "__main__":
    main()
