"""Fig. 3 (latent probes on NoisyShape, all baselines / ablations) and Table 4.

Left three panels: color / shape / size probe accuracy from H0 to H15 of
open-loop imagination (mean ± std over seeds as shaded band; H0 is the probe
fitted on the imagination set). Right panel: H0 position MSE (x + y, current
probes), bars sorted by MSE, dashed line = chance. Afterwards Table 4 is
printed (mean ± std over seeds, np.std with ddof=0, as in the paper scripts).

R2-Dreamer and Iso-Dream++ come from their own code bases; pass their probe
result files with ``--r2dreamer`` / ``--isodream`` (Iso-Dream++ files hold a
'controllable' and a 'combined' entry, both are plotted).

Example (release layout, runs/noisyshape/<variant>_s<seed>):
    python analysis/noisyshape/plot_noisyshape_probes.py --runs runs --out fig3_noisyshape_probes.pdf \\
        --r2dreamer r2/probe_results/s{0,1,2}.json --isodream iso/s{0,1,2}/results.json
Original paper runs:
    python analysis/noisyshape/plot_noisyshape_probes.py --runs logdir_paper --name_map paper \\
        --isodream logdir_paper/noisyshapev2/isodream_dmccfg_s{0,1,2}/probe_results/results.json
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import _common as C

BENCH = "noisyshape"
# (label, release variant, color, marker); order = plotting order of the paper figure
METHODS = [
    ("Orig Dreamer", "dreamerv3", "#1565C0", "-s"),
    ("SS-only", "ss_only", "#43A047", "-^"),
    ("Dual-stream-only", "dualstream", "#FB8C00", "-v"),
    ("Dual-stream+SS", "dualstream_ss", "#8E24AA", "-D"),
    ("Attention+SS", "attention_only", "#00ACC1", "-P"),
    ("R2-Dreamer", "@r2dreamer", "#6D4C41", "-X"),
    ("IsoDream++ (ctrl)", "@isodream:controllable", "#00695C", "-h"),
    ("IsoDream++ (full)", "@isodream:combined", "#4DB6AC", "-H"),
    ("StarWM", "starwm", "#E53935", "-o"),
]
# Table 4 rows: (row name, variant)
TABLE_ROWS = [
    ("DreamerV3", "dreamerv3"), ("+ dual-stream only", "dualstream"), ("+ SS only", "ss_only"),
    ("+ dual-stream + SS", "dualstream_ss"), ("+ attention only", "attention_only"),
    ("StarWM", "starwm"), ("R2-Dreamer", "@r2dreamer"),
    ("Iso-Dream++ (ctrl)", "@isodream:controllable"), ("Iso-Dream++ (full)", "@isodream:combined"),
]


def load_all(args):
    res = {}
    for _, variant, _, _ in METHODS:
        if variant == "@r2dreamer":
            res[variant] = [C.load_result_file(p) for p in args.r2dreamer]
        elif variant.startswith("@isodream:"):
            res[variant] = C.load_isodream_results(args.isodream, variant.split(":")[1])
        else:
            res[variant] = C.load_results(args, BENCH, variant)
    return res


def plot(res, horizon, legend, save_path):
    methods = []
    for label, variant, color, marker in METHODS:
        r = res[variant]
        if len(r) == 0:
            methods.append((label, None, None, color, marker))
            continue
        methods.append((label, C.extract_imagined_curves(r, horizon), C.extract_position_mse(r),
                        color, marker))

    fig = plt.figure(figsize=(8.4, 1.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
    inner = outer[0].subgridspec(1, 3, wspace=0.08)
    axes = [fig.add_subplot(inner[0, i]) for i in range(3)] + [fig.add_subplot(outer[1])]
    H = np.arange(horizon + 1)

    for i, attr in enumerate(["color", "shape", "size"]):
        ax = axes[i]
        for label, curves, _, color, marker in methods:
            if curves is None:
                continue
            m, s = curves[attr]["mean"], curves[attr]["std"]
            ax.plot(H, m, marker, color=color, markersize=3, linewidth=2, label=label)
            ax.fill_between(H, m - s, m + s, color=color, alpha=0.2)
        ax.axhline(y=C.CHANCE[attr], color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.set_ylim([0.2, 1.05])
        ax.grid(True, alpha=0.2)
        ax.set_title(attr.capitalize(), fontsize=13, fontweight="bold")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        if i == 0:
            ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if i == 1:
            ax.set_xlabel("Imagination Horizon", fontsize=10, fontweight="bold")

    # Position MSE bars (MSE_x + MSE_y), sorted descending
    ax = axes[3]
    present = [(label, pos, color) for label, _, pos, color, _ in methods if pos is not None]
    present.sort(key=lambda t: t[1]["position"]["mean"], reverse=True)
    for i, (label, pos, color) in enumerate(present):
        ax.bar(i, pos["position"]["mean"], 0.6, yerr=pos["position"]["std"], color=color,
               alpha=0.85, label=label, capsize=4, error_kw={"linewidth": 1.5})
    ax.set_xticks([])
    ax.set_xlabel("Methods", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_title("Position (H0)", fontsize=13, fontweight="bold")
    ax.set_ylabel("MSE ↑", fontsize=10, fontweight="bold")
    ax.set_ylim([0, 0.085])
    ax.axhline(y=C.CHANCE_POS, color="gray", linestyle="--", alpha=0.5, label="chance")

    if legend != "none":
        handles, labels = axes[0].get_legend_handles_labels()
        if legend == "baselines":  # what the paper figure shows (other rows: table on its left)
            keep = {"R2-Dreamer", "IsoDream++ (ctrl)", "IsoDream++ (full)", "chance"}
            pairs = [(h, l) for h, l in zip(handles, labels) if l in keep]
            handles, labels = [h for h, _ in pairs], [l for _, l in pairs]
        if labels:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       ncol=min(len(labels), 5), fontsize=8, frameon=False)
    plt.tight_layout()
    C.ensure_parent(save_path)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def print_table(res, horizon):
    """Table 4 (summarize_probes.py logic): mean ± std over seeds (ddof=0).

    Two H0 variants exist in the result files: 'imag' = ``imagined_probes['0']``
    (probe fitted on the 200-episode imagination set; first point of the Fig. 3
    curves) and 'enc' = ``current_probes`` (500-episode probe set). Position
    H0 always uses the current probes. The paper's Table 4 uses 'imag' for the
    DreamerV3 / attention-only / StarWM rows and 'enc' for the dual-stream /
    SS rows.
    """
    def cur(r, a):
        v = r["current_probes"][a]
        return v["test_acc"] if isinstance(v, dict) else v

    hdr = (f"{'Model':<22}| {'Color':>15} {'Shape':>15} {'Size':>15} {'Pos MSE':>15} "
           f"| {'Color H15':>15} {'Shape H15':>15} {'Size H15':>15}")
    for h0 in ("imag", "enc"):
        print("\n" + "=" * len(hdr))
        print(f"Table 4  (H0 = {'imagined_probes[0]' if h0 == 'imag' else 'current_probes'}; "
              f"mean ± std over seeds; accuracies in %)")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for name, variant in TABLE_ROWS:
            R = res.get(variant, [])
            if not R:
                continue
            row = []
            for a in ["color", "shape", "size"]:
                vals = ([r["imagined_probes"]["0"][a] for r in R] if h0 == "imag"
                        else [cur(r, a) for r in R])
                row.append(C.fmt(vals, 100, 2))
            pos = [r["current_probes"]["pos_x"]["test_mse"] + r["current_probes"]["pos_y"]["test_mse"]
                   for r in R]
            row.append(C.fmt(pos, 1, 3))
            for a in ["color", "shape", "size"]:
                row.append(C.fmt([r["imagined_probes"][str(horizon)][a] for r in R], 100, 2))
            print(f"{name:<22}| " + " ".join(f"{v:>15}" for v in row[:4]) + " | "
                  + " ".join(f"{v:>15}" for v in row[4:]) + f"   (n={len(R)})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--r2dreamer", nargs="*", default=[], help="R2-Dreamer results.json files (one per seed)")
    p.add_argument("--isodream", nargs="*", default=[], help="Iso-Dream++ results.json files (one per seed)")
    p.add_argument("--legend", choices=["all", "baselines", "none"], default="all")
    p.add_argument("--out", default="fig3_noisyshape_probes.pdf")
    args = p.parse_args()

    res = load_all(args)
    for label, variant, _, _ in METHODS:
        print(f"  {label}: {len(res[variant])} seeds")
    if not res["starwm"] or not res["dreamerv3"]:
        raise SystemExit("missing StarWM / DreamerV3 results")
    plot(res, args.horizon, args.legend, args.out)
    print_table(res, args.horizon)


if __name__ == "__main__":
    main()
