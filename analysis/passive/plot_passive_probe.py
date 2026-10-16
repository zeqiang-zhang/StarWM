#!/usr/bin/env python3
"""Passive-object probe results: Table 16 and Fig. 28 (Appendix F.4 (i)).

Reads, for every seed directory <root>/passiveshape/starwm_s<S> (random-frame
background) and <root>/passiveshapeseq/starwm_s<S> (sequential video),
probe_passive_results.json (b_colour accuracy) and
probe_passive_position_results.json (b_x / b_y R^2), prints mean +- SE over
seeds (the Table 16 layout), writes passive_probe_summary.json and draws the
bar figure. Colours encode the background, not a method.

Example:
  python analysis/passive/plot_passive_probe.py --root runs --out_dir figures
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402

import probe_common as pc  # noqa: E402

BG = [("random", "passiveshape", "random-frame", "#6A1B9A"),
      ("seq", "passiveshapeseq", "sequential video", "#00838F")]


def load(root, group, seeds=None):
    a = {"b_color": {"H0": [], "H15": []}}
    p = {k: {"H0": [], "H15": []} for k in ("b_x", "b_y")}
    dirs = (pc.seed_dirs(os.path.join(root, group, "starwm_s")) if seeds is None
            else [os.path.join(root, group, f"starwm_s{s}") for s in seeds])
    used = []
    for d in dirs:
        fa = os.path.join(d, "probe_passive_results.json")
        fp = os.path.join(d, "probe_passive_position_results.json")
        if not (os.path.exists(fa) and os.path.exists(fp)):
            print(f"[skip] {d}: probe results missing")
            continue
        ja, jp = json.load(open(fa)), json.load(open(fp))
        for h in ("H0", "H15"):
            a["b_color"][h].append(ja[h]["b_color"])
            for k in p:
                p[k][h].append(jp[h][k]["r2"])
        used.append(d)
    return a, p, used


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: all starwm_s<digits>")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    data, summary = {}, {}
    print("\nTable 16 (mean +- SE over seeds)")
    print(f"{'background':18s} {'n':>2s}  {'b_colour H0':>14s} {'b_colour H15':>14s} {'b_x H0':>13s} "
          f"{'b_x H15':>13s} {'b_y H0':>13s} {'b_y H15':>13s}")
    for mode, group, label, _ in BG:
        a, p, used = load(args.root, group, args.seeds)
        data[mode] = (a, p)
        row, summary[mode] = [], {"runs": used}
        for key, src, fmt in (("b_color", a, "{:.3f} +- {:.3f}"), ("b_x", p, "{:.2f} +- {:.2f}"),
                              ("b_y", p, "{:.2f} +- {:.2f}")):
            for h in ("H0", "H15"):
                m, s = pc.mean_se(src[key][h])
                summary[mode][f"{key}_{h}"] = {"mean": m, "se": s, "values": src[key][h]}
                row.append(f"{'-':>13s}" if m is None else f"{fmt.format(m, s):>13s}")
        print(f"{label:18s} {len(used):2d}  " + " ".join(row))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "passive_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plt = pc.setup_mpl()
    FS_T, FS_L, FS_TK, FS_LG = 9, 8.5, 8, 8
    plt.rcParams.update({"xtick.labelsize": FS_TK, "ytick.labelsize": FS_TK})
    fig, axes = plt.subplots(1, 3, figsize=(5.40, 1.85))
    x = np.arange(2)                       # H0, H15
    w = 0.34
    panels = [("b_color", "Identity (b_colour)", "Accuracy", 0.25, (0, 1.08)),
              ("b_x", "Position b_x", "$R^2$", 0.0, (-0.42, 1.05)),
              ("b_y", "Position b_y", "$R^2$", 0.0, (-0.42, 1.05))]
    for ax, (key, title, ylab, ref, ylim) in zip(axes, panels):
        for i, (mode, _, lab, col) in enumerate(BG):
            a, p = data[mode]
            src = a if key == "b_color" else p
            ys = [pc.mean_se(src[key][h])[0] for h in ("H0", "H15")]
            es = [pc.mean_se(src[key][h])[1] for h in ("H0", "H15")]
            ys = [np.nan if v is None else v for v in ys]
            es = [0.0 if v is None else v for v in es]
            ax.bar(x + (i - 0.5) * w, ys, w, yerr=es, color=col, capsize=2,
                   error_kw={"linewidth": 1.0}, label=lab)
        ax.axhline(ref, color="gray", ls="--", alpha=0.6, lw=1.3)
        ax.set_title(title, fontsize=FS_T, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(["H0", "H15"])
        ax.set_xlim(-0.55, 1.55); ax.set_ylim(*ylim)
        ax.set_ylabel(ylab, fontsize=FS_L, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("black")
        ax.grid(True, alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].annotate("chance", (-0.5, 0.26), fontsize=7.5, color="gray", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=FS_LG, frameon=False,
               handlelength=1.4, columnspacing=1.2, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    pc.savefig(fig, os.path.join(args.out_dir, "passive_probe"))
    plt.close(fig)


if __name__ == "__main__":
    main()
