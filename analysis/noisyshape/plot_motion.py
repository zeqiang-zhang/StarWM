"""Fig. 18 and Table 11: NoisyShape with predictable (uniform linear) motion.

Fig. 18 left: position MSE (x + y) of the imagined latents for h = 1..15
(mean over seeds; band = sqrt(std_x^2 + std_y^2), as in the paper script), with
the chance level and the H0 position MSE of StarWM on random-position
NoisyShape as reference lines. Right: 20-way background-ID probe accuracy at H0
(mean ± std over seeds). Table 11 (mean ± std over seeds, ddof=0) and the
per-horizon tables of the paper script are printed afterwards.
Input: result files of ``probe_motion.py``.

Example:
    python analysis/noisyshape/plot_motion.py --runs runs --out fig18_motion.pdf
    python analysis/noisyshape/plot_motion.py --runs logdir_paper --name_map paper
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

BENCH = "noisyshapemotion"
MODELS = {  # label: (variant, color, marker)
    "StarWM": ("starwm", "#E53935", "o"),
    "Orig DreamerV3": ("dreamerv3", "#1565C0", "s"),
}
N_BG = 20


def horizon_series(runs, key, horizons):
    vals = np.array([[r["imagined_probes"][str(h)][key] for h in horizons] for r in runs])
    return vals.mean(0), vals.std(0)


def plot_position_curves(ax, data, horizons):
    for mname, (_, color, marker) in MODELS.items():
        mx, sx = horizon_series(data[mname], "pos_x", horizons)
        my, sy = horizon_series(data[mname], "pos_y", horizons)
        m, s = mx + my, np.sqrt(sx ** 2 + sy ** 2)
        ax.plot(horizons, m, color=color, marker=marker, markersize=4, label=mname, linewidth=2)
        ax.fill_between(horizons, m - s, m + s, color=color, alpha=0.15)


def plot_bg_bars(ax, data):
    for i, (mname, (_, color, _)) in enumerate(MODELS.items()):
        vals = [r["current_probes"]["bg_id"] for r in data[mname]]
        ax.bar(i, np.mean(vals), 0.5, yerr=np.std(vals), color=color, capsize=5,
               edgecolor="black", linewidth=0.8, label=mname)
    ax.axhline(y=1 / N_BG, color="gray", linestyle="--", linewidth=1, label="Chance Level")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(list(MODELS.keys()))
    ax.set_ylabel("Background ID Accuracy")
    ax.set_title(f"H0 Background ID Probe ({N_BG} classes)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")


# Chance-level position MSE (x + y) of this variant, reported in Table 11: the
# object bounces inside [0.05, 0.95]^2 and its position is uniform there, so
# chance = 2 * 0.9^2 / 12 = 0.135. The figure's grey dashed line instead shows the
# chance level of the random-position NoisyShape (C.CHANCE_POS = 0.0711), for
# comparison with the red dashed line (StarWM on random-position NoisyShape).
CHANCE_POS_MOTION = 0.9 ** 2 / 6


def plot_figure(data, horizon, noisyshape_ref, save_path):
    """The paper figure: position MSE over h = 1..H and H0 background ID."""
    horizons = list(range(1, horizon + 1))
    fig, ax3 = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax3[0]
    plot_position_curves(ax, data, horizons)
    ax.axhline(y=C.CHANCE_POS, color="gray", linestyle="--", linewidth=1, label="Chance Level")
    if noisyshape_ref is not None:
        ax.axhline(y=noisyshape_ref, color="#E53935", linestyle="--", linewidth=1,
                   label="NoisyShape (random position) H0-StarWM")
    ax.set_xlabel("Imagination Horizon")
    ax.set_ylabel("Position MSE")
    ax.set_title("Position MSE")
    ax.set_xlim(1, horizon)
    ax.set_ylim(0.0, 0.09)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plot_bg_bars(ax3[1], data)
    plt.tight_layout()
    C.ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_horizon_figure(data, horizon, save_path):
    """Supplementary: color / shape / size accuracy and position MSE over h = 1..H."""
    horizons = list(range(1, horizon + 1))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("NoisyShape (uniform motion) latent probes: StarWM vs DreamerV3",
                 fontsize=14, fontweight="bold")
    for ax, attr in zip(axes[:3], ["color", "shape", "size"]):
        for mname, (_, color, marker) in MODELS.items():
            m, s = horizon_series(data[mname], attr, horizons)
            ax.plot(horizons, m, color=color, marker=marker, markersize=4, label=mname, linewidth=2)
            ax.fill_between(horizons, m - s, m + s, color=color, alpha=0.15)
        ax.set_xlabel("Imagination Horizon")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{attr.capitalize()} Accuracy")
        ax.set_xlim(1, horizon)
        ax.set_ylim(0.9, 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax = axes[3]
    plot_position_curves(ax, data, horizons)
    ax.set_xlabel("Imagination Horizon")
    ax.set_ylabel("Position MSE")
    ax.set_title("Position MSE (x + y)")
    ax.set_xlim(1, horizon)
    ax.set_ylim(0.0, C.CHANCE_POS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    C.ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_tables(data, horizon):
    print("\n" + "=" * 100)
    print("Table 11: H0 = imagined_probes['0'] (BG ID: current probes), H15 = imagined; "
          "mean ± std over seeds; accuracies in %")
    print("=" * 100)
    print(f"{'Stage':<6} {'Model':<16} {'Color':>16} {'Shape':>16} {'Size':>16} "
          f"{'Position MSE':>18} {'BG ID':>16}")
    for stage in ["0", str(horizon)]:
        for mname in MODELS:
            runs = data[mname]
            ip = [r["imagined_probes"][stage] for r in runs]
            row = [C.fmt([x[a] for x in ip], 100, 2) for a in ["color", "shape", "size"]]
            row.append(C.fmt([x["pos_x"] + x["pos_y"] for x in ip], 1, 4))
            row.append(C.fmt([r["current_probes"]["bg_id"] for r in runs], 100, 2)
                       if stage == "0" else "---")
            print(f"{'H' + stage:<6} {mname:<16} " + " ".join(f"{v:>16}" for v in row[:3])
                  + f" {row[3]:>18} {row[4]:>16}")
    print(f"{'Chance':<23} {25:>16.2f} {25:>16.2f} {100 / 3:>16.2f} {CHANCE_POS_MOTION:>18.4f} "
          f"{100 / N_BG:>16.2f}")

    # Raw tables of the paper script (fractions, 4 decimals)
    print("\n--- H0 (current probes, 500-episode set) ---")
    keys = ["color", "shape", "size", "pos_x", "pos_y", "bg_id"]
    print(f"{'Model':<20} " + " ".join(f"{k:>16}" for k in keys))
    for mname in MODELS:
        row = [C.fmt([r["current_probes"][k] for r in data[mname]], 1, 4) for k in keys]
        print(f"{mname:<20} " + " ".join(f"{v:>16}" for v in row))
    print("\n--- Imagined horizons ---")
    keys = ["color", "shape", "size", "pos_x", "pos_y"]
    print(f"{'Model':<16} {'H':>3} " + " ".join(f"{k:>16}" for k in keys) + f" {'pos (x+y)':>16}")
    for mname in MODELS:
        for h in [0, 1, 5, 10, 15]:
            if h > horizon:
                continue
            ip = [r["imagined_probes"][str(h)] for r in data[mname]]
            row = [C.fmt([x[k] for x in ip], 1, 4) for k in keys]
            row.append(C.fmt([x["pos_x"] + x["pos_y"] for x in ip], 1, 4))
            print(f"{mname:<16} {h:>3} " + " ".join(f"{v:>16}" for v in row))
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--noisyshape_ref", type=float, default=0.066,
                   help="StarWM H0 position MSE on random-position NoisyShape drawn as reference "
                        "(paper: 0.066, Table 4); negative to omit")
    p.add_argument("--out", default="fig18_motion.pdf")
    p.add_argument("--horizon_out", default=None,
                   help="optional extra figure with all attributes over the horizon")
    args = p.parse_args()

    data = {}
    for mname, (variant, _, _) in MODELS.items():
        data[mname] = C.load_results(args, BENCH, variant)
        print(f"  {mname}: {len(data[mname])} seeds")
        if not data[mname]:
            raise SystemExit(f"missing results for {mname}")
    ref = args.noisyshape_ref if args.noisyshape_ref >= 0 else None
    plot_figure(data, args.horizon, ref, args.out)
    if args.horizon_out:
        plot_horizon_figure(data, args.horizon, args.horizon_out)
    print_tables(data, args.horizon)


if __name__ == "__main__":
    main()
