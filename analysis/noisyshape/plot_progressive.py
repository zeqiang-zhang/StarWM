"""Fig. 19 and Table 12: StarWM vs DreamerV3 across NoisyShape variants.

Rows: Shape (clean, random position), NoisyShape, NoisyShape (Seq); columns:
color / shape / size accuracy H0..H15 (mean ± std over seeds) and the H0
position MSE (x + y) bars with the chance level. Table 12 (mean ± std over
seeds, ddof=0; H0 = imagined_probes['0']) is printed afterwards and also
includes the Shape-fixed DreamerV3 row when its results exist (no position
probe there: the position is constant). ``--per_env_dir`` additionally writes
one 1x4 figure per environment.

Example:
    python analysis/noisyshape/plot_progressive.py --runs runs --out fig19_progressive.pdf
    python analysis/noisyshape/plot_progressive.py --runs logdir_paper --name_map paper
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _common as C

ENVS = ["shape", "noisyshape", "noisyshapeseq"]
NICE = {"shape": "Shape Env", "noisyshape": "Noisy Shape Env", "noisyshapeseq": "Noisy Shape Env (Seq)"}
TABLE_NAMES = {"shapefixed": "Shape (fixed)", "shape": "Shape (clean)", "noisyshape": "NoisyShape",
               "noisyshapeseq": "NoisyShape (Seq)"}
C_S, C_O = "#E53935", "#1565C0"


def plot_combined(all_s, all_o, pos_s, pos_o, envs, horizon, save_path):
    """len(envs) x 4 grid: rows = envs, cols = color / shape / size / position."""
    # The paper script had figsize=(10, ) (invalid); (10, 6) reproduces the
    # aspect ratio of the published figure.
    fig, axes = plt.subplots(len(envs), 4, figsize=(10, 2 * len(envs)), squeeze=False)
    H = np.arange(horizon + 1)
    last = len(envs) - 1
    for row, env in enumerate(envs):
        sc, oc = all_s[env], all_o[env]
        for col, attr in enumerate(["color", "shape", "size"]):
            ax = axes[row, col]
            sm, ss = sc[attr]["mean"], sc[attr]["std"]
            om, os_ = oc[attr]["mean"], oc[attr]["std"]
            ax.plot(H, sm, "-o", color=C_S, markersize=3, linewidth=2, label="StarWM")
            ax.fill_between(H, sm - ss, sm + ss, color=C_S, alpha=0.2)
            ax.plot(H, om, "-s", color=C_O, markersize=3, linewidth=2, label="Orig")
            ax.fill_between(H, om - os_, om + os_, color=C_O, alpha=0.2)
            ax.axhline(y=C.CHANCE[attr], color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.set_ylim([0.2, 1.05] if attr in ("color", "shape") else [0.3, 1.05])
            if row == 0:
                ax.set_title(attr.capitalize(), fontsize=13, fontweight="bold")
            if row == last:
                ax.set_xlabel("Imagination Horizon", fontsize=10)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f"{NICE.get(env, env)}\nAccuracy", fontsize=10, fontweight="bold")
            else:
                ax.tick_params(labelleft=False)
            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc="lower left")

        # Position MSE bars (MSE_x + MSE_y)
        ax = axes[row, 3]
        width = 0.3
        s_val, s_err = pos_s[env]["position"]["mean"], pos_s[env]["position"]["std"]
        o_val, o_err = pos_o[env]["position"]["mean"], pos_o[env]["position"]["std"]
        ax.bar(-width / 2, [s_val], width, yerr=[s_err], color=C_S, alpha=0.85, label="StarWM",
               capsize=4, error_kw={"linewidth": 1.5})
        ax.bar(width / 2, [o_val], width, yerr=[o_err], color=C_O, alpha=0.85, label="Orig",
               capsize=4, error_kw={"linewidth": 1.5})
        ax.text(-width / 2, s_val + s_err + 0.001, f"{s_val:.4f}", ha="center", fontsize=8,
                color=C_S, fontweight="bold")
        ax.text(width / 2, o_val + o_err + 0.001, f"{o_val:.4f}", ha="center", fontsize=8,
                color=C_O, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels(["Position"], fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        if row == 0:
            ax.set_title("Position (H0)", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
        if row != last:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelleft=False)
        ax.set_ylim(0, 0.08)
        ax.axhline(y=C.CHANCE_POS, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    C.ensure_parent(save_path)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_env(sc, oc, sp, op, env_name, horizon, save_path):
    """1x4 figure for a single environment."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.7), gridspec_kw={"width_ratios": [1, 1, 1, 0.6]})
    fig.suptitle(f"Latent Probe — {env_name}", fontsize=14, fontweight="bold")
    H = np.arange(horizon + 1)
    for idx, attr in enumerate(["color", "shape", "size"]):
        ax = axes[idx]
        ax.plot(H, sc[attr]["mean"], "-o", color=C_S, markersize=3, linewidth=2, label="StarWM")
        ax.fill_between(H, sc[attr]["mean"] - sc[attr]["std"], sc[attr]["mean"] + sc[attr]["std"],
                        color=C_S, alpha=0.2)
        ax.plot(H, oc[attr]["mean"], "-s", color=C_O, markersize=3, linewidth=2, label="Orig DreamerV3")
        ax.fill_between(H, oc[attr]["mean"] - oc[attr]["std"], oc[attr]["mean"] + oc[attr]["std"],
                        color=C_O, alpha=0.2)
        ax.axhline(y=C.CHANCE[attr], color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.set_xlabel("Imagination Horizon", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(attr.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylim([0.2, 1.05])
        ax.grid(True, alpha=0.2)
    ax = axes[3]
    width = 0.3
    for x, pos, color, label in [(-width / 2, sp, C_S, "StarWM"), (width / 2, op, C_O, "Orig")]:
        v, e = pos["position"]["mean"], pos["position"]["std"]
        ax.bar(x, [v], width, yerr=[e], color=color, alpha=0.85, label=label, capsize=4,
               error_kw={"linewidth": 1.5})
        ax.text(x, v + e + 0.001, f"{v:.4f}", ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["H0"], fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_title("Position (H0)", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0, 0.08)
    ax.set_ylabel("MSE", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    C.ensure_parent(save_path)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_table(res, horizon):
    """Table 12 layout; H0 = imagined_probes['0'], position = current probes (x + y)."""
    print("\n" + "=" * 96)
    print("Table 12: mean ± std over seeds (ddof=0); accuracies in %, position = MSE_x + MSE_y")
    print("=" * 96)
    for stage, title in [("0", "H0 (Encoding)"), (str(horizon), f"H{horizon} (Imagination)")]:
        print(f"\n{title}")
        print(f"{'Environment':<18} {'Model':<10} {'Color':>16} {'Shape':>16} {'Size':>16} {'Pos. MSE':>16}")
        for env in ["shapefixed"] + ENVS:
            for mname, variant in [("DreamerV3", "dreamerv3"), ("StarWM", "starwm")]:
                R = res.get((env, variant), [])
                if not R:
                    continue
                row = [C.fmt([r["imagined_probes"][stage][a] for r in R], 100, 2)
                       for a in ["color", "shape", "size"]]
                has_pos = all("pos_x" in r["current_probes"] for r in R)
                if stage == "0" and has_pos:
                    row.append(C.fmt([r["current_probes"]["pos_x"]["test_mse"]
                                      + r["current_probes"]["pos_y"]["test_mse"] for r in R], 1, 3))
                else:
                    row.append("---")
                print(f"{TABLE_NAMES[env]:<18} {mname:<10} " + " ".join(f"{v:>16}" for v in row)
                      + f"   (n={len(R)})")
        print(f"{'Chance level':<29} {25:>16.2f} {25:>16.2f} {100 / 3:>16.2f} "
              f"{(f'{C.CHANCE_POS:.3f}' if stage == '0' else '---'):>16}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--out", default="fig19_progressive.pdf")
    p.add_argument("--per_env_dir", default=None, help="also write one figure per environment here")
    args = p.parse_args()

    res = {}
    for env in ["shapefixed"] + ENVS:
        for variant in ["starwm", "dreamerv3"]:
            if env == "shapefixed" and variant == "starwm":
                continue
            res[(env, variant)] = C.load_results(args, env, variant)
            print(f"  {env}/{variant}: {len(res[(env, variant)])} seeds")

    all_s, all_o, pos_s, pos_o = {}, {}, {}, {}
    for env in ENVS:
        sr, orr = res[(env, "starwm")], res[(env, "dreamerv3")]
        if not sr or not orr:
            print(f"  Skipping {env} (missing results)")
            continue
        all_s[env], all_o[env] = (C.extract_imagined_curves(sr, args.horizon),
                                  C.extract_imagined_curves(orr, args.horizon))
        pos_s[env], pos_o[env] = C.extract_position_mse(sr), C.extract_position_mse(orr)
        if args.per_env_dir:
            plot_env(all_s[env], all_o[env], pos_s[env], pos_o[env], NICE[env], args.horizon,
                     os.path.join(args.per_env_dir, f"{env}_probe_comparison.png"))
    envs = [e for e in ENVS if e in all_s]
    if envs:
        plot_combined(all_s, all_o, pos_s, pos_o, envs, args.horizon, args.out)

    print_table(res, args.horizon)
    # summary.txt format of the paper script (fractions)
    print("\nSummary: H0 and H15 accuracy (mean ± std)")
    for env in envs:
        print(f"\n{env}:")
        for attr in ["color", "shape", "size"]:
            s, o = all_s[env][attr], all_o[env][attr]
            print(f"  {attr:>6}: StarWM H0={s['mean'][0]:.4f}±{s['std'][0]:.4f} "
                  f"H{args.horizon}={s['mean'][-1]:.4f}±{s['std'][-1]:.4f} | "
                  f"Orig H0={o['mean'][0]:.4f}±{o['std'][0]:.4f} "
                  f"H{args.horizon}={o['mean'][-1]:.4f}±{o['std'][-1]:.4f}")
        sp, op = pos_s[env]["position"], pos_o[env]["position"]
        print(f"  {'pos':>6}: StarWM MSE={sp['mean']:.6f}±{sp['std']:.6f} | "
              f"Orig MSE={op['mean']:.6f}±{op['std']:.6f}")


if __name__ == "__main__":
    main()
