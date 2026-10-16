"""Fig. 21: latent probes on NoisyTwoShape (StarWM vs DreamerV3).

Per environment row: color / shape of the large (s1) and small (s2) object
from H0 to H15 (mean ± std over seeds; here H0 is the current probe on the
H0 probe set), then the H0 position MSE (x + y) of each object as bars with
its chance level (0.0384 large, 0.150 small; Monte-Carlo estimates over each
object's size-dependent position range). Input: result files of
``probe_two_shape.py``.

Example:
    python analysis/noisyshape/plot_two_shape.py --runs runs --out fig21_two_shape.pdf
Paper runs (the figure used the probes of the final 400k checkpoints):
    python analysis/noisyshape/plot_two_shape.py --runs logdir_paper --name_map paper \\
        --results_subdir probe_results_current
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import _common as C

CLS_KEYS = ["s1_color", "s2_color", "s1_shape", "s2_shape"]
CLS_TITLES = {"s1_color": "Large color", "s2_color": "Small color",
              "s1_shape": "Large shape", "s2_shape": "Small shape"}
CHANCE_POS = {"s1": 0.0384, "s2": 0.1500}
ROW_LABELS = {"noisytwoshape": "NoisyTwoShape"}
C_S, C_O = "#E53935", "#1565C0"


def extract_curves(results_list, horizon=15):
    """H0 = current_probes[k].test_acc, H1..H15 = imagined_probes[h][k]; mean / std over seeds."""
    out = {}
    for k in CLS_KEYS:
        arr = np.array([[r["current_probes"][k]["test_acc"]]
                        + [r["imagined_probes"][str(h)][k] for h in range(1, horizon + 1)]
                        for r in results_list])
        out[k] = {"mean": arr.mean(0), "std": arr.std(0)}
    return out


def extract_pos_mse(results_list):
    """H0 position MSE summed over (x, y) per object, mean / std over seeds."""
    out = {}
    for s in ["s1", "s2"]:
        v = [r["current_probes"][f"{s}_pos_x"]["test_mse"] + r["current_probes"][f"{s}_pos_y"]["test_mse"]
             for r in results_list]
        out[s] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return out


def plot_combined(data, envs, horizon, save_path):
    top_inches, bot_inches = 1.05, 0.85  # legend above, centred x-label below
    nrows = len(envs)
    fig_h = 2.5 * nrows + top_inches + bot_inches
    fig = plt.figure(figsize=(15.0, fig_h))
    outer = fig.add_gridspec(nrows, 2, width_ratios=[4 * 1.2, 2 * 0.6 + 0.2], wspace=0.20, hspace=0.20,
                             top=1.0 - top_inches / fig_h, bottom=bot_inches / fig_h, left=0.05, right=0.99)
    handles = [Line2D([0], [0], color=C_S, marker="o", linewidth=1.7, markersize=4, label="StarWM"),
               Line2D([0], [0], color=C_O, marker="s", linewidth=1.7, markersize=4, label="Orig Dreamer"),
               Line2D([0], [0], color="gray", linestyle="--", label="chance")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.30 / fig_h),
               ncol=3, fontsize=11, frameon=False)

    H = np.arange(horizon + 1)
    bottom_axes = None
    for r, env in enumerate(envs):
        sc, oc, sp, op = data[env]["sc"], data[env]["oc"], data[env]["sp"], data[env]["op"]
        gs_lines = outer[r, 0].subgridspec(1, 4, wspace=0.06)
        line_axes = [fig.add_subplot(gs_lines[0, i]) for i in range(4)]
        if r == nrows - 1:
            bottom_axes = line_axes
        for i, key in enumerate(CLS_KEYS):
            ax = line_axes[i]
            for c, color, marker in [(sc, C_S, "-o"), (oc, C_O, "-s")]:
                m, s = c[key]["mean"], c[key]["std"]
                ax.plot(H, m, marker, color=color, markersize=3, linewidth=1.7)
                ax.fill_between(H, m - s, m + s, color=color, alpha=0.2)
            ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim([0.2, 1.05])
            ax.grid(True, alpha=0.25)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            if r == 0:
                ax.set_title(CLS_TITLES[key], fontsize=11, fontweight="bold")
            if i == 0:
                ax.set_ylabel(f"{ROW_LABELS.get(env, env)}\nAccuracy", fontsize=10, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)

        gs_bars = outer[r, 1].subgridspec(1, 2, wspace=0.40)
        for j, (label, key) in enumerate([("Large pos (H0)", "s1"), ("Small pos (H0)", "s2")]):
            ax = fig.add_subplot(gs_bars[0, j])
            width = 0.32
            s_val, s_err = sp[key]["mean"], sp[key]["std"]
            o_val, o_err = op[key]["mean"], op[key]["std"]
            ax.bar(-width / 2, [s_val], width, yerr=[s_err], color=C_S, alpha=0.85, capsize=4,
                   error_kw={"linewidth": 1.4})
            ax.bar(width / 2, [o_val], width, yerr=[o_err], color=C_O, alpha=0.85, capsize=4,
                   error_kw={"linewidth": 1.4})
            ax.axhline(y=CHANCE_POS[key], color="gray", linestyle="--", alpha=0.6)
            ax.set_xticks([])
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(0, max(s_val + s_err, o_val + o_err, CHANCE_POS[key]) * 1.30)
            ax.grid(True, alpha=0.25, axis="y")
            if r == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(r"MSE  $\uparrow$", fontsize=11, fontweight="bold")

    if bottom_axes is not None:  # one centred x-label under the line plots
        fig.canvas.draw()
        x_c = 0.5 * (bottom_axes[0].get_position().x0 + bottom_axes[-1].get_position().x1)
        y = bottom_axes[0].get_position().y0 - 0.55 / fig_h
        fig.text(x_c, y, "Imagination Horizon", ha="center", va="center", fontsize=10, fontweight="bold")
    C.ensure_parent(save_path)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(p)
    p.add_argument("--envs", nargs="+", default=["noisytwoshape"])
    p.add_argument("--out", default="fig21_two_shape.pdf")
    args = p.parse_args()

    data = {}
    for env in args.envs:
        sr, orr = C.load_results(args, env, "starwm"), C.load_results(args, env, "dreamerv3")
        print(f"{env}: StarWM seeds={len(sr)}, DreamerV3 seeds={len(orr)}")
        if not sr or not orr:
            raise SystemExit(f"missing results for {env}")
        data[env] = {"sc": extract_curves(sr, args.horizon), "oc": extract_curves(orr, args.horizon),
                     "sp": extract_pos_mse(sr), "op": extract_pos_mse(orr)}
        for name, c, pos in [("StarWM", data[env]["sc"], data[env]["sp"]),
                             ("DreamerV3", data[env]["oc"], data[env]["op"])]:
            print(f"  {name:<10} " + "  ".join(
                f"{k} H0={c[k]['mean'][0]:.4f} H{args.horizon}={c[k]['mean'][-1]:.4f}±{c[k]['std'][-1]:.4f}"
                for k in CLS_KEYS)
                + f"  pos s1={pos['s1']['mean']:.4f}±{pos['s1']['std']:.4f}"
                  f" s2={pos['s2']['mean']:.4f}±{pos['s2']['std']:.4f}")
    plot_combined(data, args.envs, args.horizon, args.out)


if __name__ == "__main__":
    main()
