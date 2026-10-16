#!/usr/bin/env python3
"""Cross-attention maps on DMC: Figs 6, 11, 14 and Table 8.

For each requested run the script picks the first seed (lowest id) that has a
checkpoint and stored episodes, runs the encoder + cross-attention of the world
model over the frames of the most recent episode file, prints per-query
attention statistics and saves a grid of 8 evenly spaced frames with the
attention of every query overlaid (8x8 map, bilinearly upsampled):

* entropy  = mean over frames of -sum_l A_l log(A_l + 1e-8)  (uniform: log 64 = 4.16)
* avg max  = mean over frames of max_l A_l                   (uniform: 1/64)

``--figure cheetah``    Fig 6:  StarWM, Cheetah Run, Clean / VH / VHR
                        (rows = queries, columns = frames)
``--figure cartpole``   Fig 11: StarWM vs StarWM-R, Cartpole Swingup VH
``--figure isolation``  Fig 14 + Table 8: ★ StarWM, ✩3 (only detach B, no SS),
                        ✩1 (no detach, with SS) on Clean / VHR x Cheetah Run /
                        Walker Walk; Table 8 is query Q0 on Cheetah Run.
(Figs 11/14: rows = frames, columns = observation + queries.) The paper's
figures are composites of these grids.

Episodes: ``--episodes auto`` (default) uses ``eval_eps`` and falls back to
``train_eps`` when a run has no stored evaluation episodes.

    python analysis/dmc/viz_attention.py --figure isolation --device cuda
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # analysis/ (wm_loader)

import numpy as np
import scipy.ndimage
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layouts import add_layout_args, layout_from_args

FIGURES = {
    # name: list of (title, setting, task, variant, output stem)
    "cheetah": [("Clean — Cheetah Run", "clean", "cheetah_run", "starwm", "attn_clean_cheetah_run"),
                ("VH — Cheetah Run", "vh", "cheetah_run", "starwm", "attn_vh_cheetah_run"),
                ("VHR — Cheetah Run", "vhr", "cheetah_run", "starwm", "attn_vhr_cheetah_run")],
    "cartpole": [("StarWM — VH — Cartpole Swingup", "vh", "cartpole_swingup", "starwm", "starwm"),
                 ("StarWM-R — VH — Cartpole Swingup", "vh", "cartpole_swingup", "starwm_r",
                  "starwm-r")],
    "isolation": [(f"{label} — {mode.upper()} — {task}", mode, task, variant,
                   f"{variant}_{mode}_{task}")
                  for label, variant in (("Star (SS)", "starwm"),
                                         ("Triangle3 (RSSM)", "iso_star3"),
                                         ("Triangle1 (SS+RSSM+Recon)", "iso_star1"))
                  for mode in ("clean", "vhr") for task in ("cheetah_run", "walker_walk")],
}
TABLE8_ROWS = [("★  StarWM (full)", "starwm"), ("✩1 no detach A/B, with SS", "iso_star1"),
               ("✩3 only detach B, no SS", "iso_star3")]


# ─── model ───────────────────────────────────────────────────────────────────
def load_wm(run_dir, checkpoint, device):
    import wm_loader
    config = wm_loader.load_run_config(run_dir, device=device)
    # logged configs store tuples as lists; the DMC wrapper needs a tuple image size
    if isinstance(getattr(config, "size", None), list):
        config.size = tuple(config.size)
    wm, config = wm_loader.load_world_model(run_dir, config=config, checkpoint=checkpoint,
                                            device=device)
    return wm


def run_attention(wm, images, batch=64):
    """images (T,64,64,3) uint8 -> attention (T, N, 64)."""
    import wm_loader
    out = [wm_loader.attention_maps(wm, images[i:i + batch]) for i in range(0, len(images), batch)]
    A = np.concatenate(out, 0)
    return A.reshape(A.shape[0], A.shape[1], -1)


def attention_stats(attn):
    """[(entropy, avg max)] per query."""
    out = []
    for q in range(attn.shape[1]):
        ent = -(attn[:, q] * np.log(attn[:, q] + 1e-8)).sum(-1).mean()
        mx = attn[:, q].reshape(-1, 64).max(1).mean()
        out.append((float(ent), float(mx)))
    return out


def pick_run(runs, episodes, seed=None):
    """First (seed, dir, episode dir name) with latest.pt and stored episodes."""
    order = {"auto": ("eval_eps", "train_eps"), "eval": ("eval_eps",), "train": ("train_eps",)}
    for s, d in runs:
        if seed is not None and s != seed:
            continue
        d = pathlib.Path(d)
        if not (d / "latest.pt").exists():
            continue
        for eps_name in order[episodes]:
            eps_dir = d / eps_name
            if eps_dir.is_dir() and list(eps_dir.glob("*.npz")):
                return s, d, eps_name
    return None, None, None


# ─── plotting ────────────────────────────────────────────────────────────────
def _frame(obs, t):
    img = obs[t].astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    return np.clip(img, 0, 1)


def plot_attention_grid(obs, attn, title, save_path, num_frames=8, num_queries=4):
    """Rows = observation + one row per query; columns = frames."""
    T = obs.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    fig, axes = plt.subplots(num_queries + 1, num_frames,
                             figsize=(2.5 * num_frames, 2.5 * (num_queries + 1)))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for col, t in enumerate(indices):
        img = _frame(obs, t)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f't={t}', fontsize=9)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('Obs', fontsize=10)
        for q in range(num_queries):
            attn_2d = attn[t, q].reshape(8, 8)
            attn_up = scipy.ndimage.zoom(attn_2d, 8, order=1)
            axes[q + 1, col].imshow(img, alpha=0.4)
            axes[q + 1, col].imshow(attn_up, cmap='hot', alpha=0.7,
                                    vmin=0, vmax=max(attn_2d.max(), 1e-6))
            axes[q + 1, col].axis('off')
            if col == 0:
                avg_max = attn[:, q].reshape(-1, 64).max(1).mean()
                axes[q + 1, col].set_ylabel(f'Q{q}\nmax={avg_max:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attention_grid_h(obs, attn, title, save_path, num_frames=8, num_queries=4):
    """Rows = frames; columns = observation + one column per query."""
    T = obs.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    ncols = 1 + num_queries
    fig, axes = plt.subplots(num_frames, ncols, figsize=(2.2 * ncols, 2.2 * num_frames))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    avg_max_per_q = [attn[:, q].reshape(-1, 64).max(1).mean() for q in range(num_queries)]
    for row, t in enumerate(indices):
        img = _frame(obs, t)
        axes[row, 0].imshow(img)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f't={t}', fontsize=10, fontweight='bold')
        if row == 0:
            axes[row, 0].set_title('Obs', fontsize=11, fontweight='bold')
        for q in range(num_queries):
            attn_2d = attn[t, q].reshape(8, 8)
            attn_up = scipy.ndimage.zoom(attn_2d, 8, order=1)
            ax = axes[row, q + 1]
            ax.imshow(img, alpha=0.4)
            ax.imshow(attn_up, cmap='hot', alpha=0.7, vmin=0, vmax=max(attn_2d.max(), 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'Q{q}\n(avg max={avg_max_per_q[q]:.2f})',
                             fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--figure", choices=sorted(FIGURES), required=True)
    ap.add_argument("--episodes", choices=["auto", "eval", "train"], default="auto",
                    help="episode store to visualise (auto: eval_eps, else train_eps)")
    ap.add_argument("--seed", type=int, default=None,
                    help="use this seed instead of the first one with data")
    ap.add_argument("--checkpoint", default="latest.pt", help="relative to the run directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default=os.path.join("figures", "attention"))
    args = ap.parse_args()
    layout = layout_from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {args.device}")

    stats = {}
    for title, setting, task, variant, stem in FIGURES[args.figure]:
        seed, run_dir, eps_name = pick_run(layout.dmc(setting, task, variant), args.episodes,
                                           args.seed)
        if run_dir is None:
            print(f"SKIP {variant}/{setting}/{task}: no checkpoint + episodes found")
            continue
        full_title = f"{title} (seed={seed}, eps={eps_name})"
        print(f"\n{'=' * 68}\n  {full_title}\n  {run_dir}\n{'=' * 68}")
        wm = load_wm(run_dir, args.checkpoint, args.device)
        eps_files = sorted((run_dir / eps_name).glob("*.npz"))
        obs = np.load(str(eps_files[-1]))["image"]
        print(f"  Episode: {obs.shape[0]} steps, shape={obs.shape}  ({eps_files[-1].name})")
        attn = run_attention(wm, obs)
        st = attention_stats(attn)
        for q, (ent, mx) in enumerate(st):
            print(f"    Q{q}: entropy={ent:.3f}, avg_max={mx:.3f}")
        stats[(variant, setting, task)] = st
        nq = attn.shape[1]
        if args.figure == "cheetah":
            plot_attention_grid(obs, attn, full_title,
                                os.path.join(args.out_dir, f"{stem}.png"), 8, nq)
        else:
            plot_attention_grid_h(obs, attn, full_title,
                                  os.path.join(args.out_dir, f"{stem}_seed{seed}.png"
                                               if args.figure == "cartpole" else f"{stem}.png"),
                                  8, nq)
        del wm

    if args.figure == "isolation":
        print("\nTable 8: attention statistics of query Q0 on Cheetah Run")
        print(f"{'':28s}{'Clean entropy':>15s}{'Clean max':>11s}{'VHR entropy':>13s}{'VHR max':>9s}")
        for label, variant in TABLE8_ROWS:
            cells = []
            for setting in ("clean", "vhr"):
                st = stats.get((variant, setting, "cheetah_run"))
                cells += ["--", "--"] if st is None else [f"{st[0][0]:.2f}", f"{st[0][1]:.2f}"]
            print(f"{label:28s}{cells[0]:>15s}{cells[1]:>11s}{cells[2]:>13s}{cells[3]:>9s}")
    print("\nDone.")


if __name__ == "__main__":
    main()
