"""Fig. 22: StarWM cross-attention maps on NoisyTwoShape.

Collects 2 random-policy episodes x 20 steps (env seed 42, policy seed 0),
picks ``--num_frames`` evenly spaced frames and plots, per frame, the
observation (ground-truth centres: circle = large object s1, square = small
object s2) followed by the per-query attention maps (8x8, bilinearly
upsampled, overlaid on the frame). Per-query attention entropy and mean peak
are printed. The paper figure shows three of these rows.

Example:
    python analysis/noisyshape/viz_two_shape_attention.py runs/noisytwoshape/starwm_s0 \\
        --davis_path $DAVIS_DIR --out fig22_two_shape_attention.png
"""
import argparse
import os

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _probe_common as pc


def collect_frames(env, num_episodes=2, steps_per_ep=20, seed=0):
    rng = np.random.RandomState(seed)
    frames, infos = [], []
    for _ in range(num_episodes):
        env.reset()
        for _ in range(steps_per_ep):
            obs, _r, _d, _t, info = env.step(rng.randint(0, env.action_space.n))
            frames.append(obs["image"])
            infos.append(info)
    return np.array(frames), infos


def plot_attention_grid(obs, attn, info_list, title, save_path, num_frames=8):
    """Rows = frames, cols = [obs, Q0, ..., Q_{N-1}]. attn: (T, N, 8, 8)."""
    T, nq = attn.shape[:2]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    n_cols = nq + 1
    fig, axes = plt.subplots(num_frames, n_cols, figsize=(2.5 * n_cols, 2.5 * num_frames), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    headers = ["Obs (○ s1, ▢ s2)"] + [
        f"Q{q}\nmax={attn[:, q].reshape(T, -1).max(1).mean():.3f}" for q in range(nq)]
    for row, t in enumerate(indices):
        img = obs[t].astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        S = img.shape[0]
        axes[row, 0].imshow(img)
        if info_list is not None:
            inf = info_list[t]
            axes[row, 0].plot(inf["s1_position_x"] * S, inf["s1_position_y"] * S, marker="o",
                              mfc="none", mec="red", mew=1.5, ms=14)
            axes[row, 0].plot(inf["s2_position_x"] * S, inf["s2_position_y"] * S, marker="s",
                              mfc="none", mec="cyan", mew=1.5, ms=10)
        axes[row, 0].axis("off")
        axes[row, 0].text(-0.12, 0.5, f"t={t}", fontsize=10, transform=axes[row, 0].transAxes,
                          ha="right", va="center")
        for q in range(nq):
            a = attn[t, q]
            a_up = scipy.ndimage.zoom(a, S // a.shape[0], order=1)
            axes[row, q + 1].imshow(img, alpha=0.4)
            axes[row, q + 1].imshow(a_up, cmap="hot", alpha=0.7, vmin=0, vmax=max(a.max(), 1e-6))
            axes[row, q + 1].axis("off")
        if row == 0:
            for c, h in enumerate(headers):
                axes[row, c].set_title(h, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="StarWM run directory trained on noisytwoshape")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"))
    p.add_argument("--env_seed", type=int, default=42)
    p.add_argument("--policy_seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=2)
    p.add_argument("--steps_per_episode", type=int, default=20)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--no_title", action="store_true")
    p.add_argument("--device", default=pc.default_device())
    p.add_argument("--out", default="fig22_two_shape_attention.png")
    args = p.parse_args()

    import shapeenv
    wm, config = pc.load_model(args.run_dir, args.checkpoint, args.device)
    if wm is None:
        raise SystemExit(1)
    if config.model not in ("starwm", "attention_only"):
        raise SystemExit(f"{args.run_dir} is a {config.model} run: no attention maps")
    env_name = config.task.split("_", 1)[0]
    kw = {"davis_path": args.davis_path} if env_name in shapeenv.USES_DAVIS else {}
    env = shapeenv.make(env_name, seed=args.env_seed, **kw)
    frames, infos = collect_frames(env, args.num_episodes, args.steps_per_episode, args.policy_seed)
    print(f"  Frames: {frames.shape}")
    attn = pc.wm_loader.attention_maps(wm, frames)  # (T, N, 8, 8)

    print("Attention stats:")
    flat = attn.reshape(attn.shape[0], attn.shape[1], -1)
    for q in range(attn.shape[1]):
        ent = -(flat[:, q] * np.log(flat[:, q] + 1e-8)).sum(-1).mean()
        print(f"    Q{q}: entropy={ent:.3f}, avg_max={flat[:, q].max(1).mean():.3f}")
    title = None if args.no_title else (
        f"StarWM {env_name} cross-attention   ○ = s1 (large, red), ▢ = s2 (small, cyan)")
    plot_attention_grid(frames, attn, infos, title, args.out, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
