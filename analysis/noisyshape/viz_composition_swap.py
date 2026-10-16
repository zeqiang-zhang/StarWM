"""Fig. 4: cross-sample composition with a StarWM model trained on NoisyShape.

For pairs of frames (X = "A", Y = "B") every frame is encoded independently
(T = 1, is_first) and the three components of the dual-stream decoder are
recombined across the pair before decoding:
  bg  F^bg      = MLP_bg(z_res)                       (residual stream)
  fg  F^fg      = A^T e,  e = MLP_fg(RSSM feat)       (dynamics stream, N entity tokens)
  mask          = clamp(sum_q A_q, 0, 1)              (attention)
  decode(fg * mask + bg * (1 - mask))
Rows of the output grid (one column per pair):
  A input | A recon | A fg mask | A bg only | B input | B fg only |
  A bg + B fg + A mask | A bg + B fg + B mask | A bg + (A_A^T e_B) + A mask
The yellow outlines mark where B's shape is expected (at B's position in
row 8, at A's position in row 9). Panels (a)-(c) of the paper figure are
rows 1/5, 4/6 and 9/8 of this grid. z_res is sampled (torch seed ``--seed``).

Example:
    python analysis/noisyshape/viz_composition_swap.py runs/noisyshape/starwm_s0 \\
        --davis_path $DAVIS_DIR --out fig4_composition_swap.png
"""
import argparse
import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

import _probe_common as pc

SIZE_FRACS = [0.10, 0.18, 0.26]  # radius / image size of the three NoisyShape sizes


def load_model(run_dir, checkpoint, device):
    """wm_loader with the reward-head fix of the paper script: checkpoints whose
    reward head has a single output were trained with an 'mse' reward head."""
    config = pc.wm_loader.load_run_config(run_dir, device=device)
    sd = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu",
                    weights_only=False)["agent_state_dict"]
    w = next((v for k, v in sd.items() if k.endswith("heads.reward.mean_layer.weight")), None)
    if w is not None and w.shape[0] == 1:
        config.reward_head = {**dict(config.reward_head), "dist": "mse"}
    del sd
    wm, config = pc.wm_loader.load_world_model(run_dir, config=config, checkpoint=checkpoint,
                                               device=device)
    if config.model != "starwm":
        raise SystemExit(f"{run_dir} is a {config.model} run; the swap needs the StarWM decoder")
    return wm


@torch.no_grad()
def forward_once(wm, imgs_np, device):
    """Encode N independent frames as T=1 sequences; return the decoder components."""
    N = imgs_np.shape[0]
    B, T = N, 1
    batch = {
        "image": imgs_np[:, None],
        "action": np.zeros((B, T, wm._config.num_actions), np.float32),
        "is_first": np.ones((B, T), np.float32),
        "is_terminal": np.zeros((B, T), np.float32),
    }
    data = wm.preprocess(batch)
    f8_proj, raw_embed, _ = wm._encode_with_f8(data)
    pooled_for_rssm, _, attn_map = wm._compute_attention(f8_proj)  # attn: (BT, N_q, L)
    embed_dyn = wm.proj_dyn(pooled_for_rssm.reshape(B, T, -1))
    post, _ = wm.dynamics.observe(embed_dyn, data["action"], data["is_first"])
    rssm_feat = wm.dynamics.get_feat(post)
    embed_res, _ = wm._compute_residual(f8_proj, raw_embed, attn_map)

    S, Nq, BT = wm.assembly_spatial, wm.num_queries, B * T
    L = S * S
    A = attn_map.reshape(BT, Nq, L)
    fg_tokens = wm.mlp_fg(rssm_feat.reshape(BT, -1)).reshape(BT, Nq, wm.assembly_channels)
    fg_fmap = torch.matmul(A.transpose(-1, -2), fg_tokens).permute(0, 2, 1).reshape(
        BT, wm.assembly_channels, S, S)
    bg_fmap = wm.mlp_bg(embed_res.reshape(BT, -1)).reshape(BT, wm.assembly_channels, S, S)
    fg_mask = torch.clamp(A.sum(dim=1), 0.0, 1.0).reshape(BT, 1, S, S)
    return {
        "fg_tokens": fg_tokens, "fg_fmap": fg_fmap, "bg_fmap": bg_fmap, "fg_mask": fg_mask,
        "attn": A,
        "recon": decode_mix(wm, fg_fmap, bg_fmap, fg_mask),
        "bg_only": decode_mix(wm, fg_fmap, bg_fmap, torch.zeros_like(fg_mask)),
    }


@torch.no_grad()
def decode_mix(wm, fg_fmap, bg_fmap, fg_mask):
    out = wm.conv_decoder(fg_fmap * fg_mask + bg_fmap * (1.0 - fg_mask))
    out = out.reshape(-1, *out.shape[-3:])
    return torch.clamp(out.permute(0, 2, 3, 1), 0, 1)


def sample_frames(n_pairs, seed, davis_path):
    """2 * n_pairs frames, each after 2..7 random steps of a fresh episode.

    The paper script tried to seed ``env.action_space`` before sampling, which
    has no effect (the property returns a new Space each time), so its actions
    were unseeded; here they come from ``RandomState(seed + episode)``.
    """
    import shapeenv
    env = shapeenv.make("noisyshape", seed=seed, davis_path=davis_path)
    frames, attrs = [], []
    for ep in range(n_pairs * 2):
        env.reset()
        rng = np.random.RandomState(seed + ep)
        for _ in range(rng.randint(2, 8)):
            obs, _, _, _, _ = env.step(rng.randint(0, env.action_space.n))
        frames.append(obs["image"])
        attrs.append({"shape_id": env._shape_id, "size_id": env._size_id, "pos": env._pos})
    return np.array(frames), attrs


def draw_gt_outline(ax, shape_id, size_id, pos_x, pos_y, img_size=64, color="#FFD700", lw=2):
    """Yellow outline of a ground-truth shape (same geometry as the environment)."""
    r = SIZE_FRACS[size_id] * img_size
    cx, cy = pos_x * img_size, pos_y * img_size
    if shape_id == 0:  # circle
        patch = Circle((cx, cy), r, fill=False, edgecolor=color, linewidth=lw)
    elif shape_id == 1:  # triangle
        pts = [(cx + r * math.cos(math.pi / 2 + 2 * math.pi * i / 3),
                cy - r * math.sin(math.pi / 2 + 2 * math.pi * i / 3)) for i in range(3)]
        patch = Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=lw)
    elif shape_id == 2:  # square
        half = r * math.sqrt(2) / 2
        patch = Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                          edgecolor=color, linewidth=lw)
    elif shape_id == 3:  # pentagon
        pts = [(cx + r * math.cos(math.pi / 2 + 2 * math.pi * i / 5),
                cy - r * math.sin(math.pi / 2 + 2 * math.pi * i / 5)) for i in range(5)]
        patch = Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=lw)
    else:
        return
    ax.add_patch(patch)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="StarWM run directory trained on noisyshape")
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"))
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7, help="env / frame-sampling seed")
    p.add_argument("--torch_seed", type=int, default=0, help="seed for sampling z_res")
    p.add_argument("--title", default="StarWM decomposition / swap")
    p.add_argument("--device", default=pc.default_device())
    p.add_argument("--out", default="fig4_composition_swap.png")
    args = p.parse_args()

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    wm = load_model(args.run_dir, args.checkpoint, args.device)
    imgs, attrs = sample_frames(args.n_pairs, args.seed, args.davis_path)
    print(f"Sampled frames: {imgs.shape}")
    feats = forward_once(wm, imgs, args.device)

    P = args.n_pairs
    idx_A, idx_B = np.arange(0, 2 * P, 2), np.arange(1, 2 * P, 2)
    fg_B, bg_A = feats["fg_fmap"][idx_B], feats["bg_fmap"][idx_A]
    mask_A, mask_B = feats["fg_mask"][idx_A], feats["fg_mask"][idx_B]
    # A's background + B's dynamics content (at B's attention locations) + A's / B's mask
    mix_A_mask = decode_mix(wm, fg_B, bg_A, mask_A).cpu().numpy()
    mix_B_mask = decode_mix(wm, fg_B, bg_A, mask_B).cpu().numpy()
    # B's entity tokens re-projected with A's attention: A_A^T e_B
    S, C = wm.assembly_spatial, wm.assembly_channels
    fg_reproj = torch.matmul(feats["attn"][idx_A].transpose(-1, -2), feats["fg_tokens"][idx_B])
    fg_reproj = fg_reproj.permute(0, 2, 1).reshape(-1, C, S, S)
    mix_reproj = decode_mix(wm, fg_reproj, bg_A, mask_A).cpu().numpy()
    recon_A = feats["recon"][idx_A].cpu().numpy()
    bg_only_A = feats["bg_only"][idx_A].cpu().numpy()
    fg_only_B = decode_mix(wm, fg_B, torch.zeros_like(feats["bg_fmap"][idx_B]),
                           torch.ones_like(mask_B)).cpu().numpy()
    A_imgs, B_imgs = imgs[idx_A] / 255.0, imgs[idx_B] / 255.0
    mask_A_np = mask_A.squeeze(1).cpu().numpy()

    row_labels = ["A (input)", "A recon", "A fg mask", r"A bg only ($F^{BG}_A$)", "B (input)",
                  r"B fg only ($F^{FG}_B$)", "A bg + B fg + A mask", "A bg + B fg + B mask",
                  r"A bg + (A attn$^\top$ · B tokens) + A mask"]
    n_rows = len(row_labels)
    fig, axes = plt.subplots(n_rows, P, figsize=(2.2 * P, 2.2 * n_rows), squeeze=False)
    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    H = imgs.shape[1]
    for i in range(P):
        axes[0, i].imshow(A_imgs[i])
        axes[1, i].imshow(recon_A[i])
        axes[2, i].imshow(A_imgs[i])
        axes[2, i].imshow(np.kron(mask_A_np[i], np.ones((H // S, H // S))), cmap="hot",
                          alpha=0.6, vmin=0, vmax=1)
        axes[3, i].imshow(bg_only_A[i])
        axes[4, i].imshow(B_imgs[i])
        axes[5, i].imshow(fg_only_B[i])
        axes[6, i].imshow(mix_A_mask[i])
        axes[7, i].imshow(mix_B_mask[i])
        axes[8, i].imshow(mix_reproj[i])
        a_A, a_B = attrs[idx_A[i]], attrs[idx_B[i]]
        draw_gt_outline(axes[7, i], a_B["shape_id"], a_B["size_id"], *a_B["pos"], img_size=H)
        draw_gt_outline(axes[8, i], a_B["shape_id"], a_B["size_id"], *a_A["pos"], img_size=H)
        for r in range(n_rows):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    for r, lab in enumerate(row_labels):
        axes[r, 0].set_ylabel(lab, fontsize=10, fontweight="bold", rotation=0, ha="right",
                              va="center", labelpad=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
