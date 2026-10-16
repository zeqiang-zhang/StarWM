#!/usr/bin/env python3
"""CALVIN attention figure, Fig. 26 (Appendix F.3 (iii)).

Top row: --n_frames held-out frames of one 500-frame validation episode
(t = 20, 20+stride, ...). Rows below: the attention map of each of the N=4
queries (8x8, bilinearly zoomed to 64x64, 'hot' colormap, per-panel
normalisation) over the dimmed frame. Defaults reproduce the paper figure
(6 frames, stride 80, episode index 3).

Example:
  python calvin/viz_attention_calvin_paperstyle.py \
      --ckpt runs/calvin/starwm/checkpoints/wm_step180000.pt --data_dir <conv>/D_val \
      --device cuda:0 --out figures/calvin_attention.png
"""
import argparse
import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from calvin_attn_metric_proj import load_wm  # noqa: E402
from train_calvin_offline import build_config  # noqa: E402


@torch.no_grad()
def get_attn(wm, cfg, imgs):
    """imgs (T,64,64,3) uint8 -> raw attention (T,N,S,S)."""
    dev = cfg.device
    T = imgs.shape[0]
    data = {"image": torch.tensor(imgs[None], device=dev), "action": torch.zeros(1, T, 7, device=dev),
            "is_first": torch.zeros(1, T, device=dev), "is_terminal": torch.zeros(1, T, device=dev)}
    data = wm.preprocess(data)
    f8, _, _ = wm._encode_with_f8(data)
    _, _, attn = wm._compute_attention(f8)                             # (T, N, L)
    return attn.reshape(T, wm.num_queries, wm.f8_spatial, wm.f8_spatial).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True, help="converted held-out split (calvin_convert.py)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--ep", type=int, default=3, help="index into the sorted 500-frame episodes")
    ap.add_argument("--stride", type=int, default=80)
    ap.add_argument("--out", default="figures/calvin_attention.png")
    ap.add_argument("--title", default="CALVIN (held-out val) — StarWM cross-attention")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.ndimage

    cfg = build_config("starwm", {"device": a.device, "seed": 0})
    wm = load_wm(a.ckpt, cfg)
    files = sorted(f for f in glob.glob(os.path.join(a.data_dir, "*.npz")) if f.endswith("-500.npz"))
    imgs_all = np.load(files[a.ep % len(files)])["image"]            # (500,64,64,3)
    idx = np.arange(a.n_frames) * a.stride + 20
    idx = idx[idx < imgs_all.shape[0]]
    imgs = imgs_all[idx]
    attn = get_attn(wm, cfg, imgs)                                   # (n,N,S,S)
    n, N = imgs.shape[0], attn.shape[1]

    # one row of observations, then one row per query; 'hot' over a dimmed frame
    nrow, ncol = N + 1, n
    TEXTWIDTH_IN = 5.5   # natural width == paper \textwidth
    cell = TEXTWIDTH_IN / ncol
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(TEXTWIDTH_IN, cell * nrow + 0.30))
    fig.suptitle(a.title, fontsize=11, fontweight="bold")
    zoom = imgs.shape[1] // attn.shape[-1]                           # 8x8 -> 64x64
    for col in range(n):
        img = imgs[col].astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        axes[0][col].imshow(img)
        axes[0][col].set_title(f"t={idx[col]}", fontsize=10)
        axes[0][col].axis("off")
        for q in range(N):
            attn_2d = attn[col, q]
            attn_up = scipy.ndimage.zoom(attn_2d, zoom, order=1)
            axes[q + 1][col].imshow(img, alpha=0.4)
            axes[q + 1][col].imshow(attn_up, cmap="hot", alpha=0.7, vmin=0, vmax=max(attn_2d.max(), 1e-6))
            axes[q + 1][col].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print("[viz] saved", a.out, flush=True)


if __name__ == "__main__":
    main()
