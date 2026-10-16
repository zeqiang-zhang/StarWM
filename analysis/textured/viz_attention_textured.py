#!/usr/bin/env python3
"""Attention mass on the object for the textured ladder (Appendix F.3 (i), text
after Table 13: "attention mass ... 0.82 / 0.57 / 0.37 across L1-L3, 7.0x /
4.9x / 3.2x above uniform").

Renders --n_frames consecutive frames of one random-policy rollout
(env seed 777, policy seed 1), computes StarWM's cross-attention maps
(8x8, bilinearly upsampled to 64x64 and renormalised) and reports, against a
disk of the object's radius around its centre ("object mask"):
  COMBINED   - attention mass of the sum over queries inside the mask;
  BEST-query - per frame, the largest single-query mass inside the mask;
and both divided by the mask's share of the image (= uniform attention).
The paper quotes the BEST-query numbers. Also saves an overlay figure.

Example:
  python analysis/textured/viz_attention_textured.py --logdir runs/texnoisyshape/L3/starwm_s0 \
      --tex_level 3 --device cuda:0 --out figures/attention_textured_L3.png
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import probe_common as pc  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402  (StarWM root, put on sys.path by probe_common)
from shapeenv import TexturedNoisyShapeEnv  # noqa: E402


def objmask(base):
    w = h = 64
    cx, cy = base._pos[0] * w, base._pos[1] * h
    r = base._get_radius_pixels() * 1.0
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def up64(m):
    u = np.array(Image.fromarray(m.astype(np.float32)).resize((64, 64), Image.BILINEAR))
    return u / u.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--checkpoint", default="latest.pt")
    ap.add_argument("--tex_level", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--davis_path", default=pc.default_davis())
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="figure path (default figures/attention_textured_L<L><tag>.png)")
    ap.add_argument("--json", default=None, help="optional json with the metrics")
    a = ap.parse_args()
    pc.require_davis(ap, a.davis_path)

    wm, cfg = pc.load_model(a.logdir, a.checkpoint, a.device, presets=("texnoisyshape",))
    base = TexturedNoisyShapeEnv(size=(64, 64), seed=777, davis_path=a.davis_path,
                                 level=a.tex_level, alpha=0.5)
    env = wrappers.OneHotAction(base)
    rng = np.random.RandomState(1)
    env.reset()
    frames, masks = [], []
    for _ in range(a.n_frames):
        act = np.zeros(4, np.float32); act[rng.randint(4)] = 1.0
        o = env.step(act)[0]
        frames.append(o["image"]); masks.append(objmask(base))
    frames, masks = np.array(frames), np.array(masks)          # (F,64,64,3), (F,64,64)

    A = pc.wm_loader.attention_maps(wm, frames)                 # (F,N,8,8)
    comb = A.sum(1)
    F, N = a.n_frames, A.shape[1]
    onobj = np.array([up64(comb[i])[masks[i]].sum() for i in range(F)])
    area = np.array([masks[i].mean() for i in range(F)])
    perq = np.array([[up64(A[i, q])[masks[i]].sum() for q in range(N)] for i in range(F)])
    best = perq.max(1)
    print(f'L{a.tex_level}{a.tag}: object_area={area.mean():.3f} | COMBINED on-obj {onobj.mean():.3f} '
          f'({onobj.mean()/area.mean():.1f}x) | BEST-query on-obj {best.mean():.3f} '
          f'({best.mean()/area.mean():.1f}x) | per-query means ' + str(np.round(perq.mean(0), 2).tolist()))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(
            logdir=a.logdir, checkpoint=a.checkpoint, tex_level=a.tex_level, n_frames=F,
            object_area=float(area.mean()), combined=float(onobj.mean()),
            best=float(best.mean()), best_x_uniform=float(best.mean() / area.mean())), indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, F, figsize=(1.7 * F, 3.6))
    for i in range(F):
        ax[0, i].imshow(frames[i]); ax[0, i].set_xticks([]); ax[0, i].set_yticks([])
        up = np.array(Image.fromarray(comb[i].astype(np.float32)).resize((64, 64), Image.BILINEAR))
        ax[1, i].imshow(frames[i]); ax[1, i].imshow(up, cmap="jet", alpha=0.5)
        ax[1, i].contour(masks[i], colors="w", linewidths=0.8)
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])
        ax[1, i].set_title(f"{onobj[i]:.2f}", fontsize=7)
    ax[0, 0].set_ylabel("input", fontsize=9); ax[1, 0].set_ylabel("attn overlay", fontsize=9)
    fig.suptitle(f"StarWM cross-attention on textured L{a.tex_level}  (white=object; title=on-object "
                 f"frac; mean {onobj.mean():.2f}, {onobj.mean()/area.mean():.1f}x uniform)", fontsize=9)
    fig.tight_layout()
    out = pathlib.Path(a.out or f"figures/attention_textured_L{a.tex_level}{a.tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
