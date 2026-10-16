#!/usr/bin/env python3
"""Delayed action: attention mass on the object (Appendix F.5, Table 19, Attn. mass / xUniform).

Renders --n_frames consecutive frames of one random-policy NoisyShape rollout
under ActionDelay(k) (env seed 555, policy seed 2), computes StarWM's
cross-attention maps (8x8, bilinearly upsampled to 64x64 and renormalised)
and scores them against a disk of the object's radius around its centre:
  object_area - mean share of the image covered by the disk (uniform baseline);
  COMBINED    - attention mass of the sum over queries inside the disk;
  BEST-query  - per frame, the largest single-query mass inside the disk.
"(Nx)" = mass / object_area. Table 19 reports BEST-query and its xUniform.
Also saves an overlay figure.

Example:
  python analysis/delay/viz_attention_delay.py --logdir runs/delay/k2_shifted_s0 --k 2 \
      --checkpoint checkpoints/agent_step180000.pt --out figures/attention_delay_k2.png
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import probe_common as pc  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402  (StarWM root, put on sys.path by probe_common)
from shapeenv import NoisyShapeEnv  # noqa: E402


def mask(base):
    w = h = 64
    cx, cy = base._pos[0] * w, base._pos[1] * h
    rr = base._get_radius_pixels()
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr


def up64(m):
    u = np.array(Image.fromarray(m.astype(np.float32)).resize((64, 64), Image.BILINEAR))
    return u / u.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--k", type=int, required=True, help="action delay of the rollout")
    ap.add_argument("--checkpoint", default="checkpoints/agent_step180000.pt")
    ap.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR", ""))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--out", default=None, help="figure (default figures/attention_delay_k<k>.png)")
    ap.add_argument("--json", default=None, help="optional json with the metrics")
    a = ap.parse_args()
    pc.require_davis(ap, a.davis_path)

    wm, cfg = pc.load_model(a.logdir, a.checkpoint, a.device, presets=("noisyshape",))
    base = NoisyShapeEnv(size=(64, 64), seed=555, davis_path=a.davis_path)
    env = wrappers.OneHotAction(base)
    if a.k > 0:
        env = wrappers.ActionDelay(env, a.k)
    rng = np.random.RandomState(2)
    env.reset()
    frames, masks = [], []
    for _ in range(a.n_frames):
        act = np.zeros(4, np.float32); act[rng.randint(4)] = 1.0
        o = env.step(act)[0]
        frames.append(o["image"]); masks.append(mask(base))
    frames, masks = np.array(frames), np.array(masks)

    A = pc.wm_loader.attention_maps(wm, frames)                 # (F,N,8,8)
    comb = A.sum(1)
    F, N = a.n_frames, A.shape[1]
    onobj = np.array([up64(comb[i])[masks[i]].sum() for i in range(F)])
    area = np.array([masks[i].mean() for i in range(F)])
    best = np.array([max(up64(A[i, q])[masks[i]].sum() for q in range(N)) for i in range(F)])
    print(f'delay k={a.k}: object_area={area.mean():.3f} | COMBINED on-obj {onobj.mean():.3f} '
          f'({onobj.mean()/area.mean():.1f}x) | BEST-query {best.mean():.3f} ({best.mean()/area.mean():.1f}x)')
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(
            logdir=a.logdir, checkpoint=a.checkpoint, k=a.k, n_frames=F,
            object_area=float(area.mean()), combined=float(onobj.mean()),
            best=float(best.mean()), best_x_uniform=float(best.mean() / area.mean())), indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, F, figsize=(1.6 * F, 3.4))
    for i in range(F):
        ax[0, i].imshow(frames[i]); ax[0, i].axis("off")
        up = np.array(Image.fromarray(comb[i].astype(np.float32)).resize((64, 64), Image.BILINEAR))
        ax[1, i].imshow(frames[i]); ax[1, i].imshow(up, cmap="jet", alpha=0.5)
        ax[1, i].contour(masks[i], colors="w", linewidths=0.8); ax[1, i].axis("off")
        ax[1, i].set_title(f"{best[i]:.2f}", fontsize=7)
    fig.suptitle(f"StarWM attention · NoisyShape delay k={a.k} (best-query on-obj {best.mean():.2f}, "
                 f"{best.mean()/area.mean():.1f}x uniform)", fontsize=9)
    fig.tight_layout()
    out = pathlib.Path(a.out or f"figures/attention_delay_k{a.k}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
