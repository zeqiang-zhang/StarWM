#!/usr/bin/env python3
"""Delayed action: held-out attribute decodability (Appendix F.5, Table 19, Shape / Size columns).

Probes colour / shape / size of the rendered object from two StarWM features
on the held-out ActionDelay(k) trajectories of collect_delay_trajs.py
(300 episodes x 201 frames):
  [h;z]           - RSSM features (filtered with the issued actions);
  pooled_for_attn - the attention bottleneck (per-frame, no recurrence).
MLP probe, 70/30 stratified split (shapeenv.probe). Table 19 reports [h;z].

Example:
  python analysis/delay/probe_delay_attr.py --logdir runs/delay/k2_shifted_s0 --k 2 \
      --checkpoint checkpoints/agent_step180000.pt --traj runs/traj_cache/delay_k2_traj.npz
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import probe_common as pc  # noqa: E402
from shapeenv.probe import train_clf  # noqa: E402

NC = {"color": 4, "shape": 4, "size": 3}


def pooled_feats(wm, cfg, imgs, acts, isfirst, bs=16):
    """pooled_for_attn (N*T1, N_q*C) of every frame."""
    dev = cfg.device
    N, T1 = imgs.shape[0], imgs.shape[1]
    out = []
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N); B = e - s
            data = {"image": torch.tensor(imgs[s:e], device=dev),
                    "action": torch.zeros(B, T1, acts.shape[-1], device=dev),
                    "is_first": torch.tensor(isfirst[s:e], device=dev, dtype=torch.float32),
                    "is_terminal": torch.zeros(B, T1, device=dev)}
            data = wm.preprocess(data)
            f8, _, _ = wm._encode_with_f8(data)
            _, pa, _ = wm._compute_attention(f8)
            out.append(pa.reshape(B * T1, -1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--k", type=int, required=True, help="action delay of the probe trajectories")
    ap.add_argument("--checkpoint", default="checkpoints/agent_step180000.pt",
                    help="relative to --logdir (Table 19: agent_step180000.pt)")
    ap.add_argument("--traj", default=None, help="default runs/traj_cache/delay_k<k>_traj.npz")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None, help="optional json with the accuracies")
    a = ap.parse_args()

    wm, cfg = pc.load_model(a.logdir, a.checkpoint, a.device, presets=("noisyshape",))
    z = np.load(a.traj or f"runs/traj_cache/delay_k{a.k}_traj.npz")
    imgs, acts, isf = z["imgs"], z["acts"], z["isfirst"]
    lab = {k: z[k].reshape(-1).astype(int) for k in NC}
    hz = pc.encode_h0(wm, cfg, imgs, acts, isf)          # [h;z]  (N*T1, F)
    pf = pooled_feats(wm, cfg, imgs, acts, isf)          # pooled_for_attn (N*T1, D)
    name = pathlib.Path(a.logdir).name
    print(f"=== {name} (k={a.k}) : held-out attribute decodability (paper 70/30 probe) ===")
    res = {}
    for feat, fn in ((hz, "[h;z]"), (pf, "pooled_for_attn")):
        accs = {}
        for at in ("color", "shape", "size"):
            _, te, _ = train_clf(feat.astype(np.float32), lab[at], NC[at], a.device)
            accs[at] = te
        res[fn] = accs
        print(f'  {fn:16s}: color={accs["color"]:.3f} shape={accs["shape"]:.3f} size={accs["size"]:.3f}')
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            dict(logdir=a.logdir, checkpoint=a.checkpoint, k=a.k, **res), indent=2))


if __name__ == "__main__":
    main()
