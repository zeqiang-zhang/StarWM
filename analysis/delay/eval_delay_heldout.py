#!/usr/bin/env python3
"""Delayed action: inverse-head accuracy on replay vs fresh held-out rollouts (Appendix F.5).

Supports the statement in Appendix F.5 that the training-time inverse-model
accuracy is not a valid metric: with a per-frame random object position each
replay frame is a unique fingerprint, so the inverse head can memorise
frame -> action on the replay buffer. The script measures the inverse head of
a checkpoint with the window it was trained with (predict a_j from
(e_{j+s}, e_{j+s+1}), s = inv_delay; s = 0 is the adjacent window) on
  * TRAIN frames:    windows sampled from the last 200 files of <logdir>/train_eps;
  * HELD-OUT frames: fresh ActionDelay(k) rollouts with an unseen seed.
Chance is 0.25. Not a reported table; used to justify Table 19's metrics.

Example:
  python analysis/delay/eval_delay_heldout.py --logdir runs/delay/k4_adjacent_s0 \
      --checkpoint checkpoints/agent_step180000.pt --device cuda:0
Writes <logdir>/eval_delay_heldout.json.
"""
import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import probe_common as pc  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402
from shapeenv import NoisyShapeEnv  # noqa: E402


def inv_acc(wm, cfg, img, act, isf, shift):
    """Inverse-head accuracy with the shifted window: predict a_j from (e_{j+s}, e_{j+s+1})."""
    dev = cfg.device
    B, T = act.shape[0], act.shape[1]
    data = {"image": torch.tensor(img, device=dev),
            "action": torch.tensor(act, dtype=torch.float32, device=dev),
            "is_first": torch.tensor(isf, device=dev),
            "is_terminal": torch.zeros(B, T, dtype=torch.bool, device=dev)}
    with torch.no_grad():
        data = wm.preprocess(data)
        f8, _, _ = wm._encode_with_f8(data)
        _, pooled_attn, _ = wm._compute_attention(f8)
        pab = pooled_attn.reshape(B, T, -1)
        inv_in = torch.cat([pab[:, shift:-1], pab[:, shift + 1:]], -1)
        pred = wm.attn_inv_head(inv_in.reshape(-1, inv_in.shape[-1])).argmax(-1)
        tgt = data["action"][:, 1:T - shift].argmax(-1).reshape(-1)
        return float((pred == tgt).float().mean())


def rollout(action_delay, davis_path, seed, B=24, T=64):
    """B fresh episodes of T steps; action[t] is the issued action that precedes image[t]."""
    base = NoisyShapeEnv(size=(64, 64), seed=seed, davis_path=davis_path)
    env = wrappers.OneHotAction(base)
    if action_delay > 0:
        env = wrappers.ActionDelay(env, action_delay)
    rng = np.random.RandomState(seed + 1)
    IM, ACT, IF = [], [], []
    for _ in range(B):
        env.reset()
        im, act_ep, isf = [], [], []
        for t in range(T):
            a = rng.randint(0, 4); oh = np.zeros(4, np.float32); oh[a] = 1.0
            o, *_ = env.step(oh)
            im.append(o["image"]); act_ep.append(oh); isf.append(t == 0)
        IM.append(np.stack(im)); ACT.append(np.stack(act_ep)); IF.append(np.array(isf))
    return np.stack(IM), np.stack(ACT), np.stack(IF)


def sample_buffer(logdir, B=24, T=64, seed=0):
    fs = sorted(glob.glob(f"{logdir}/train_eps/*.npz"))[-200:]
    rng = np.random.RandomState(seed)
    IM, ACT, IF = [], [], []
    for _ in range(B):
        d = np.load(fs[rng.randint(len(fs))]); L = len(d["action"]); s = rng.randint(0, L - T)
        IM.append(d["image"][s:s + T]); ACT.append(d["action"][s:s + T]); IF.append(d["is_first"][s:s + T])
    return np.stack(IM), np.stack(ACT), np.stack(IF)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--checkpoint", default="latest.pt")
    ap.add_argument("--action_delay", type=int, default=None, help="default: from the run config")
    ap.add_argument("--inv_delay", type=int, default=None,
                    help="window shift the model was trained with (default: from the run config)")
    ap.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR", ""))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_batches", type=int, default=6)
    ap.add_argument("--heldout_seed", type=int, default=99991)
    ap.add_argument("--out", default=None, help="default <logdir>/eval_delay_heldout.json")
    args = ap.parse_args()
    pc.require_davis(ap, args.davis_path)

    wm, cfg = pc.load_model(args.logdir, args.checkpoint, args.device, presets=("noisyshape",))
    k = cfg.action_delay if args.action_delay is None else args.action_delay
    shift = cfg.inv_delay if args.inv_delay is None else args.inv_delay

    tr = np.mean([inv_acc(wm, cfg, *sample_buffer(args.logdir, seed=s), shift)
                  for s in range(args.n_batches)])
    ho = np.mean([inv_acc(wm, cfg, *rollout(k, args.davis_path, args.heldout_seed + s), shift)
                  for s in range(args.n_batches)])
    res = dict(logdir=args.logdir, checkpoint=args.checkpoint, action_delay=k, inv_delay=shift,
               window="shifted-adjacent" if shift > 0 else "adjacent",
               train_inv_acc=round(float(tr), 4), heldout_inv_acc=round(float(ho), 4), chance=0.25)
    print(json.dumps(res, indent=2))
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.logdir) / "eval_delay_heldout.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
