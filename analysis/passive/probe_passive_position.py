#!/usr/bin/env python3
"""Passive-object position probe (Appendix F.4 (i), Table 16 and Fig. 28, b_x / b_y columns).

B's only per-episode-varying attributes are its colour (probe_passive.py) and
its position (x, y in [0.15, 0.85], static within an episode). This regresses
b_x and b_y from [h;z] with the paper's MLP regressor (70/30 split) and reports
held-out MSE and R^2 = 1 - MSE / Var(y_test):
  H0  : regressor trained and tested on the current set (500 episodes, seed 42);
  H15 : regressor trained on the imagined set's real H0 features
        (200 episodes, seed 12345) and applied to all features after 15
        imagined steps; the target is the unchanged (static) position.

Example:
  python analysis/passive/probe_passive_position.py --logdir runs/passiveshapeseq/starwm_s0 \
      --mode seq --traj_cache runs/traj_cache/passive_seq_pos.npz --device cuda:0
Writes <logdir>/probe_passive_position_results.json. Use a different cache
file than probe_passive.py (the label sets differ).
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import probe_common as pc  # noqa: E402
from probe_passive import ENV, add_common_args, get_sets  # noqa: E402
from shapeenv.probe import train_reg  # noqa: E402


def _predict(model, X):
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(dev)).cpu().numpy().reshape(-1)


def score_h0(feats, y, device):
    """Held-out (30%) MSE + R^2 with the paper regressor and split."""
    _, te_mse, _ = train_reg(feats, y, device)
    _, _, _, y_te = train_test_split(feats, y, test_size=0.3, random_state=42)
    var = float(np.var(y_te))
    return float(te_mse), (1.0 - te_mse / var if var > 0 else float("nan"))


def score_h15(rf, y_train, imf_last, y_target, device):
    """Train on real H0 feats, predict on 15-step imagined feats; MSE + R^2 vs the static target."""
    _, _, model = train_reg(rf, y_train, device)
    pred = _predict(model, imf_last)
    mse = float(np.mean((pred - y_target) ** 2))
    var = float(np.var(y_target))
    return mse, (1.0 - mse / var if var > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    args = ap.parse_args()
    if not (args.traj_cache and os.path.exists(args.traj_cache)):
        pc.require_davis(ap, args.davis_path)
    (ci, ca, cf, cl), (ii, ia, iff, il) = get_sets(args, "pos")
    if args.collect_only:
        print("collect_only: done.")
        return
    if not args.logdir:
        ap.error("--logdir required unless --collect_only")

    wm, cfg = pc.load_model(args.logdir, args.checkpoint, args.device,
                            presets=(ENV[args.mode],), model=args.model)
    res = {"logdir": args.logdir, "checkpoint": args.checkpoint, "mode": args.mode,
           "model": cfg.model, "H0": {}, "H15": {}}

    f0 = pc.encode_h0(wm, cfg, ci, ca, cf)
    for dim in ("b_x", "b_y"):
        y = cl[dim].reshape(-1).astype(np.float32)
        mse, r2 = score_h0(f0, y, args.device)
        res["H0"][dim] = {"mse": round(mse, 5), "r2": round(r2, 4)}
        print(f"  H0  {dim}: MSE={mse:.5f}  R2={r2:.4f}", flush=True)

    rf, imf, ima = pc.encode_imagined(wm, cfg, ii, ia, iff, args.horizon)
    for dim in ("b_x", "b_y"):
        y = il[dim].reshape(-1).astype(np.float32)
        mse, r2 = score_h15(rf, y, imf[args.horizon - 1], y, args.device)
        res["H15"][dim] = {"mse": round(mse, 5), "r2": round(r2, 4)}
        print(f"  H15 {dim}: MSE={mse:.5f}  R2={r2:.4f}", flush=True)

    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(args.logdir) / "probe_passive_position_results.json")
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
