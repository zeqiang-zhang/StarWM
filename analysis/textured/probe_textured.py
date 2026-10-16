#!/usr/bin/env python3
"""Latent probe on the textured foreground ladder (Appendix F.3 (i), Table 13, Fig. 24).

Paper probe protocol (Appendix C.2) with the texture ID in place of colour:
  * two independent random-policy trajectory sets, 200-step episodes:
      current : 500 episodes (seed 42)    -> H0 probes (texture / shape / size
                accuracy, position MSE = MSE_x + MSE_y)
      imagined: 200 episodes (seed 12345) -> H15 probes
  * H15 = transfer: a probe trained on the imagined set's real H0 features is
    applied to the features after 15 open-loop imagination steps (uniform
    random actions) and scored against labels propagated through the action
    semantics (shapeenv.labels.rollout_labels);
  * feature = [h;z]; MLP probe, 70/30 split (shapeenv.probe).
The trajectory sets are model-free; cache them once per level with --traj_cache.

Example:
  python analysis/textured/probe_textured.py --collect_only --tex_level 3 \
      --traj_cache runs/traj_cache/tex_L3.npz
  python analysis/textured/probe_textured.py --logdir runs/texnoisyshape/L3/starwm_s0 \
      --tex_level 3 --traj_cache runs/traj_cache/tex_L3.npz --device cuda:0
Writes <logdir>/probe_textured_results.json.
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

import probe_common as pc  # noqa: E402
from shapeenv.collect import collect_trajectories  # noqa: E402
from shapeenv.labels import rollout_labels  # noqa: E402
from shapeenv.probe import train_clf, train_reg  # noqa: E402

NCLASS = {"texture": 4, "shape": 4, "size": 3}


def collect(n_eps, davis_path, seed, level, alpha, ep_len=200):
    """Random-policy rollouts; env and policy are both seeded with ``seed``."""
    imgs, acts, isfirst, lab = collect_trajectories(
        "texnoisyshape", n_eps, env_seed=seed, action_seed=seed, episode_length=ep_len,
        size=(64, 64), davis_path=davis_path, level=level, alpha=alpha)
    labels = {"texture": lab["texture_id"], "shape": lab["shape_id"], "size": lab["size_id"],
              "pos_x": lab["position_x"], "pos_y": lab["position_y"]}
    return imgs, acts, isfirst, labels


def get_trajectories(args):
    """Load cached (curr, imag) trajectory sets, else collect + optionally cache."""
    if args.traj_cache and os.path.exists(args.traj_cache):
        z = np.load(args.traj_cache, allow_pickle=True)
        print(f"loaded cached trajectories from {args.traj_cache} "
              f"(curr {z['curr_imgs'].shape[0]}ep, imag {z['imag_imgs'].shape[0]}ep)")
        curr = (z["curr_imgs"], z["curr_acts"], z["curr_isfirst"], z["curr_labels"].item())
        imag = (z["imag_imgs"], z["imag_acts"], z["imag_isfirst"], z["imag_labels"].item())
        return curr, imag
    print(f"collecting current {args.n_curr}ep (seed {pc.CURR_SEED}) on texnoisyshape L{args.tex_level} ...")
    curr = collect(args.n_curr, args.davis_path, pc.CURR_SEED, args.tex_level, args.tex_alpha)
    print(f"collecting imagined {args.n_imag}ep (seed {pc.IMAG_SEED}) ...")
    imag = collect(args.n_imag, args.davis_path, pc.IMAG_SEED, args.tex_level, args.tex_alpha)
    if args.traj_cache:
        os.makedirs(os.path.dirname(args.traj_cache) or ".", exist_ok=True)
        np.savez_compressed(
            args.traj_cache,
            curr_imgs=curr[0], curr_acts=curr[1], curr_isfirst=curr[2],
            curr_labels=np.array(curr[3], dtype=object),
            imag_imgs=imag[0], imag_acts=imag[1], imag_isfirst=imag[2],
            imag_labels=np.array(imag[3], dtype=object))
        print(f"cached trajectories to {args.traj_cache}")
    return curr, imag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", default=None, help="run directory")
    ap.add_argument("--checkpoint", default="latest.pt", help="relative to --logdir")
    ap.add_argument("--tex_level", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--tex_alpha", type=float, default=0.5)
    ap.add_argument("--davis_path", default=pc.default_davis(),
                    help="DAVIS JPEGImages/480p (default: $DAVIS_DIR)")
    ap.add_argument("--n_curr", type=int, default=500, help="current-set episodes (H0 probes)")
    ap.add_argument("--n_imag", type=int, default=200, help="imagined-set episodes (H15 probes)")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=None, choices=["starwm", "dreamerv3"],
                    help="override the model type logged with the run")
    ap.add_argument("--traj_cache", default=None,
                    help="npz to cache/reuse the (curr, imag) trajectory sets of this level")
    ap.add_argument("--collect_only", action="store_true",
                    help="only build the trajectory cache (no model, CPU)")
    ap.add_argument("--out", default=None,
                    help="result json (default <logdir>/probe_textured_results.json)")
    args = ap.parse_args()
    if not (args.traj_cache and os.path.exists(args.traj_cache)):
        pc.require_davis(ap, args.davis_path)

    curr, imag = get_trajectories(args)
    if args.collect_only:
        print("collect_only: cache ready.")
        return
    if not args.logdir:
        ap.error("--logdir required unless --collect_only")

    wm, cfg = pc.load_model(args.logdir, args.checkpoint, args.device,
                            presets=("texnoisyshape",), model=args.model)
    results = {"logdir": args.logdir, "checkpoint": args.checkpoint, "tex_level": args.tex_level,
               "model": cfg.model, "n_curr": args.n_curr, "n_imag": args.n_imag,
               "horizon": args.horizon, "H0": {}, "H15": {}}

    # ---- H0 probes on the current set ------------------------------------------
    ci, ca, cf, cl = curr
    feats0 = pc.encode_h0(wm, cfg, ci, ca, cf)
    clab = {k: cl[k].reshape(-1) for k in cl}
    for attr, nc in NCLASS.items():
        _, te, _ = train_clf(feats0, clab[attr].astype(int), nc, args.device)
        results["H0"][attr] = round(float(te), 4)
        print(f"  H0  {attr:8s} acc = {te:.4f}")
    _, mse_x, _ = train_reg(feats0, clab["pos_x"].astype(np.float32), args.device)
    _, mse_y, _ = train_reg(feats0, clab["pos_y"].astype(np.float32), args.device)
    results["H0"]["position_mse"] = round(float(mse_x) + float(mse_y), 5)
    print(f"  H0  position MSE = {results['H0']['position_mse']:.5f} (chance ~0.071)")

    # ---- H15 probes on the imagined set: train on real H0, transfer -------------
    ii, ia, iff, il = imag
    rf, imf, ima = pc.encode_imagined(wm, cfg, ii, ia, iff, args.horizon)
    ilab = {k: il[k].reshape(-1) for k in il}
    start = {"color": ilab["texture"].astype(int), "shape": ilab["shape"].astype(int),
             "size": ilab["size"].astype(int)}
    prop = rollout_labels(start, ima, args.horizon)
    for attr, nc in NCLASS.items():
        key = "color" if attr == "texture" else attr
        _, _, clf = train_clf(rf, start[key].astype(int), nc, args.device)
        pred = clf.predict(imf[args.horizon - 1])
        y_true = np.asarray(prop[args.horizon][key]).astype(int)
        acc = float(accuracy_score(y_true, pred))
        results["H15"][attr] = round(acc, 4)
        print(f"  H15 {attr:8s} acc = {acc:.4f}  (H0 probe transferred through {args.horizon} imagined steps)")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.logdir) / "probe_textured_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
