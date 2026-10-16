#!/usr/bin/env python3
"""Passive-object identity probe (Appendix F.4 (i), Table 16 and Fig. 28, b_colour columns).

PassiveShapeEnv adds an object B whose colour and position are drawn once per
episode and never respond to actions. This probes whether [h;z] encodes B's
colour (4-way, target) and, as controls, the action-driven object A's
colour / shape / size, at H0 and after 15 imagination steps (H15 = H0 probe
transferred; B is static, so its H15 label equals its H0 label).
  --mode random : PassiveShapeEnv     (i.i.d. background frame per step)
  --mode seq    : PassiveShapeSeqEnv  (sequential video background)
Protocol as in probe_textured.py: 500 current (seed 42) + 200 imagined
(seed 12345) episodes of 200 steps, MLP probe, 70/30 split.

Example:
  python analysis/passive/probe_passive.py --logdir runs/passiveshape/starwm_s0 --mode random \
      --traj_cache runs/traj_cache/passive_random_attr.npz --device cuda:0
Writes <logdir>/probe_passive_results.json.
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "textured"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

import probe_common as pc  # noqa: E402
from shapeenv.collect import collect_trajectories  # noqa: E402
from shapeenv.labels import rollout_labels  # noqa: E402
from shapeenv.probe import train_clf  # noqa: E402

ENV = {"random": "passiveshape", "seq": "passiveshapeseq"}
NC = {"b_color": 4, "a_color": 4, "a_shape": 4, "a_size": 3}


def collect(mode, kind, n_eps, davis_path, seed, ep_len=200):
    """Random-policy rollouts (env and policy seeded with ``seed``).
    kind='attr' -> b_color/a_color/a_shape/a_size; kind='pos' -> b_x/b_y (float32)."""
    imgs, acts, isfirst, lab = collect_trajectories(
        ENV[mode], n_eps, env_seed=seed, action_seed=seed, episode_length=ep_len,
        size=(64, 64), davis_path=davis_path)
    if kind == "attr":
        labels = {"b_color": lab["b_color_id"], "a_color": lab["color_id"],
                  "a_shape": lab["shape_id"], "a_size": lab["size_id"]}
    else:
        labels = {"b_x": lab["b_position_x"].astype(np.float32),
                  "b_y": lab["b_position_y"].astype(np.float32)}
    return imgs, acts, isfirst, labels


def get_sets(args, kind):
    """(curr, imag) sets from --traj_cache if present, else collected (and cached)."""
    need = "b_color" if kind == "attr" else "b_x"
    if args.traj_cache and os.path.exists(args.traj_cache):
        z = np.load(args.traj_cache, allow_pickle=True)
        print(f"loaded cache {args.traj_cache}", flush=True)
        curr = (z["c_i"], z["c_a"], z["c_f"], z["c_l"].item())
        imag = (z["i_i"], z["i_a"], z["i_f"], z["i_l"].item())
        if need not in curr[3]:
            raise SystemExit(f"{args.traj_cache} has labels {sorted(curr[3])}; this probe needs "
                             f"'{need}'. Use separate caches for probe_passive.py and "
                             "probe_passive_position.py.")
        return curr, imag
    print(f"collecting {args.n_curr} curr + {args.n_imag} imag ({args.mode}) ...", flush=True)
    curr = collect(args.mode, kind, args.n_curr, args.davis_path, pc.CURR_SEED)
    imag = collect(args.mode, kind, args.n_imag, args.davis_path, pc.IMAG_SEED)
    if args.traj_cache:
        os.makedirs(os.path.dirname(args.traj_cache) or ".", exist_ok=True)
        np.savez_compressed(args.traj_cache,
                            c_i=curr[0], c_a=curr[1], c_f=curr[2], c_l=np.array(curr[3], object),
                            i_i=imag[0], i_a=imag[1], i_f=imag[2], i_l=np.array(imag[3], object))
        print(f"cached -> {args.traj_cache}", flush=True)
    return curr, imag


def add_common_args(ap):
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--checkpoint", default="latest.pt", help="relative to --logdir")
    ap.add_argument("--mode", default="random", choices=["random", "seq"])
    ap.add_argument("--davis_path", default=pc.default_davis(),
                    help="DAVIS JPEGImages/480p (default: $DAVIS_DIR)")
    ap.add_argument("--n_curr", type=int, default=500)
    ap.add_argument("--n_imag", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=None, choices=["starwm", "dreamerv3"],
                    help="override the model type logged with the run")
    ap.add_argument("--traj_cache", default=None, help="npz to cache/reuse the trajectory sets")
    ap.add_argument("--collect_only", action="store_true")
    ap.add_argument("--out", default=None, help="result json (default inside --logdir)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    args = ap.parse_args()
    if not (args.traj_cache and os.path.exists(args.traj_cache)):
        pc.require_davis(ap, args.davis_path)
    curr, imag = get_sets(args, "attr")
    if args.collect_only:
        print("collect_only: done.")
        return
    if not args.logdir:
        ap.error("--logdir required unless --collect_only")

    wm, cfg = pc.load_model(args.logdir, args.checkpoint, args.device,
                            presets=(ENV[args.mode],), model=args.model)
    results = {"logdir": args.logdir, "checkpoint": args.checkpoint, "mode": args.mode,
               "model": cfg.model, "H0": {}, "H15": {}}

    # ---- H0 on the current set ----
    ci, ca, cf, cl = curr
    f0 = pc.encode_h0(wm, cfg, ci, ca, cf)
    clab = {k: cl[k].reshape(-1).astype(int) for k in cl}
    for attr, nc in NC.items():
        _, te, _ = train_clf(f0, clab[attr], nc, args.device)
        results["H0"][attr] = round(float(te), 4)
        print(f"  H0  {attr:8s} acc = {te:.4f}")

    # ---- H15 transfer on the imagined set: probe trained on real H0 feats, applied to imagined ----
    ii, ia, iff, il = imag
    rf, imf, ima = pc.encode_imagined(wm, cfg, ii, ia, iff, args.horizon)
    ilab = {k: il[k].reshape(-1).astype(int) for k in il}
    # A's attributes propagate through the action semantics; b_color is static.
    prop = rollout_labels({"color": ilab["a_color"], "shape": ilab["a_shape"], "size": ilab["a_size"]},
                          ima, args.horizon)
    h15_targets = {
        "b_color": ilab["b_color"],
        "a_color": np.asarray(prop[args.horizon]["color"]).astype(int),
        "a_shape": np.asarray(prop[args.horizon]["shape"]).astype(int),
        "a_size": np.asarray(prop[args.horizon]["size"]).astype(int),
    }
    for attr, nc in NC.items():
        _, _, clf = train_clf(rf, ilab[attr], nc, args.device)
        pred = clf.predict(imf[args.horizon - 1])
        acc = float(accuracy_score(h15_targets[attr], pred))
        results["H15"][attr] = round(acc, 4)
        print(f"  H15 {attr:8s} acc = {acc:.4f}")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.logdir) / "probe_passive_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
