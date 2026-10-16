"""Latent probes on NoisyShape with uniform motion (Appendix E.6, Fig. 18 / Table 11).

Same protocol as ``probe_noisyshape.py`` (500 + 200 cached random-policy
episodes, MLP probes, 70/30 split, 15-step open-loop imagination) with two
changes for the ``noisyshapemotion`` environment:
  * Position follows linear motion with boundary reflection, so it is
    predictable: position probes are evaluated at every horizon against labels
    rolled forward with ``shapeenv.labels.rollout_labels_motion``.
  * Background ID (20 DAVIS clips, re-sampled every step, unpredictable) is
    probed at H0 only (20-way classification on the 500-episode set).

Output: ``<run_dir>/probe_results/results.json`` =
``{<model>: {"current_probes": {color, shape, size, pos_x, pos_y, bg_id}: float,
"imagined_probes": {"0".."15": {color, shape, size, pos_x, pos_y}}}}``
(accuracies / test MSEs as plain floats, as in the paper's result files).

Example:
    python analysis/noisyshape/probe_motion.py runs/noisyshapemotion/*_s[012] \\
        --cache_dir runs/probe_cache --davis_path $DAVIS_DIR
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm

import _probe_common as pc
from shapeenv.labels import rollout_labels_motion
from shapeenv.probe import train_clf, train_reg

LABELS = {"color_id": "color", "shape_id": "shape", "size_id": "size",
          "position_x": "pos_x", "position_y": "pos_y",
          "velocity_x": "vel_x", "velocity_y": "vel_y", "background_id": "bg_id"}
CLS = [("color", 4), ("shape", 4), ("size", 3)]
POS = ["pos_x", "pos_y"]


def evaluate_model(wm, config, curr_data, imag_data, device, horizon=15, batch_size=64,
                   imag_action_seed=0):
    ci, ca, cf, cl = curr_data
    ii, ia, isf, il = imag_data
    fcl = pc.rename_labels(cl, LABELS)
    fil = pc.rename_labels(il, LABELS)

    # ---- H0 (current) ----
    print("  Encoding current trajectories...")
    feats = pc.wm_loader.encode_episodes(wm, config, ci, ca, cf, batch_size=batch_size)
    print(f"  Encoded: {feats.shape}")
    h0 = {}
    for nm, nc in CLS:
        _, h0[nm], _ = train_clf(feats, fcl[nm], nc, device)
        print(f"    {nm}: test_acc={h0[nm]:.4f}")
    for nm in POS:
        _, h0[nm], _ = train_reg(feats, fcl[nm], device)
        print(f"    {nm}: test_mse={h0[nm]:.6f}")
    if fcl["bg_id"].min() < 0:  # env fell back to noise backgrounds (no DAVIS frames found)
        print("    WARNING: no DAVIS backgrounds (background_id < 0): skipping the bg_id probe")
    else:
        n_bg = int(fcl["bg_id"].max()) + 1
        _, h0["bg_id"], _ = train_clf(feats, fcl["bg_id"].astype(np.int64), n_bg, device)
        print(f"    bg_id: test_acc={h0['bg_id']:.4f} ({n_bg} classes)")
    del feats

    # ---- imagined H1..H15 ----
    print("  Encoding + imagining...")
    sampler = pc.action_sampler(config.num_actions, imag_action_seed)
    rf, imf, ima = pc.wm_loader.encode_episodes(wm, config, ii, ia, isf, batch_size=batch_size,
                                                horizon=horizon, action_sampler=sampler)
    print(f"  real={rf.shape}, imagined={imf.shape}")
    corrected = rollout_labels_motion(fil, ima, horizon)
    ip, clfs, regs = {"0": {}}, {}, {}
    for nm, nc in CLS:
        _, ip["0"][nm], clfs[nm] = train_clf(rf, fil[nm], nc, device)
    for nm in POS:
        _, ip["0"][nm], regs[nm] = train_reg(rf, fil[nm], device)
    for h in tqdm(range(horizon), desc="  Evaluating horizons"):
        hf = imf[h]
        ip[str(h + 1)] = {}
        for nm, _ in CLS:
            ip[str(h + 1)][nm] = float(accuracy_score(corrected[h + 1][nm], clfs[nm].predict(hf)))
        for nm in POS:
            ip[str(h + 1)][nm] = float(mean_squared_error(corrected[h + 1][nm], regs[nm].predict(hf)))
    for h in range(1, horizon + 1):
        s = ip[str(h)]
        print(f"    H{h}: color={s['color']:.4f} shape={s['shape']:.4f} size={s['size']:.4f} "
              f"pos_x={s['pos_x']:.6f} pos_y={s['pos_y']:.6f}")
    return {"current_probes": h0, "imagined_probes": ip}


def main():
    args = pc.make_parser(__doc__).parse_args()
    env_name = args.env or pc.infer_env(args.run_dirs[0])
    if env_name != "noisyshapemotion":
        raise SystemExit(f"probe_motion.py expects --env noisyshapemotion (got {env_name})")
    print(f"Environment: {env_name}, device: {args.device}")
    curr_data, imag_data = pc.get_probe_data(args, env_name)

    for rd in args.run_dirs:
        rp = pc.results_path(rd, args.results_subdir)
        if os.path.exists(rp) and not args.force:
            print(f"\nSKIP {rd} (results exist; --force to overwrite)")
            continue
        print(f"\n---- {rd} ----")
        wm, config = pc.load_model(rd, args.checkpoint, args.device)
        if wm is None:
            continue
        torch.manual_seed(args.probe_seed)
        results = evaluate_model(wm, config, curr_data, imag_data, args.device,
                                 horizon=args.horizon, batch_size=args.batch_size,
                                 imag_action_seed=args.imag_action_seed)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as f:
            json.dump({config.model: results}, f, indent=2)
        print(f"  Saved to {rp}")
        del wm
        pc.free()
    print("\nAll probe analyses complete.")


if __name__ == "__main__":
    main()
