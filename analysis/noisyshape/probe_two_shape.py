"""Latent probes on NoisyTwoShape (Appendix E.9, Fig. 21).

Protocol of the paper's two-object probe script:
  * Smaller probe sets than single-object NoisyShape: 200 episodes for H0 and
    100 episodes for imagination (env seeds 42 / 12345), 200 steps each.
  * H0 ("current_probes"): color and shape of both objects (4-way each) and
    the x / y position of both objects (regression).
  * Imagination: classification probes re-fitted on the 100-episode set and
    evaluated after h = 1..15 open-loop steps with uniform random actions
    over the 8 actions; labels rolled forward with
    ``shapeenv.labels.rollout_labels_two_shape``. Position is H0 only.
  * The random policy uses ``RandomState(env_seed)`` and imagination uses
    ``RandomState(0)``, exactly as the original script.
s1 = large object, s2 = small object.

Output: ``<run_dir>/probe_results/results.json`` =
``{<model>: {"current_probes": {s1_color: {test_acc}, ..., s1_pos_x: {test_mse}, ...},
"imagined_probes": {"0".."15": {s1_color, s2_color, s1_shape, s2_shape}}}}``.

Example:
    python analysis/noisyshape/probe_two_shape.py runs/noisytwoshape/*_s[012] \\
        --cache_dir runs/probe_cache --davis_path $DAVIS_DIR
"""
import json
import os

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import _probe_common as pc
from shapeenv.labels import rollout_labels_two_shape
from shapeenv.probe import train_clf, train_reg

CLS_KEYS = ["s1_color", "s2_color", "s1_shape", "s2_shape"]
POS_KEYS = ["s1_pos_x", "s1_pos_y", "s2_pos_x", "s2_pos_y"]
LABELS = {"s1_color_id": "s1_color", "s2_color_id": "s2_color",
          "s1_shape_id": "s1_shape", "s2_shape_id": "s2_shape",
          "s1_position_x": "s1_pos_x", "s1_position_y": "s1_pos_y",
          "s2_position_x": "s2_pos_x", "s2_position_y": "s2_pos_y"}


def evaluate_model(wm, config, curr_data, imag_data, device, horizon=15, batch_size=64,
                   imag_action_seed=0):
    ci, ca, cf, cl = curr_data
    ii, ia, isf, il = imag_data
    fcl = pc.rename_labels(cl, LABELS)
    fil = pc.rename_labels(il, LABELS)

    print("  Encoding current trajectories (H0 probes)...")
    feats = pc.wm_loader.encode_episodes(wm, config, ci, ca, cf, batch_size=batch_size)
    print(f"  Encoded: {feats.shape}")
    cp = {}
    for nm in CLS_KEYS:
        _, tea, _ = train_clf(feats, fcl[nm], 4, device)
        cp[nm] = {"test_acc": tea}
        print(f"    {nm:8s}: acc={tea:.4f}")
    for nm in POS_KEYS:
        _, tem, _ = train_reg(feats, fcl[nm], device)
        cp[nm] = {"test_mse": tem}
        print(f"    {nm:8s}: mse={tem:.6f}")
    del feats

    print(f"  Encoding + imagining (H0..H{horizon})...")
    sampler = pc.action_sampler(config.num_actions, imag_action_seed)
    rf, imf, ima = pc.wm_loader.encode_episodes(wm, config, ii, ia, isf, batch_size=batch_size,
                                                horizon=horizon, action_sampler=sampler)
    print(f"  real={rf.shape}, imagined={imf.shape}")
    corrected = rollout_labels_two_shape(fil, ima, horizon)
    ip, clfs = {"0": {}}, {}
    for nm in CLS_KEYS:
        _, ip["0"][nm], clfs[nm] = train_clf(rf, fil[nm], 4, device)
    for h in tqdm(range(horizon), desc="  Eval horizons"):
        hf = imf[h]
        ip[str(h + 1)] = {nm: float(accuracy_score(corrected[h + 1][nm], clfs[nm].predict(hf)))
                          for nm in CLS_KEYS}
    for h in range(1, horizon + 1):
        s = ip[str(h)]
        print(f"    H{h:2d}: s1_color={s['s1_color']:.3f} s2_color={s['s2_color']:.3f}  "
              f"s1_shape={s['s1_shape']:.3f} s2_shape={s['s2_shape']:.3f}")
    return {"current_probes": cp, "imagined_probes": ip}


def main():
    args = pc.make_parser(__doc__, num_curr=200, num_imag=100).parse_args()
    env_name = args.env or pc.infer_env(args.run_dirs[0])
    if env_name != "noisytwoshape":
        raise SystemExit(f"probe_two_shape.py expects --env noisytwoshape (got {env_name})")
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
