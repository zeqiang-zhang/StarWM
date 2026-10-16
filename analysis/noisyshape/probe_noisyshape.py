"""Latent probes on Shape / Shape-fixed / NoisyShape / NoisyShape-Seq (Appendix C.2).

Produces the per-run ``probe_results/results.json`` behind Fig. 3 / Table 4
(NoisyShape, all model variants), Fig. 12 (gradient isolation), Fig. 15 (SS
ablation) and Fig. 19 / Table 12 (progressive distraction). Works for every
``--model`` type (starwm, dreamerv3, dualstream, attention_only and their
presets).

Protocol (identical to the paper's probe script):
  * 500 random-policy episodes (env seed 42) + a separate 200-episode set
    (env seed 12345), 200 steps each, collected once per environment and cached
    in ``--cache_dir`` (shared by all models).
  * H0 ("current_probes"): MLP probes on the filtered RSSM features
    feat=[h; z] of the 500-episode set: color (4), shape (4), size (3)
    classification and position x / y regression, 70/30 split.
  * Imagination ("imagined_probes"): probes are re-fitted on the filtered
    features of the 200-episode set (H0 entry ``'0'`` = their test score), then
    applied to the features after h = 1..15 open-loop ``img_step`` calls with
    uniform random actions; labels at h are rolled forward with the env's
    action logic (``shapeenv.labels.rollout_labels``).
  * Position at h >= 1 is scored against the position of the start frame (it is
    re-sampled every step, so this entry is not meaningful and not reported).
  * Shape-fixed: the position is constant (0.5, 0.5), so the position probes
    are skipped (as in the paper) and the JSON has no ``pos_x`` / ``pos_y``.

Output: ``<run_dir>/probe_results/results.json`` =
``{<model>: {"current_probes": {color: {train_acc, test_acc}, ..., pos_x:
{train_mse, test_mse}, ...}, "imagined_probes": {"0": {...}, ..., "15": {...}}}}``.

Example:
    python analysis/noisyshape/probe_noisyshape.py runs/noisyshape/*_s[012] \\
        --cache_dir runs/probe_cache --davis_path $DAVIS_DIR
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm

import _probe_common as pc
from shapeenv.labels import rollout_labels
from shapeenv.probe import train_clf, train_reg

ENVS = ("shape", "shapefixed", "noisyshape", "noisyshapeseq")
LABELS = {"color_id": "color", "shape_id": "shape", "size_id": "size",
          "position_x": "pos_x", "position_y": "pos_y"}
CLS = [("color", 4), ("shape", 4), ("size", 3)]
POS = ["pos_x", "pos_y"]


def evaluate_model(wm, config, curr_data, imag_data, device, horizon=15, batch_size=64,
                   imag_action_seed=0, with_position=True):
    ci, ca, cf, cl = curr_data
    ii, ia, isf, il = imag_data
    fcl = pc.rename_labels(cl, LABELS)
    fil = pc.rename_labels(il, LABELS)
    pos = POS if with_position else []

    print("  Encoding current trajectories...")
    feats = pc.wm_loader.encode_episodes(wm, config, ci, ca, cf, batch_size=batch_size)
    print(f"  Encoded: {feats.shape}")
    cp = {}
    for nm, nc in CLS:
        tra, tea, _ = train_clf(feats, fcl[nm], nc, device)
        cp[nm] = {"train_acc": tra, "test_acc": tea}
        print(f"    {nm}: test_acc={tea:.4f}")
    for nm in pos:
        trm, tem, _ = train_reg(feats, fcl[nm], device)
        cp[nm] = {"train_mse": trm, "test_mse": tem}
        print(f"    {nm}: test_mse={tem:.6f}")
    del feats

    print("  Encoding + imagining...")
    sampler = pc.action_sampler(config.num_actions, imag_action_seed)
    rf, imf, ima = pc.wm_loader.encode_episodes(wm, config, ii, ia, isf, batch_size=batch_size,
                                                horizon=horizon, action_sampler=sampler)
    print(f"  real={rf.shape}, imagined={imf.shape}")
    corrected = rollout_labels(fil, ima, horizon)
    ip, clfs, regs = {"0": {}}, {}, {}
    for nm, nc in CLS:
        _, tea, clfs[nm] = train_clf(rf, fil[nm], nc, device)
        ip["0"][nm] = tea
    for nm in pos:
        _, tem, regs[nm] = train_reg(rf, fil[nm], device)
        ip["0"][nm] = tem
    for h in tqdm(range(horizon), desc="  Evaluating horizons"):
        hf = imf[h]
        ip[str(h + 1)] = {}
        for nm, _ in CLS:
            ip[str(h + 1)][nm] = float(accuracy_score(corrected[h + 1][nm], clfs[nm].predict(hf)))
        for nm in pos:  # scored against the start-frame position (see module docstring)
            ip[str(h + 1)][nm] = float(mean_squared_error(fil[nm], regs[nm].predict(hf)))
    for h in range(1, horizon + 1):
        s = ip[str(h)]
        print(f"    H{h}: color={s['color']:.4f} shape={s['shape']:.4f} size={s['size']:.4f}")
    return {"current_probes": cp, "imagined_probes": ip}


def main():
    args = pc.make_parser(__doc__).parse_args()
    env_name = args.env or pc.infer_env(args.run_dirs[0])
    if env_name not in ENVS:
        raise SystemExit(f"--env must be one of {ENVS} (got {env_name}); "
                         "use probe_motion.py / probe_two_shape.py for the other variants")
    print(f"Environment: {env_name}, device: {args.device}")
    curr_data, imag_data = pc.get_probe_data(args, env_name)
    # Shape-fixed: constant position -> no position probes
    with_position = bool(np.ptp(curr_data[3]["position_x"]) > 0 or np.ptp(curr_data[3]["position_y"]) > 0)
    if not with_position:
        print("  position is constant in this environment: skipping position probes")

    for rd in args.run_dirs:
        rp = pc.results_path(rd, args.results_subdir)
        if os.path.exists(rp) and not args.force:
            print(f"\nSKIP {rd} (results exist; --force to overwrite)")
            continue
        print(f"\n---- {rd} ----")
        wm, config = pc.load_model(rd, args.checkpoint, args.device)
        if wm is None:
            continue
        run_env = config.task.split("_", 1)[0]
        if run_env != env_name:
            print(f"WARNING: {rd} was trained on {run_env}, probing on {env_name}")
        torch.manual_seed(args.probe_seed)
        results = evaluate_model(wm, config, curr_data, imag_data, args.device,
                                 horizon=args.horizon, batch_size=args.batch_size,
                                 imag_action_seed=args.imag_action_seed,
                                 with_position=with_position)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as f:
            json.dump({config.model: results}, f, indent=2)
        print(f"  Saved to {rp}")
        del wm
        pc.free()
    print("\nAll probe analyses complete.")


if __name__ == "__main__":
    main()
