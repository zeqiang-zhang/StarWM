"""Shared helpers for the NoisyShape-family plotting scripts (no torch needed).

Result files are ``<runs>/<benchmark>/<variant>_s<seed>/<results_subdir>/results.json``
with schema ``{<model key>: {"current_probes": ..., "imagined_probes": ...}}``
(written by ``probe_noisyshape.py`` / ``probe_motion.py`` / ``probe_two_shape.py``).
The model key is ignored when reading, so result files of the paper runs
(keys ``starwm``, ``original``, ``icr``, ``crossattn_single``, ``result``, ...)
load the same way. ``--name_map paper`` maps release benchmark / variant names
onto the directory names of the paper runs.
"""
import json
import os

import numpy as np

# release benchmark -> run directory of the paper runs
PAPER_BENCHMARKS = {
    "shape": "shapev2", "shapefixed": "shapevfixed", "noisyshape": "noisyshapev2",
    "noisyshapeseq": "noisyshapev2seq", "noisyshapemotion": "noisyshapev4",
    "noisytwoshape": "noisytwoshape",
}
# release variant -> run name of the paper runs
PAPER_VARIANTS = {
    "dreamerv3": "orig", "starwm": "starwm_detach", "dualstream": "icr_only",
    "ss_only": "selfsup_only", "dualstream_ss": "icr_selfsup",
    "attention_only": "crossattn_single",
    "iso_star1": "no_detach_v2_step2", "iso_star2": "no_detach_no_ss_v2_step2",
    "iso_star3": "no_detach_no_ss", "iso_star4": "no_detach",
    "inv_only": "inv_only", "ctr_only": "ctr_only",
}

CHANCE = {"color": 0.25, "shape": 0.25, "size": 1 / 3}
# Chance-level position MSE (x + y) of NoisyShape; = shapeenv.labels.position_chance_mse()
CHANCE_POS = 0.0711


def add_common_args(parser, horizon=True):
    parser.add_argument("--runs", default="runs",
                        help="root directory holding <benchmark>/<variant>_s<seed> run dirs")
    parser.add_argument("--name_map", choices=["release", "paper"], default="release",
                        help="'paper' reads the directory layout of the original paper runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results_subdir", default="probe_results")
    if horizon:
        parser.add_argument("--horizon", type=int, default=15)


def run_dir(root, benchmark, variant, seed, name_map="release"):
    if name_map == "paper":
        benchmark = PAPER_BENCHMARKS.get(benchmark, benchmark)
        variant = PAPER_VARIANTS.get(variant, variant)
    return os.path.join(root, benchmark, f"{variant}_s{seed}")


def load_result_file(path):
    """One results.json -> its single {'current_probes', 'imagined_probes'} entry."""
    with open(path) as f:
        d = json.load(f)
    return d[list(d.keys())[0]]


def load_results(args, benchmark, variant):
    """Per-seed result dicts of one (benchmark, variant); missing seeds are skipped."""
    out = []
    for s in args.seeds:
        rp = os.path.join(run_dir(args.runs, benchmark, variant, s, args.name_map),
                          args.results_subdir, "results.json")
        if not os.path.exists(rp):
            print(f"WARNING: {rp} not found")
            continue
        out.append(load_result_file(rp))
    return out


def load_isodream_results(paths, kind):
    """Iso-Dream++ probe files: ``{<logdir>: {'controllable': {...}, 'combined': {...}}}``,
    where H0 entries are plain floats. ``kind`` selects the latent slice; returns
    per-seed dicts in the schema used by :func:`extract_imagined_curves` /
    :func:`extract_position_mse`."""
    out = []
    for rp in paths:
        sub = load_result_file(rp)[kind]
        cp = sub["current_probes"]
        cp_wrapped = {**cp, "pos_x": {"test_mse": cp["pos_x"]},
                      "pos_y": {"test_mse": cp["pos_y"]}}
        out.append({"current_probes": cp_wrapped, "imagined_probes": sub["imagined_probes"]})
    return out


def extract_position_mse(results_list):
    """H0 position MSE: per seed MSE_x + MSE_y (current probes), then mean / std over seeds."""
    mse_x_list, mse_y_list = [], []
    for res in results_list:
        cp = res["current_probes"]
        mse_x_list.append(cp["pos_x"]["test_mse"])
        mse_y_list.append(cp["pos_y"]["test_mse"])
    sum_list = [x + y for x, y in zip(mse_x_list, mse_y_list)]
    return {"position": {"mean": np.mean(sum_list), "std": np.std(sum_list)}}


def extract_imagined_curves(results_list, horizon=15):
    """Mean / std (over seeds, ddof=0) of color/shape/size accuracy at H0..H<horizon>.
    H0 is ``imagined_probes['0']`` (probe fitted on the imagination set)."""
    attrs = ["color", "shape", "size"]
    curves = {a: [] for a in attrs}
    for res in results_list:
        ip = res["imagined_probes"]
        for a in attrs:
            curves[a].append([ip[str(h)][a] for h in range(horizon + 1)])
    out = {}
    for a in attrs:
        arr = np.array(curves[a])  # (n_seeds, horizon+1)
        out[a] = {"mean": arr.mean(0), "std": arr.std(0)}
    return out


def mean_std(values, scale=1.0):
    v = np.asarray(values, dtype=np.float64) * scale
    return float(v.mean()), float(v.std())


def fmt(values, scale=100.0, digits=2):
    """'mean ± std' (std over seeds, ddof=0) or '---' if empty."""
    if values is None or len(values) == 0:
        return "---"
    m, s = mean_std(values, scale)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
