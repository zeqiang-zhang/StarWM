#!/usr/bin/env python3
"""Aggregate statistics of the DMC benchmark (rliable; Fig 1 right, Fig 10).

Compute half of the pipeline; plotting is done by ``plot_rliable.py`` (Fig 10)
and ``plot_iqm_scatter.py`` (Fig 1, right), which only read the files written here.

Per seed, the score is the mean training-episode return over the last 50K env
steps of the run. Every task is normalised by DreamerV3's *Clean* score on that
task (mean over seeds of the same statistic), giving a recovery ratio. rliable's
stratified bootstrap then gives Median / IQM / Mean / Optimality Gap with 95% CIs
(10,000 reps), performance profiles (81 thresholds in [0, 1.5]) and the
probability-of-improvement matrix (2,000 reps). As in the paper, one
``RandomState(0)`` is passed to every rliable bootstrap (order VH, VHR, VH+VHR),
but rliable's ``StratifiedBootstrap`` draws its resampling indices from NumPy's
*global* RNG, so the confidence intervals (not the point estimates) change by
about +-0.003 from run to run. ``--np_seed S`` seeds the global RNG to make them
repeatable (not done for the paper). Only seeds present for every method of a
task are used.

Outputs under ``--out_root`` (default ``figures/rliable``)::

    scores_table.csv, baselines.json
    {vh,vhr,combined}/{score_arrays,aggregate,profile,poi}.npz, tasks.json

Takes ~15 min on one CPU core.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from collections import OrderedDict

from rliable import library as rly
from rliable import metrics

from common import TASKS, paper_task, load_metric
from layouts import add_layout_args, layout_from_args, TRAIN_METRIC

RNG = np.random.RandomState(0)

MODES = ["vh", "vhr"]
# method label -> (source, name); the labels are the keys of the saved arrays
METHOD_SOURCE = OrderedDict([
    ("StarWM", ("dmc", "starwm")),
    ("StarWM-R", ("dmc", "starwm_r")),
    ("Orig DreamerV3", ("dmc", "dreamerv3")),
    ("R2Dreamer", ("baseline", "r2dreamer")),
    ("DenoisedMDP", ("baseline", "denoised_mdp")),
    ("IsoDreamer", ("baseline", "isodream")),
])
METHODS = list(METHOD_SOURCE)
LATE_WINDOW_STEPS = 50_000


def seed_dirs(layout, method, mode, task):
    source, name = METHOD_SOURCE[method]
    runs = layout.dmc(mode, task, name) if source == "dmc" else layout.baseline(name, mode, task)
    return dict(runs)


def seed_score(run_dir, metric):
    """Mean of the metric over the last 50K steps of the run (NaN if no data)."""
    if run_dir is None or not os.path.isdir(run_dir):
        return np.nan
    steps, vals = load_metric(run_dir, metric)
    if steps is None or len(steps) < 2:
        return np.nan
    cutoff = steps[-1] - LATE_WINDOW_STEPS
    mask = steps >= cutoff
    if mask.sum() < 1:
        return float(vals[-1])
    return float(np.mean(vals[mask]))


def metric_of(method):
    source, name = METHOD_SOURCE[method]
    return TRAIN_METRIC["starwm" if source == "dmc" else name]


# ─── Score table & normalisation ─────────────────────────────────────────────
def build_score_table(layout, seeds):
    rows = []
    for method in METHODS:
        for mode in MODES:
            for task in TASKS:
                sd = seed_dirs(layout, method, mode, task)
                for seed in seeds:
                    s = seed_score(sd.get(seed), metric_of(method))
                    # tasks are stored under the paper's names: the bootstrap strata
                    # are ordered by name, so this keeps the random draws identical
                    rows.append(dict(method=method, mode=mode, env=paper_task(task),
                                     seed=seed, raw_score=s))
    return pd.DataFrame(rows)


def dreamer_clean_baselines(layout, seeds):
    """Per task: mean over seeds of DreamerV3's late-window score in Clean."""
    base = {}
    for task in TASKS:
        sd = dict(layout.dmc("clean", task, "dreamerv3"))
        scores = [s for s in (seed_score(sd.get(seed), TRAIN_METRIC["dreamerv3"])
                              for seed in seeds) if np.isfinite(s)]
        base[paper_task(task)] = float(np.mean(scores)) if scores else np.nan
    return base


def normalize_table(df, baselines):
    df = df.copy()
    df["norm_score"] = np.nan
    for env, base in baselines.items():
        if not np.isfinite(base) or base <= 0:
            continue
        idx = df["env"] == env
        df.loc[idx, "norm_score"] = df.loc[idx, "raw_score"] / base
    return df


def to_score_arrays(df, modes_list):
    """{method: (n_seeds, n_tasks)} using, per task, the seeds every method has."""
    sub = df[df["mode"].isin(modes_list)].copy()
    sub["task_id"] = sub["mode"] + "/" + sub["env"]
    task_ids = sorted(sub["task_id"].unique())

    common_seeds_by_task = {}
    for tid in task_ids:
        seeds_ok = None
        for m in METHODS:
            seeds_with_data = set(
                sub[(sub.task_id == tid) & (sub.method == m) &
                    np.isfinite(sub.norm_score)]["seed"].tolist())
            seeds_ok = seeds_with_data if seeds_ok is None else (seeds_ok & seeds_with_data)
        common_seeds_by_task[tid] = sorted(seeds_ok) if seeds_ok else []

    min_seeds = min((len(v) for v in common_seeds_by_task.values()), default=0)
    if min_seeds == 0:
        usable_tasks = [t for t, v in common_seeds_by_task.items() if v]
        if not usable_tasks:
            return {}, []
        min_seeds = min(len(common_seeds_by_task[t]) for t in usable_tasks)
        task_ids = usable_tasks

    score_arrays = {}
    for m in METHODS:
        mat = np.full((min_seeds, len(task_ids)), np.nan, dtype=np.float64)
        for j, tid in enumerate(task_ids):
            for i, sd in enumerate(common_seeds_by_task[tid][:min_seeds]):
                row = sub[(sub.task_id == tid) & (sub.method == m) & (sub.seed == sd)]
                if len(row) == 1 and np.isfinite(row.norm_score.iloc[0]):
                    mat[i, j] = row.norm_score.iloc[0]
        score_arrays[m] = mat
    return score_arrays, task_ids


# ─── Bootstrap & save ────────────────────────────────────────────────────────
def compute_aggregate(score_arrays):
    """(point estimates, CIs) of [Median, IQM, Mean, Optimality Gap]."""
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ])
    return rly.get_interval_estimates(score_arrays, aggregate_func, reps=10000,
                                      random_state=RNG)


def compute_poi(score_arrays, methods_present):
    """M[i, j] = P(method_i > method_j)."""
    pairs = OrderedDict()
    for a in methods_present:
        for b in methods_present:
            if a != b:
                pairs[f"{a},{b}"] = (score_arrays[a], score_arrays[b])
    poi, _ = rly.get_interval_estimates(pairs, metrics.probability_of_improvement,
                                        reps=2000, random_state=RNG)
    n = len(methods_present)
    M = np.full((n, n), np.nan)
    for k, val in poi.items():
        a, b = k.split(",")
        M[methods_present.index(a), methods_present.index(b)] = val
    np.fill_diagonal(M, 0.5)
    return M


def run_block(score_arrays, tasks_used, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    if not score_arrays:
        print(f"[{title}] no usable data")
        return
    n_seeds, n_tasks = next(iter(score_arrays.values())).shape
    print(f"\n[{title}] {len(score_arrays)} methods x {n_seeds} seeds x "
          f"{n_tasks} tasks ({tasks_used})")
    np.savez(os.path.join(out_dir, "score_arrays.npz"), **score_arrays)
    with open(os.path.join(out_dir, "tasks.json"), "w") as f:
        json.dump(list(tasks_used), f, indent=2)

    aggr_pts, aggr_cis = compute_aggregate(score_arrays)
    metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
    print(f"\n{title}  (point [95% CI])")
    print("-" * 80)
    print(f"{'method':<20}" + "".join(f"{n:>22}" for n in metric_names))
    for m in METHODS:
        if m not in aggr_pts:
            continue
        line = f"{m:<20}"
        for k in range(4):
            lo, hi = aggr_cis[m][:, k]
            line += f"  {aggr_pts[m][k]:.3f} [{lo:.3f},{hi:.3f}]".rjust(22)
        print(line)
    dump = {}
    for m in aggr_pts:
        dump[f"{m}__point"] = aggr_pts[m]
        dump[f"{m}__ci"] = aggr_cis[m]
    dump["__metric_names__"] = np.array(metric_names)
    np.savez(os.path.join(out_dir, "aggregate.npz"), **dump)

    thresholds = np.linspace(0.0, 1.5, 81)
    prof_pts, prof_cis = rly.create_performance_profile(score_arrays, thresholds)
    dump = {"__thresholds__": thresholds}
    for m in prof_pts:
        dump[f"{m}__point"] = prof_pts[m]
        dump[f"{m}__ci"] = prof_cis[m]
    np.savez(os.path.join(out_dir, "profile.npz"), **dump)

    methods_present = [m for m in METHODS if m in score_arrays]
    poi_mat = compute_poi(score_arrays, methods_present)
    np.savez(os.path.join(out_dir, "poi.npz"), matrix=poi_mat,
             methods=np.array(methods_present))
    print(f"  -> saved score_arrays.npz / aggregate.npz / profile.npz / poi.npz under {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--out_root", default=os.path.join("figures", "rliable"))
    ap.add_argument("--np_seed", type=int, default=None,
                    help="seed NumPy's global RNG (makes the bootstrap CIs repeatable; "
                         "unset for the paper)")
    args = ap.parse_args()
    layout = layout_from_args(args)
    if args.np_seed is not None:
        np.random.seed(args.np_seed)

    if args.seeds:
        seeds = sorted(args.seeds)
    elif layout.kind == "paper":
        seeds = list(range(6))          # the paper's runs use seed ids 0-5
    else:
        found = set()
        for m in METHODS:
            for mode in MODES:
                for task in TASKS:
                    found |= set(seed_dirs(layout, m, mode, task))
        for task in TASKS:
            found |= {s for s, _ in layout.dmc("clean", task, "dreamerv3")}
        seeds = sorted(found)

    print("Loading scores …")
    df = build_score_table(layout, seeds)
    print("\nComputing per-task baselines from Orig DreamerV3 *clean* …")
    baselines = dreamer_clean_baselines(layout, seeds)
    for env, b in baselines.items():
        print(f"  {env:<22} baseline = {b:.2f}")
    df = normalize_table(df, baselines)
    print("\nPer-method/mode normalized mean:")
    print(df.groupby(["method", "mode"])["norm_score"].agg(["count", "mean"]))

    os.makedirs(args.out_root, exist_ok=True)
    csv_path = os.path.join(args.out_root, "scores_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nNormalized scores table -> {csv_path}")
    with open(os.path.join(args.out_root, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)

    for modes, sub, title in ((["vh"], "vh", "VH"), (["vhr"], "vhr", "VHR"),
                              (["vh", "vhr"], "combined", "VH + VHR")):
        sa, tasks = to_score_arrays(df, modes)
        run_block(sa, tasks, os.path.join(args.out_root, sub), title)
    print("\nDone.")


if __name__ == "__main__":
    main()
