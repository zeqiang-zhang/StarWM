"""Shared helpers for the DMC analysis scripts in this directory.

Every function here reproduces, operation for operation, the code that produced
the paper's figures and tables, so that the numbers come out identical. Two
families of conventions are used in the paper and both live here:

* Learning curves, Table 5, Tables 7/10, Figs 5/7/13/16/17/20 ("4K grid"):
  per-seed series from ``metrics.jsonl`` (duplicate steps averaged), smoothed
  with a padded moving average (:func:`moving_average`), linearly interpolated
  onto a common 4,000-step grid (:func:`interp_to_grid`; points beyond a seed's
  last step are NaN), then mean and *population* std (``np.nanstd``, ddof=0)
  across seeds. A table cell is the mean of the last 10 grid points
  (:func:`final_window`), ending either at the last step where ANY seed has
  data (``rule="any"``) or where ALL seeds have data (``rule="all"``).

* Stress tests, Tables 9/14/17, Figs 25/29 ("EMA-0.3"): eval returns,
  smoothed with an exponential moving average of weight 0.3 (:func:`ema`).
  Curves are interpolated on a 200-point grid spanning the steps covered by
  every seed (:func:`agg_seeds_se`); table cells are the EMA value at a step
  matched across the compared methods (:func:`matched_step`, :func:`at_step`),
  reported as mean +- standard error (std with ddof=1 / sqrt(n)).

Run directories are found with :func:`seed_runs`, which only accepts a seed
suffix ``_s<digits>`` (release layout) or ``_<digit>`` (paper layout); any other
suffix (e.g. an archived ``walker_s0_old``) is ignored rather than silently mixed
into an aggregate.

``metrics.jsonl`` is appended to when a run is resumed, so a run that was
restarted from scratch into the same directory produces a second curve that the
loaders would sort into the first one. Every file is therefore scanned for
backward step jumps when read; jumps larger than :data:`JUMP_WARN_STEPS` are
reported on stderr (numbers are never altered). Scan a whole tree with

    python analysis/dmc/common.py --scan runs/
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ─── Tasks ───────────────────────────────────────────────────────────────────
# Release task keys (``train.py --task dmc_<task>``), in the column order of the
# paper's figures (alphabetical by the paper's run names).
TASKS = ["cup_catch", "cartpole_swingup", "cheetah_run",
         "finger_spin", "walker_run", "walker_walk"]
# Directory names used by the paper's own runs.
PAPER_TASK = {"cup_catch": "ball_in_cup_catch"}
TASK_LABELS = {
    "cup_catch": "Ball In Cup\nCatch", "cartpole_swingup": "Cartpole\nSwingup",
    "cheetah_run": "Cheetah\nRun", "finger_spin": "Finger\nSpin",
    "walker_run": "Walker\nRun", "walker_walk": "Walker\nWalk",
}
# StarWM-R: per-task weight of the attention reward head (``--attn_reward_scale``).
STARWM_R_TASK_SCALE = {
    "cup_catch": 100, "cartpole_swingup": 50, "cheetah_run": 50,
    "finger_spin": 30, "walker_run": 30, "walker_walk": 30,
}


def paper_task(task):
    """Release task key -> name used in the paper's run directories."""
    return PAPER_TASK.get(task, task)


def release_task(task):
    """Accept either naming and return the release task key."""
    inv = {v: k for k, v in PAPER_TASK.items()}
    task = task[len("dmc_"):] if task.startswith("dmc_") else task
    return inv.get(task, task)


# ─── Seed discovery ──────────────────────────────────────────────────────────
_SEED_SUFFIX = {"s": re.compile(r"_s(\d+)$"), "d": re.compile(r"_(\d)$")}


def seed_runs(prefix, style="s", seeds=None):
    """Seed directories of one cell, as a list of ``(seed, path)`` sorted by seed.

    ``prefix`` is the run path without the seed suffix. ``style="s"`` accepts
    ``<prefix>_s<digits>`` (release layout and the stress-test runs),
    ``style="d"`` accepts ``<prefix>_<one digit>`` (the paper's main runs, seeds
    0-5). Anything else after the prefix is rejected, e.g. ``<prefix>_s0_old``
    or ``<prefix>_noNCE_0`` (a different variant sharing the prefix).
    ``seeds`` optionally restricts the result to the given seed ids.
    """
    pat = _SEED_SUFFIX[style]
    stem = os.path.basename(prefix)
    out = []
    for d in glob.glob(glob.escape(prefix) + "_*"):
        name = os.path.basename(d)
        m = pat.match(name[len(stem):]) if name.startswith(stem) else None
        if m is None or m.start() != 0 or not os.path.isdir(d):
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        out.append((seed, d))
    return sorted(out)


# ─── metrics.jsonl ───────────────────────────────────────────────────────────
JUMP_WARN_STEPS = 50_000   # larger rewinds than a checkpoint interval => likely a restart
_ROWS = {}


def backward_jumps(rows):
    """``[(row_index, previous_step, step), ...]`` where the logged step decreases."""
    out, prev = [], None
    for i, d in enumerate(rows):
        if "step" not in d:
            continue
        s = d["step"]
        if prev is not None and s < prev:
            out.append((i, prev, s))
        prev = s
    return out


def read_jsonl(path):
    """Rows (dicts) of a metrics.jsonl, malformed lines skipped; None if missing.

    Cached per path. Warns once per file about large backward step jumps
    (a run restarted into the same directory).
    """
    if path in _ROWS:
        return _ROWS[path]
    if not os.path.isfile(path):
        _ROWS[path] = None
        return None
    rows = []
    with open(path, "r") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                rows.append(d)
    big = [j for j in backward_jumps(rows) if j[1] - j[2] > JUMP_WARN_STEPS]
    if big:
        i, a, b = max(big, key=lambda j: j[1] - j[2])
        print(f"WARNING: {path}: {len(big)} backward step jump(s) > {JUMP_WARN_STEPS} "
              f"(largest {a} -> {b} at row {i}); the run was probably restarted into "
              f"the same directory and its curves are interleaved.", file=sys.stderr)
    _ROWS[path] = rows
    return rows


def load_step_mean(logdir, key):
    """``(steps, values)`` of ``key``; rows logged at the same step are averaged.

    Steps sorted ascending. ``(None, None)`` if the file or key is missing.
    (Loader of the learning-curve and rliable scripts.)
    """
    rows = read_jsonl(os.path.join(logdir, "metrics.jsonl"))
    if rows is None:
        return None, None
    step_returns = defaultdict(list)
    for d in rows:
        if key in d and "step" in d:
            step_returns[d["step"]].append(d[key])
    if not step_returns:
        return None, None
    steps = sorted(step_returns.keys())
    returns = [np.mean(step_returns[s]) for s in steps]
    return np.array(steps), np.array(returns)


def load_eval_rows(logdir, key="eval_return"):
    """``(steps, values)`` of every row holding ``key`` (missing step -> 0), sorted
    by step with ``np.argsort``; no averaging of duplicates. ``(None, None)`` if
    nothing is found. (Loader of the gradient-isolation figures.)"""
    rows = read_jsonl(os.path.join(logdir, "metrics.jsonl"))
    if rows is None:
        return None, None
    steps, returns = [], []
    for d in rows:
        if key in d:
            steps.append(d.get("step", 0))
            returns.append(d[key])
    if not steps:
        return None, None
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(returns)[order]


# Eval-return keys in precedence order: StarWM / DreamerV3 log ``eval_return``,
# R2-Dreamer logs ``episode/eval_score``. Train-return keys are deliberately
# absent so train and eval rows are never mixed.
RETURN_KEYS = ("eval_return", "eval/return", "eval_return_mean",
               "episode/eval_score", "eval_episode_reward", "return")


def load_series(logdir, key="return"):
    """``(steps, values)`` for rows containing ``key`` (``"return"`` = first of
    :data:`RETURN_KEYS` present in the row), sorted by step; empty arrays if the
    file is missing. (Loader of the EMA-0.3 stress-test scripts.)"""
    rows = read_jsonl(os.path.join(logdir, "metrics.jsonl"))
    s, v = [], []
    if rows is None:
        return np.array([]), np.array([])
    keys = (key,) if key not in ("return",) else RETURN_KEYS
    for d in rows:
        for k in keys:
            if k in d and "step" in d:
                s.append(d["step"]); v.append(d[k]); break
    order = np.argsort(s)
    return np.array(s)[order], np.array(v, dtype=float)[order]


# ─── TensorBoard (DenoisedMDP) ───────────────────────────────────────────────
def has_tfevents(d):
    return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "events.out.tfevents.*")))


def find_tfevents_dir(run_dir):
    """``run_dir`` itself if it holds event files, else the first (sorted) nested
    directory that does; None otherwise."""
    if has_tfevents(run_dir):
        return run_dir
    for c in sorted(glob.glob(os.path.join(glob.escape(run_dir), "**"), recursive=True)):
        if has_tfevents(c):
            return c
    return None


def load_tb_scalar(logdir, tag):
    """``(steps, values)`` of a TensorBoard scalar, sorted by step; (None, None)
    if unavailable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None
    if not has_tfevents(logdir):
        return None, None
    try:
        acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
        acc.Reload()
    except Exception:
        return None, None
    if tag not in acc.Tags().get("scalars", []):
        return None, None
    events = acc.Scalars(tag)
    if not events:
        return None, None
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], values[order]


def load_metric(run_dir, metric):
    """Dispatch on ``metric`` = ``"jsonl:<key>"`` or ``"tb:<tag>"``."""
    kind, key = metric.split(":", 1)
    if kind == "tb":
        return load_tb_scalar(run_dir, key)
    return load_step_mean(run_dir, key)


# ─── Smoothing ───────────────────────────────────────────────────────────────
def moving_average(values, window):
    """Trailing moving average; the series is left-padded with its first value so
    the output has the same length. Series shorter than ``window`` are returned
    unchanged."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def ema(v, alpha=0.3):
    """Exponential moving average ``out[i] = alpha*v[i] + (1-alpha)*out[i-1]``."""
    if len(v) == 0:
        return v
    out = np.empty_like(v, dtype=float); out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


# ─── 4K-grid aggregation (learning curves, Table 5/7/10) ─────────────────────
GRID_STEP = 4000


def collect_curves(run_dirs, metric, window):
    """Load and smooth every seed. Returns ``(all_steps, all_returns)`` for the
    seeds with more than one logged point."""
    all_steps, all_returns = [], []
    for d in run_dirs:
        steps, returns = load_metric(d, metric)
        if steps is not None and len(steps) > 1:
            all_steps.append(steps)
            all_returns.append(moving_average(returns, window))
    return all_steps, all_returns


def interp_to_grid(all_steps, all_returns, step=GRID_STEP):
    """Interpolate seeds onto ``arange(min first step, max last step + 1, step)``.

    Grid points after a seed's last step are NaN for that seed. Returns
    ``(grid, mean, std, min, max, all_seeds_mask)`` with NaN-aware statistics
    over seeds (std is the population std, ddof=0), or None.
    """
    if not all_steps:
        return None
    min_step = min(s[0] for s in all_steps)
    max_step = max(s[-1] for s in all_steps)
    if max_step <= min_step:
        return None
    common = np.arange(min_step, max_step + 1, step)
    interpolated = np.array([np.interp(common, s, r, right=np.nan)
                             for s, r in zip(all_steps, all_returns)])
    mean = np.nanmean(interpolated, axis=0)
    std = np.nanstd(interpolated, axis=0)
    min_vals = np.nanmin(interpolated, axis=0)
    max_vals = np.nanmax(interpolated, axis=0)
    all_seeds_mask = ~np.isnan(interpolated).any(axis=0)
    return common, mean, std, min_vals, max_vals, all_seeds_mask


def final_window(common, mean, std, all_seeds_mask, rule="any", window=10):
    """Table value: mean of the last ``window`` grid points of the seed-mean curve
    (and of the seed-std curve).

    ``rule="any"``: the window ends at the last grid point where at least one
    seed has data (NaN-aware averaging). ``rule="all"``: it ends at the last grid
    point where every seed has data. Returns ``(step, mean, std)`` or None.
    """
    if rule == "any":
        idx = np.where(~np.isnan(mean))[0]
        agg = np.nanmean
    elif rule == "all":
        idx = np.where(all_seeds_mask)[0]
        agg = np.mean
    else:
        raise ValueError(rule)
    if len(idx) == 0:
        return None
    li = idx[-1]
    ws = max(0, li - (window - 1))
    return int(common[li]), float(agg(mean[ws:li + 1])), float(agg(std[ws:li + 1]))


# ─── EMA-0.3 aggregation (stress tests) ──────────────────────────────────────
def last_step(run):
    s, _ = load_series(run, "return")
    return int(s[-1]) if len(s) else 0


def comparable(runs, frac=0.5):
    """Drop seeds trained far less than the longest one (last step < ``frac`` x
    the longest seed's last step), so that an early-stage seed does not clip the
    common curve of :func:`agg_seeds_se`."""
    last = {r: last_step(r) for r in runs}
    hi = max(last.values()) if last else 0
    return [r for r in runs if last[r] >= frac * hi]


def agg_seeds_se(logdirs, key="return", smooth=0.0, grid=None, n_points=200):
    """Interpolate each seed's (optionally EMA-smoothed) series onto a common grid
    of ``n_points`` steps spanning the range covered by every seed. Returns
    ``(grid, mean, se)`` with se = std(ddof=1)/sqrt(n) (0 for a single seed)."""
    series = [load_series(d, key) for d in logdirs]
    series = [(s, v) for s, v in series if len(s) > 1]
    if not series:
        return np.array([]), np.array([]), np.array([])
    if grid is None:
        lo = max(s[0] for s, _ in series)
        hi = min(s[-1] for s, _ in series)
        if hi <= lo:
            hi = max(s[-1] for s, _ in series); lo = min(s[0] for s, _ in series)
        grid = np.linspace(lo, hi, n_points)
    mat = []
    for s, v in series:
        if smooth:
            v = ema(v, smooth)
        mat.append(np.interp(grid, s, v))
    mat = np.vstack(mat)
    mean = mat.mean(0)
    se = mat.std(0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros_like(mean)
    return grid, mean, se


def matched_step(groups):
    """Common step of a comparison: min over groups of the shortest seed's last
    step (``groups`` = lists of run dirs)."""
    return min(min(last_step(r) for r in rs) for rs in groups)


def at_step(runs, step, smooth=0.3):
    """EMA-smoothed return of each seed interpolated at ``step``; seeds whose last
    step is below 0.98 x ``step`` are skipped. Returns ``(mean, se, n)`` or None."""
    vals = []
    for r in runs:
        s, v = load_series(r, "return")
        if len(s) < 2 or s[-1] < step * 0.98:
            continue
        v = ema(v, smooth)
        vals.append(float(np.interp(step, s, v)))
    if not vals:
        return None
    return np.mean(vals), (np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0), len(vals)


# ─── misc ────────────────────────────────────────────────────────────────────
def savefig_all(fig_or_plt, out_base, formats, **kw):
    """Save ``out_base.<ext>`` for each ext in ``formats`` (in that order)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    for ext in formats:
        path = f"{out_base}.{ext}"
        fig_or_plt.savefig(path, **kw)
        print(f"Saved: {path}")


def scan(root, threshold=0):
    """Print every metrics.jsonl under ``root`` with backward step jumps."""
    n = 0
    for p in sorted(glob.glob(os.path.join(glob.escape(root), "**", "metrics.jsonl"),
                              recursive=True)):
        rows = []
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    rows.append(d)
        jumps = [j for j in backward_jumps(rows) if j[1] - j[2] > threshold]
        if jumps:
            n += 1
            i, a, b = max(jumps, key=lambda j: j[1] - j[2])
            print(f"{len(jumps):5d} jump(s), largest {a} -> {b} (row {i})  {p}")
    print(f"{n} file(s) with backward step jumps > {threshold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", nargs="+", metavar="DIR", required=True,
                    help="scan metrics.jsonl files under these directories for backward step jumps")
    ap.add_argument("--threshold", type=int, default=0,
                    help=f"only list jumps larger than this many steps (loaders warn above {JUMP_WARN_STEPS})")
    a = ap.parse_args()
    for r in a.scan:
        scan(r, a.threshold)
