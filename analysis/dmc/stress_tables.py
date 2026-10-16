#!/usr/bin/env python3
"""Tables 9, 14 and 17: hyper-parameter sensitivity, low-contrast agent and
DMC Reacher.

All three use the EMA-0.3 convention: each seed's evaluation-return series
(StarWM / DreamerV3: ``eval_return``; R2-Dreamer: ``episode/eval_score``) is
smoothed with an exponential moving average (weight 0.3 on the new point) and
read off at a step *matched* across the compared methods: the minimum over
methods of the shortest seed's last logged step. Seeds that were trained for
less than half as long as the longest seed of their method are dropped first,
and seeds that stop before 98% of the matched step are skipped. Cells are
mean +- standard error (std with ddof=1 / sqrt(n)).

    Table 9   Walker Walk VHR, StarWM with one hyper-parameter changed
              (runs: linv0.3, linv3.0, bg_noanneal, bg_final2.0)
    Table 14  low-contrast Walker Walk (``--recolor_agent True``), VH and VHR
    Table 17  Reacher easy with DAVIS backgrounds (``--pixel_video_bg True``), VH and VHR

    python analysis/dmc/stress_tables.py [--tables 9 14 17]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import comparable, matched_step, at_step
from layouts import add_layout_args, layout_from_args, dirs

LABEL = {"starwm": "StarWM", "starwm_r": "StarWM-R", "dreamerv3": "DreamerV3",
         "r2dreamer": "R2-Dreamer"}
SENSITIVITY = [("bg_final2.0", "beta_bg final = 2.0 (4x increase)"),
               ("bg_noanneal", "beta_bg annealing removed"),
               ("linv0.3", "lambda_inv = 0.3 (3x reduction)"),
               ("linv3.0", "lambda_inv = 3.0 (3x increase)")]


def compare(groups, title):
    """``groups``: {name: [run dirs]}. Prints and returns {name: (mean, se, n)}."""
    data = {m: comparable(rs) for m, rs in groups.items()}
    data = {m: rs for m, rs in data.items() if rs}
    if not data:
        print(f"\n### {title}: no data")
        return {}, None
    mstep = matched_step(data.values())
    print(f"\n### {title}   matched@{mstep // 1000}k")
    res = {}
    w = max(len(m) for m in data)
    for m, rs in data.items():
        got = at_step(rs, mstep)
        if got:
            res[m] = got
            print(f"   {m:{w}s}: {got[0]:6.0f} ± {got[1]:<4.0f} (n={got[2]})   "
                  f"[{got[0]:.2f} ± {got[1]:.2f}]")
    if res:
        win = max(res, key=lambda k: res[k][0])
        print(f"   -> best at matched step: {win}")
    return res, mstep


def table9(layout):
    print("\n=== Table 9: one-at-a-time perturbations (StarWM, Walker Walk VHR) ===")
    for k, desc in SENSITIVITY:
        print(f"   {k:12s} = {desc}")
    compare({k: dirs(layout.sensitivity(k)) for k, _ in SENSITIVITY},
            "sensitivity (matched step across settings)")


def table14(layout, tasks):
    print("\n=== Table 14: low-contrast agent ===")
    for task in tasks:
        for setting in ("vh", "vhr"):
            compare({LABEL[m]: dirs(layout.lowcontrast(setting, task, m))
                     for m in ("starwm", "dreamerv3", "r2dreamer")},
                    f"low-contrast {task} {setting.upper()}")


def table17(layout):
    print("\n=== Table 17: DMC Reacher ===")
    for setting in ("vh", "vhr"):
        compare({LABEL[m]: dirs(layout.reacher(setting, m))
                 for m in ("starwm", "starwm_r", "dreamerv3", "r2dreamer")},
                f"Reacher {setting.upper()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_layout_args(ap)
    ap.add_argument("--tables", nargs="+", type=int, choices=[9, 14, 17], default=[9, 14, 17])
    ap.add_argument("--lowcontrast_tasks", nargs="+", default=["walker_walk"],
                    help="tasks of Table 14 (the paper reports walker_walk)")
    args = ap.parse_args()
    layout = layout_from_args(args)
    if 9 in args.tables:
        table9(layout)
    if 14 in args.tables:
        table14(layout, args.lowcontrast_tasks)
    if 17 in args.tables:
        table17(layout)


if __name__ == "__main__":
    main()
