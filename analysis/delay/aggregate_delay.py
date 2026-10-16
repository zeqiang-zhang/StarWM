#!/usr/bin/env python3
"""Delayed action: Table 19 (Appendix F.5) from the per-run sweep records.

Each run of the sweep leaves one text record <sweep_dir>/k<k>_<arm>_s<seed>.txt
(written by run_delay_sweep.sh):
    ARM k=1 arm=shifted seed=0
      [h;z]           : color=1.000 shape=0.998 size=1.000
      pooled_for_attn : color=0.999 shape=0.930 size=0.992
    delay k=1: object_area=0.067 | COMBINED on-obj 0.493 (7.4x) | BEST-query 0.842 (12.6x)
This parses them into <sweep_dir>/aggregate.json
    {"k1_shifted": [{"seed", "hz_color", "hz_shape", "hz_size", "pl_shape",
                     "area", "best", "xuni"}, ...], ...}
(values exactly as printed, i.e. rounded to 3 / 1 decimals) and prints
Table 19: BEST-query attention mass (mean +- SE), its xUniform (mean of the
per-seed ratios), and [h;z] shape / size accuracy (mean +- SE). Arm names
"adjacent"/"nofix" and "shifted"/"fixshift" are both accepted.

Example:
  python analysis/delay/aggregate_delay.py --sweep_dir runs/delay_sweep
"""
import argparse
import glob
import json
import os
import re

import numpy as np

RE_HEAD = re.compile(r"ARM k=(\d+) arm=(\S+) seed=(\d+)")
RE_FEAT = re.compile(r"^\s*(\[h;z\]|pooled_for_attn)\s*: color=([\d.]+) shape=([\d.]+) size=([\d.]+)")
RE_ATTN = re.compile(r"object_area=([\d.]+) \| COMBINED on-obj ([\d.]+) \(([\d.]+)x\) \| "
                     r"BEST-query ([\d.]+) \(([\d.]+)x\)")
ARM_LABEL = {"adjacent": "Adjacent", "nofix": "Adjacent", "shifted": "Shifted", "fixshift": "Shifted"}


def parse(path):
    rec, key = {}, None
    for line in open(path):
        m = RE_HEAD.search(line)
        if m:
            key = f"k{m.group(1)}_{m.group(2)}"
            rec["seed"] = int(m.group(3))
            continue
        m = RE_FEAT.search(line)
        if m:
            if m.group(1) == "[h;z]":
                rec["hz_color"], rec["hz_shape"], rec["hz_size"] = map(float, m.group(2, 3, 4))
            else:
                rec["pl_shape"] = float(m.group(3))
            continue
        m = RE_ATTN.search(line)
        if m:
            rec["area"], rec["best"], rec["xuni"] = float(m.group(1)), float(m.group(4)), float(m.group(5))
    missing = {"seed", "hz_color", "hz_shape", "hz_size", "pl_shape", "area", "best", "xuni"} - set(rec)
    if key is None or missing:
        raise ValueError(f"{path}: incomplete record (missing {sorted(missing) or 'header'})")
    order = ["seed", "hz_color", "hz_shape", "hz_size", "pl_shape", "area", "best", "xuni"]
    return key, {k: rec[k] for k in order}


def aggregate(sweep_dir):
    agg = {}
    for p in sorted(glob.glob(os.path.join(sweep_dir, "k*_*_s*.txt"))):
        key, rec = parse(p)
        agg.setdefault(key, []).append(rec)
    for key in agg:
        agg[key].sort(key=lambda r: r["seed"])
    return agg


def _ms(v):
    a = np.asarray(v, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan"))


def table(agg):
    def sort_key(key):
        k, arm = key[1:].split("_", 1)
        return int(k), ARM_LABEL.get(arm, arm) != "Adjacent", arm
    print("\nTable 19 (mean +- SE over seeds; xUniform = mean of per-seed ratios)")
    print(f"{'k':>2s}  {'window':9s} {'n':>2s}  {'area':>5s}  {'attn mass':>14s} {'xUni':>6s}  "
          f"{'shape':>14s}  {'size':>14s}  {'color':>6s}")
    for key in sorted(agg, key=sort_key):
        rows = agg[key]
        k, arm = key[1:].split("_", 1)
        win = "Reference" if k == "0" else ARM_LABEL.get(arm, arm)

        def fmt(field):
            m, s = _ms([r[field] for r in rows])
            return f"{m:.3f}" + ("" if np.isnan(s) else f" +- {s:.3f}")
        xuni = np.mean([r["xuni"] for r in rows])
        area = np.mean([r["area"] for r in rows])
        col = np.mean([r["hz_color"] for r in rows])
        print(f"{k:>2s}  {win:9s} {len(rows):2d}  {area:5.3f}  {fmt('best'):>14s} {xuni:5.1f}x  "
              f"{fmt('hz_shape'):>14s}  {fmt('hz_size'):>14s}  {col:6.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep_dir", default="runs/delay_sweep")
    ap.add_argument("--out", default=None, help="default <sweep_dir>/aggregate.json")
    args = ap.parse_args()
    agg = aggregate(args.sweep_dir)
    if not agg:
        raise SystemExit(f"no k*_*_s*.txt records in {args.sweep_dir}")
    out = args.out or os.path.join(args.sweep_dir, "aggregate.json")
    with open(out, "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", out)
    table(agg)


if __name__ == "__main__":
    main()
