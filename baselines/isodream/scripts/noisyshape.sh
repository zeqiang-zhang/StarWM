#!/usr/bin/env bash
# Iso-Dream++ on NoisyShape + imagination probes (StarWM paper: Fig. 3, Sec. 4.1).
# `noisyshape` config = the official `dmc` hyper-parameters with discrete (one-hot)
# actions, 200-step episodes, no action repeat, 2e5 env steps.
# Usage: noisyshape.sh [train|probe|all]   (default: all)
# Needs DAVIS_DIR and the shapeenv package.
# Output: $RUNS_DIR/baselines/isodream/noisyshape/noisyshape_s<seed>[/probe_results/results.json]
#   results.json holds two feature sets: "controllable" ([stoch; deter], Iso-Dream++ (ctrl)
#   in Fig. 3) and "combined" (+ the non-controllable branch, Iso-Dream++ (full)).
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

STAGE="${1:-all}"
need_davis
need_shapeenv
SRC="$(baseline_src isodream)/iso_rl"
BASE="$RUNS_DIR/baselines/isodream/noisyshape/noisyshape"

train() {  # SEED RUN_DIR
  (cd "$SRC" && exec "$PYTHON" -u dreamer.py --configs defaults noisyshape \
     --logdir "$2" --seed "$1" --device cuda:0 --steps 200000 --davis_path "$DAVIS_DIR" ${EXTRA[@]+"${EXTRA[@]}"})
}

probe() {  # SEED RUN_DIR
  (cd "$SRC" && exec "$PYTHON" run_probes_isodream.py --logdir "$2" --device cuda:0 \
     --davis_path "$DAVIS_DIR")
}

probe_all() {
  local s
  for s in $SEEDS; do
    echo "[$(date '+%F %T')] probe seed $s"
    probe "$s" "${BASE}_s${s}" > "${BASE}_s${s}.probe.log" 2>&1 \
      || die "probe failed for seed $s (see ${BASE}_s${s}.probe.log)"
  done
}

case "$STAGE" in
  train) for_each_seed "$BASE" train ;;
  probe) probe_all ;;
  all)   for_each_seed "$BASE" train; probe_all ;;
  *)     die "usage: $0 [train|probe|all]" ;;
esac
