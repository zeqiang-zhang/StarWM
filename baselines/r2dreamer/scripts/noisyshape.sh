#!/usr/bin/env bash
# R2-Dreamer on NoisyShape + imagination probes (StarWM paper: Fig. 3, Sec. 4.1).
# 2e5 env steps per seed (env/noisyshape.yaml). These runs use the replay-buffer fix
# buffer.clone_shifted_action=True (see ../README.md); the DMC runs do not.
# Usage: noisyshape.sh [train|probe|all] [extra probe args, e.g. --force]   (default: all)
# Needs DAVIS_DIR and the shapeenv package.
# Output: $RUNS_DIR/baselines/r2dreamer/noisyshape/noisyshape_s<seed>[/probe_results/results.json]
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

STAGE="${1:-all}"; shift || true
need_davis
need_shapeenv
SRC="$(baseline_src r2dreamer)"
BASE="$RUNS_DIR/baselines/r2dreamer/noisyshape"

train() {  # SEED RUN_DIR
  run_expect "$2/latest.pt" in_dir "$SRC" "$PYTHON" train.py env=noisyshape buffer.clone_shifted_action=True \
    env.distracting_background_path="$DAVIS_DIR" seed="$1" logdir="$2" ${EXTRA[@]+"${EXTRA[@]}"}
}

probe() {
  # shellcheck disable=SC2086
  (cd "$SRC" && "$PYTHON" run_probes.py --logdir "$BASE" --run_prefix noisyshape \
     --seeds $SEEDS --davis_path "$DAVIS_DIR" --device cuda:0 "$@")
}

case "$STAGE" in
  train) for_each_seed "$BASE/noisyshape" train ;;
  probe) probe "$@" ;;
  all)   for_each_seed "$BASE/noisyshape" train; probe "$@" ;;
  *)     die "usage: $0 [train|probe|all] [extra probe args]" ;;
esac
