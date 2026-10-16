#!/usr/bin/env bash
# R2-Dreamer on the small-object DMC variants (StarWM paper: App. "Spatial Granularity &
# Small Objects"). The `_small` tasks widen the side-camera field of view so the agent
# appears at half its linear size; physics, reward and actions are unchanged.
#   clean: no distractor,                1M env steps
#   vh   : coherent video_hard background, 3M env steps
#   vhr  : random video_hard frames,       1M env steps
# Usage: small_objects.sh {clean|vh|vhr} [TASK ...]
#   TASK in cheetah_run_small walker_walk_small (default: both)
# Needs DMCGB_DATA_DIR for vh/vhr. Output: $RUNS_DIR/baselines/r2dreamer/small_<setting>/<task>_s<seed>
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"; shift || true
case "$SETTING" in
  clean) FLAGS=(env.steps=1.01e6) ;;
  vh)    FLAGS=(env.steps=3.01e6 env.video_hard=True); need_dmcgb ;;
  vhr)   FLAGS=(env.steps=1.01e6 env.video_hard=True env.video_random_frames=True); need_dmcgb ;;
  *)     die "usage: $0 {clean|vh|vhr} [TASK ...]" ;;
esac
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(cheetah_run_small walker_walk_small)
SRC="$(baseline_src r2dreamer)"

train() {  # SEED RUN_DIR TASK
  run_expect "$2/latest.pt" in_dir "$SRC" "$PYTHON" train.py env=dmc_vision env.task="dmc_$3" \
    "${FLAGS[@]}" seed="$1" logdir="$2" ${EXTRA[@]+"${EXTRA[@]}"}
}

for task in "${TASKS[@]}"; do
  for_each_seed "$RUNS_DIR/baselines/r2dreamer/small_$SETTING/$task" train "$task"
done
