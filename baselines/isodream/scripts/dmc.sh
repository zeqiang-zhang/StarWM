#!/usr/bin/env bash
# Iso-Dream++ on DMC with the DMC-GB2 video distractors (StarWM paper: Fig. 5, Table 5).
# Official `dmc` config (action_repeat 2); --steps counts env steps.
#   vh : temporally coherent video_hard background, 3M env steps
#   vhr: random video_hard frame at every step,       1M env steps
# Usage: dmc.sh {vh|vhr} [TASK ...]
#   TASK in walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch (default: all six)
# Needs DMCGB_DATA_DIR. Output: $RUNS_DIR/baselines/isodream/<vh|vhr>/<task>_s<seed>
# Iso-Dream resumes from <run>/latest_model.pt and the stored episodes if restarted.
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"; shift || true
case "$SETTING" in
  vh)  FLAGS=(--steps 3000000 --use_video_hard True) ;;
  vhr) FLAGS=(--steps 1000000 --use_video_hard True --video_random_frames True) ;;
  *)   die "usage: $0 {vh|vhr} [TASK ...]" ;;
esac
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch)
need_dmcgb
SRC="$(baseline_src isodream)/iso_rl"

# Iso-Dream maps the domain "cup" to ball_in_cup, so Ball-in-cup Catch is dmc_cup_catch.
train() {  # SEED RUN_DIR TASK
  (cd "$SRC" && exec "$PYTHON" -u dreamer.py --configs defaults dmc --task "dmc_$3" \
     --logdir "$2" --seed "$1" --device cuda:0 "${FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"})
}

for task in "${TASKS[@]}"; do
  for_each_seed "$RUNS_DIR/baselines/isodream/$SETTING/$task" train "$task"
done
