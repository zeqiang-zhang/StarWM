#!/usr/bin/env bash
# R2-Dreamer on DMC with the DMC-GB2 video distractors (StarWM paper: Fig. 5, Table 5).
#   vh : temporally coherent video_hard background, 3M env steps
#   vhr: random video_hard frame at every step,       1M env steps
# Usage: dmc.sh {vh|vhr} [TASK ...]
#   TASK in walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch (default: all six)
# Needs DMCGB_DATA_DIR. Output: $RUNS_DIR/baselines/r2dreamer/<vh|vhr>/<task>_s<seed>
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"; shift || true
case "$SETTING" in
  vh)  FLAGS=(env.steps=3.01e6 env.video_hard=True env.video_random_frames=False) ;;
  vhr) FLAGS=(env.steps=1.01e6 env.video_hard=True env.video_random_frames=True) ;;
  *)   die "usage: $0 {vh|vhr} [TASK ...]" ;;
esac
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch)
need_dmcgb
SRC="$(baseline_src r2dreamer)"

# R2-Dreamer splits "<domain>_<task>" at the last underscore: Ball-in-cup is ball_in_cup_catch.
r2_task() { if [ "$1" = cup_catch ]; then echo ball_in_cup_catch; else echo "$1"; fi; }

train() {  # SEED RUN_DIR TASK
  run_expect "$2/latest.pt" in_dir "$SRC" "$PYTHON" train.py env=dmc_vision \
    env.task="dmc_$(r2_task "$3")" "${FLAGS[@]}" seed="$1" logdir="$2" ${EXTRA[@]+"${EXTRA[@]}"}
}

for task in "${TASKS[@]}"; do
  for_each_seed "$RUNS_DIR/baselines/r2dreamer/$SETTING/$task" train "$task"
done
