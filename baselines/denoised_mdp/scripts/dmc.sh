#!/usr/bin/env bash
# DenoisedMDP on DMC with the DMC-GB2 video distractors (StarWM paper: Fig. 5, Table 5).
# Default DenoisedMDP config; env spec <domain>_<task>_{vh,vhr} (added by the patch).
#   vh : temporally coherent video_hard background, 3M env steps
#   vhr: random video_hard frame at every step,       1M env steps
# learning.test.enable_visualize=False disables test episodes and videos, as in the
# paper runs; returns are the training-episode returns (TensorBoard tag `train_reward`).
# Usage: dmc.sh {vh|vhr} [TASK ...]
#   TASK in walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch (default: all six)
# Needs DMCGB_DATA_DIR. Output: $RUNS_DIR/baselines/denoised_mdp/<vh|vhr>/<task>_s<seed>/
#   dmc_<spec>/<auto-named run folder>/ (DenoisedMDP names the innermost folder itself).
# DenoisedMDP resumes from its last checkpoint if restarted with the same output dir.
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"; shift || true
case "$SETTING" in
  vh)  STEPS=3000000 ;;
  vhr) STEPS=1000000 ;;
  *)   die "usage: $0 {vh|vhr} [TASK ...]" ;;
esac
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(walker_walk walker_run cheetah_run cartpole_swingup finger_spin cup_catch)
need_dmcgb
SRC="$(baseline_src denoised_mdp)"

dmdp_task() { if [ "$1" = cup_catch ]; then echo ball_in_cup_catch; else echo "$1"; fi; }

train() {  # SEED RUN_DIR TASK
  mkdir -p "$2"   # DenoisedMDP requires output_base_dir to exist
  (cd "$SRC" && exec "$PYTHON" -u main.py env.kind=dmc env.spec="$(dmdp_task "$3")_$SETTING" \
     learning.exploration.total_steps=$STEPS learning.test.num_episodes=1 \
     learning.test.enable_visualize=False seed="$1" output_base_dir="$2" ${EXTRA[@]+"${EXTRA[@]}"})
}

for task in "${TASKS[@]}"; do
  for_each_seed "$RUNS_DIR/baselines/denoised_mdp/$SETTING/$task" train "$task"
done
