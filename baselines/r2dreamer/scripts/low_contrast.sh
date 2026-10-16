#!/usr/bin/env bash
# R2-Dreamer on low-contrast Walker Walk (StarWM paper: App. "Separability", Table
# "Low-contrast Walker Walk return"). recolor_agent sets every agent geom to the
# desaturated RGBA env.recolor_rgba = [0.52, 0.47, 0.42, 1.0] on top of video_hard.
#   vh : coherent video_hard background, 3M env steps
#   vhr: random video_hard frames,       1M env steps
# Usage: low_contrast.sh {vh|vhr}
# Needs DMCGB_DATA_DIR. Output: $RUNS_DIR/baselines/r2dreamer/lowcontrast_<setting>/walker_walk_s<seed>
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"
case "$SETTING" in
  vh)  FLAGS=(env.steps=3.01e6 env.video_hard=True env.video_random_frames=False) ;;
  vhr) FLAGS=(env.steps=1.01e6 env.video_hard=True env.video_random_frames=True) ;;
  *)   die "usage: $0 {vh|vhr}" ;;
esac
need_dmcgb
SRC="$(baseline_src r2dreamer)"

train() {  # SEED RUN_DIR
  run_expect "$2/latest.pt" in_dir "$SRC" "$PYTHON" train.py env=dmc_vision env.task=dmc_walker_walk \
    "${FLAGS[@]}" env.recolor_agent=True seed="$1" logdir="$2" ${EXTRA[@]+"${EXTRA[@]}"}
}

for_each_seed "$RUNS_DIR/baselines/r2dreamer/lowcontrast_$SETTING/walker_walk" train
