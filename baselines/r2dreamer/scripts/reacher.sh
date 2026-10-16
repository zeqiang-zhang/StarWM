#!/usr/bin/env bash
# R2-Dreamer on DMC Reacher Easy with DAVIS video backgrounds (StarWM paper: App.
# "Passive objects", Table "DMC Reacher episode returns"). Reacher's top-down camera
# does not see the video_hard backcube, so DAVIS frames are composited into every
# background pixel (floor / walls / sky, from the segmentation render) instead.
#   vh : coherent DAVIS video (one frame per observation, new video each episode), 3M env steps
#   vhr: random DAVIS frame for every observation,                                1M env steps
# Usage: reacher.sh {vh|vhr}
# Needs DAVIS_DIR. Output: $RUNS_DIR/baselines/r2dreamer/reacher_<setting>/reacher_easy_s<seed>
source "$(dirname "${BASH_SOURCE[0]}")/../../common/launch_lib.sh"

SETTING="${1:-}"
case "$SETTING" in
  vh)  FLAGS=(env.steps=3.01e6 env.pixel_video_bg=True env.video_random_frames=False) ;;
  vhr) FLAGS=(env.steps=1.01e6 env.pixel_video_bg=True env.video_random_frames=True) ;;
  *)   die "usage: $0 {vh|vhr}" ;;
esac
need_davis
SRC="$(baseline_src r2dreamer)"

train() {  # SEED RUN_DIR
  run_expect "$2/latest.pt" in_dir "$SRC" "$PYTHON" train.py env=dmc_vision env.task=dmc_reacher_easy \
    "${FLAGS[@]}" env.distracting_background_path="$DAVIS_DIR" seed="$1" logdir="$2" ${EXTRA[@]+"${EXTRA[@]}"}
}

for_each_seed "$RUNS_DIR/baselines/r2dreamer/reacher_$SETTING/reacher_easy" train
