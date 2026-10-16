# Shared helpers for the baseline launchers in baselines/<method>/scripts/.
# Source it; do not execute it.
#
# Environment variables (all optional unless a launcher says otherwise):
#   BASELINES_SRC   directory holding the patched checkouts made by
#                   setup_baselines.sh (default: baselines/upstream)
#   RUNS_DIR        output root (default: ./runs); runs are written to
#                   $RUNS_DIR/baselines/<method>/<setting>/<task>_s<seed>
#   SEEDS           seeds to run (default: "0 1 2")
#   PARALLEL        1 = start all seeds of a task concurrently, as for the paper
#                   (3 seeds packed on one GPU); 0 = one after another (default)
#   PYTHON          python executable (default: python)
#   EXTRA_ARGS      extra arguments appended to every training command, e.g.
#                   EXTRA_ARGS="model.compile=False" for R2-Dreamer (default: none)
#   DMCGB_DATA_DIR  DMC-GB2 data dir containing video_hard/video{0..99}.mp4 (VH/VHR)
#   DAVIS_DIR       DAVIS 2017 JPEGImages/480p directory (NoisyShape, Reacher)
# Select the GPU with CUDA_VISIBLE_DEVICES.

set -euo pipefail

BASELINES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_SRC="${BASELINES_SRC:-$BASELINES_ROOT/upstream}"
RUNS_DIR="${RUNS_DIR:-runs}"
case "$RUNS_DIR" in /*) ;; *) RUNS_DIR="$PWD/$RUNS_DIR" ;; esac  # absolute: launchers cd
SEEDS="${SEEDS:-0 1 2}"
PARALLEL="${PARALLEL:-0}"
PYTHON="${PYTHON:-python}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Path of a patched checkout (r2dreamer | isodream | denoised_mdp).
baseline_src() {
  local d="$BASELINES_SRC/$1"
  [ -d "$d" ] || die "$d not found. Run baselines/setup_baselines.sh first (or set BASELINES_SRC)."
  echo "$d"
}

need_dmcgb() {
  [ -n "${DMCGB_DATA_DIR:-}" ] && [ -d "$DMCGB_DATA_DIR/video_hard" ] \
    || die "set DMCGB_DATA_DIR to the DMC-GB2 data directory (must contain video_hard/)."
  export DMCGB_DATA_DIR
}

need_davis() {
  [ -n "${DAVIS_DIR:-}" ] && [ -d "$DAVIS_DIR" ] \
    || die "set DAVIS_DIR to the DAVIS 2017 JPEGImages/480p directory."
  export DAVIS_DIR
}

need_shapeenv() {
  "$PYTHON" -c "import shapeenv" 2>/dev/null \
    || die "the shapeenv package is not importable (pip install -e <path to shapeenv>)."
}

# in_dir DIR CMD...: run CMD inside DIR (in a subshell).
in_dir() { (cd "$1" && shift && exec "$@"); }

# run_expect FILE CMD...: run CMD; succeed iff it (re)wrote FILE. Used for R2-Dreamer,
# whose train.py exits non-zero at interpreter shutdown (environment-worker teardown)
# even after a complete run; latest.pt is written only at the very end of training.
run_expect() {
  local file="$1" stamp rc=0; shift
  stamp="$(mktemp)"
  "$@" || echo "(exit code $? ignored; checking for $file)"
  [ "$file" -nt "$stamp" ] || rc=1
  rm -f "$stamp"
  return "$rc"
}

# for_each_seed RUN_BASE FN [ARGS...]
#   For every seed in $SEEDS calls `FN SEED RUN_DIR [ARGS...]` with
#   RUN_DIR=RUN_BASE_s<seed>; stdout/stderr go to RUN_DIR.log.
for_each_seed() {
  local base="$1" fn="$2"; shift 2
  local s out fail=0 pids=()
  mkdir -p "$(dirname "$base")"
  for s in $SEEDS; do
    out="${base}_s${s}"
    echo "[$(date '+%F %T')] seed $s -> $out  (log: $out.log)"
    if [ "$PARALLEL" = "1" ]; then
      "$fn" "$s" "$out" "$@" > "$out.log" 2>&1 &
      pids+=("$!")
    else
      "$fn" "$s" "$out" "$@" > "$out.log" 2>&1 || { echo "  seed $s FAILED (see $out.log)"; fail=$((fail + 1)); }
    fi
  done
  for p in "${pids[@]+"${pids[@]}"}"; do
    wait "$p" || fail=$((fail + 1))
  done
  [ "$fail" -eq 0 ] || die "$fail seed(s) failed under $(dirname "$base")"
}
