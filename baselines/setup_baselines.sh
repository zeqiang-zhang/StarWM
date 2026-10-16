#!/usr/bin/env bash
# Build the patched baseline checkouts used in the StarWM paper.
#
# For each requested baseline this
#   1. clones the official repository and checks out the pinned commit,
#   2. applies baselines/<name>/<name>.patch (git apply --check first),
#   3. installs the files shared with StarWM from $STARWM_ROOT/envs
#      (video_hard_wrapper.py + the DMC-GB2 dmcgb/ module, and for R2-Dreamer
#      small_variant.py) into the checkout's `envs` package,
#   4. copies the NoisyShape probe scripts (R2-Dreamer, Iso-Dream++).
#
# Usage: baselines/setup_baselines.sh [--target DIR] [r2dreamer] [isodream] [denoised_mdp]
#   default: all three, into $BASELINES_SRC (default: baselines/upstream).
#   Set R2DREAMER_URL / ISODREAM_URL / DENOISED_MDP_URL to clone from a mirror.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARWM_ROOT="${STARWM_ROOT:-$(dirname "$HERE")}"
TARGET="${BASELINES_SRC:-$HERE/upstream}"

R2DREAMER_URL="${R2DREAMER_URL:-https://github.com/NM512/r2dreamer.git}"
R2DREAMER_COMMIT=1fadce46c40dc59cf5a7407e4a296a3c6e4d1416
ISODREAM_URL="${ISODREAM_URL:-https://github.com/panmt/MBRL_with_Isolated_Imaginations.git}"
ISODREAM_COMMIT=3f60cc6bbbf4a8da4f2b48b233868d15064998b5
DENOISED_MDP_URL="${DENOISED_MDP_URL:-https://github.com/facebookresearch/denoised_mdp.git}"
DENOISED_MDP_COMMIT=bc68709750e5a3ea375077f0da6d569bc2217665

die() { echo "ERROR: $*" >&2; exit 1; }

NAMES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    r2dreamer|isodream|denoised_mdp) NAMES+=("$1"); shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ ${#NAMES[@]} -gt 0 ] || NAMES=(r2dreamer isodream denoised_mdp)

for f in envs/video_hard_wrapper.py envs/small_variant.py envs/dmcgb/__init__.py envs/dmcgb/dmcgb_photometric.py; do
  [ -f "$STARWM_ROOT/$f" ] || die "$STARWM_ROOT/$f not found (set STARWM_ROOT to the StarWM repository root)."
done
mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"

clone_at() {  # URL COMMIT DEST
  [ ! -e "$3" ] || die "$3 already exists; remove it or use --target."
  echo "== cloning $1 @ $2 -> $3"
  git clone --quiet "$1" "$3"
  git -C "$3" -c advice.detachedHead=false checkout --quiet "$2"
}

apply_patch() {  # DEST PATCH
  echo "== applying $(basename "$2")"
  git -C "$1" apply --check "$2"
  git -C "$1" apply "$2"
}

# install_shared_envs ENVS_DIR [--small]: copy the StarWM env files into a package dir.
install_shared_envs() {
  local dst="$1"
  mkdir -p "$dst/dmcgb"
  cp "$STARWM_ROOT/envs/video_hard_wrapper.py" "$dst/"
  cp "$STARWM_ROOT/envs/dmcgb/__init__.py" "$STARWM_ROOT/envs/dmcgb/dmcgb_photometric.py" "$dst/dmcgb/"
  [ "${2:-}" != "--small" ] || cp "$STARWM_ROOT/envs/small_variant.py" "$dst/"
  [ -f "$dst/__init__.py" ] || : > "$dst/__init__.py"
  echo "== installed shared env files into $dst"
}

setup_r2dreamer() {
  local d="$TARGET/r2dreamer"
  clone_at "$R2DREAMER_URL" "$R2DREAMER_COMMIT" "$d"
  apply_patch "$d" "$HERE/r2dreamer/r2dreamer.patch"
  install_shared_envs "$d/envs" --small
  cp "$HERE/r2dreamer/probes/run_probes.py" "$d/"
}

setup_isodream() {
  local d="$TARGET/isodream"
  clone_at "$ISODREAM_URL" "$ISODREAM_COMMIT" "$d"
  apply_patch "$d" "$HERE/isodream/isodream.patch"
  install_shared_envs "$d/iso_rl/envs"
  cp "$HERE/isodream/probes/run_probes_isodream.py" "$d/iso_rl/"
}

setup_denoised_mdp() {
  local d="$TARGET/denoised_mdp"
  # The robodesk submodule is not needed for DMC and is not cloned.
  clone_at "$DENOISED_MDP_URL" "$DENOISED_MDP_COMMIT" "$d"
  apply_patch "$d" "$HERE/denoised_mdp/denoised_mdp.patch"
  # Top-level `envs` package next to main.py (imported by denoised_mdp/envs/dmc/simple_dmc.py).
  install_shared_envs "$d/envs"
}

for n in "${NAMES[@]}"; do
  "setup_$n"
done

echo
echo "Done. Checkouts in $TARGET"
[ "$TARGET" = "$HERE/upstream" ] || echo "Launchers look in \$BASELINES_SRC: export BASELINES_SRC=$TARGET"
