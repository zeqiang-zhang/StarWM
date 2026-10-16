#!/usr/bin/env bash
# Delayed action, Appendix F.5 / Table 19: probe every run of the sweep and aggregate.
#
# For each run directory <RUNS>/k<k>_<arm>_s<seed> (arm = adjacent | shifted) this runs
#   probe_delay_attr.py   (held-out colour/shape/size decodability, [h;z] and pooled_for_attn)
#   viz_attention_delay.py (BEST-query attention mass on the object, xUniform)
# at the matched checkpoint and writes the record <OUT>/k<k>_<arm>_s<seed>.txt, then calls
# aggregate_delay.py (-> <OUT>/aggregate.json + Table 19).
#
# Usage (from the StarWM root, after collect_delay_trajs.py):
#   bash analysis/delay/run_delay_sweep.sh [RUNS=runs/delay] [OUT=runs/delay_sweep] [DEVICE=cuda:0]
# Environment overrides:
#   CKPT     checkpoint of the delayed runs        (default checkpoints/agent_step180000.pt)
#   K0_CKPT  checkpoint of the k=0 reference runs  (default latest.pt; see README)
#   TRAJ     directory with delay_k<k>_traj.npz    (default runs/traj_cache)
#   DAVIS_DIR  DAVIS JPEGImages/480p (needed by viz_attention_delay.py)
set -euo pipefail
RUNS=${1:-runs/delay}
OUT=${2:-runs/delay_sweep}
DEV=${3:-cuda:0}
CKPT=${CKPT:-checkpoints/agent_step180000.pt}
K0_CKPT=${K0_CKPT:-latest.pt}
TRAJ=${TRAJ:-runs/traj_cache}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"

for d in "$RUNS"/k*_*_s*; do
  name=$(basename "$d")
  [[ $name =~ ^k([0-9]+)_([a-z]+)_s([0-9]+)$ ]] || continue
  k=${BASH_REMATCH[1]}; arm=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
  ck=$CKPT; [ "$k" = 0 ] && ck=$K0_CKPT
  if [ ! -f "$d/$ck" ]; then echo "skip $name (no $ck)"; continue; fi
  {
    echo "ARM k=$k arm=$arm seed=$s"
    "$PY" "$HERE/probe_delay_attr.py" --logdir "$d" --k "$k" --checkpoint "$ck" \
        --traj "$TRAJ/delay_k${k}_traj.npz" --device "$DEV" 2>&1 | grep -E "\[h;z\]|pooled_for_attn"
    "$PY" "$HERE/viz_attention_delay.py" --logdir "$d" --k "$k" --checkpoint "$ck" \
        --device "$DEV" --out "$OUT/attention_$name.png" 2>&1 | grep -E "^delay k="
  } > "$OUT/$name.txt"
  echo "done $name"
done

"$PY" "$HERE/aggregate_delay.py" --sweep_dir "$OUT"
