#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIRECTION="${1:-}"
GPU_ID="${STAGE2_GPU_ID:-0}"

case "$DIRECTION" in
  shuffle|mlp|transformer) ;;
  *) echo "usage: $0 {shuffle|mlp|transformer}" >&2; exit 2 ;;
esac

if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing remote launch outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p results/cel_stage2_environment
LOG="results/cel_stage2_environment/launcher_${DIRECTION}.stdout.log"
PID_FILE="results/cel_stage2_environment/launcher_${DIRECTION}.pid"

if pgrep -f "run_stage2_candidate.sh $DIRECTION" >/dev/null; then
  echo "Stage 2 $DIRECTION candidate is already running" >&2
  exit 1
fi
if [[ -e "$LOG" ]]; then
  echo "refusing to overwrite launcher log: $LOG" >&2
  exit 1
fi

nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/cel_stage2_environment/run_stage2_candidate.sh "$DIRECTION" \
  > "$LOG" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$PID_FILE"
echo "direction=$DIRECTION"
echo "gpu_id=$GPU_ID"
echo "pid=$pid"
echo "log=$LOG"
