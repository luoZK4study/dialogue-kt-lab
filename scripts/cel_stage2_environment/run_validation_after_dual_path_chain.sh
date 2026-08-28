#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
ROUND_ID="${1:-}"
CANDIDATE_ID="${2:-}"
LAUNCHER_PID="${3:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "$ROUND_ID" || -z "$CANDIDATE_ID" || -z "$LAUNCHER_PID" ]]; then
  echo "usage: $0 <round_id> <candidate_id> <launcher_pid>" >&2
  exit 2
fi
if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing Stage 2 validation outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
RESULT_ROOT="$ROOT_DIR/results/cel_stage2_environment/dual_path/$ROUND_ID"
LAUNCHER_LOG="$RESULT_ROOT/launcher.stdout.log"
SUCCESS_MARKER="END $ROUND_ID training chain"

while ! grep -Fq "$SUCCESS_MARKER" "$LAUNCHER_LOG"; do
  if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] training chain ended without success marker; skipping validation" >&2
    exit 1
  fi
  sleep 60
done

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $ROUND_ID validation"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/cel_stage2_environment/evaluate_dual_path_candidate.py \
  "$ROUND_ID" "$CANDIDATE_ID" validation --cuda "$GPU_ID"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $ROUND_ID validation audit"
python scripts/cel_stage2_environment/audit_dual_path_candidate.py \
  "$ROUND_ID" "$CANDIDATE_ID" --require-validation

echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $ROUND_ID validation and audit"
