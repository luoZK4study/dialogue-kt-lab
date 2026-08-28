#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_REMOTE_ROOT="${TASK_CONDITIONED_EXPECTED_REMOTE_ROOT:-/home/user4/dialogue-kt}"
FORCE_RERUN="${TASK_CONDITIONED_FORCE_RERUN:-0}"
CANDIDATE="${1:-}"

source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh <candidate_key>

Environment:
  TASK_CONDITIONED_GPU_ID=<id>                Override the default GPU for the selected candidate
  TASK_CONDITIONED_FORCE_RERUN=1              Allow relaunch even if metrics already exist; backs up old log/metrics first
  TASK_CONDITIONED_EXPECTED_REMOTE_ROOT=...   Expected remote repo root (default: /home/user4/dialogue-kt)
EOF
}

if [[ -z "$CANDIDATE" ]]; then
  usage >&2
  exit 2
fi

if [[ "$ROOT_DIR" != "$EXPECTED_REMOTE_ROOT" ]]; then
  echo "refusing launch outside expected remote repo root: root_dir=$ROOT_DIR expected=$EXPECTED_REMOTE_ROOT" >&2
  exit 1
fi

formal_candidate_require_key "$CANDIDATE" || {
  usage >&2
  exit 2
}

MODEL_NAME="${FORMAL_MODEL_NAME[$CANDIDATE]}"
MODEL_PATTERN="${FORMAL_MODEL_PATTERN[$CANDIDATE]}"
RUN_SCRIPT_REL="${FORMAL_RUN_SCRIPT_REL[$CANDIDATE]}"
STDOUT_LOG_REL="${FORMAL_STDOUT_LOG_REL[$CANDIDATE]}"
METRICS_REL="${FORMAL_METRICS_REL[$CANDIDATE]}"
GPU_ID="${TASK_CONDITIONED_GPU_ID:-${FORMAL_DEFAULT_GPU_ID[$CANDIDATE]}}"

cd "$ROOT_DIR"

if ps -ef | grep -E "$MODEL_PATTERN" | grep -v grep >/dev/null; then
  echo "Remote process already running for $MODEL_NAME; refusing duplicate launch" >&2
  exit 1
fi

if [[ -e "$METRICS_REL" && "$FORCE_RERUN" != "1" ]]; then
  echo "Remote metrics already exist for $MODEL_NAME; set TASK_CONDITIONED_FORCE_RERUN=1 to relaunch explicitly" >&2
  exit 1
fi

mkdir -p results/cel_stage1_last_layer results/cel_stage1_last_layer/metrics
ts="$(date +%Y%m%d_%H%M%S)"
if [[ -e "$STDOUT_LOG_REL" ]]; then
  mv "$STDOUT_LOG_REL" "$STDOUT_LOG_REL.bak.$ts"
fi
if [[ "$FORCE_RERUN" == "1" && -e "$METRICS_REL" ]]; then
  mv "$METRICS_REL" "$METRICS_REL.bak.$ts"
fi

nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$RUN_SCRIPT_REL" > "$STDOUT_LOG_REL" 2>&1 < /dev/null &
pid="$!"

echo "launched_model=$MODEL_NAME"
echo "candidate_key=$CANDIDATE"
echo "gpu_id=$GPU_ID"
echo "pid=$pid"
echo "stdout_log=$STDOUT_LOG_REL"
echo "metrics_path=$METRICS_REL"
