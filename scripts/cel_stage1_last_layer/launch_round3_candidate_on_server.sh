#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"
SYNC_CODE_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/sync_code_to_server.sh"
QUEUE_GUARD_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_guard.py"
CONTEXT_HELPER="$ROOT_DIR/scripts/cel_stage1_last_layer/print_formal_candidate_context.py"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

CANDIDATE="${1:-}"
SYNC_CODE_BEFORE_LAUNCH="${TASK_CONDITIONED_SYNC_CODE_BEFORE_LAUNCH:-1}"
FORCE_RERUN="${TASK_CONDITIONED_FORCE_RERUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh <candidate_key>

Environment:
  TASK_CONDITIONED_SYNC_CODE_BEFORE_LAUNCH=1  Sync current local code before launch (default: 1)
  TASK_CONDITIONED_FORCE_RERUN=1              Allow relaunch even if synced metrics already exist
  TASK_CONDITIONED_GPU_ID=<id>                Override the default GPU for the selected candidate
  TASK_CONDITIONED_SSH_TARGET=<target>        Override SSH target, e.g. 3090
  TASK_CONDITIONED_SSH_OPTS=<opts>            Override SSH opts; set empty when using SSH config alias
  TASK_CONDITIONED_REMOTE_ROOT=<path>         Override remote repo root
EOF
}

if [[ -z "$CANDIDATE" ]]; then
  usage >&2
  exit 2
fi

formal_candidate_require_key "$CANDIDATE" || {
  usage >&2
  exit 2
}

MODEL_NAME="${FORMAL_MODEL_NAME[$CANDIDATE]}"
MODEL_PATTERN="${FORMAL_MODEL_PATTERN[$CANDIDATE]}"
REMOTE_SCRIPT="${FORMAL_RUN_SCRIPT_REL[$CANDIDATE]}"
REMOTE_LOG="${FORMAL_STDOUT_LOG_REL[$CANDIDATE]}"
REMOTE_METRICS="${FORMAL_METRICS_REL[$CANDIDATE]}"
GPU_ID="${TASK_CONDITIONED_GPU_ID:-${FORMAL_DEFAULT_GPU_ID[$CANDIDATE]}}"
REMOTE_HELPER="scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh"

python3 "$QUEUE_GUARD_SCRIPT" --mode assert-launch --candidate-label "$CANDIDATE"
python3 "$CONTEXT_HELPER" "$CANDIDATE"

echo "Preparing Round 3 launch for $MODEL_NAME on GPU $GPU_ID"

if [[ "$SYNC_CODE_BEFORE_LAUNCH" == "1" ]]; then
  echo "Syncing local code to SSH before launch"
  bash "$SYNC_CODE_SCRIPT"
fi

if ssh_run "$SSH_TARGET" "ps -ef | grep -E '$MODEL_PATTERN' | grep -v grep >/dev/null"; then
  echo "Remote process already running for $MODEL_NAME; refusing duplicate launch" >&2
  exit 1
fi

if ssh_run "$SSH_TARGET" "test -e '$REMOTE_ROOT/$REMOTE_METRICS'"; then
  if [[ "$FORCE_RERUN" != "1" ]]; then
    echo "Remote metrics already exist for $MODEL_NAME; set TASK_CONDITIONED_FORCE_RERUN=1 to relaunch explicitly" >&2
    exit 1
  fi
fi

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
TASK_CONDITIONED_GPU_ID="$GPU_ID" \
TASK_CONDITIONED_FORCE_RERUN="$FORCE_RERUN" \
TASK_CONDITIONED_EXPECTED_REMOTE_ROOT="$REMOTE_ROOT" \
  bash "$REMOTE_HELPER" "$CANDIDATE"
EOF
)"

ssh_run "$SSH_TARGET" "$REMOTE_CMD"

echo "Launched $MODEL_NAME on SSH"
echo "Next steps:"
echo "  1. Monitor with bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once"
echo "  2. Finalize closeout with bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh $CANDIDATE"
echo "  3. Review from rebuilt authoritative surfaces with bash scripts/cel_stage1_last_layer/review_formal_candidate.sh $CANDIDATE"
