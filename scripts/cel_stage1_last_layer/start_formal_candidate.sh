#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIFF_DOC="$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md"
CONTEXT_HELPER="$ROOT_DIR/scripts/cel_stage1_last_layer/print_formal_candidate_context.py"
READINESS_HELPER="$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_rerun_readiness.py"
MANUAL_FALLBACK_HELPER="$ROOT_DIR/scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

CANDIDATE_KEY="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/start_formal_candidate.sh <candidate_key>

Purpose:
  Run the strict full-train start sequence for one formal candidate:
  1. preflight
  2. SSH launch

Examples:
  bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh
  bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 v21
  bash scripts/cel_stage1_last_layer/start_formal_candidate.sh v21
  TASK_CONDITIONED_SSH_TARGET=3090 TASK_CONDITIONED_SSH_OPTS= bash scripts/cel_stage1_last_layer/start_formal_candidate.sh v21
EOF
}

if [[ -z "$CANDIDATE_KEY" ]]; then
  usage >&2
  exit 2
fi

formal_candidate_require_key "$CANDIDATE_KEY" || {
  usage >&2
  exit 2
}

MODEL_NAME="${FORMAL_MODEL_NAME[$CANDIDATE_KEY]}"

echo "== Strict Formal Start =="
echo "candidate_key=$CANDIDATE_KEY"
echo "model_name=$MODEL_NAME"
echo "candidate_brief=$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md"
echo "candidate_config_diff=$CONFIG_DIFF_DOC"
echo

python3 "$CONTEXT_HELPER" "$CANDIDATE_KEY"
echo
python3 "$READINESS_HELPER" "$CANDIDATE_KEY"
echo
bash "$ROOT_DIR/scripts/cel_stage1_last_layer/preflight_strict_full_train.sh"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_guard.py" --mode assert-launch --candidate-label "$CANDIDATE_KEY"
echo
if ! bash "$ROOT_DIR/scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh" "$CANDIDATE_KEY"; then
  echo
  echo "Strict formal SSH launch failed in the current environment."
  echo "Manual fallback steps:"
  bash "$MANUAL_FALLBACK_HELPER" "$CANDIDATE_KEY"
  exit 1
fi
