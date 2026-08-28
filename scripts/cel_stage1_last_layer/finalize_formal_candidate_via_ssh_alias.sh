#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ALIAS="3090"

SSH_ALIAS="${1:-}"
CANDIDATE_KEY="${2:-}"
SKIP_REFRESH="${TASK_CONDITIONED_FINALIZE_SKIP_REFRESH:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh <ssh_alias> <candidate_key>

Purpose:
  Run the explicit strict formal finalize wrapper through a local SSH config
  alias without manually exporting SSH override environment variables.

Examples:
  bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v21
  TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh --default-3090 v22

Notes:
  - This wrapper sets `TASK_CONDITIONED_SSH_TARGET=<ssh_alias>` and
    `TASK_CONDITIONED_SSH_OPTS=` so the strict loop uses the local SSH config.
  - When `TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1`, the wrapper skips the SSH
    config validation because closeout stays on the current local snapshot.
EOF
}

if [[ -z "$SSH_ALIAS" || -z "$CANDIDATE_KEY" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SSH_ALIAS" == "--default-3090" ]]; then
  SSH_ALIAS="$DEFAULT_ALIAS"
fi

echo "== Strict Formal Alias Finalize Candidate =="
echo "ssh_alias=$SSH_ALIAS"
echo "candidate_key=$CANDIDATE_KEY"
echo "skip_refresh=$SKIP_REFRESH"
echo

if [[ "$SKIP_REFRESH" != "1" ]]; then
  TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
  TASK_CONDITIONED_SSH_OPTS= \
    bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"
  echo
fi

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh" "$CANDIDATE_KEY"
