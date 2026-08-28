#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ALIAS="3090"

SSH_ALIAS="${1:-}"
CANDIDATE_KEY="${2:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh <ssh_alias> <candidate_key>

Purpose:
  Start one strict formal candidate through a local SSH config alias without
  manually exporting SSH override environment variables.

Examples:
  bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 v21
  bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh --default-3090 v21

Notes:
  - This wrapper sets `TASK_CONDITIONED_SSH_TARGET=<ssh_alias>` and
    `TASK_CONDITIONED_SSH_OPTS=` so the strict loop uses the local SSH config.
  - The alias must already resolve correctly under `ssh -G <ssh_alias>`.
  - Use `bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`
    to print the repository's recommended alias block.
EOF
}

if [[ -z "$SSH_ALIAS" || -z "$CANDIDATE_KEY" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SSH_ALIAS" == "--default-3090" ]]; then
  SSH_ALIAS="$DEFAULT_ALIAS"
fi

echo "== Strict Formal Alias Start =="
echo "ssh_alias=$SSH_ALIAS"
echo "candidate_key=$CANDIDATE_KEY"
echo

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"

echo

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate.sh" "$CANDIDATE_KEY"
