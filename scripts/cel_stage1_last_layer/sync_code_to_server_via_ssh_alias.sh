#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ALIAS="3090"

SSH_ALIAS="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh <ssh_alias>

Purpose:
  Sync the current strict-formal local code surface to the SSH server through
  a local SSH config alias without manually exporting SSH override variables.

Examples:
  bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh 3090
  bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh --default-3090
EOF
}

if [[ -z "$SSH_ALIAS" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SSH_ALIAS" == "--default-3090" ]]; then
  SSH_ALIAS="$DEFAULT_ALIAS"
fi

echo "== Strict Formal Alias Sync Code =="
echo "ssh_alias=$SSH_ALIAS"
echo

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"

echo

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_code_to_server.sh"
