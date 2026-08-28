#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ALIAS="3090"

SSH_ALIAS="${1:-}"
MODE="${2:-once}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh <ssh_alias> [once|watch|skip-sync]

Purpose:
  Run the unified strict-formal monitor entrypoint through a local SSH config
  alias without manually exporting SSH override environment variables.

Examples:
  bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once
  TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch
  bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh --default-3090 skip-sync

Notes:
  - This wrapper sets `TASK_CONDITIONED_SSH_TARGET=<ssh_alias>` and
    `TASK_CONDITIONED_SSH_OPTS=` so the strict loop uses the local SSH config.
  - `skip-sync` is local-snapshot only and therefore does not require SSH.
EOF
}

if [[ -z "$SSH_ALIAS" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SSH_ALIAS" == "--default-3090" ]]; then
  SSH_ALIAS="$DEFAULT_ALIAS"
fi

case "$MODE" in
  once|watch|skip-sync)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "== Strict Formal Alias Monitor =="
echo "ssh_alias=$SSH_ALIAS"
echo "mode=$MODE"
echo

if [[ "$MODE" != "skip-sync" ]]; then
  TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
  TASK_CONDITIONED_SSH_OPTS= \
    bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"
  echo
fi

TASK_CONDITIONED_SSH_TARGET="$SSH_ALIAS" \
TASK_CONDITIONED_SSH_OPTS= \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate.sh" "$MODE"
