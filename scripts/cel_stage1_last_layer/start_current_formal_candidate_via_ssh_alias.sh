#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${1:-3090}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh [ssh_alias]

Purpose:
  Resolve the current launchable strict formal candidate from authoritative
  local state, then launch it through the SSH alias wrapper.

Examples:
  bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh
  bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

helper_output="$(
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py" --format env
)"

declare -A STATE=()
while IFS='=' read -r key value; do
  [[ -n "$key" ]] || continue
  STATE["$key"]="$value"
done <<<"$helper_output"

decision="${STATE[current_formal_decision]:-inspect_authoritative_surfaces}"
launch_key="${STATE[suggested_launch_key]:-none}"

echo "== Current Formal Alias Start =="
echo "ssh_alias=$SSH_ALIAS"
echo "current_formal_decision=$decision"
echo "suggested_launch_key=$launch_key"

if [[ "$launch_key" == "none" ]]; then
  echo
  echo "No current launch candidate is available from authoritative local state."
  echo "Inspect the current helper output:"
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
  exit 1
fi

echo
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
echo
bash "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS" "$launch_key"
