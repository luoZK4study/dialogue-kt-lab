#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh

Purpose:
  Resolve the current launchable strict formal candidate from authoritative
  local state, then print the manual SSH fallback launch steps for that
  candidate.
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

echo "== Current Formal Manual Launch =="
echo "current_formal_decision=$decision"
echo "suggested_launch_key=$launch_key"

if [[ "$launch_key" == "none" ]]; then
  echo
  echo "No current manual-launch candidate is available from authoritative local state."
  echo "Inspect the current helper output:"
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
  exit 1
fi

echo
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
echo
bash "$ROOT_DIR/scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh" "$launch_key"
