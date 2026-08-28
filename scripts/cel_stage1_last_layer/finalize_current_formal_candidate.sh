#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh

Purpose:
  Resolve the current strict formal candidate from authoritative local state,
  then run the standard finalize closeout wrapper for that candidate.
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
closeout_key="${STATE[suggested_finalize_key]:-none}"

echo "== Current Formal Finalize =="
echo "current_formal_decision=$decision"
echo "suggested_finalize_key=$closeout_key"

if [[ "$closeout_key" == "none" ]]; then
  echo
  echo "No current closeout candidate is available from authoritative local state."
  echo "Inspect the current helper output:"
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
  exit 1
fi

echo
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
echo
bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh" "$closeout_key"
