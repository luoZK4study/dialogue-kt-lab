#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ALIAS="3090"

SSH_ALIAS="${1:-}"
MODE="${2:-run}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh <ssh_alias> [--dry-run]

Purpose:
  Resolve the current strict-formal next action from authoritative local state,
  then execute the corresponding alias-based wrapper in one step.

Examples:
  bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
  bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
  bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh --default-3090

Behavior:
  - `monitor_active_full_train` -> monitor once via alias
  - `launch_next_candidate` -> start current candidate via alias
  - `review_recorded_or_completed_formal_candidate` -> finalize current + review current via alias
  - `design_next_formal_candidate` -> print the next-candidate design guidance and stop
  - `winner_or_done` -> finalize current + review current via alias
  - `freeze_recorded_then_rerun` -> snapshot-finalize recorded non-winner if present, then start current candidate via alias
  - `--dry-run` prints the resolved command(s) without executing them
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$SSH_ALIAS" ]]; then
  usage >&2
  exit 2
fi

if [[ "$SSH_ALIAS" == "--default-3090" ]]; then
  SSH_ALIAS="$DEFAULT_ALIAS"
fi

if [[ "$MODE" != "run" && "$MODE" != "--dry-run" ]]; then
  usage >&2
  exit 2
fi

helper_output="$(
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py" --format env --ssh-alias "$SSH_ALIAS"
)"

declare -A STATE=()
while IFS='=' read -r key value; do
  [[ -n "$key" ]] || continue
  STATE["$key"]="$value"
done <<<"$helper_output"

decision="${STATE[current_formal_decision]:-inspect_authoritative_surfaces}"
launch_key="${STATE[suggested_launch_key]:-none}"
recorded_non_winner="${STATE[current_recorded_non_winner_target]:-none}"

echo "== Current Formal Alias Next Action =="
echo "ssh_alias=$SSH_ALIAS"
echo "current_formal_decision=$decision"
echo "suggested_launch_key=$launch_key"
echo "current_recorded_non_winner_target=$recorded_non_winner"
echo "mode=$MODE"
echo

python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py" --ssh-alias "$SSH_ALIAS"
echo

run_or_print() {
  if [[ "$MODE" == "--dry-run" ]]; then
    printf 'dry_run:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

case "$decision" in
  monitor_active_full_train)
    run_or_print bash "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS" once
    ;;
  launch_next_candidate)
    run_or_print bash "$ROOT_DIR/scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS"
    ;;
  review_recorded_or_completed_formal_candidate|winner_or_done)
    run_or_print bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS"
    if [[ "$MODE" != "--dry-run" ]]; then
      echo
    fi
    run_or_print bash "$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS"
    ;;
  design_next_formal_candidate)
    echo "No alias auto-launch is defined for current_formal_decision=$decision." >&2
    echo "The current formal queue is complete; scaffold the next one-variable candidate first." >&2
    exit 3
    ;;
  freeze_recorded_then_rerun)
    if [[ "$recorded_non_winner" != "none" ]]; then
      if [[ "$MODE" == "--dry-run" ]]; then
        printf 'dry_run: TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1'
        printf ' %q' bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS" "$recorded_non_winner"
        printf '\n'
      else
        TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 \
          bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS" "$recorded_non_winner"
        echo
      fi
    fi
    run_or_print bash "$ROOT_DIR/scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh" "$SSH_ALIAS"
    ;;
  *)
    echo "No executable alias automation is defined for current_formal_decision=$decision." >&2
    echo "Inspect the printed helper output and choose the explicit wrapper manually." >&2
    exit 1
    ;;
esac
