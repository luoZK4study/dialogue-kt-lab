#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
DEFAULT_ALIAS="3090"
PID_FILE="$RESULT_DIR/task_conditioned_controller.pid"
RUNNER_LOG="$RESULT_DIR/task_conditioned_controller_runner.log"
STDOUT_LOG="$RESULT_DIR/task_conditioned_controller_stdout.log"
REFRESH_LOG="$RESULT_DIR/task_conditioned_refresh.log"
REGISTRY_SH="$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

SSH_ALIAS="${1:-$DEFAULT_ALIAS}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh [ssh_alias]

Purpose:
  Print one compact health snapshot for the current strict formal runtime:
  - local authoritative formal state
  - background controller pid/aliveness
  - latest local controller / refresh events
  - remote latest progress / metrics presence / GPU usage
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$REGISTRY_SH"

declare -A STATE=()
helper_output="$(
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py" --format env --ssh-alias "$SSH_ALIAS"
)"
while IFS='=' read -r key value; do
  [[ -n "$key" ]] || continue
  STATE["$key"]="$value"
done <<<"$helper_output"

snapshot_key="${STATE[current_active_target]:-none}"
if [[ "$snapshot_key" == "none" ]]; then
  snapshot_key="${STATE[current_completed_target]:-none}"
fi
if [[ "$snapshot_key" == "none" ]]; then
  snapshot_key="${STATE[current_recorded_non_winner_target]:-none}"
fi

snapshot_model="${STATE[current_active_model]:-none}"
if [[ "$snapshot_model" == "none" ]]; then
  snapshot_model="${STATE[current_completed_model]:-none}"
fi
if [[ "$snapshot_model" == "none" ]]; then
  snapshot_model="${STATE[current_recorded_non_winner_model]:-none}"
fi

snapshot_log_rel="none"
snapshot_metrics_rel="none"
if [[ "$snapshot_key" != "none" && -n "${FORMAL_STDOUT_LOG_REL[$snapshot_key]+x}" ]]; then
  snapshot_log_rel="${FORMAL_STDOUT_LOG_REL[$snapshot_key]}"
fi
if [[ "$snapshot_key" != "none" && -n "${FORMAL_METRICS_REL[$snapshot_key]+x}" ]]; then
  snapshot_metrics_rel="${FORMAL_METRICS_REL[$snapshot_key]}"
fi

controller_pid="none"
controller_alive="no"
controller_cmd="none"
controller_child_cmd="none"
controller_next_cycle_estimate="none"
if [[ -f "$PID_FILE" ]]; then
  controller_pid="$(tr -d ' \n\r' < "$PID_FILE")"
  if [[ -n "$controller_pid" && "$controller_pid" != "none" ]] && kill -0 "$controller_pid" 2>/dev/null; then
    controller_alive="yes"
    controller_cmd="$(ps -p "$controller_pid" -o args= 2>/dev/null | sed 's/^[[:space:]]*//')"
    child_pid="$(pgrep -P "$controller_pid" | head -n 1 || true)"
    if [[ -n "$child_pid" ]]; then
      controller_child_cmd="$(ps -p "$child_pid" -o args= 2>/dev/null | sed 's/^[[:space:]]*//')"
    fi
  fi
fi

echo "== Strict Formal Runtime Health =="
echo "generated_at=$(date '+%F %T %z')"
echo "ssh_alias=$SSH_ALIAS"
echo "controller_pid=$controller_pid"
echo "controller_alive=$controller_alive"
echo "controller_cmd=${controller_cmd:-none}"
echo "controller_child_cmd=${controller_child_cmd:-none}"

if [[ -f "$RUNNER_LOG" ]]; then
  echo "latest_controller_runner_event=$(tail -n 1 "$RUNNER_LOG")"
fi
if [[ -f "$STDOUT_LOG" ]]; then
  latest_controller_stdout_event="$(tail -n 1 "$STDOUT_LOG")"
  echo "latest_controller_stdout_event=$latest_controller_stdout_event"
  if [[ "$latest_controller_stdout_event" =~ ^\[([0-9]{4}-[0-9]{2}-[0-9]{2}\ [0-9:]{8})\]\ controller\ sleeping\ ([0-9]+)s ]]; then
    controller_base_epoch="$(date -d "${BASH_REMATCH[1]}" '+%s' 2>/dev/null || true)"
    if [[ -n "${controller_base_epoch:-}" ]]; then
      controller_next_cycle_epoch="$((controller_base_epoch + BASH_REMATCH[2]))"
      controller_next_cycle_estimate="$(date -d "@$controller_next_cycle_epoch" '+%F %T %z' 2>/dev/null || true)"
      controller_next_cycle_estimate="${controller_next_cycle_estimate:-none}"
    fi
  fi
fi
if [[ -f "$REFRESH_LOG" ]]; then
  echo "latest_refresh_event=$(tail -n 1 "$REFRESH_LOG")"
fi
echo "controller_next_cycle_estimate=$controller_next_cycle_estimate"

echo
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
echo

echo "== Remote Runtime Snapshot =="
ssh "$SSH_ALIAS" "python3 - <<'PY' \"$snapshot_key\" \"$snapshot_model\" \"$snapshot_log_rel\" \"$snapshot_metrics_rel\"
import sys
from pathlib import Path

snapshot_key = sys.argv[1]
snapshot_model = sys.argv[2]
snapshot_log_rel = sys.argv[3]
snapshot_metrics_rel = sys.argv[4]
repo_root = Path('/home/user4/dialogue-kt')
stdout_log = repo_root / snapshot_log_rel if snapshot_log_rel != 'none' else None
metrics_path = repo_root / snapshot_metrics_rel if snapshot_metrics_rel != 'none' else None

def latest_line(lines, predicate):
    matches = [line for line in lines if predicate(line)]
    return matches[-1] if matches else 'none'

if stdout_log is not None and stdout_log.exists():
    text = stdout_log.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
else:
    lines = []

print('snapshot_target=' + snapshot_key)
print('snapshot_model=' + snapshot_model)
print('remote_log_present=' + ('yes' if stdout_log is not None and stdout_log.exists() else 'no'))
print('latest_epoch_line=' + latest_line(lines, lambda line: line.startswith('Epoch ')))
print('latest_training_line=' + latest_line(lines, lambda line: 'Training:' in line))
print('latest_validation_line=' + latest_line(lines, lambda line: ('Validation:' in line or 'Validating:' in line)))
print('latest_testing_line=' + latest_line(lines, lambda line: 'Testing:' in line))
print('metrics_file_present=' + ('yes' if metrics_path is not None and metrics_path.exists() else 'no'))
if metrics_path is not None:
    print('metrics_file=' + metrics_path.name)
PY"
echo

echo "== Remote GPU Snapshot =="
ssh "$SSH_ALIAS" "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader" || true
