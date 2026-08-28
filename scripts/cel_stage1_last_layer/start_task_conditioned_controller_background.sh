#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
RUNNER_LOG="$RESULT_DIR/task_conditioned_controller_runner.log"
CONTROLLER_STDOUT_LOG="$RESULT_DIR/task_conditioned_controller_stdout.log"
ALIAS_CONTROLLER_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh"
DIRECT_CONTROLLER_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh"
PID_FILE="$RESULT_DIR/task_conditioned_controller.pid"

SSH_ALIAS="${1:-3090}"
SLEEP_SECS="${TASK_CONDITIONED_CONTROLLER_SLEEP_SECS:-600}"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_CONTROLLER_SYNC_TIMEOUT_SECS:-240}"
FAILURE_BACKOFF_SECS="${TASK_CONDITIONED_CONTROLLER_FAILURE_BACKOFF_SECS:-300}"
POST_ACTION_SLEEP_SECS="${TASK_CONDITIONED_CONTROLLER_POST_ACTION_SLEEP_SECS:-60}"

mkdir -p "$RESULT_DIR"

if pgrep -f "$DIRECT_CONTROLLER_SCRIPT" >/dev/null 2>&1; then
  echo "task_conditioned controller already running" | tee -a "$RUNNER_LOG"
  exit 0
fi

controller_cmd=(
  env
  TASK_CONDITIONED_CONTROLLER_SLEEP_SECS="$SLEEP_SECS"
  TASK_CONDITIONED_CONTROLLER_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS"
  TASK_CONDITIONED_CONTROLLER_FAILURE_BACKOFF_SECS="$FAILURE_BACKOFF_SECS"
  TASK_CONDITIONED_CONTROLLER_POST_ACTION_SLEEP_SECS="$POST_ACTION_SLEEP_SECS"
)

if [[ -n "$SSH_ALIAS" ]] && ssh -G "$SSH_ALIAS" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] launching task_conditioned controller via alias=$SSH_ALIAS :: sleep=${SLEEP_SECS}s" | tee -a "$RUNNER_LOG"
  controller_cmd+=(bash "$ALIAS_CONTROLLER_SCRIPT" "$SSH_ALIAS")
else
  if [[ -n "$SSH_ALIAS" ]]; then
    echo "[$(date '+%F %T')] alias '$SSH_ALIAS' not resolvable; falling back to direct controller path" | tee -a "$RUNNER_LOG"
  fi
  echo "[$(date '+%F %T')] launching task_conditioned controller via direct ssh_config.sh path :: sleep=${SLEEP_SECS}s" | tee -a "$RUNNER_LOG"
  controller_cmd+=(bash "$DIRECT_CONTROLLER_SCRIPT")
fi

nohup setsid "${controller_cmd[@]}" >> "$CONTROLLER_STDOUT_LOG" 2>&1 < /dev/null &

controller_pid="$!"
disown "$controller_pid" 2>/dev/null || true
printf '%s\n' "$controller_pid" > "$PID_FILE"

echo "[$(date '+%F %T')] launched controller pid=$controller_pid" | tee -a "$RUNNER_LOG"
