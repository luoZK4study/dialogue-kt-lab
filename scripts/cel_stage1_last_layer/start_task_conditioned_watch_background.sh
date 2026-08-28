#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
RUNNER_LOG="$RESULT_DIR/task_conditioned_watch_runner.log"
WATCH_STDOUT_LOG="$RESULT_DIR/task_conditioned_watch_stdout.log"
WATCH_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate.sh"
PID_FILE="$RESULT_DIR/task_conditioned_watch.pid"

SLEEP_SECS="${WATCH_SLEEP_SECS:-600}"
FAST_SLEEP_SECS="${WATCH_FAST_SLEEP_SECS:-300}"
SYNC_TIMEOUT_SECS="${WATCH_SYNC_TIMEOUT_SECS:-240}"

mkdir -p "$RESULT_DIR"

if pgrep -f "$WATCH_SCRIPT" >/dev/null 2>&1; then
  echo "task_conditioned watcher already running" | tee -a "$RUNNER_LOG"
  exit 0
fi

echo "[$(date '+%F %T')] launching task_conditioned watcher via monitor_formal_candidate.sh watch :: sleep=${SLEEP_SECS}s fast_sleep=${FAST_SLEEP_SECS}s" | tee -a "$RUNNER_LOG"
nohup setsid env \
  TASK_CONDITIONED_MONITOR_SLEEP_SECS="$SLEEP_SECS" \
  TASK_CONDITIONED_MONITOR_FAST_SLEEP_SECS="$FAST_SLEEP_SECS" \
  TASK_CONDITIONED_MONITOR_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" \
  bash "$WATCH_SCRIPT" watch >> "$WATCH_STDOUT_LOG" 2>&1 < /dev/null &

watch_pid="$!"
disown "$watch_pid" 2>/dev/null || true
printf '%s\n' "$watch_pid" > "$PID_FILE"

echo "[$(date '+%F %T')] launched watcher pid=$watch_pid" | tee -a "$RUNNER_LOG"
