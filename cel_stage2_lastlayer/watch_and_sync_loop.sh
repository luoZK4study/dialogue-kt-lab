#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
LOG_FILE="$ROOT_DIR/results/cel_stage2_lastlayer/watch_sync.log"
SYNC_SCRIPT="$ROOT_DIR/scripts/cel_stage2_lastlayer/sync_results_from_server.sh"
STATUS_SCRIPT="$ROOT_DIR/scripts/cel_stage2_lastlayer/generate_status_summary.py"

mkdir -p "$ROOT_DIR/results/cel_stage2_lastlayer"

while true; do
  bash "$SYNC_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$STATUS_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  latest_event="$(grep -E '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}' "$ROOT_DIR/results/cel_stage2_lastlayer/loop.log" | tail -n 1 || true)"
  echo "[$(date '+%F %T')] synced baseline_recert and cel_stage2_lastlayer :: ${latest_event:-no-loop-event-yet}" >> "$LOG_FILE"
  if grep -q "experiment loop finished" "$ROOT_DIR/results/cel_stage2_lastlayer/loop.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] loop finished; stopping local watcher" >> "$LOG_FILE"
    break
  fi
  sleep 300
done
