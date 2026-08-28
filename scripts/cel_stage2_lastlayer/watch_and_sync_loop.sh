#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
LOG_FILE="$ROOT_DIR/results/cel_stage1_last_layer/watch_sync.log"
SYNC_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server.sh"
STATUS_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/generate_status_summary.py"
COMPARISON_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/generate_comparison_report.py"
DETAILED_ANALYSIS_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/generate_detailed_analysis.py"
BASELINE_RECORD_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/update_baseline_record.py"
STAGE1_RECORD_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py"
SLEEP_SECS="${WATCH_SLEEP_SECS:-60}"
FOLLOWUP_DONE_PATTERN="followup loop found a CEL model that beats baseline|followup loop completed; no CEL model beat baseline by continuation margin|existing CEL result already beats baseline by continuation margin; no followup runs needed|adapter_v2 recovery check beats baseline by continuation margin; followup finished|selector-only round beats baseline by continuation margin; followup finished|early prediction-token batch1 beats baseline by continuation margin; followup finished|early task-conditioned prediction-token batch beats baseline by continuation margin; followup finished|early vector-predshift batch1 beats baseline by continuation margin; followup finished|early task-conditioned vector-predshift batch beats baseline by continuation margin; followup finished|early pre-lm-head vector-predshift batch1 beats baseline by continuation margin; followup finished|early task-conditioned pre-lm-head vector-predshift batch beats baseline by continuation margin; followup finished|direct pre-lm-head selector-only batch1 beats baseline by continuation margin; followup finished|direct task-conditioned pre-lm-head selector-only batch beats baseline by continuation margin; followup finished|scalar-gate followup beats baseline by continuation margin; followup finished|vector-shift batch1 beats baseline by continuation margin; followup finished|vector-shift batch2 beats baseline by continuation margin; followup finished|conservative scalar batch beats baseline by continuation margin; followup finished|conservative low-lr phase3 beats baseline by continuation margin; followup finished|phase4 stabilization batch beats baseline by continuation margin; followup finished|phase5 warm-start batch1 beats baseline by continuation margin; followup finished|phase5 warm-start batch2 beats baseline by continuation margin; followup finished|phase6 prediction-token batch1 beats baseline by continuation margin; followup finished|phase6 task-conditioned prediction-token batch beats baseline by continuation margin; followup finished|baseline metrics missing; exiting followup loop"

mkdir -p "$ROOT_DIR/results/cel_stage1_last_layer"

while true; do
  bash "$SYNC_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$STATUS_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$COMPARISON_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$DETAILED_ANALYSIS_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$BASELINE_RECORD_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  python "$STAGE1_RECORD_SCRIPT" >> "$LOG_FILE" 2>&1 || true
  latest_event="$(grep -E '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}' "$ROOT_DIR/results/cel_stage1_last_layer/loop.log" | tail -n 1 || true)"
  echo "[$(date '+%F %T')] synced baseline and cel_stage1_last_layer :: ${latest_event:-no-loop-event-yet}" >> "$LOG_FILE"
  if grep -Eq "$FOLLOWUP_DONE_PATTERN" "$ROOT_DIR/results/cel_stage1_last_layer/followup.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] followup finished; stopping local watcher" >> "$LOG_FILE"
    break
  fi
  sleep "$SLEEP_SECS"
done
