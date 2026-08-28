#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
SLEEP_SECS="${WATCH_SLEEP_SECS:-120}"
METRICS_DIR="$ROOT_DIR/results/cel_stage1_last_layer/metrics"
LOG_FILE="$ROOT_DIR/results/cel_stage1_last_layer/task_conditioned_watch.log"

while true; do
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server.sh" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_status_summary.py" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_comparison_report.py" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_detailed_analysis.py" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/update_baseline_record.py" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py" || true
  python "$ROOT_DIR/scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py" || true

  have_v11=0
  have_v12=0
  [[ -f "$METRICS_DIR/metrics_cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b.txt" ]] && have_v11=1
  [[ -f "$METRICS_DIR/metrics_cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b.txt" ]] && have_v12=1
  echo "[$(date '+%F %T')] task_conditioned watcher sync complete :: v11=$have_v11 v12=$have_v12" >> "$LOG_FILE"

  if [[ "$have_v11" -eq 1 && "$have_v12" -eq 1 ]]; then
    echo "[$(date '+%F %T')] both task_conditioned calibration metrics detected; stopping watcher"
    break
  fi

  sleep "$SLEEP_SECS"
done
