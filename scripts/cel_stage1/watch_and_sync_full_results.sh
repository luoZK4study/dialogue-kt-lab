#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
LOG_FILE="$ROOT_DIR/results/cel_stage1/watch_sync.log"
SYNC_SCRIPT="$ROOT_DIR/scripts/cel_stage1/sync_results_from_server.sh"
SUMMARY_SCRIPT="$ROOT_DIR/scripts/cel_stage1/summarize_metrics.py"

EXPECTED_MODELS=(
  "cel_mlp_qwen3_1.7b"
  "cel_adapter_qwen3_1.7b"
  "cel_task_conditioned_qwen3_1.7b"
)

mkdir -p "$ROOT_DIR/results/cel_stage1"

count_ready() {
  local ready=0
  local model
  for model in "${EXPECTED_MODELS[@]}"; do
    if [[ -f "$ROOT_DIR/results/cel_stage1/metrics/metrics_${model}.txt" ]] \
      && [[ -f "$ROOT_DIR/results/cel_stage1/qual/qual_${model}.csv" ]] \
      && [[ -f "$ROOT_DIR/results/cel_stage1/kcs/kcs_${model}.json" ]]; then
      ready=$((ready + 1))
    fi
  done
  echo "$ready"
}

log_status() {
  {
    date +"[%F %T] syncing CEL full results"
    echo "ready: $(count_ready)/3"
  } >> "$LOG_FILE"
}

main() {
  while true; do
    bash "$SYNC_SCRIPT" >> "$LOG_FILE" 2>&1 || true
    log_status

    if [[ "$(count_ready)" == "3" ]]; then
      {
        date +"[%F %T] all CEL full results synced"
        python "$SUMMARY_SCRIPT" \
          "results/cel_stage1/metrics/metrics_cel_mlp_qwen3_1.7b.txt" \
          "results/cel_stage1/metrics/metrics_cel_adapter_qwen3_1.7b.txt" \
          "results/cel_stage1/metrics/metrics_cel_task_conditioned_qwen3_1.7b.txt" || true
      } >> "$LOG_FILE" 2>&1
      break
    fi

    sleep 180
  done
}

main "$@"
