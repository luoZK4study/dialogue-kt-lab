#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
CEL_DIR="$ROOT_DIR/results/cel_stage1"
METRICS_DIR="$CEL_DIR/metrics"
QUAL_DIR="$CEL_DIR/qual"
KCS_DIR="$CEL_DIR/kcs"
LOG_FILE="$CEL_DIR/collect_server_results.log"

MODELS=(
  "cel_mlp_qwen3_1.7b"
  "cel_adapter_qwen3_1.7b"
  "cel_task_conditioned_qwen3_1.7b"
)

mkdir -p "$METRICS_DIR" "$QUAL_DIR" "$KCS_DIR"

have_all_results() {
  local model
  for model in "${MODELS[@]}"; do
    [[ -f "$METRICS_DIR/metrics_${model}.txt" ]] || return 1
    [[ -f "$QUAL_DIR/qual_${model}.csv" ]] || return 1
    [[ -f "$KCS_DIR/kcs_${model}.json" ]] || return 1
  done
  return 0
}

log_status() {
  {
    date +"[%F %T] collecting CEL results"
    local model
    for model in "${MODELS[@]}"; do
      echo "  $model:"
      if [[ -f "$METRICS_DIR/metrics_${model}.txt" ]]; then
        echo "    metrics: ready"
      else
        echo "    metrics: waiting"
      fi
      if [[ -f "$QUAL_DIR/qual_${model}.csv" ]]; then
        echo "    qual: ready"
      else
        echo "    qual: waiting"
      fi
      if [[ -f "$KCS_DIR/kcs_${model}.json" ]]; then
        echo "    kcs: ready"
      else
        echo "    kcs: waiting"
      fi
    done
  } >> "$LOG_FILE"
  return 0
}

copy_available_results() {
  return 0
}

write_summary() {
  python "$ROOT_DIR/scripts/cel_stage1/summarize_metrics.py" \
    "$CEL_DIR/metrics/metrics_cel_mlp_qwen3_1.7b.txt" \
    "$CEL_DIR/metrics/metrics_cel_adapter_qwen3_1.7b.txt" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_qwen3_1.7b.txt" \
    > "$CEL_DIR/final_metrics_table.md"
}

update_record() {
  python "$ROOT_DIR/scripts/cel_stage1/update_experiment_record.py" \
    --record "$CEL_DIR/CEL_Stage1_DialogueKT_实验记录.md" \
    --metrics \
    "$CEL_DIR/metrics/metrics_cel_mlp_qwen3_1.7b.txt" \
    "$CEL_DIR/metrics/metrics_cel_adapter_qwen3_1.7b.txt" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_qwen3_1.7b.txt"
}

main() {
  {
    date +"[%F %T] server-side CEL collector started"
    echo "root: $ROOT_DIR"
  } >> "$LOG_FILE"

  while true; do
    if have_all_results; then
      break
    fi
    copy_available_results
    log_status
    sleep 180
  done

  copy_available_results
  write_summary
  update_record

  {
    date +"[%F %T] all CEL results copied and summarized"
    cat "$CEL_DIR/final_metrics_table.md"
  } >> "$LOG_FILE"
}

main "$@"
