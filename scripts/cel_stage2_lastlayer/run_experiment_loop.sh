#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
BASELINE_DIR="$ROOT_DIR/results/baseline"
CEL_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
LOG_FILE="$CEL_DIR/loop.log"
STEP_LOG_DIR="$CEL_DIR/step_logs"
PRIMARY_CLEAR_MARGIN="${PRIMARY_CLEAR_MARGIN:-0.20}"

mkdir -p "$BASELINE_DIR/metrics" "$BASELINE_DIR/qual" "$BASELINE_DIR/kcs"
mkdir -p "$CEL_DIR/metrics" "$CEL_DIR/qual" "$CEL_DIR/kcs"
mkdir -p "$STEP_LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

step_log_path() {
  local name="$1"
  echo "$STEP_LOG_DIR/${name}.log"
}

extract_overall_auc() {
  python3 - "$1" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"Overall.*?\n.*?\nAcc:\s+[0-9.]+,\s+AUC:\s+([0-9.]+)", text, re.S)
if not m:
    raise SystemExit(1)
print(m.group(1))
PY
}

clear_beats_baseline() {
  python3 - "$1" "$2" "$PRIMARY_CLEAR_MARGIN" <<'PY'
import sys
auc = float(sys.argv[1])
baseline = float(sys.argv[2])
margin = float(sys.argv[3])
sys.exit(0 if auc > baseline + margin else 1)
PY
}

raw_beats_baseline() {
  python3 - "$1" "$2" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
}

run_step() {
  local name="$1"
  shift
  local step_log
  step_log="$(step_log_path "$name")"
  log "START $name"
  : > "$step_log"
  "$@" >> "$step_log" 2>&1
  log "END $name"
}

run_step_if_missing() {
  local name="$1"
  local metrics_file="$2"
  shift 2
  if [[ -s "$metrics_file" ]]; then
    log "SKIP $name (existing metrics: $metrics_file)"
    return
  fi
  run_step "$name" "$@"
}

run_step_async_into() {
  local pid_var="$1"
  local name="$2"
  shift 2
  local step_log
  step_log="$(step_log_path "$name")"
  log "START $name"
  : > "$step_log"
  (
    "$@" >> "$step_log" 2>&1
    local status=$?
    if [[ $status -eq 0 ]]; then
      log "END $name"
    else
      log "FAIL $name (exit=$status)"
    fi
    exit "$status"
  ) &
  local pid=$!
  printf -v "$pid_var" '%s' "$pid"
}

run_step_async_if_missing_into() {
  local pid_var="$1"
  local name="$2"
  local metrics_file="$3"
  shift 3
  if [[ -s "$metrics_file" ]]; then
    log "SKIP $name (existing metrics: $metrics_file)"
    printf -v "$pid_var" ''
    return 0
  fi
  run_step_async_into "$pid_var" "$name" "$@"
}

wait_for_step() {
  local name="$1"
  local pid="$2"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if wait "$pid"; then
    return 0
  fi
  local status=$?
  log "ABORT $name (exit=$status)"
  exit "$status"
}

main() {
  cd "$ROOT_DIR"

  local baseline_metrics="$BASELINE_DIR/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt"
  run_step_if_missing "baseline" "$baseline_metrics" bash scripts/cel_stage1_last_layer/run_baseline.sh
  local baseline_auc
  baseline_auc="$(extract_overall_auc "$baseline_metrics")"
  log "baseline overall auc = $baseline_auc"
  log "primary continuation margin = +$PRIMARY_CLEAR_MARGIN over baseline"

  local mlp_pid adapter_pid
  run_step_async_if_missing_into mlp_pid \
    "cel_mlp_v1" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v1_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v1.sh
  run_step_async_if_missing_into adapter_pid \
    "cel_adapter_v1" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v1_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v1.sh

  wait_for_step "cel_adapter_v1" "$adapter_pid"
  run_step_if_missing "cel_task_conditioned_v1" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_v1.sh
  wait_for_step "cel_mlp_v1" "$mlp_pid"

  local best_model="cel_adapter_lastlayer_v1_qwen3_1.7b"
  local best_auc="0"
  local model
  for model in \
    cel_mlp_lastlayer_v1_qwen3_1.7b \
    cel_adapter_lastlayer_v1_qwen3_1.7b \
    cel_task_conditioned_lastlayer_v1_qwen3_1.7b
  do
    local metrics_file="$CEL_DIR/metrics/metrics_${model}.txt"
    local auc
    auc="$(extract_overall_auc "$metrics_file")"
    log "$model overall auc = $auc"
    if python3 - "$auc" "$best_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      best_auc="$auc"
      best_model="$model"
    fi
  done

  log "best round1 cel model = $best_model (auc=$best_auc)"

  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "round1 already beat baseline by continuation margin; skipping adapter_v2"
  else
    if raw_beats_baseline "$best_auc" "$baseline_auc"; then
      log "round1 beat raw baseline but not continuation margin; still launching adapter_v2"
    else
      log "round1 did not beat baseline; launching adapter_v2"
    fi
    run_step_if_missing "cel_adapter_v2" \
      "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v2_qwen3_1.7b.txt" \
      bash scripts/cel_stage1_last_layer/run_adapter_v2.sh
    log "adapter_v2 overall auc = $(extract_overall_auc "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v2_qwen3_1.7b.txt")"
  fi

  log "experiment loop finished"
}

main "$@"
