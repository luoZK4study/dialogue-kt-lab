#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
BASELINE_DIR="$ROOT_DIR/results/baseline_recert"
CEL_DIR="$ROOT_DIR/results/cel_stage2_lastlayer"
LOG_FILE="$CEL_DIR/loop.log"

mkdir -p "$BASELINE_DIR/metrics" "$BASELINE_DIR/qual" "$BASELINE_DIR/kcs"
mkdir -p "$CEL_DIR/metrics" "$CEL_DIR/qual" "$CEL_DIR/kcs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

extract_overall_auc() {
  python - "$1" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"Overall.*?\n.*?\nAcc:\s+[0-9.]+,\s+AUC:\s+([0-9.]+)", text, re.S)
if not m:
    raise SystemExit(1)
print(m.group(1))
PY
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@" >> "$LOG_FILE" 2>&1
  log "END $name"
}

main() {
  cd "$ROOT_DIR"

  run_step "baseline_recert" bash scripts/cel_stage2_lastlayer/run_baseline_recert.sh

  local baseline_metrics="$BASELINE_DIR/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt"
  local baseline_auc
  baseline_auc="$(extract_overall_auc "$baseline_metrics")"
  log "baseline overall auc = $baseline_auc"

  run_step "cel_mlp_v1" bash scripts/cel_stage2_lastlayer/run_mlp_v1.sh
  run_step "cel_adapter_v1" bash scripts/cel_stage2_lastlayer/run_adapter_v1.sh
  run_step "cel_task_conditioned_v1" bash scripts/cel_stage2_lastlayer/run_task_conditioned_v1.sh

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
    if python - "$auc" "$best_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      best_auc="$auc"
      best_model="$model"
    fi
  done

  log "best round1 cel model = $best_model (auc=$best_auc)"

  if python - "$best_auc" "$baseline_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    log "round1 already beat baseline; skipping adapter_v2"
  else
    log "round1 did not beat baseline; launching adapter_v2"
    run_step "cel_adapter_v2" bash scripts/cel_stage2_lastlayer/run_adapter_v2.sh
    log "adapter_v2 overall auc = $(extract_overall_auc "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v2_qwen3_1.7b.txt")"
  fi

  log "experiment loop finished"
}

main "$@"
