#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
BASELINE_DIR="$ROOT_DIR/results/baseline_recert"
CEL_DIR="$ROOT_DIR/results/cel_stage2_lastlayer"
LOG_FILE="$CEL_DIR/followup.log"

mkdir -p "$BASELINE_DIR/metrics" "$CEL_DIR/metrics"

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

extract_gate_abs_mean() {
  python - "$1" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"CEL Diagnostics:\s+.*?gate_abs_mean:\s*([0-9.]+)", text)
print(m.group(1) if m else "0.0")
PY
}

wait_for_primary_loop() {
  while pgrep -f "bash scripts/cel_stage2_lastlayer/run_experiment_loop.sh" >/dev/null; do
    log "waiting for primary loop to finish"
    sleep 300
  done
}

best_existing_model() {
  python - "$CEL_DIR/metrics" <<'PY'
import re
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
best_model = None
best_auc = -1.0
for path in sorted(metrics_dir.glob("metrics_*.txt")):
    text = path.read_text(encoding="utf-8")
    m = re.search(r"Overall.*?\n.*?\nAcc:\s+[0-9.]+,\s+AUC:\s+([0-9.]+)", text, re.S)
    if not m:
        continue
    auc = float(m.group(1))
    if auc > best_auc:
        best_auc = auc
        best_model = path.stem.replace("metrics_", "", 1)
if best_model is None:
    raise SystemExit(1)
print(best_model)
print(best_auc)
PY
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@" >> "$LOG_FILE" 2>&1
  log "END $name"
}

pick_adapter_followup() {
  local metrics_file="$1"
  local gate_abs_mean
  gate_abs_mean="$(extract_gate_abs_mean "$metrics_file")"
  if python - "$gate_abs_mean" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) >= 0.12 else 1)
PY
  then
    echo "lowgamma"
  else
    echo "highgamma"
  fi
}

main() {
  cd "$ROOT_DIR"
  wait_for_primary_loop

  local baseline_metrics="$BASELINE_DIR/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt"
  if [[ ! -f "$baseline_metrics" ]]; then
    log "baseline metrics missing; exiting followup loop"
    exit 1
  fi

  local baseline_auc
  baseline_auc="$(extract_overall_auc "$baseline_metrics")"
  log "baseline overall auc = $baseline_auc"

  local best_model best_auc
  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best existing cel model = $best_model (auc=$best_auc)"

  if python - "$best_auc" "$baseline_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    log "existing CEL result already beats baseline; no followup runs needed"
    exit 0
  fi

  run_step "cel_mlp_v2_selector_only" bash scripts/cel_stage2_lastlayer/run_mlp_v2_selector_only.sh
  run_step "cel_adapter_v3_selector_only" bash scripts/cel_stage2_lastlayer/run_adapter_v3_selector_only.sh
  run_step "cel_task_conditioned_v2_selector_only" bash scripts/cel_stage2_lastlayer/run_task_conditioned_v2_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after selector-only round = $best_model (auc=$best_auc)"

  if python - "$best_auc" "$baseline_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    log "selector-only round beats baseline; followup finished"
    exit 0
  fi

  local best_metrics_file="$CEL_DIR/metrics/metrics_${best_model}.txt"
  local adapter_followup
  adapter_followup="$(pick_adapter_followup "$best_metrics_file")"
  log "adapter followup chosen = $adapter_followup"

  if [[ "$adapter_followup" == "lowgamma" ]]; then
    run_step "cel_adapter_v4_selector_only_lowgamma" bash scripts/cel_stage2_lastlayer/run_adapter_v4_selector_only_lowgamma.sh
  else
    run_step "cel_adapter_v5_selector_only_highgamma" bash scripts/cel_stage2_lastlayer/run_adapter_v5_selector_only_highgamma.sh
  fi

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL = $best_model (auc=$best_auc)"
  if python - "$best_auc" "$baseline_auc" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    log "followup loop found a CEL model that beats baseline"
  else
    log "followup loop completed; no CEL model beat baseline yet"
  fi
}

main "$@"
