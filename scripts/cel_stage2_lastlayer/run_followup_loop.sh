#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
BASELINE_DIR="$ROOT_DIR/results/baseline"
CEL_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
LOG_FILE="$CEL_DIR/followup.log"
STEP_LOG_DIR="$CEL_DIR/step_logs"
WAIT_SLEEP_SECS="${FOLLOWUP_WAIT_SLEEP_SECS:-180}"
FOLLOWUP_CLEAR_MARGIN="${FOLLOWUP_CLEAR_MARGIN:-0.20}"

mkdir -p "$BASELINE_DIR/metrics" "$CEL_DIR/metrics"
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

extract_gate_abs_mean() {
  python3 - "$1" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"CEL Diagnostics:\s+.*?gate_abs_mean:\s*([0-9.]+)", text)
if m:
    print(m.group(1))
else:
    m = re.search(r"CEL Diagnostics:\s+.*?shift_abs_mean:\s*([0-9.]+)", text)
    print(m.group(1) if m else "0.0")
PY
}

beats_baseline() {
  python3 - "$1" "$2" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
}

clear_beats_baseline() {
  python3 - "$1" "$2" "$FOLLOWUP_CLEAR_MARGIN" <<'PY'
import sys
auc = float(sys.argv[1])
baseline = float(sys.argv[2])
margin = float(sys.argv[3])
sys.exit(0 if auc > baseline + margin else 1)
PY
}

wait_for_primary_loop() {
  while pgrep -f "bash scripts/cel_stage1_last_layer/run_experiment_loop.sh" >/dev/null; do
    log "waiting for primary loop to finish"
    sleep "$WAIT_SLEEP_SECS"
  done
}

best_existing_model() {
  python3 - "$CEL_DIR/metrics" <<'PY'
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
    return 0
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

pick_adapter_followup() {
  local metrics_file="$1"
  local gate_abs_mean
  gate_abs_mean="$(extract_gate_abs_mean "$metrics_file")"
  if python3 - "$gate_abs_mean" <<'PY'
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
  log "followup continuation margin = +$FOLLOWUP_CLEAR_MARGIN over baseline"

  local best_model best_auc
  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best existing cel model = $best_model (auc=$best_auc)"

  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "existing CEL result already beats baseline by continuation margin; no followup runs needed"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi
  if beats_baseline "$best_auc" "$baseline_auc"; then
    log "best existing CEL is above raw baseline but below continuation margin; continuing followup"
  fi

  run_step_if_missing \
    "cel_adapter_v2_followup_recover" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v2_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_adapter_v2.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after adapter_v2 recovery check = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "adapter_v2 recovery check beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_pid adapter_pid
  run_step_async_if_missing_into mlp_pid \
    "cel_mlp_v2_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v2_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v2_selector_only.sh
  run_step_async_if_missing_into adapter_pid \
    "cel_adapter_v3_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v3_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v3_selector_only.sh
  wait_for_step "cel_mlp_v2_selector_only" "$mlp_pid"
  wait_for_step "cel_adapter_v3_selector_only" "$adapter_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after selector-only round = $best_model (auc=$best_auc)"

  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "selector-only round beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v6_pid adapter_v16_pid
  run_step_async_if_missing_into mlp_v6_pid \
    "cel_mlp_v6_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v6_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v6_predshift_selector_only.sh
  run_step_async_if_missing_into adapter_v16_pid \
    "cel_adapter_v16_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v16_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v16_predshift_selector_only.sh
  wait_for_step "cel_mlp_v6_predshift_selector_only" "$mlp_v6_pid"
  wait_for_step "cel_adapter_v16_predshift_selector_only" "$adapter_v16_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early prediction-token batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early prediction-token batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v7_pid adapter_v17_pid
  run_step_async_if_missing_into mlp_v7_pid \
    "cel_mlp_v7_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v7_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v7_vector_shift_predshift_selector_only.sh
  run_step_async_if_missing_into adapter_v17_pid \
    "cel_adapter_v17_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v17_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v17_vector_shift_predshift_selector_only.sh
  wait_for_step "cel_mlp_v7_vector_shift_predshift_selector_only" "$mlp_v7_pid"
  wait_for_step "cel_adapter_v17_vector_shift_predshift_selector_only" "$adapter_v17_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early vector-predshift batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early vector-predshift batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v8_pid adapter_v18_pid
  run_step_async_if_missing_into mlp_v8_pid \
    "cel_mlp_v8_prelm_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v8_prelm_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v8_prelm_vector_shift_predshift_selector_only.sh
  run_step_async_if_missing_into adapter_v18_pid \
    "cel_adapter_v18_prelm_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v18_prelm_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v18_prelm_vector_shift_predshift_selector_only.sh
  wait_for_step "cel_mlp_v8_prelm_vector_shift_predshift_selector_only" "$mlp_v8_pid"
  wait_for_step "cel_adapter_v18_prelm_vector_shift_predshift_selector_only" "$adapter_v18_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early pre-lm-head vector-predshift batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early pre-lm-head vector-predshift batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v9_pid adapter_v19_pid
  run_step_async_if_missing_into mlp_v9_pid \
    "cel_mlp_v9_prelm_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v9_prelm_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v9_prelm_selector_only.sh
  run_step_async_if_missing_into adapter_v19_pid \
    "cel_adapter_v19_prelm_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v19_prelm_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v19_prelm_selector_only.sh
  wait_for_step "cel_mlp_v9_prelm_selector_only" "$mlp_v9_pid"
  wait_for_step "cel_adapter_v19_prelm_selector_only" "$adapter_v19_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after direct pre-lm-head selector-only batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "direct pre-lm-head selector-only batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  # Task-conditioned runs are much slower than the MLP/adapter family, so keep
  # them after the highest-value predshift / pre-lm-head checks.
  run_step_if_missing \
    "cel_task_conditioned_v2_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v2_selector_only_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v2_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after delayed task-conditioned selector-only batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "selector-only round beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  run_step_if_missing \
    "cel_task_conditioned_v7_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v7_predshift_selector_only_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v7_predshift_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early task-conditioned prediction-token batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early task-conditioned prediction-token batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  run_step_if_missing \
    "cel_task_conditioned_v8_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v8_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v8_vector_shift_predshift_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early task-conditioned vector-predshift batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early task-conditioned vector-predshift batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  run_step_if_missing \
    "cel_task_conditioned_v9_prelm_vector_shift_predshift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v9_prelm_vector_shift_predshift_selector_only_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v9_prelm_vector_shift_predshift_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after early task-conditioned pre-lm-head vector-predshift batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "early task-conditioned pre-lm-head vector-predshift batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  run_step_if_missing \
    "cel_task_conditioned_v10_prelm_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v10_prelm_selector_only_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v10_prelm_selector_only.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after direct task-conditioned pre-lm-head selector-only batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "direct task-conditioned pre-lm-head selector-only batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local best_metrics_file="$CEL_DIR/metrics/metrics_${best_model}.txt"
  local adapter_followup
  adapter_followup="$(pick_adapter_followup "$best_metrics_file")"
  log "adapter followup chosen = $adapter_followup"

  if [[ "$adapter_followup" == "lowgamma" ]]; then
    run_step_if_missing \
      "cel_adapter_v4_selector_only_lowgamma" \
      "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v4_selector_only_lowgamma_qwen3_1.7b.txt" \
      bash scripts/cel_stage1_last_layer/run_adapter_v4_selector_only_lowgamma.sh
  else
    run_step_if_missing \
      "cel_adapter_v5_selector_only_highgamma" \
      "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v5_selector_only_highgamma_qwen3_1.7b.txt" \
      bash scripts/cel_stage1_last_layer/run_adapter_v5_selector_only_highgamma.sh
  fi

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after scalar-gate followup = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "scalar-gate followup beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local adapter_v6_pid task_shift_pid
  run_step_async_if_missing_into adapter_v6_pid \
    "cel_adapter_v6_vector_shift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v6_vector_shift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_adapter_v6_vector_shift_selector_only.sh
  run_step_async_if_missing_into task_shift_pid \
    "cel_task_conditioned_v3_vector_shift_selector_only" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v3_vector_shift_selector_only_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_v3_vector_shift_selector_only.sh
  wait_for_step "cel_adapter_v6_vector_shift_selector_only" "$adapter_v6_pid"
  wait_for_step "cel_task_conditioned_v3_vector_shift_selector_only" "$task_shift_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after vector-shift batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "vector-shift batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local adapter_v7_pid adapter_v8_pid
  run_step_async_if_missing_into adapter_v7_pid \
    "cel_adapter_v7_vector_shift_selector_only_highgamma" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v7_vector_shift_selector_only_highgamma_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_adapter_v7_vector_shift_selector_only_highgamma.sh
  run_step_async_if_missing_into adapter_v8_pid \
    "cel_adapter_v8_vector_shift_selector_only_ultralowgamma" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v8_vector_shift_selector_only_ultralowgamma_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v8_vector_shift_selector_only_ultralowgamma.sh
  wait_for_step "cel_adapter_v7_vector_shift_selector_only_highgamma" "$adapter_v7_pid"
  wait_for_step "cel_adapter_v8_vector_shift_selector_only_ultralowgamma" "$adapter_v8_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL after vector-shift batch2 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "vector-shift batch2 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v3_pid adapter_v9_pid
  run_step_async_if_missing_into mlp_v3_pid \
    "cel_mlp_v3_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v3_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v3_selector_only_tinylr.sh
  run_step_async_if_missing_into adapter_v9_pid \
    "cel_adapter_v9_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v9_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v9_selector_only_tinylr.sh
  wait_for_step "cel_mlp_v3_selector_only_tinylr" "$mlp_v3_pid"
  wait_for_step "cel_adapter_v9_selector_only_tinylr" "$adapter_v9_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after conservative scalar batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "conservative scalar batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local adapter_v10_pid task_v4_pid
  run_step_async_if_missing_into adapter_v10_pid \
    "cel_adapter_v10_vector_shift_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v10_vector_shift_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_adapter_v10_vector_shift_selector_only_tinylr.sh
  run_step_async_if_missing_into task_v4_pid \
    "cel_task_conditioned_v4_vector_shift_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v4_vector_shift_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_v4_vector_shift_selector_only_tinylr.sh
  wait_for_step "cel_adapter_v10_vector_shift_selector_only_tinylr" "$adapter_v10_pid"
  wait_for_step "cel_task_conditioned_v4_vector_shift_selector_only_tinylr" "$task_v4_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL after conservative low-lr phase3 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "conservative low-lr phase3 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local adapter_v11_pid adapter_v12_pid
  run_step_async_if_missing_into adapter_v11_pid \
    "cel_adapter_v11_vector_shift_selector_only_tinylr_norm" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v11_vector_shift_selector_only_tinylr_norm_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_adapter_v11_vector_shift_selector_only_tinylr_norm.sh
  run_step_async_if_missing_into adapter_v12_pid \
    "cel_adapter_v12_selector_only_tinylr_nowd" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v12_selector_only_tinylr_nowd_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v12_selector_only_tinylr_nowd.sh
  wait_for_step "cel_adapter_v11_vector_shift_selector_only_tinylr_norm" "$adapter_v11_pid"
  wait_for_step "cel_adapter_v12_selector_only_tinylr_nowd" "$adapter_v12_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL after phase4 stabilization batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "phase4 stabilization batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v4_pid adapter_v13_pid
  run_step_async_if_missing_into mlp_v4_pid \
    "cel_mlp_v4_vector_shift_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v4_vector_shift_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v4_vector_shift_selector_only_tinylr.sh
  run_step_async_if_missing_into adapter_v13_pid \
    "cel_adapter_v13_vector_shift_selector_only_tinylr_nowd" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v13_vector_shift_selector_only_tinylr_nowd_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v13_vector_shift_selector_only_tinylr_nowd.sh
  wait_for_step "cel_mlp_v4_vector_shift_selector_only_tinylr" "$mlp_v4_pid"
  wait_for_step "cel_adapter_v13_vector_shift_selector_only_tinylr_nowd" "$adapter_v13_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after phase5 warm-start batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "phase5 warm-start batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local adapter_v14_pid task_v5_pid
  run_step_async_if_missing_into adapter_v14_pid \
    "cel_adapter_v14_vector_shift_selector_only_epoch3" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v14_vector_shift_selector_only_epoch3_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_adapter_v14_vector_shift_selector_only_epoch3.sh
  run_step_async_if_missing_into task_v5_pid \
    "cel_task_conditioned_v5_vector_shift_selector_only_tinylr_nowd" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v5_vector_shift_selector_only_tinylr_nowd_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_v5_vector_shift_selector_only_tinylr_nowd.sh
  wait_for_step "cel_adapter_v14_vector_shift_selector_only_epoch3" "$adapter_v14_pid"
  wait_for_step "cel_task_conditioned_v5_vector_shift_selector_only_tinylr_nowd" "$task_v5_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL after phase5 warm-start batch2 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "phase5 warm-start batch2 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  local mlp_v5_pid adapter_v15_pid
  run_step_async_if_missing_into mlp_v5_pid \
    "cel_mlp_v5_predshift_selector_only_tinylr" \
    "$CEL_DIR/metrics/metrics_cel_mlp_lastlayer_v5_predshift_selector_only_tinylr_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=0 bash scripts/cel_stage1_last_layer/run_mlp_v5_predshift_selector_only_tinylr.sh
  run_step_async_if_missing_into adapter_v15_pid \
    "cel_adapter_v15_predshift_selector_only_tinylr_nowd" \
    "$CEL_DIR/metrics/metrics_cel_adapter_lastlayer_v15_predshift_selector_only_tinylr_nowd_qwen3_1.7b.txt" \
    env CUDA_VISIBLE_DEVICES=1 bash scripts/cel_stage1_last_layer/run_adapter_v15_predshift_selector_only_tinylr_nowd.sh
  wait_for_step "cel_mlp_v5_predshift_selector_only_tinylr" "$mlp_v5_pid"
  wait_for_step "cel_adapter_v15_predshift_selector_only_tinylr_nowd" "$adapter_v15_pid"

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "best CEL after phase6 prediction-token batch1 = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "phase6 prediction-token batch1 beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
    exit 0
  fi

  run_step_if_missing \
    "cel_task_conditioned_v6_predshift_selector_only_tinylr_nowd" \
    "$CEL_DIR/metrics/metrics_cel_task_conditioned_lastlayer_v6_predshift_selector_only_tinylr_nowd_qwen3_1.7b.txt" \
    bash scripts/cel_stage1_last_layer/run_task_conditioned_v6_predshift_selector_only_tinylr_nowd.sh

  readarray -t best_info < <(best_existing_model)
  best_model="${best_info[0]}"
  best_auc="${best_info[1]}"
  log "final best CEL after phase6 task-conditioned prediction-token batch = $best_model (auc=$best_auc)"
  if clear_beats_baseline "$best_auc" "$baseline_auc"; then
    log "phase6 task-conditioned prediction-token batch beats baseline by continuation margin; followup finished"
    log "followup loop found a CEL model that beats baseline"
  else
    if beats_baseline "$best_auc" "$baseline_auc"; then
      log "best CEL beat raw baseline but not continuation margin (best_auc=$best_auc, baseline_auc=$baseline_auc, margin=$FOLLOWUP_CLEAR_MARGIN)"
    fi
    log "followup loop completed; no CEL model beat baseline by continuation margin"
  fi
}

main "$@"
