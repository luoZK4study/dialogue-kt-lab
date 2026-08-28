#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
BASELINE_DIR="$ROOT_DIR/results/baseline"
CEL_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
LOG_FILE="$CEL_DIR/supervisor.log"
SLEEP_SECS="${SUPERVISOR_SLEEP_SECS:-60}"

PRIMARY_CMD='bash scripts/cel_stage1_last_layer/run_experiment_loop.sh'
FOLLOWUP_CMD='bash scripts/cel_stage1_last_layer/run_followup_loop.sh'

PRIMARY_DONE_PATTERN='experiment loop finished'
FOLLOWUP_DONE_PATTERN='followup loop found a CEL model that beats baseline|followup loop completed; no CEL model beat baseline by continuation margin|existing CEL result already beats baseline by continuation margin; no followup runs needed|adapter_v2 recovery check beats baseline by continuation margin; followup finished|selector-only round beats baseline by continuation margin; followup finished|early prediction-token batch1 beats baseline by continuation margin; followup finished|early task-conditioned prediction-token batch beats baseline by continuation margin; followup finished|early vector-predshift batch1 beats baseline by continuation margin; followup finished|early task-conditioned vector-predshift batch beats baseline by continuation margin; followup finished|early pre-lm-head vector-predshift batch1 beats baseline by continuation margin; followup finished|early task-conditioned pre-lm-head vector-predshift batch beats baseline by continuation margin; followup finished|direct pre-lm-head selector-only batch1 beats baseline by continuation margin; followup finished|direct task-conditioned pre-lm-head selector-only batch beats baseline by continuation margin; followup finished|scalar-gate followup beats baseline by continuation margin; followup finished|vector-shift batch1 beats baseline by continuation margin; followup finished|vector-shift batch2 beats baseline by continuation margin; followup finished|conservative scalar batch beats baseline by continuation margin; followup finished|conservative low-lr phase3 beats baseline by continuation margin; followup finished|phase4 stabilization batch beats baseline by continuation margin; followup finished|phase5 warm-start batch1 beats baseline by continuation margin; followup finished|phase5 warm-start batch2 beats baseline by continuation margin; followup finished|phase6 prediction-token batch1 beats baseline by continuation margin; followup finished|phase6 task-conditioned prediction-token batch beats baseline by continuation margin; followup finished|baseline metrics missing; exiting followup loop'

mkdir -p "$BASELINE_DIR/metrics" "$CEL_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

proc_running() {
  local cmd="$1"
  pgrep -xf "$cmd" >/dev/null 2>&1
}

start_primary() {
  log "starting primary loop"
  cd "$ROOT_DIR"
  nohup bash scripts/cel_stage1_last_layer/run_experiment_loop.sh > "$CEL_DIR/loop_stdout.log" 2>&1 < /dev/null &
}

start_followup() {
  log "starting followup loop"
  cd "$ROOT_DIR"
  nohup bash scripts/cel_stage1_last_layer/run_followup_loop.sh > "$CEL_DIR/followup_stdout.log" 2>&1 < /dev/null &
}

loop_finished() {
  grep -q "$PRIMARY_DONE_PATTERN" "$CEL_DIR/loop.log" 2>/dev/null
}

followup_finished() {
  grep -Eq "$FOLLOWUP_DONE_PATTERN" "$CEL_DIR/followup.log" 2>/dev/null
}

main() {
  cd "$ROOT_DIR"
  log "server supervisor started"
  while true; do
    if ! proc_running "$PRIMARY_CMD"; then
      if loop_finished; then
        :
      else
        start_primary
      fi
    fi

    if ! proc_running "$FOLLOWUP_CMD"; then
      if followup_finished; then
        :
      elif [[ -f "$CEL_DIR/loop.log" ]]; then
        start_followup
      fi
    fi

    sleep "$SLEEP_SECS"
  done
}

main "$@"
