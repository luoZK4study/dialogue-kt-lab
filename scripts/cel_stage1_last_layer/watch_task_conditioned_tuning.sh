#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLEEP_SECS="${WATCH_SLEEP_SECS:-600}"
FAST_SLEEP_SECS="${WATCH_FAST_SLEEP_SECS:-300}"
SYNC_TIMEOUT_SECS="${WATCH_SYNC_TIMEOUT_SECS:-240}"
MAX_CYCLES="${WATCH_MAX_CYCLES:-0}"
STATUS_JSON="$ROOT_DIR/results/cel_stage1_last_layer/task_conditioned_status.json"
STATUS_MD="$ROOT_DIR/results/cel_stage1_last_layer/STATUS.md"
LOG_FILE="$ROOT_DIR/results/cel_stage1_last_layer/task_conditioned_watch.log"
REFRESH_LOG="$ROOT_DIR/results/cel_stage1_last_layer/task_conditioned_refresh.log"
REFRESH_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh"
PYTHON_BIN="${DIALOGUE_KT_PYTHON:-python3}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

on_exit() {
  local status="$?"
  if [[ "$status" -ne 0 ]]; then
    log "task_conditioned watcher exited unexpectedly with status=$status"
  fi
}

trap on_exit EXIT

refresh_local_state() {
  TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" bash "$REFRESH_SCRIPT" || true
}

read_watch_state() {
  "$PYTHON_BIN" - <<'PY' "$STATUS_JSON" "$STATUS_MD" "$REFRESH_LOG"
import json
import re
import sys
from pathlib import Path

status_json = Path(sys.argv[1])
status_md = Path(sys.argv[2])
refresh_log = Path(sys.argv[3])

next_action = "missing"
winner_found = "false"
round3_done = "false"
round3_summary = "unknown"
stage = "unknown"
live_phase = "unknown"
sleep_hint = "standard"
refresh_sync = "unknown"
progress_pct = None
recommended_poll_secs = ""

if status_json.exists():
    data = json.loads(status_json.read_text(encoding="utf-8"))
    next_action = str(data.get("next_action", "missing"))
    winner_found = "true" if data.get("winner_found") else "false"
    rounds = data.get("rounds") or []
    if len(rounds) >= 3:
        round3_models = rounds[2].get("models") or []
        statuses = [str(model.get("status", "pending")) for model in round3_models]
        if statuses:
            round3_summary = ",".join(statuses)
            round3_done = "true" if all(status in {"done", "failed"} for status in statuses) else "false"

if status_md.exists():
    text = status_md.read_text(encoding="utf-8", errors="ignore")
    stage_match = re.search(r"- Stage: `([^`]+)`", text)
    if stage_match:
        stage = stage_match.group(1)
    phase_match = re.search(r"- Live phase: `([^`]+)`", text)
    if phase_match:
        live_phase = phase_match.group(1)
        if live_phase in {"validation", "validating", "testing"}:
            sleep_hint = "fast"
    progress_match = re.search(r"- Live [^`]+ progress: `([^`]+)`", text)
    if progress_match:
        pct_match = re.search(r"\((\d+)%\)", progress_match.group(1))
        if pct_match:
            progress_pct = int(pct_match.group(1))
    if progress_pct is not None and progress_pct >= 95:
        sleep_hint = "fast"
    poll_match = re.search(r"- Recommended task-conditioned poll interval: `(\d+)s` \(([^`]+)\)", text)
    if poll_match:
        recommended_poll_secs = poll_match.group(1)

if refresh_log.exists():
    for line in reversed(refresh_log.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = re.search(r"task_conditioned refresh complete :: sync=([a-z_]+)", line)
        if match:
            refresh_sync = match.group(1)
            break

print(next_action)
print(winner_found)
print(round3_done)
print(round3_summary)
print(stage)
print(live_phase)
print(sleep_hint)
print(refresh_sync)
print(recommended_poll_secs)
PY
}

cycle_count=0
while true; do
  cycle_count="$((cycle_count + 1))"
  cycle_started_at="$(date +%s)"
  log "task_conditioned watcher cycle start"
  refresh_local_state

  if [[ ! -f "$STATUS_JSON" ]]; then
    log "task_conditioned watcher sync complete :: status_json=missing"
    sleep "$SLEEP_SECS"
    continue
  fi

  mapfile -t state < <(read_watch_state)
  next_action="${state[0]}"
  winner_found="${state[1]}"
  round3_done="${state[2]}"
  round3_summary="${state[3]}"
  stage="${state[4]}"
  live_phase="${state[5]}"
  sleep_hint="${state[6]}"
  refresh_sync="${state[7]}"
  recommended_poll_secs="${state[8]}"
  cycle_finished_at="$(date +%s)"
  cycle_duration="$((cycle_finished_at - cycle_started_at))"

  log "task_conditioned watcher sync complete :: refresh_sync=$refresh_sync stage=$stage next_action=$next_action live_phase=$live_phase round3=$round3_summary winner=$winner_found recommended_poll=${recommended_poll_secs:-unknown}s duration=${cycle_duration}s"

  if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
    log "task_conditioned watcher reached max_cycles=$MAX_CYCLES; stopping"
    break
  fi

  if [[ "$winner_found" == "true" || "$next_action" == "done" ]]; then
    log "strict full-train winner detected; stopping watcher"
    break
  fi

  if [[ "$next_action" == "launch_round3" ]]; then
    log "Round 3 is still pending launch; stopping watcher so controller or manual launch can start the run"
    break
  fi

  if [[ "$round3_done" == "true" && "$next_action" == "manual_decide" ]]; then
    log "Round 3 finished without an automatic winner; stopping watcher for manual analysis"
    break
  fi

  next_sleep_secs=""
  if [[ -n "$recommended_poll_secs" ]]; then
    next_sleep_secs="$recommended_poll_secs"
  elif [[ "$sleep_hint" == "fast" ]]; then
    next_sleep_secs="$FAST_SLEEP_SECS"
  else
    next_sleep_secs="$SLEEP_SECS"
  fi

  log "sleeping ${next_sleep_secs}s before next watcher cycle"
  sleep "$next_sleep_secs"
done
