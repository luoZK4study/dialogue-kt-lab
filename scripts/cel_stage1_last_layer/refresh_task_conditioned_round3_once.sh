#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS:-240}"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
REFRESH_LOG="$RESULT_DIR/task_conditioned_refresh.log"
REBUILD_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/rebuild_task_conditioned_reports.sh"
LOCK_FILE="$RESULT_DIR/.task_conditioned_refresh.lock"
LOCK_TIMEOUT_SECS="${TASK_CONDITIONED_REFRESH_LOCK_TIMEOUT_SECS:-900}"
PYTHON_BIN="${DIALOGUE_KT_PYTHON:-python3}"
SYNC_STATUS="unknown"
SYNC_EXIT_CODE=0

log() {
  echo "[$(date '+%F %T')] $*" >> "$REFRESH_LOG"
}

run_sync() {
  local rc=0
  if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "${SYNC_TIMEOUT_SECS}" bash "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server.sh"
    rc="$?"
    set -e
    if [[ "$rc" -eq 124 ]]; then
      SYNC_STATUS="timeout"
    elif [[ "$rc" -eq 0 ]]; then
      SYNC_STATUS="ok"
    else
      SYNC_STATUS="failed"
    fi
  else
    set +e
    bash "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server.sh"
    rc="$?"
    set -e
    if [[ "$rc" -eq 0 ]]; then
      SYNC_STATUS="ok"
    else
      SYNC_STATUS="failed"
    fi
  fi
  SYNC_EXIT_CODE="$rc"
}

mkdir -p "$RESULT_DIR"
exec 9>"$LOCK_FILE"
flock -w "$LOCK_TIMEOUT_SECS" 9

cycle_started_at="$(date +%s)"
log "task_conditioned refresh start :: sync_timeout=${SYNC_TIMEOUT_SECS}s"
run_sync
log "task_conditioned refresh sync :: status=${SYNC_STATUS} exit=${SYNC_EXIT_CODE}"

# Rebuild authoritative task-conditioned state first; downstream reports read it.
bash "$REBUILD_SCRIPT"

"$PYTHON_BIN" - <<'PY' "$RESULT_DIR/task_conditioned_status.json" "$RESULT_DIR/STATUS.md" "$REFRESH_LOG" "$cycle_started_at" "$SYNC_STATUS" "$SYNC_EXIT_CODE"
import json
import re
import sys
from pathlib import Path

status_json = Path(sys.argv[1])
status_md = Path(sys.argv[2])
log_path = Path(sys.argv[3])
cycle_started_at = int(sys.argv[4])
sync_status = sys.argv[5]
sync_exit_code = sys.argv[6]

next_action = "missing"
winner_found = "false"
round3_summary = "unknown"
stage = "unknown"
live_phase = "unknown"

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

if status_md.exists():
    text = status_md.read_text(encoding="utf-8", errors="ignore")
    stage_match = re.search(r"- Stage: `([^`]+)`", text)
    if stage_match:
        stage = stage_match.group(1)
    phase_match = re.search(r"- Live phase: `([^`]+)`", text)
    if phase_match:
        live_phase = phase_match.group(1)

from time import time
duration = int(time()) - cycle_started_at
with log_path.open("a", encoding="utf-8") as f:
    f.write(
    f"[{__import__('datetime').datetime.now().strftime('%F %T')}] "
        f"task_conditioned refresh complete :: sync={sync_status} sync_exit={sync_exit_code} "
        f"stage={stage} next_action={next_action} "
        f"live_phase={live_phase} round3={round3_summary} winner={winner_found} duration={duration}s\n"
    )
PY

# Refresh completion is part of the authoritative status surface, so rebuild
# once more after appending the final refresh-log summary. Route this through
# the shared rebuild script so concurrent refresh/finalize/review paths stay
# serialized under one lock.
bash "$REBUILD_SCRIPT"
