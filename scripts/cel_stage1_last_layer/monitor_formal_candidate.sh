#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-once}"
WATCH_CYCLES="${TASK_CONDITIONED_MONITOR_MAX_CYCLES:-0}"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_MONITOR_SYNC_TIMEOUT_SECS:-240}"
DEFAULT_SLEEP_SECS="${TASK_CONDITIONED_MONITOR_SLEEP_SECS:-600}"
FAST_SLEEP_SECS="${TASK_CONDITIONED_MONITOR_FAST_SLEEP_SECS:-300}"
STATUS_MD="$ROOT_DIR/results/cel_stage1_last_layer/STATUS.md"
STRICT_REPORT="$ROOT_DIR/results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md"
AUDIT_JSON="$ROOT_DIR/results/cel_stage1_last_layer/formal_experiment_audit.json"

print_local_snapshot_summary() {
  python3 - <<'PY' "$STATUS_MD" "$STRICT_REPORT" "$AUDIT_JSON" "$DEFAULT_SLEEP_SECS" "$FAST_SLEEP_SECS"
import json
import re
import sys
from datetime import datetime
from pathlib import Path

status_md = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
strict_report = Path(sys.argv[2]).read_text(encoding="utf-8", errors="ignore")
audit_json_path = Path(sys.argv[3])
default_sleep_secs = sys.argv[4]
fast_sleep_secs = sys.argv[5]

audit = None
if audit_json_path.exists():
    try:
        audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        audit = None

def extract(pattern: str, text: str, default: str = "unknown") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def parse_monitor_timestamp(text: str) -> datetime | None:
    text = text.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2} [0-9:]+$", text):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def format_wait_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0s"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"

stage = extract(r"- Stage: `([^`]+)`", status_md)
next_action = extract(r"- Task-conditioned next action: `([^`]+)`", status_md)
winner = extract(r"- Task-conditioned winner found: `([^`]+)`", status_md)
sync_freshness = extract(r"- Task-conditioned sync freshness: `([^`]+)`", status_md)
poll_interval = extract(r"- Recommended task-conditioned poll interval: `([^`]+)`", status_md, default="")
next_monitor_after = extract(r"- Suggested next task-conditioned monitor after: `([^`]+)`", status_md, default="")
authoritative_remote_refresh_event = extract(r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`", status_md, default="")
latest_refresh_attempt = extract(r"- Latest task-conditioned refresh attempt: `([^`]+)`", status_md, default="")
active_timing_summary = extract(r"- 当前[^：\n]*计时：([^\n]+)", status_md, default="")

if not poll_interval:
    if next_action in {"manual_decide", "done", "launch_round3"}:
        poll_interval = "not_applicable"
    elif "validation" in status_md.lower() or "testing" in status_md.lower():
        poll_interval = f"{fast_sleep_secs}s_fallback"
    else:
        poll_interval = f"{default_sleep_secs}s_fallback"

counts = (audit or {}).get("counts") or {}
formal_audit_summary = ",".join(f"{key}={value}" for key, value in sorted(counts.items())) or "unknown"
rows = (audit or {}).get("rows") or []
rerun_queue = ",".join(str(row.get("label")) for row in rows if row.get("audit") == "needs_rerun") or "none"
recorded_non_winner_queue = ",".join(
    str(row.get("label"))
    for row in rows
    if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
) or "none"

decision_lines = []
capture = False
for raw_line in strict_report.splitlines():
    if raw_line.strip() == "## Decision":
        capture = True
        continue
    if capture and raw_line.startswith("## "):
        break
    if capture and raw_line.startswith("- "):
        decision_lines.append(raw_line)

print(f"stage={stage}")
print(f"next_action={next_action}")
print(f"winner_found={winner}")
print(f"sync_freshness={sync_freshness}")
print(f"recommended_poll_interval={poll_interval}")
if authoritative_remote_refresh_event and authoritative_remote_refresh_event != "unknown":
    print(f"authoritative_remote_refresh_event={authoritative_remote_refresh_event}")
if latest_refresh_attempt and latest_refresh_attempt != "unknown":
    print(f"latest_refresh_attempt={latest_refresh_attempt}")
if next_monitor_after:
    print(f"suggested_next_monitor_after={next_monitor_after}")
    next_monitor_after_dt = parse_monitor_timestamp(next_monitor_after)
    if next_monitor_after_dt is not None:
        delta_secs = int((next_monitor_after_dt - datetime.now()).total_seconds())
        due_now = delta_secs <= 0
        print(f"monitor_due_now={due_now}")
        if due_now:
            print("seconds_until_suggested_monitor=0")
            print("remaining_wait_text=0s")
        else:
            print(f"seconds_until_suggested_monitor={delta_secs}")
            print(f"remaining_wait_text={format_wait_duration(delta_secs)}")
if active_timing_summary:
    print(f"active_timing_summary={active_timing_summary}")
print(f"formal_audit_summary={formal_audit_summary}")
print(f"rerun_queue={rerun_queue}")
print(f"recorded_non_winner_queue={recorded_non_winner_queue}")
if decision_lines:
    print("decision_summary:")
    for line in decision_lines:
        print(line)
PY
}

print_current_next_action() {
  echo "current_formal_next_action:"
  python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh [once|watch|skip-sync]

Purpose:
  Provide one monitor entrypoint for the strict full-train loop.

Modes:
  once   Run one authoritative refresh/rebuild cycle
  watch  Run the existing watcher with the configured cadence
  skip-sync  Read the current local markdown snapshot without attempting SSH sync

Environment:
  TASK_CONDITIONED_MONITOR_SYNC_TIMEOUT_SECS  Sync timeout for each refresh (default: 240)
  TASK_CONDITIONED_MONITOR_SLEEP_SECS         Default watcher interval in seconds (default: 600)
  TASK_CONDITIONED_MONITOR_FAST_SLEEP_SECS    Fast watcher interval in seconds (default: 300)
  TASK_CONDITIONED_MONITOR_MAX_CYCLES         Optional watcher cycle cap; 0 means unlimited

Examples:
  bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once
  TASK_CONDITIONED_MONITOR_MAX_CYCLES=3 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
EOF
}

case "$MODE" in
  once)
    echo "== Strict Formal Monitor =="
    echo "mode=once"
    TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" \
      bash "$ROOT_DIR/scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh"
    print_local_snapshot_summary
    print_current_next_action
    ;;
  watch)
    echo "== Strict Formal Monitor =="
    echo "mode=watch"
    echo "max_cycles=$WATCH_CYCLES"
    echo "sleep_secs=$DEFAULT_SLEEP_SECS"
    echo "fast_sleep_secs=$FAST_SLEEP_SECS"
    WATCH_MAX_CYCLES="$WATCH_CYCLES" \
      WATCH_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" \
      WATCH_SLEEP_SECS="$DEFAULT_SLEEP_SECS" \
      WATCH_FAST_SLEEP_SECS="$FAST_SLEEP_SECS" \
      bash "$ROOT_DIR/scripts/cel_stage1_last_layer/watch_task_conditioned_tuning.sh"
    ;;
  skip-sync)
    echo "== Strict Formal Monitor =="
    echo "mode=skip-sync"
    print_local_snapshot_summary
    print_current_next_action
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
