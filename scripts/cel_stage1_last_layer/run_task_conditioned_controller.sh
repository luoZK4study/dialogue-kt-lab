#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
STATUS_JSON="$RESULT_DIR/task_conditioned_status.json"
STATUS_MD="$RESULT_DIR/STATUS.md"
LOG_FILE="$RESULT_DIR/task_conditioned_controller.log"
PRINT_STATE_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
START_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate.sh"
FINALIZE_CURRENT_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh"
REVIEW_CURRENT_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate.sh"
FINALIZE_EXPLICIT_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh"
# Fallback controller interval. The authoritative cadence comes from the
# generated STATUS.md recommended poll interval whenever it is present.
SLEEP_SECS="${TASK_CONDITIONED_CONTROLLER_SLEEP_SECS:-600}"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_CONTROLLER_SYNC_TIMEOUT_SECS:-240}"
FAILURE_BACKOFF_SECS="${TASK_CONDITIONED_CONTROLLER_FAILURE_BACKOFF_SECS:-300}"
POST_ACTION_SLEEP_SECS="${TASK_CONDITIONED_CONTROLLER_POST_ACTION_SLEEP_SECS:-60}"
MAX_CYCLES="${TASK_CONDITIONED_CONTROLLER_MAX_CYCLES:-0}"

REFRESH_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh"

mkdir -p "$RESULT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

refresh_local_state() {
  TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" bash "$REFRESH_SCRIPT" >> "$LOG_FILE" 2>&1 || true
}

declare -A STATE=()
PRE_REFRESH_WAIT_SECS=""
PRE_REFRESH_NEXT_MONITOR="none"
PRE_REFRESH_REMAINING_WAIT="none"

load_current_state() {
  local helper_output
  helper_output="$(
    python3 "$PRINT_STATE_SCRIPT" --format env
  )"

  STATE=()
  while IFS='=' read -r key value; do
    [[ -n "$key" ]] || continue
    STATE["$key"]="$value"
  done <<<"$helper_output"
}

state_value() {
  local key="$1"
  local value="${STATE[$key]:-none}"
  printf '%s\n' "$value"
}

parse_duration_secs() {
  local text="${1:-}"
  if [[ "$text" =~ ^([0-9]+)s$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf '\n'
}

controller_sleep_secs() {
  local decision="$1"
  local until_due
  local poll_interval
  local parsed

  if [[ "$decision" == "monitor_active_full_train" ]]; then
    until_due="$(state_value seconds_until_suggested_monitor)"
    if [[ "$until_due" =~ ^[0-9]+$ && "$until_due" -gt 0 ]]; then
      printf '%s\n' "$until_due"
      return 0
    fi
  fi

  poll_interval="$(state_value recommended_poll_interval)"
  parsed="$(parse_duration_secs "$poll_interval")"
  if [[ -n "$parsed" ]]; then
    printf '%s\n' "$parsed"
    return 0
  fi

  printf '%s\n' "$SLEEP_SECS"
}

pre_refresh_wait_secs() {
  PRE_REFRESH_WAIT_SECS=""
  PRE_REFRESH_NEXT_MONITOR="none"
  PRE_REFRESH_REMAINING_WAIT="none"

  if [[ ! -f "$STATUS_JSON" ]]; then
    return 0
  fi

  load_current_state

  local decision
  local sync_freshness
  local monitor_due_now
  local until_due

  decision="$(state_value current_formal_decision)"
  sync_freshness="$(state_value sync_freshness)"
  monitor_due_now="$(state_value monitor_due_now)"
  until_due="$(state_value seconds_until_suggested_monitor)"

  if [[ "$decision" != "monitor_active_full_train" ]]; then
    return 0
  fi

  if [[ "$sync_freshness" != "fresh_remote" ]]; then
    return 0
  fi

  if [[ "$monitor_due_now" == "False" && "$until_due" =~ ^[0-9]+$ && "$until_due" -gt 0 ]]; then
    PRE_REFRESH_WAIT_SECS="$until_due"
    PRE_REFRESH_NEXT_MONITOR="$(state_value suggested_next_monitor_after)"
    PRE_REFRESH_REMAINING_WAIT="$(state_value remaining_wait_text)"
  fi
}

run_logged_action() {
  local description="$1"
  shift
  log "$description"
  if "$@" >> "$LOG_FILE" 2>&1; then
    log "action complete :: $description"
    return 0
  fi
  local status="$?"
  log "action failed status=$status :: $description"
  return "$status"
}

launch_current_candidate() {
  local launch_key
  launch_key="$(state_value suggested_launch_key)"
  if [[ "$launch_key" == "none" ]]; then
    log "current formal state does not expose a launchable candidate"
    return 1
  fi
  run_logged_action \
    "launching strict full-train candidate key=$launch_key via guarded start wrapper" \
    bash "$START_SCRIPT" "$launch_key"
}

freeze_recorded_then_rerun() {
  local recorded_key
  recorded_key="$(state_value current_recorded_non_winner_target)"
  if [[ "$recorded_key" != "none" ]]; then
    run_logged_action \
      "freezing recorded non-winner key=$recorded_key before relaunch" \
      env TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash "$FINALIZE_EXPLICIT_SCRIPT" "$recorded_key" || return 1
  fi
  launch_current_candidate
}

closeout_current_candidate() {
  local decision="$1"
  run_logged_action \
    "finalizing current formal candidate from authoritative local state" \
    env TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash "$FINALIZE_CURRENT_SCRIPT" || return 1
  run_logged_action \
    "reviewing current formal candidate from authoritative local state" \
    env TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash "$REVIEW_CURRENT_SCRIPT" || return 1
  if [[ "$decision" == "winner_or_done" ]]; then
    log "winner_or_done closeout completed; stopping controller"
    return 10
  fi
  return 0
}

log_monitor_snapshot() {
  local active_target
  local active_model
  local next_monitor
  local remaining_wait
  local timing_summary

  active_target="$(state_value current_active_target)"
  active_model="$(state_value current_active_model)"
  next_monitor="$(state_value suggested_next_monitor_after)"
  remaining_wait="$(state_value remaining_wait_text)"
  timing_summary="$(state_value active_timing_summary)"

  log "monitoring active strict full-train target=${active_target:-none} model=${active_model:-none}"
  if [[ "$timing_summary" != "none" ]]; then
    log "timing summary :: $timing_summary"
  fi
  if [[ "$next_monitor" != "none" ]]; then
    log "next authoritative monitor after=$next_monitor remaining_wait=${remaining_wait:-unknown}"
  fi
}

main() {
  log "task_conditioned controller started"
  local cycle_count=0
  while true; do
    cycle_count="$((cycle_count + 1))"
    pre_refresh_wait_secs
    if [[ -n "$PRE_REFRESH_WAIT_SECS" ]]; then
      log "authoritative local state is still fresh; next monitor after=${PRE_REFRESH_NEXT_MONITOR:-unknown} remaining_wait=${PRE_REFRESH_REMAINING_WAIT:-unknown}; sleeping ${PRE_REFRESH_WAIT_SECS}s without refresh"
      sleep "$PRE_REFRESH_WAIT_SECS"
      if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
        log "task_conditioned controller reached max_cycles=$MAX_CYCLES; stopping"
        break
      fi
      continue
    fi
    log "refreshing local state"
    refresh_local_state

    if [[ ! -f "$STATUS_JSON" ]]; then
      next_sleep_secs="$SLEEP_SECS"
      log "status json not ready; sleeping ${next_sleep_secs}s"
      sleep "$next_sleep_secs"
      continue
    fi

    load_current_state

    decision="$(state_value current_formal_decision)"
    stage="$(state_value stage)"
    next_action="$(state_value next_action)"
    sync_freshness="$(state_value sync_freshness)"
    recommended_poll_interval="$(state_value recommended_poll_interval)"
    log "current state :: stage=$stage next_action=$next_action decision=$decision sync_freshness=$sync_freshness recommended_poll=$recommended_poll_interval"

    action_status=0
    case "$decision" in
      monitor_active_full_train)
        log_monitor_snapshot
        ;;
      launch_next_candidate)
        if launch_current_candidate; then
          next_sleep_secs="$POST_ACTION_SLEEP_SECS"
          log "controller sleeping ${next_sleep_secs}s after launch action"
          sleep "$next_sleep_secs"
          if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
            log "task_conditioned controller reached max_cycles=$MAX_CYCLES; stopping"
            break
          fi
          continue
        fi
        action_status=$?
        ;;
      freeze_recorded_then_rerun)
        if freeze_recorded_then_rerun; then
          next_sleep_secs="$POST_ACTION_SLEEP_SECS"
          log "controller sleeping ${next_sleep_secs}s after freeze-and-rerun action"
          sleep "$next_sleep_secs"
          if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
            log "task_conditioned controller reached max_cycles=$MAX_CYCLES; stopping"
            break
          fi
          continue
        fi
        action_status=$?
        ;;
      review_recorded_or_completed_formal_candidate|winner_or_done)
        if closeout_current_candidate "$decision"; then
          if [[ "$decision" == "winner_or_done" ]]; then
            break
          fi
          next_sleep_secs="$POST_ACTION_SLEEP_SECS"
          log "controller sleeping ${next_sleep_secs}s after closeout action"
          sleep "$next_sleep_secs"
          if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
            log "task_conditioned controller reached max_cycles=$MAX_CYCLES; stopping"
            break
          fi
          continue
        fi
        action_status=$?
        if [[ "$action_status" == "10" ]]; then
          break
        fi
        ;;
      design_next_formal_candidate)
        log "current state has no active strict full-train or rerun; stop controller and design the next one-variable formal candidate"
        break
        ;;
      inspect_authoritative_surfaces)
        log "current state requires manual inspection of authoritative surfaces before any automated action; stopping controller"
        break
        ;;
      *)
        log "unknown current_formal_decision=$decision; stopping controller for manual inspection"
        break
        ;;
    esac

    if [[ "$action_status" -ne 0 ]]; then
      next_sleep_secs="$FAILURE_BACKOFF_SECS"
      log "controller action backoff ${next_sleep_secs}s after failure"
    else
      next_sleep_secs="$(controller_sleep_secs "$decision")"
      log "controller sleeping ${next_sleep_secs}s before next cycle"
    fi

    if [[ "$MAX_CYCLES" != "0" && "$cycle_count" -ge "$MAX_CYCLES" ]]; then
      log "task_conditioned controller reached max_cycles=$MAX_CYCLES; stopping"
      break
    fi

    sleep "$next_sleep_secs"
  done
}

main "$@"
