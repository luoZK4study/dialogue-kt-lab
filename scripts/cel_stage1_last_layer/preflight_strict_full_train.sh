#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS_JSON="$ROOT_DIR/results/cel_stage1_last_layer/task_conditioned_status.json"
FORMAL_AUDIT_JSON="$ROOT_DIR/results/cel_stage1_last_layer/formal_experiment_audit.json"
REBUILD_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/rebuild_task_conditioned_reports.sh"
PREFLIGHT_ALIAS="${TASK_CONDITIONED_PREFLIGHT_SSH_ALIAS-3090}"
PREFLIGHT_ALIAS_REQUIRED="${TASK_CONDITIONED_PREFLIGHT_SSH_ALIAS_REQUIRED:-0}"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

alias_is_explicitly_configured() {
  [[ "${TASK_CONDITIONED_PREFLIGHT_SSH_ALIAS+x}" == "x" ]]
}

echo "== Strict Full-Train Preflight =="

require_file "$ROOT_DIR/dialogue_kt/training.py"
require_file "$ROOT_DIR/dialogue_kt/main.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/current_formal_state.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/task_conditioned_failure_utils.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_experiment_audit.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_method_protocol.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_briefs.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_config_diffs.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/print_formal_candidate_context.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_rerun_readiness.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/scaffold_formal_candidate.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_registry_alignment.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_surface_consistency.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_probability_bce_safety.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/check_sync_manifest_coverage.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_guard.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_state.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.py"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/review_formal_candidate.sh"
require_file "$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate.sh"
require_file "$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md"
require_file "$ROOT_DIR/results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md"

for key in "${FORMAL_CANDIDATE_KEYS[@]}"; do
  require_file "$ROOT_DIR/${FORMAL_RUN_SCRIPT_REL[$key]}"
done

echo "rebuilding local task-conditioned reports"
bash "$REBUILD_SCRIPT"

echo "local ssh config check (direct/default path)"
bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"

if [[ -n "$PREFLIGHT_ALIAS" ]]; then
  if ssh -G "$PREFLIGHT_ALIAS" >/dev/null 2>&1; then
    echo "local ssh config check (alias path: $PREFLIGHT_ALIAS)"
    TASK_CONDITIONED_SSH_TARGET="$PREFLIGHT_ALIAS" \
    TASK_CONDITIONED_SSH_OPTS= \
      bash "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh"
  elif [[ "$PREFLIGHT_ALIAS_REQUIRED" == "1" || alias_is_explicitly_configured ]]; then
    echo "preflight failed: unable to resolve configured SSH alias '$PREFLIGHT_ALIAS' for alias-path validation" >&2
    exit 1
  else
    echo "note: skipping alias-path ssh config check because alias '$PREFLIGHT_ALIAS' is not resolvable in this shell"
  fi
fi

require_file "$STATUS_JSON"
require_file "$FORMAL_AUDIT_JSON"
require_file "$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_METHOD_PROTOCOL.md"
require_file "$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md"
require_file "$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md"

python3 - <<'PY' "$STATUS_JSON" "$FORMAL_AUDIT_JSON" "${FORMAL_MODEL_NAME[@]}"
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
expected_models = set(sys.argv[3:])
data = json.loads(status_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
rounds = data.get("rounds") or []
round3 = None
for entry in rounds:
    if str(entry.get("title", "")).startswith("Round 3"):
        round3 = entry
        break

if round3 is None:
    raise SystemExit("Round 3 state missing from task_conditioned_status.json")

models = [row.get("model") for row in (round3.get("models") or [])]
if set(models) != expected_models:
    raise SystemExit(f"unexpected Round 3 model set: {models}")

winner_found = bool(data.get("winner_found"))
next_action = str(data.get("next_action"))
rows = audit.get("rows") or []
recorded = [row.get("label") for row in rows if row.get("audit") == "recorded"]
needs_rerun = [row.get("label") for row in rows if row.get("audit") == "needs_rerun"]
print(f"next_action={next_action}")
print(f"winner_found={winner_found}")
print(f"round3_models={','.join(models)}")
if next_action == "manual_decide" and needs_rerun:
    recorded_text = ",".join(label for label in recorded if label) or "none"
    needs_rerun_text = ",".join(label for label in needs_rerun if label)
    print(f"formal_loop_hint=freeze_recorded({recorded_text})_rerun_first({needs_rerun_text})")
elif next_action == "wait_round3":
    print("formal_loop_hint=continue_sync_and_monitor_current_full_train")
PY

echo "py_compile check"
python3 -m py_compile \
  "$ROOT_DIR/dialogue_kt/main.py" \
  "$ROOT_DIR/dialogue_kt/training.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_registry_alignment.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_surface_consistency.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_probability_bce_safety.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/current_formal_state.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_sync_manifest_coverage.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_guard.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_state.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/task_conditioned_failure_utils.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_strict_full_train_report.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_experiment_audit.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_method_protocol.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_briefs.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_config_diffs.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/print_formal_candidate_context.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_rerun_readiness.py" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/scaffold_formal_candidate.py"

echo "registry alignment check"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_candidate_registry_alignment.py"

echo "formal surface consistency check"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/check_formal_surface_consistency.py"

echo "probability BCE safety check"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/check_probability_bce_safety.py"

echo "sync manifest coverage check"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/check_sync_manifest_coverage.py"

echo "formal queue summary"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_queue_guard.py" --mode summary

echo "current formal next action"
python3 "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"

echo "shell syntax check"
bash -n \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/check_ssh_formal_config.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/monitor_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/review_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/review_current_formal_candidate.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_code_to_server.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/sync_results_from_server.sh" \
  "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"

echo "preflight passed"
