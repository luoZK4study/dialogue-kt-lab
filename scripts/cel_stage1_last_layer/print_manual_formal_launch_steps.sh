#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL_DOC="$ROOT_DIR/results/cel_stage1_last_layer/FORMAL_METHOD_PROTOCOL.md"
RUNBOOK_DOC="$ROOT_DIR/results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md"
PRINT_STATE_SCRIPT="$ROOT_DIR/scripts/cel_stage1_last_layer/print_current_formal_next_action.py"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

CANDIDATE_KEY="${1:-}"
FORCE_RERUN="${TASK_CONDITIONED_FORCE_RERUN:-0}"
GPU_ID="${TASK_CONDITIONED_GPU_ID:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh <candidate_key>

Purpose:
  Print a strict full-train manual SSH fallback sequence for environments that
  cannot directly run the automated launch wrapper.

Environment:
  TASK_CONDITIONED_SSH_TARGET=<target>  Override SSH target, e.g. 3090
  TASK_CONDITIONED_SSH_OPTS=<opts>      Override SSH opts; set empty when using SSH config alias
  TASK_CONDITIONED_REMOTE_ROOT=<path>   Override remote repo root
EOF
}

if [[ -z "$CANDIDATE_KEY" ]]; then
  usage >&2
  exit 2
fi

formal_candidate_require_key "$CANDIDATE_KEY" || {
  usage >&2
  exit 2
}

MODEL_NAME="${FORMAL_MODEL_NAME[$CANDIDATE_KEY]}"
RUN_SCRIPT_REL="${FORMAL_RUN_SCRIPT_REL[$CANDIDATE_KEY]}"
STDOUT_LOG_REL="${FORMAL_STDOUT_LOG_REL[$CANDIDATE_KEY]}"
METRICS_REL="${FORMAL_METRICS_REL[$CANDIDATE_KEY]}"
MODEL_PATTERN="${FORMAL_MODEL_PATTERN[$CANDIDATE_KEY]}"
DEFAULT_GPU_ID="${FORMAL_DEFAULT_GPU_ID[$CANDIDATE_KEY]}"
REMOTE_HELPER_REL="scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh"
if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$DEFAULT_GPU_ID"
fi

current_formal_decision="unknown"
current_rerun_target="none"
current_recorded_non_winner="none"
current_active_target="none"
if helper_output="$(python3 "$PRINT_STATE_SCRIPT" --format env 2>/dev/null)"; then
  while IFS='=' read -r key value; do
    [[ -n "$key" ]] || continue
    case "$key" in
      current_formal_decision) current_formal_decision="$value" ;;
      current_rerun_target) current_rerun_target="$value" ;;
      current_recorded_non_winner_target) current_recorded_non_winner="$value" ;;
      current_active_target) current_active_target="$value" ;;
    esac
  done <<<"$helper_output"
fi

cat <<EOF
== Manual Strict Formal Launch Fallback ==
candidate_key=$CANDIDATE_KEY
model_name=$MODEL_NAME
default_gpu_id=$DEFAULT_GPU_ID
selected_gpu_id=$GPU_ID
force_rerun=$FORCE_RERUN
strict_scope=SSH train + val + test
strict_scope_guard=no_ckpt_only_no_post_hoc_no_frozen_retraining
duplicate_process_guard=enabled
existing_metrics_guard=$([[ "$FORCE_RERUN" == "1" ]] && echo "override_with_backup" || echo "enabled")
ssh_target=$SSH_TARGET
ssh_opts=$SSH_OPTS
remote_root=$REMOTE_ROOT
formal_method_protocol=$PROTOCOL_DOC
strict_runbook=$RUNBOOK_DOC

Run the following from a terminal that can reach the SSH server:

1. Local preflight
   bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh

2. Sync current local strict-loop code to SSH
   bash scripts/cel_stage1_last_layer/sync_code_to_server.sh
   bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh 3090

   If you prefer your local SSH config alias, for example:
   TASK_CONDITIONED_SSH_TARGET=3090 TASK_CONDITIONED_SSH_OPTS= bash scripts/cel_stage1_last_layer/sync_code_to_server.sh

3. SSH into the server
   $(ssh_target_command)

4. On SSH, move into the repo and check whether the run is already active
   cd $REMOTE_ROOT
   ps -ef | grep -E "$MODEL_PATTERN" | grep -v grep

5. Still on SSH, check whether synced formal metrics already exist
   cd $REMOTE_ROOT
   test -e "$METRICS_REL" && echo "metrics_present" || echo "metrics_missing"

6. Launch policy
   - If step 4 finds an active process, stop and do not launch a duplicate run.
   - If step 5 shows metrics already exist and force_rerun=0, stop and keep the recorded result.
   - Only continue with a relaunch when you explicitly intend a strict rerun of the same candidate.

7. If the duplicate-process and existing-metrics guards are satisfied, launch through the shared remote helper
   cd $REMOTE_ROOT
   TASK_CONDITIONED_GPU_ID="$GPU_ID" TASK_CONDITIONED_FORCE_RERUN="$FORCE_RERUN" bash "$REMOTE_HELPER_REL" "$CANDIDATE_KEY"

   Notes:
   - This shared helper enforces the same duplicate-process guard as the automated SSH launcher.
   - If force_rerun=1, it also backs up the old log and old metrics before relaunch.

8. On SSH, verify the log is moving
   tail -n 40 "$STDOUT_LOG_REL"

9. Monitoring cadence after launch
   - Early long training: poll about every 900s.
   - Stable mid-run training: poll about every 600s.
   - Validation / Testing, >=95% progress, or short ETA: poll about every 300s.
   - Authoritative cadence source of truth remains STATUS.md / STRICT_FULL_TRAIN_REPORT.md after each refresh.

10. Back on local, use the unified monitor entrypoint for one refresh cycle at a time
   bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once
   TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
   bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once
   TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch

11. After the SSH run completes, close out in the required order
   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090
   bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090
   bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh $CANDIDATE_KEY
   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh $CANDIDATE_KEY
   bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 $CANDIDATE_KEY
   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 $CANDIDATE_KEY

12. If the current local environment still cannot SSH during closeout, fall back to snapshot-only review
   TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
   TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
   TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh $CANDIDATE_KEY
   TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh $CANDIDATE_KEY
   TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 $CANDIDATE_KEY
   TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 $CANDIDATE_KEY

Expected remote artifacts after completion:
  - $METRICS_REL
  - $STDOUT_LOG_REL

Closeout rule:
  - Do not judge winner / non-winner from raw log output alone.
  - Only treat the run as formally analyzed after finalize -> audit -> review is complete.
  - If the latest authoritative refresh event is not sync=ok, local markdown movement is still snapshot-only.

Current strict queue reminder:
  - current formal decision: $current_formal_decision
  - rerun-first candidate: $current_rerun_target
  - active full-train candidate: $current_active_target
  - recorded non-winner: $current_recorded_non_winner
EOF
