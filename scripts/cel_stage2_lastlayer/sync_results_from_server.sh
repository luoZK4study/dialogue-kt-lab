#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
SSH_TARGET="user4@119.29.183.125"
SSH_OPTS="-i ~/.ssh/id_rsa_u4 -p 22049"
REMOTE_ROOT="/home/user4/dialogue-kt"

sync_dir_with_fallback() {
  local local_dir="$1"
  shift
  local rel_path
  for rel_path in "$@"; do
    if rsync -avz -e "ssh $SSH_OPTS" \
      "$SSH_TARGET:$REMOTE_ROOT/$rel_path/" \
      "$local_dir/"; then
      return 0
    fi
  done
  return 0
}

sync_log_file() {
  local log_name="$1"
  local remote_path="$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1_last_layer/$log_name"
  local legacy_remote_path="$SSH_TARGET:$REMOTE_ROOT/results/cel_stage2_lastlayer/$log_name"
  local local_dir="$ROOT_DIR/results/cel_stage1_last_layer/"

  # Logs are often still growing while the remote run is active. Append-verify
  # plus a quiet second pass reduces one-cycle staleness in local status files.
  rsync -avz --append-verify -e "ssh $SSH_OPTS" \
    "$remote_path" \
    "$local_dir" || rsync -avz --append-verify -e "ssh $SSH_OPTS" \
    "$legacy_remote_path" \
    "$local_dir" || true

  rsync -az --append-verify -e "ssh $SSH_OPTS" \
    "$remote_path" \
    "$local_dir" || rsync -az --append-verify -e "ssh $SSH_OPTS" \
    "$legacy_remote_path" \
    "$local_dir" || true
}

mkdir -p \
  "$ROOT_DIR/results/baseline/metrics" \
  "$ROOT_DIR/results/baseline/qual" \
  "$ROOT_DIR/results/baseline/kcs" \
  "$ROOT_DIR/results/cel_stage1_last_layer/metrics" \
  "$ROOT_DIR/results/cel_stage1_last_layer/qual" \
  "$ROOT_DIR/results/cel_stage1_last_layer/kcs" \
  "$ROOT_DIR/results/cel_stage1_last_layer/step_logs"

sync_dir_with_fallback "$ROOT_DIR/results/baseline/metrics" \
  "results/baseline/metrics" \
  "results/baseline_recert/metrics"

sync_dir_with_fallback "$ROOT_DIR/results/baseline/qual" \
  "results/baseline/qual" \
  "results/baseline_recert/qual"

sync_dir_with_fallback "$ROOT_DIR/results/baseline/kcs" \
  "results/baseline/kcs" \
  "results/baseline_recert/kcs"

sync_dir_with_fallback "$ROOT_DIR/results/cel_stage1_last_layer/metrics" \
  "results/cel_stage1_last_layer/metrics" \
  "results/cel_stage2_lastlayer/metrics"

sync_dir_with_fallback "$ROOT_DIR/results/cel_stage1_last_layer/qual" \
  "results/cel_stage1_last_layer/qual" \
  "results/cel_stage2_lastlayer/qual"

sync_dir_with_fallback "$ROOT_DIR/results/cel_stage1_last_layer/kcs" \
  "results/cel_stage1_last_layer/kcs" \
  "results/cel_stage2_lastlayer/kcs"

sync_dir_with_fallback "$ROOT_DIR/results/cel_stage1_last_layer/step_logs" \
  "results/cel_stage1_last_layer/step_logs" \
  "results/cel_stage2_lastlayer/step_logs"

for log_name in \
  loop.log \
  followup.log \
  loop_stdout.log \
  followup_stdout.log \
  supervisor.log \
  supervisor_stdout.log \
  task_conditioned_v11_cal_bias_only.stdout.log \
  task_conditioned_v12_cal_affine_only.stdout.log \
  task_conditioned_v13_selector_cal_bias.stdout.log \
  task_conditioned_v14_selector_cal_affine.stdout.log; do
  sync_log_file "$log_name"
done
