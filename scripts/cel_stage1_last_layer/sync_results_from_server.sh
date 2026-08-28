#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"
REMOTE_EXISTING_PATHS=""

LOG_FILES=(
  loop.log
  followup.log
  loop_stdout.log
  followup_stdout.log
  supervisor.log
  supervisor_stdout.log
  task_conditioned_v11_cal_bias_only.stdout.log
  task_conditioned_v12_cal_affine_only.stdout.log
  task_conditioned_v13_selector_cal_bias.stdout.log
  task_conditioned_v14_selector_cal_affine.stdout.log
  task_conditioned_v17_selector_cal_bias_memfix.stdout.log
  task_conditioned_v15_fullv1_cal_bias_only.stdout.log
  task_conditioned_v16_fullv1_cal_affine_only.stdout.log
  task_conditioned_v18_fullv1_fixed_bias_eval.stdout.log
  task_conditioned_v19_fullv1_cal_bias_only_tinylr.stdout.log
  task_conditioned_v20_fullv1_valfit_bias.stdout.log
  task_conditioned_v26_bootstrap.stdout.log
  task_conditioned_v26_calibrator_warmup.stdout.log
  task_conditioned_v26_selftrained_joint.stdout.log
)

for key in "${FORMAL_CANDIDATE_KEYS[@]}"; do
  LOG_FILES+=("${FORMAL_STDOUT_LOG_NAME[$key]}")
done

build_remote_manifest() {
  local manifest
  if ! manifest="$(
    {
      printf '%s\n' \
        "results/baseline/metrics" \
        "results/baseline_recert/metrics" \
        "results/baseline/qual" \
        "results/baseline_recert/qual" \
        "results/baseline/kcs" \
        "results/baseline_recert/kcs" \
        "results/cel_stage1_last_layer/metrics" \
        "results/cel_stage2_lastlayer/metrics" \
        "results/cel_stage1_last_layer/qual" \
        "results/cel_stage2_lastlayer/qual" \
        "results/cel_stage1_last_layer/kcs" \
        "results/cel_stage2_lastlayer/kcs" \
        "results/cel_stage1_last_layer/step_logs" \
        "results/cel_stage2_lastlayer/step_logs" \
        "results/cel_stage1_last_layer/V26_SELFTRAINED_AUDIT.json" \
        "results/cel_stage1_last_layer/V26_SELFTRAINED_AUDIT.md"
      local log_name
      for log_name in "${LOG_FILES[@]}"; do
        printf 'results/cel_stage1_last_layer/%s\n' "$log_name"
        printf 'results/cel_stage2_lastlayer/%s\n' "$log_name"
      done
    } | ssh_run "$SSH_TARGET" \
      "cd '$REMOTE_ROOT' && while IFS= read -r rel_path; do if [ -e \"\$rel_path\" ]; then printf '%s\n' \"\$rel_path\"; fi; done; true"
  )"; then
    REMOTE_EXISTING_PATHS=""
    return 1
  fi
  REMOTE_EXISTING_PATHS="$manifest"
}

remote_path_exists() {
  local rel_path="$1"
  grep -Fqx "$rel_path" <<<"$REMOTE_EXISTING_PATHS"
}

sync_dir_with_fallback() {
  local local_dir="$1"
  shift
  local rel_path
  for rel_path in "$@"; do
    if ! remote_path_exists "$rel_path"; then
      continue
    fi
    rsync -az -e "$(rsync_ssh_command)" \
      "$SSH_TARGET:$REMOTE_ROOT/$rel_path/" \
      "$local_dir/"
    return 0
  done
  return 0
}

sync_log_file() {
  local log_name="$1"
  local remote_rel="results/cel_stage1_last_layer/$log_name"
  local legacy_remote_rel="results/cel_stage2_lastlayer/$log_name"
  local local_dir="$ROOT_DIR/results/cel_stage1_last_layer/"
  local local_path="$local_dir/$log_name"

  sync_one_log_path() {
    local active_remote_rel="$1"
    local remote_size
    remote_size="$(ssh_run "$SSH_TARGET" "cd '$REMOTE_ROOT' && wc -c < '$active_remote_rel'" 2>/dev/null)"
    remote_size="${remote_size//[[:space:]]/}"
    if [[ -z "$remote_size" ]]; then
      return 1
    fi

    if [[ -f "$local_path" ]]; then
      local local_size
      local_size="$(wc -c < "$local_path")"
      if [[ "$local_size" -gt "$remote_size" ]]; then
        rm -f "$local_path"
      fi
    fi

    rsync -az --append-verify -e "$(rsync_ssh_command)" \
      "$SSH_TARGET:$REMOTE_ROOT/$active_remote_rel" \
      "$local_dir"
    rsync -az --append-verify -e "$(rsync_ssh_command)" \
      "$SSH_TARGET:$REMOTE_ROOT/$active_remote_rel" \
      "$local_dir"
    return 0
  }

  # Logs are often still growing while the remote run is active. Append-verify
  # plus a quiet second pass reduces one-cycle staleness in local status files.
  if remote_path_exists "$remote_rel"; then
    sync_one_log_path "$remote_rel"
    return 0
  fi

  if remote_path_exists "$legacy_remote_rel"; then
    sync_one_log_path "$legacy_remote_rel"
  fi
}

sync_result_file() {
  local file_name="$1"
  local remote_rel="results/cel_stage1_last_layer/$file_name"
  if ! remote_path_exists "$remote_rel"; then
    return 0
  fi
  rsync -az -e "$(rsync_ssh_command)" \
    "$SSH_TARGET:$REMOTE_ROOT/$remote_rel" \
    "$ROOT_DIR/results/cel_stage1_last_layer/"
}

mkdir -p \
  "$ROOT_DIR/results/baseline/metrics" \
  "$ROOT_DIR/results/baseline/qual" \
  "$ROOT_DIR/results/baseline/kcs" \
  "$ROOT_DIR/results/cel_stage1_last_layer/metrics" \
  "$ROOT_DIR/results/cel_stage1_last_layer/qual" \
  "$ROOT_DIR/results/cel_stage1_last_layer/kcs" \
  "$ROOT_DIR/results/cel_stage1_last_layer/step_logs"

build_remote_manifest

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

for log_name in "${LOG_FILES[@]}"; do
  sync_log_file "$log_name"
done

sync_result_file "V26_SELFTRAINED_AUDIT.json"
sync_result_file "V26_SELFTRAINED_AUDIT.md"
