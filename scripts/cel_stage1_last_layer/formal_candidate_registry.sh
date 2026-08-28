#!/usr/bin/env bash

# Shared formal Round 3 candidate registry for shell-side orchestration.

FORMAL_CANDIDATE_KEYS=(v21 v22 v23 v24 v25 v26)

declare -A FORMAL_MODEL_NAME=(
  [v21]="cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b"
  [v22]="cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b"
  [v23]="cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b"
  [v24]="cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b"
  [v25]="cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b"
  [v26]="cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b"
)

declare -A FORMAL_MODEL_PATTERN=(
  [v21]="task_conditioned_lastlayer_v21"
  [v22]="task_conditioned_lastlayer_v22"
  [v23]="task_conditioned_lastlayer_v23_joint_bias_posinit"
  [v24]="task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain"
  [v25]="task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x"
  [v26]="run_task_conditioned_v26_selftrained|task_conditioned_lastlayer_v26_"
)

declare -A FORMAL_RUN_SCRIPT_NAME=(
  [v21]="run_task_conditioned_v21_joint_bias_fulltrain.sh"
  [v22]="run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh"
  [v23]="run_task_conditioned_v23_joint_bias_posinit.sh"
  [v24]="run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh"
  [v25]="run_task_conditioned_v25_joint_bias_calibratorlr10x.sh"
  [v26]="run_task_conditioned_v26_selftrained.sh"
)

declare -A FORMAL_RUN_SCRIPT_REL=(
  [v21]="scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh"
  [v22]="scripts/cel_stage1_last_layer/run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh"
  [v23]="scripts/cel_stage1_last_layer/run_task_conditioned_v23_joint_bias_posinit.sh"
  [v24]="scripts/cel_stage1_last_layer/run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh"
  [v25]="scripts/cel_stage1_last_layer/run_task_conditioned_v25_joint_bias_calibratorlr10x.sh"
  [v26]="scripts/cel_stage1_last_layer/run_task_conditioned_v26_selftrained.sh"
)

declare -A FORMAL_STDOUT_LOG_NAME=(
  [v21]="task_conditioned_v21_joint_bias_fulltrain.stdout.log"
  [v22]="task_conditioned_v22_fullv1_joint_bias_tinylr.stdout.log"
  [v23]="task_conditioned_v23_joint_bias_posinit.stdout.log"
  [v24]="task_conditioned_v24_selectorinit_joint_bias_fulltrain.stdout.log"
  [v25]="task_conditioned_v25_joint_bias_calibratorlr10x.stdout.log"
  [v26]="task_conditioned_v26_selftrained.stdout.log"
)

declare -A FORMAL_STDOUT_LOG_REL=(
  [v21]="results/cel_stage1_last_layer/task_conditioned_v21_joint_bias_fulltrain.stdout.log"
  [v22]="results/cel_stage1_last_layer/task_conditioned_v22_fullv1_joint_bias_tinylr.stdout.log"
  [v23]="results/cel_stage1_last_layer/task_conditioned_v23_joint_bias_posinit.stdout.log"
  [v24]="results/cel_stage1_last_layer/task_conditioned_v24_selectorinit_joint_bias_fulltrain.stdout.log"
  [v25]="results/cel_stage1_last_layer/task_conditioned_v25_joint_bias_calibratorlr10x.stdout.log"
  [v26]="results/cel_stage1_last_layer/task_conditioned_v26_selftrained.stdout.log"
)

declare -A FORMAL_METRICS_REL=(
  [v21]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b.txt"
  [v22]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b.txt"
  [v23]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b.txt"
  [v24]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b.txt"
  [v25]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b.txt"
  [v26]="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b.txt"
)

declare -A FORMAL_DEFAULT_GPU_ID=(
  [v21]="0"
  [v22]="1"
  [v23]="0"
  [v24]="0"
  [v25]="0"
  [v26]="0"
)

formal_candidate_require_key() {
  local key="$1"
  if [[ -z "${FORMAL_MODEL_NAME[$key]+x}" ]]; then
    echo "unknown formal candidate key: $key" >&2
    return 1
  fi
}
