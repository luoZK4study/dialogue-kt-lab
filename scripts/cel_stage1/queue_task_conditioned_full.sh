#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
LOG_DIR="$ROOT_DIR/results/cel_stage1"
LOG_FILE="$LOG_DIR/task_conditioned_queue.log"

mkdir -p "$LOG_DIR"

wait_for_slot() {
  while true; do
    local mlp_pid adapter_pid
    mlp_pid="$(pgrep -f "model_name cel_mlp_qwen3_1.7b" || true)"
    adapter_pid="$(pgrep -f "model_name cel_adapter_qwen3_1.7b" || true)"

    if [[ -z "$mlp_pid" ]]; then
      echo 0
      return 0
    fi
    if [[ -z "$adapter_pid" ]]; then
      echo 1
      return 0
    fi

    {
      date +"[%F %T] waiting for CEL full-train slot..."
      echo "  mlp pid: ${mlp_pid}"
      echo "  adapter pid: ${adapter_pid}"
    } >> "$LOG_FILE"
    sleep 60
  done
}

main() {
  {
    date +"[%F %T] queue script started"
    echo "root: $ROOT_DIR"
  } >> "$LOG_FILE"

  cd "$ROOT_DIR"
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate luo_2
  export WANDB_MODE=disabled
  export CUBLAS_WORKSPACE_CONFIG=:4096:8

  local gpu_id
  gpu_id="$(wait_for_slot)"

  {
    date +"[%F %T] launching task-conditioned full training"
    echo "gpu: $gpu_id"
  } >> "$LOG_FILE"

  CUDA_VISIBLE_DEVICES="$gpu_id" python -m dialogue_kt.main train \
    --dataset mathdial --model_type lmkt \
    --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
    --model_name cel_task_conditioned_qwen3_1.7b \
    --epochs 2 --quantize 0 \
    --cel_mode task_conditioned --cel_layer_idx 12 \
    --cel_selector_hidden_dim 512 --cel_gamma 1.0 \
    >> "$LOG_FILE" 2>&1
}

main "$@"
