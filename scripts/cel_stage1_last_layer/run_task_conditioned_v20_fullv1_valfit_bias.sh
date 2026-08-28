#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/cel_stage1_last_layer/fit_val_bias_calibration.py \
  --source_model_name cel_task_conditioned_lastlayer_v1_qwen3_1.7b \
  --target_model_name cel_task_conditioned_lastlayer_v20_fullv1_valfit_bias_qwen3_1.7b \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --result_subdir cel_stage1_last_layer
