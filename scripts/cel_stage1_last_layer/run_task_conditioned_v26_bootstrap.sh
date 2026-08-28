#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m dialogue_kt.main train \
  --dataset mathdial \
  --model_type lmkt \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --model_name cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b \
  --result_subdir cel_stage1_last_layer/v26_stages/bootstrap \
  --skip_test_after_train 1 \
  --epochs 2 \
  --lr 0.0002 \
  --wd 0.01 \
  --quantize 0 \
  --cel_mode task_conditioned \
  --cel_layer_idx -1 \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 0.30 \
  --cel_use_norm 0
