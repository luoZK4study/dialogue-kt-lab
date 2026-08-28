#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m dialogue_kt.main train \
  --dataset mathdial \
  --model_type lmkt \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --model_name cel_mlp_qwen3_1.7b \
  --epochs 2 \
  --quantize 0 \
  --cel_mode mlp \
  --cel_layer_idx 12 \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 1.0
