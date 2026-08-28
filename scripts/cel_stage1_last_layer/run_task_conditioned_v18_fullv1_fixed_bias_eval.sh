#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"
SOURCE_MODEL="cel_task_conditioned_lastlayer_v1_qwen3_1.7b"
MODEL_NAME="cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b"
SOURCE_CKPT_DIR="$ROOT_DIR/saved_models/$SOURCE_MODEL"
TARGET_CKPT_DIR="$ROOT_DIR/saved_models/$MODEL_NAME"

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$TARGET_CKPT_DIR"
cp -a "$SOURCE_CKPT_DIR"/. "$TARGET_CKPT_DIR"/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m dialogue_kt.main test \
  --dataset mathdial \
  --model_type lmkt \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --model_name "$MODEL_NAME" \
  --result_subdir cel_stage1_last_layer \
  --quantize 0 \
  --cel_mode task_conditioned \
  --cel_layer_idx -1 \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 0.30 \
  --cel_use_norm 0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331
