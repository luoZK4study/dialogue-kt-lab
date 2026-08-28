#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user4/dialogue-kt"

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python -m dialogue_kt.main train \
  --dataset mathdial \
  --model_type lmkt \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --pt_model_name cel_task_conditioned_lastlayer_v1_qwen3_1.7b \
  --cel_selector_init_model_name cel_task_conditioned_lastlayer_v1_qwen3_1.7b \
  --model_name cel_task_conditioned_lastlayer_v19_fullv1_cal_bias_only_tinylr_qwen3_1.7b \
  --result_subdir cel_stage1_last_layer \
  --epochs 1 \
  --lr 0.00001 \
  --wd 0.0 \
  --quantize 0 \
  --cel_mode task_conditioned \
  --cel_layer_idx -1 \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 0.30 \
  --cel_use_norm 0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331 \
  --cel_train_calibrator_only 1
