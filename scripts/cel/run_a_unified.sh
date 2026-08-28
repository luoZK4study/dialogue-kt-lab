#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
MODEL_NAME="${MODEL_NAME:-cel_a_unified_qwen3_1.7b}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
PATIENCE="${PATIENCE:-1}"
MIN_DELTA="${MIN_DELTA:-0.0}"

cd "$ROOT_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

exec "$PYTHON_BIN" -m dialogue_kt.main train \
  --dataset mathdial \
  --model_type lmkt \
  --base_model "$BASE_MODEL" \
  --model_name "$MODEL_NAME" \
  --result_subdir a \
  --quantize 0 \
  --max_epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --cel_mode task_conditioned \
  --cel_layer_idx -1 \
  --cel_hook_site last_block \
  --cel_hook_timing pre_block \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 0.30 \
  --cel_use_norm 0 \
  --cel_injection_variant scalar_gate \
  --cel_application_mode token_residual \
  --cel_output_calibration none
