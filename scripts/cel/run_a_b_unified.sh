#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
CANDIDATE_ID="${CANDIDATE_ID:-a_b_unified_contextual_transformer}"
MODEL_NAME="${MODEL_NAME:-cel_${CANDIDATE_ID}_qwen3_1.7b}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
PATIENCE="${PATIENCE:-1}"
MIN_DELTA="${MIN_DELTA:-0.0}"

cd "$ROOT_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OPTIONAL_LR_ARGS=()
if [[ -n "${SELECTOR_LR:-}" ]]; then
  OPTIONAL_LR_ARGS+=(--cel_selector_lr "$SELECTOR_LR")
fi
if [[ -n "${ENVIRONMENT_LR:-}" ]]; then
  OPTIONAL_LR_ARGS+=(--cel_environment_lr "$ENVIRONMENT_LR")
fi

exec "$PYTHON_BIN" -m dialogue_kt.main train \
  --dataset mathdial \
  --model_type lmkt \
  --base_model "$BASE_MODEL" \
  --model_name "$MODEL_NAME" \
  --result_subdir a_b \
  --quantize 0 \
  --max_epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --min_delta "$MIN_DELTA" \
  --model_init_seed "${MODEL_INIT_SEED:-1221}" \
  --cel_mode task_conditioned \
  --cel_layer_idx -1 \
  --cel_hook_site last_block \
  --cel_hook_timing pre_block \
  --cel_selector_hidden_dim 512 \
  --cel_gamma 0.30 \
  --cel_use_norm 0 \
  --cel_injection_variant scalar_gate \
  --cel_application_mode token_residual \
  --cel_stage2_enabled 1 \
  --cel_stage2_fresh_init 1 \
  --cel_stage2_candidate_id "$CANDIDATE_ID" \
  --cel_env_mode contextual_transformer \
  --cel_env_split_mode complementary \
  --cel_env_beta "${ENV_BETA:-0.10}" \
  --cel_env_hidden_dim "${ENV_HIDDEN_DIM:-1024}" \
  --cel_env_num_layers "${ENV_NUM_LAYERS:-4}" \
  --cel_env_num_heads "${ENV_NUM_HEADS:-8}" \
  --cel_env_ffn_dim "${ENV_FFN_DIM:-4096}" \
  --cel_env_drop "${ENV_DROP:-0.10}" \
  --cel_env_output_postprocess centered_rms \
  --cel_env_output_ratio "${ENV_OUTPUT_RATIO:-1.0}" \
  --cel_env_output_init_std "${ENV_OUTPUT_INIT_STD:-0.01}" \
  --cel_stage2_lambda_r "${LAMBDA_R:-1.0}" \
  --cel_stage2_lambda_m "${LAMBDA_M:-1.0}" \
  --cel_stage2_lambda_cons "${LAMBDA_CONS:-0.10}" \
  "${OPTIONAL_LR_ARGS[@]}" \
  --cel_output_calibration none
