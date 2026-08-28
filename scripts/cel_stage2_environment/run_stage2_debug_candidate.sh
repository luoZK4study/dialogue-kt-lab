#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
DIRECTION="${1:-shuffle}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
BASE_MODEL="${STAGE2_BASE_MODEL:-/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B}"
MODEL_SEED="${STAGE2_MODEL_SEED:-9321}"

case "$DIRECTION" in
  shuffle|mlp|transformer) ;;
  *) echo "usage: $0 {shuffle|mlp|transformer}" >&2; exit 2 ;;
esac

CANDIDATE_ID="${STAGE2_CANDIDATE_ID:-stage2_debug_${DIRECTION}_hookfix_seed${MODEL_SEED}}"
ENV_BETA="${STAGE2_ENV_BETA:-0.10}"
ENV_TOPK_RATIO="${STAGE2_ENV_TOPK_RATIO:-0.20}"
ENV_HIDDEN_DIM="${STAGE2_ENV_HIDDEN_DIM:-512}"
ENV_NUM_LAYERS="${STAGE2_ENV_NUM_LAYERS:-1}"
ENV_NUM_HEADS="${STAGE2_ENV_NUM_HEADS:-4}"
ENV_FFN_DIM="${STAGE2_ENV_FFN_DIM:-1024}"
ENV_OUTPUT_POSTPROCESS="${STAGE2_ENV_OUTPUT_POSTPROCESS:-none}"
ENV_OUTPUT_RATIO="${STAGE2_ENV_OUTPUT_RATIO:-0.10}"
ENV_OUTPUT_INIT_STD="${STAGE2_ENV_OUTPUT_INIT_STD:-0.001}"

RESULT_ROOT="$ROOT_DIR/results/cel_stage2_environment/debug/$CANDIDATE_ID"
STAGE_LOG_DIR="$RESULT_ROOT/stages"
BOOTSTRAP_MODEL="cel_${CANDIDATE_ID}_bootstrap_qwen3_1.7b"
WARMUP_MODEL="cel_${CANDIDATE_ID}_calibrator_warmup_qwen3_1.7b"
FINAL_MODEL="cel_${CANDIDATE_ID}_joint_qwen3_1.7b"

if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing Stage 2 debug launch outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to overwrite Stage 2 debug results: $RESULT_ROOT" >&2
  exit 1
fi
for model_name in "$BOOTSTRAP_MODEL" "$WARMUP_MODEL" "$FINAL_MODEL"; do
  if [[ -e "$ROOT_DIR/saved_models/$model_name" ]]; then
    echo "refusing to overwrite Stage 2 debug checkpoint: saved_models/$model_name" >&2
    exit 1
  fi
done
mkdir -p "$STAGE_LOG_DIR"

COMMON_ARGS=(
  --dataset mathdial
  --model_type lmkt
  --base_model "$BASE_MODEL"
  --quantize 0
  --debug
  --model_init_seed "$MODEL_SEED"
  --cel_mode task_conditioned
  --cel_layer_idx -1
  --cel_hook_site last_block
  --cel_hook_timing pre_block
  --cel_selector_hidden_dim 512
  --cel_gamma 0.30
  --cel_use_norm 0
  --cel_injection_variant scalar_gate
  --cel_application_mode token_residual
  --cel_stage2_enabled 1
  --cel_stage2_fresh_init 1
  --cel_stage2_candidate_id "$CANDIDATE_ID"
  --cel_env_mode "$DIRECTION"
  --cel_env_beta "$ENV_BETA"
  --cel_env_split_mode topk_abs
  --cel_env_topk_ratio "$ENV_TOPK_RATIO"
  --cel_env_hidden_dim "$ENV_HIDDEN_DIM"
  --cel_env_num_layers "$ENV_NUM_LAYERS"
  --cel_env_num_heads "$ENV_NUM_HEADS"
  --cel_env_ffn_dim "$ENV_FFN_DIM"
  --cel_env_drop 0.10
  --cel_env_output_postprocess "$ENV_OUTPUT_POSTPROCESS"
  --cel_env_output_ratio "$ENV_OUTPUT_RATIO"
  --cel_env_output_init_std "$ENV_OUTPUT_INIT_STD"
  --cel_env_shuffle_seed "$MODEL_SEED"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START debug $DIRECTION bootstrap"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dialogue_kt.main train \
  "${COMMON_ARGS[@]}" \
  --cel_stage2_phase bootstrap \
  --model_name "$BOOTSTRAP_MODEL" \
  --result_subdir "cel_stage2_environment/debug/$CANDIDATE_ID/stages/bootstrap" \
  --skip_test_after_train 1 \
  --epochs 1 \
  --lr 0.0002 \
  --wd 0.01 \
  2>&1 | tee "$STAGE_LOG_DIR/bootstrap.stdout.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START debug $DIRECTION calibrator warmup"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dialogue_kt.main train \
  "${COMMON_ARGS[@]}" \
  --cel_stage2_phase calibrator_warmup \
  --cel_stage2_parent_model_name "$BOOTSTRAP_MODEL" \
  --pt_model_name "$BOOTSTRAP_MODEL" \
  --cel_selector_init_model_name "$BOOTSTRAP_MODEL" \
  --cel_environment_init_model_name "$BOOTSTRAP_MODEL" \
  --cel_require_exact_selector_init 1 \
  --cel_require_exact_environment_init 1 \
  --model_name "$WARMUP_MODEL" \
  --result_subdir "cel_stage2_environment/debug/$CANDIDATE_ID/stages/calibrator_warmup" \
  --skip_test_after_train 1 \
  --epochs 1 \
  --lr 0.0002 \
  --wd 0.0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331 \
  --cel_train_calibrator_only 1 \
  2>&1 | tee "$STAGE_LOG_DIR/calibrator_warmup.stdout.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START debug $DIRECTION strict joint + final test"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dialogue_kt.main train \
  "${COMMON_ARGS[@]}" \
  --cel_stage2_phase joint \
  --cel_stage2_parent_model_name "$WARMUP_MODEL" \
  --pt_model_name "$WARMUP_MODEL" \
  --cel_selector_init_model_name "$WARMUP_MODEL" \
  --cel_calibrator_init_model_name "$WARMUP_MODEL" \
  --cel_environment_init_model_name "$WARMUP_MODEL" \
  --cel_require_exact_selector_init 1 \
  --cel_require_exact_environment_init 1 \
  --cel_require_complete_checkpoint 1 \
  --model_name "$FINAL_MODEL" \
  --result_subdir "cel_stage2_environment/debug/$CANDIDATE_ID/joint" \
  --epochs 1 \
  --lr 0.00001 \
  --wd 0.0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331 \
  2>&1 | tee "$RESULT_ROOT/joint.stdout.log"

python scripts/cel_stage2_environment/audit_stage2_candidate.py \
  "$CANDIDATE_ID" --debug-layout
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END debug $DIRECTION candidate"
