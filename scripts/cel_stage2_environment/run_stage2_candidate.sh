#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
DIRECTION="${1:-}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
BASE_MODEL="${STAGE2_BASE_MODEL:-/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B}"

case "$DIRECTION" in
  shuffle)
    DEFAULT_CANDIDATE_ID="stage2_shuffle_topk20_beta010_seed1221"
    ENV_MODE="shuffle"
    DEFAULT_ENV_BETA="0.10"
    DEFAULT_OUTPUT_POSTPROCESS="none"
    DEFAULT_OUTPUT_RATIO="0.10"
    DEFAULT_OUTPUT_INIT_STD="0.001"
    ;;
  mlp)
    DEFAULT_CANDIDATE_ID="stage2_mlp_topk20_beta005_seed1221"
    ENV_MODE="mlp"
    DEFAULT_ENV_BETA="0.05"
    DEFAULT_OUTPUT_POSTPROCESS="none"
    DEFAULT_OUTPUT_RATIO="0.10"
    DEFAULT_OUTPUT_INIT_STD="0.001"
    ;;
  transformer)
    DEFAULT_CANDIDATE_ID="stage2_transformer_topk20_beta010_centeredrms010_seed1221"
    ENV_MODE="transformer"
    DEFAULT_ENV_BETA="0.10"
    DEFAULT_OUTPUT_POSTPROCESS="centered_rms"
    DEFAULT_OUTPUT_RATIO="0.10"
    DEFAULT_OUTPUT_INIT_STD="0.01"
    ;;
  *)
    echo "usage: $0 {shuffle|mlp|transformer}" >&2
    exit 2
    ;;
esac

CANDIDATE_ID="${STAGE2_CANDIDATE_ID:-$DEFAULT_CANDIDATE_ID}"
MODEL_SEED="${STAGE2_MODEL_SEED:-1221}"
ENV_BETA="${STAGE2_ENV_BETA:-$DEFAULT_ENV_BETA}"
ENV_TOPK_RATIO="${STAGE2_ENV_TOPK_RATIO:-0.20}"
ENV_HIDDEN_DIM="${STAGE2_ENV_HIDDEN_DIM:-512}"
ENV_NUM_LAYERS="${STAGE2_ENV_NUM_LAYERS:-1}"
ENV_NUM_HEADS="${STAGE2_ENV_NUM_HEADS:-4}"
ENV_FFN_DIM="${STAGE2_ENV_FFN_DIM:-1024}"
ENV_OUTPUT_POSTPROCESS="${STAGE2_ENV_OUTPUT_POSTPROCESS:-$DEFAULT_OUTPUT_POSTPROCESS}"
ENV_OUTPUT_RATIO="${STAGE2_ENV_OUTPUT_RATIO:-$DEFAULT_OUTPUT_RATIO}"
ENV_OUTPUT_INIT_STD="${STAGE2_ENV_OUTPUT_INIT_STD:-$DEFAULT_OUTPUT_INIT_STD}"

RESULT_DIR="$ROOT_DIR/results/cel_stage2_environment"
STAGE_RESULT_DIR="$RESULT_DIR/$CANDIDATE_ID/stages"
BOOTSTRAP_MODEL="cel_${CANDIDATE_ID}_bootstrap_qwen3_1.7b"
WARMUP_MODEL="cel_${CANDIDATE_ID}_calibrator_warmup_qwen3_1.7b"
FINAL_MODEL="cel_${CANDIDATE_ID}_joint_qwen3_1.7b"
BOOTSTRAP_LOG="$STAGE_RESULT_DIR/bootstrap.stdout.log"
WARMUP_LOG="$STAGE_RESULT_DIR/calibrator_warmup.stdout.log"
JOINT_LOG="$RESULT_DIR/${CANDIDATE_ID}.stdout.log"
METRICS_PATH="$RESULT_DIR/metrics/metrics_${FINAL_MODEL}.txt"

if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing formal Stage 2 launch outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$STAGE_RESULT_DIR" "$RESULT_DIR/metrics" "$RESULT_DIR/kcs" "$RESULT_DIR/qual"
for model_name in "$BOOTSTRAP_MODEL" "$WARMUP_MODEL" "$FINAL_MODEL"; do
  if [[ -e "$ROOT_DIR/saved_models/$model_name" ]]; then
    echo "refusing to overwrite Stage 2 checkpoint: saved_models/$model_name" >&2
    exit 1
  fi
done
if [[ -e "$METRICS_PATH" ]]; then
  echo "refusing to overwrite Stage 2 metrics: $METRICS_PATH" >&2
  exit 1
fi

COMMON_ARGS=(
  --dataset mathdial
  --model_type lmkt
  --base_model "$BASE_MODEL"
  --quantize 0
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
  --cel_env_mode "$ENV_MODE"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START Stage 2 $DIRECTION bootstrap from raw Qwen3 base"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dialogue_kt.main train \
  "${COMMON_ARGS[@]}" \
  --cel_stage2_phase bootstrap \
  --model_name "$BOOTSTRAP_MODEL" \
  --result_subdir "cel_stage2_environment/$CANDIDATE_ID/stages/bootstrap" \
  --skip_test_after_train 1 \
  --epochs 2 \
  --lr 0.0002 \
  --wd 0.01 \
  2>&1 | tee "$BOOTSTRAP_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START Stage 2 $DIRECTION calibrator warmup"
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
  --result_subdir "cel_stage2_environment/$CANDIDATE_ID/stages/calibrator_warmup" \
  --skip_test_after_train 1 \
  --epochs 1 \
  --lr 0.0002 \
  --wd 0.0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331 \
  --cel_train_calibrator_only 1 \
  2>&1 | tee "$WARMUP_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START Stage 2 $DIRECTION strict joint + final test"
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
  --result_subdir cel_stage2_environment \
  --epochs 1 \
  --lr 0.00001 \
  --wd 0.0 \
  --cel_output_calibration bias \
  --cel_calibrator_init_bias 0.331 \
  2>&1 | tee "$JOINT_LOG"

python scripts/cel_stage2_environment/audit_stage2_candidate.py "$CANDIDATE_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END Stage 2 $DIRECTION candidate"
