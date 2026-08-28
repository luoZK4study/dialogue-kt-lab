#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
ROUND_ID="${1:-}"
if [[ -z "$ROUND_ID" ]]; then
  echo "usage: $0 <round_id>" >&2
  exit 2
fi

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
BASE_MODEL="${STAGE2_BASE_MODEL:-/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B}"
MODEL_SEED="${STAGE2_MODEL_SEED:-1221}"
CANDIDATE_ID="${STAGE2_CANDIDATE_ID:-stage2_dual_${ROUND_ID}_contextual_transformer_seed${MODEL_SEED}}"
START_PHASE="${STAGE2_START_PHASE:-a_bootstrap}"

ENV_BETA="${STAGE2_ENV_BETA:-0.10}"
ENV_HIDDEN_DIM="${STAGE2_ENV_HIDDEN_DIM:-1024}"
ENV_NUM_LAYERS="${STAGE2_ENV_NUM_LAYERS:-4}"
ENV_NUM_HEADS="${STAGE2_ENV_NUM_HEADS:-8}"
ENV_FFN_DIM="${STAGE2_ENV_FFN_DIM:-4096}"
ENV_DROP="${STAGE2_ENV_DROP:-0.10}"
ENV_OUTPUT_RATIO="${STAGE2_ENV_OUTPUT_RATIO:-1.0}"
ENV_OUTPUT_INIT_STD="${STAGE2_ENV_OUTPUT_INIT_STD:-0.01}"
LAMBDA_R="${STAGE2_LAMBDA_R:-1.0}"
LAMBDA_M="${STAGE2_LAMBDA_M:-1.0}"
LAMBDA_CONS="${STAGE2_LAMBDA_CONS:-0.10}"
BETA_START_RATIO="${STAGE2_BETA_START_RATIO:-0.20}"
CONSISTENCY_RAMP="${STAGE2_CONSISTENCY_RAMP:-0.25}"
BOOTSTRAP_LR="${STAGE2_A_BOOTSTRAP_LR:-0.00020}"
SELECTOR_LR="${STAGE2_SELECTOR_LR:-0.00001}"
ENVIRONMENT_LR="${STAGE2_ENVIRONMENT_LR:-0.00010}"
CALIBRATOR_LR="${STAGE2_CALIBRATOR_LR:-0.00010}"

if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing formal dual-path launch outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESULT_ROOT="$ROOT_DIR/results/cel_stage2_environment/dual_path/$ROUND_ID"
STAGE_ROOT="$RESULT_ROOT/stages"
mkdir -p "$STAGE_ROOT"

A_BOOTSTRAP="cel_${CANDIDATE_ID}_a_bootstrap_qwen3_1.7b"
CAL_WARMUP="cel_${CANDIDATE_ID}_calibrator_warmup_qwen3_1.7b"
A_JOINT="cel_${CANDIDATE_ID}_a_joint_qwen3_1.7b"
B_WARMUP="cel_${CANDIDATE_ID}_b_warmup_qwen3_1.7b"
FINAL_MODEL="cel_${CANDIDATE_ID}_joint_qwen3_1.7b"

case "$START_PHASE" in
  a_bootstrap) START_PHASE_INDEX=0 ;;
  calibrator_warmup) START_PHASE_INDEX=1 ;;
  a_joint) START_PHASE_INDEX=2 ;;
  b_warmup) START_PHASE_INDEX=3 ;;
  joint) START_PHASE_INDEX=4 ;;
  *)
    echo "invalid STAGE2_START_PHASE: $START_PHASE" >&2
    exit 2
    ;;
esac

PHASE_NAMES=(a_bootstrap calibrator_warmup a_joint b_warmup joint)
MODEL_NAMES=("$A_BOOTSTRAP" "$CAL_WARMUP" "$A_JOINT" "$B_WARMUP" "$FINAL_MODEL")

require_completed_phase() {
  local phase="$1"
  local model_name="$2"
  local expected_parent="$3"
  local model_dir="$ROOT_DIR/saved_models/$model_name"
  local required_files=(adapter_model.safetensors cel_selector.pt cel_stage2_manifest.json)
  if [[ "$phase" != "a_bootstrap" ]]; then
    required_files+=(cel_calibrator.pt)
  fi
  if [[ "$phase" == "b_warmup" || "$phase" == "joint" ]]; then
    required_files+=(cel_environment.pt)
  fi
  for file_name in "${required_files[@]}"; do
    if [[ ! -f "$model_dir/$file_name" ]]; then
      echo "cannot resume: missing saved_models/$model_name/$file_name" >&2
      exit 1
    fi
  done
  python - "$model_dir/cel_stage2_manifest.json" "$phase" "$model_name" \
    "$CANDIDATE_ID" "$expected_parent" "$MODEL_SEED" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, phase, model_name, candidate_id, expected_parent, model_seed = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
expected = {
    "phase": phase,
    "model_name": model_name,
    "candidate_id": candidate_id,
    "parent_model_name": expected_parent or None,
    "model_init_seed": int(model_seed),
    "fresh_init_required": True,
    "objective": "dual_path_hr_hm_js",
}
mismatches = {
    key: {"expected": value, "actual": manifest.get(key)}
    for key, value in expected.items()
    if manifest.get(key) != value
}
if mismatches:
    raise SystemExit(f"cannot resume from incompatible manifest {manifest_path}: {mismatches}")
PY
}

for phase_index in "${!PHASE_NAMES[@]}"; do
  phase="${PHASE_NAMES[$phase_index]}"
  model_name="${MODEL_NAMES[$phase_index]}"
  if (( phase_index < START_PHASE_INDEX )); then
    case "$phase" in
      a_bootstrap) expected_parent="" ;;
      calibrator_warmup) expected_parent="$A_BOOTSTRAP" ;;
      a_joint) expected_parent="$CAL_WARMUP" ;;
      b_warmup) expected_parent="$A_JOINT" ;;
      joint) expected_parent="$B_WARMUP" ;;
    esac
    require_completed_phase "$phase" "$model_name" "$expected_parent"
  elif [[ -e "$ROOT_DIR/saved_models/$model_name" ]]; then
    echo "refusing to overwrite checkpoint: saved_models/$model_name" >&2
    exit 1
  fi
done

if (( START_PHASE_INDEX > 0 )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] RESUME $ROUND_ID from phase=$START_PHASE"
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
  --cel_env_mode contextual_transformer
  --cel_env_split_mode complementary
  --cel_env_beta "$ENV_BETA"
  --cel_env_hidden_dim "$ENV_HIDDEN_DIM"
  --cel_env_num_layers "$ENV_NUM_LAYERS"
  --cel_env_num_heads "$ENV_NUM_HEADS"
  --cel_env_ffn_dim "$ENV_FFN_DIM"
  --cel_env_drop "$ENV_DROP"
  --cel_env_output_postprocess centered_rms
  --cel_env_output_ratio "$ENV_OUTPUT_RATIO"
  --cel_env_output_init_std "$ENV_OUTPUT_INIT_STD"
  --cel_stage2_lambda_r "$LAMBDA_R"
  --cel_stage2_lambda_m "$LAMBDA_M"
  --cel_stage2_lambda_cons "$LAMBDA_CONS"
  --cel_stage2_beta_start_ratio "$BETA_START_RATIO"
  --cel_stage2_consistency_ramp_fraction "$CONSISTENCY_RAMP"
  --cel_calibrator_lr "$CALIBRATOR_LR"
)
if [[ "${STAGE2_DEBUG:-0}" == "1" ]]; then
  COMMON_ARGS+=(--debug)
fi

run_phase() {
  local phase="$1"
  local model_name="$2"
  local log_path="$3"
  shift 3
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $ROUND_ID phase=$phase model=$model_name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dialogue_kt.main train \
    "${COMMON_ARGS[@]}" \
    --cel_stage2_phase "$phase" \
    --model_name "$model_name" \
    --result_subdir "cel_stage2_environment/dual_path/$ROUND_ID/stages/$phase" \
    --skip_test_after_train 1 \
    "$@" 2>&1 | tee "$log_path"
}

if (( START_PHASE_INDEX <= 0 )); then
  run_phase a_bootstrap "$A_BOOTSTRAP" "$STAGE_ROOT/a_bootstrap.stdout.log" \
    --epochs 2 --lr "$BOOTSTRAP_LR" --wd 0.01
fi

if (( START_PHASE_INDEX <= 1 )); then
  run_phase calibrator_warmup "$CAL_WARMUP" "$STAGE_ROOT/calibrator_warmup.stdout.log" \
    --cel_stage2_parent_model_name "$A_BOOTSTRAP" \
    --pt_model_name "$A_BOOTSTRAP" \
    --cel_selector_init_model_name "$A_BOOTSTRAP" \
    --cel_require_exact_selector_init 1 \
    --cel_output_calibration bias \
    --cel_calibrator_init_bias 0.331 \
    --cel_train_calibrator_only 1 \
    --epochs 1 --lr 0.0002 --wd 0.0
fi

if (( START_PHASE_INDEX <= 2 )); then
  run_phase a_joint "$A_JOINT" "$STAGE_ROOT/a_joint.stdout.log" \
    --cel_stage2_parent_model_name "$CAL_WARMUP" \
    --pt_model_name "$CAL_WARMUP" \
    --cel_selector_init_model_name "$CAL_WARMUP" \
    --cel_calibrator_init_model_name "$CAL_WARMUP" \
    --cel_require_exact_selector_init 1 \
    --cel_require_complete_checkpoint 1 \
    --cel_output_calibration bias \
    --cel_calibrator_init_bias 0.331 \
    --cel_selector_lr "$SELECTOR_LR" \
    --epochs 1 --lr 0.00001 --wd 0.0
fi

if (( START_PHASE_INDEX <= 3 )); then
  run_phase b_warmup "$B_WARMUP" "$STAGE_ROOT/b_warmup.stdout.log" \
    --cel_stage2_parent_model_name "$A_JOINT" \
    --pt_model_name "$A_JOINT" \
    --cel_selector_init_model_name "$A_JOINT" \
    --cel_calibrator_init_model_name "$A_JOINT" \
    --cel_require_exact_selector_init 1 \
    --cel_require_complete_checkpoint 1 \
    --cel_output_calibration bias \
    --cel_calibrator_init_bias 0.331 \
    --cel_train_environment_only 1 \
    --cel_environment_lr "$ENVIRONMENT_LR" \
    --epochs 1 --lr "$ENVIRONMENT_LR" --wd 0.0
fi

if (( START_PHASE_INDEX <= 4 )); then
  run_phase joint "$FINAL_MODEL" "$RESULT_ROOT/joint.stdout.log" \
    --cel_stage2_parent_model_name "$B_WARMUP" \
    --pt_model_name "$B_WARMUP" \
    --cel_selector_init_model_name "$B_WARMUP" \
    --cel_calibrator_init_model_name "$B_WARMUP" \
    --cel_environment_init_model_name "$B_WARMUP" \
    --cel_require_exact_selector_init 1 \
    --cel_require_exact_environment_init 1 \
    --cel_require_complete_checkpoint 1 \
    --cel_output_calibration bias \
    --cel_calibrator_init_bias 0.331 \
    --cel_selector_lr "$SELECTOR_LR" \
    --cel_environment_lr "$ENVIRONMENT_LR" \
    --epochs 1 --lr 0.00001 --wd 0.0
fi

cat > "$RESULT_ROOT/round_config.env" <<EOF
ROUND_ID=$ROUND_ID
CANDIDATE_ID=$CANDIDATE_ID
MODEL_SEED=$MODEL_SEED
ENV_BETA=$ENV_BETA
ENV_HIDDEN_DIM=$ENV_HIDDEN_DIM
ENV_NUM_LAYERS=$ENV_NUM_LAYERS
ENV_NUM_HEADS=$ENV_NUM_HEADS
ENV_FFN_DIM=$ENV_FFN_DIM
ENV_DROP=$ENV_DROP
ENV_OUTPUT_RATIO=$ENV_OUTPUT_RATIO
ENV_OUTPUT_INIT_STD=$ENV_OUTPUT_INIT_STD
LAMBDA_R=$LAMBDA_R
LAMBDA_M=$LAMBDA_M
LAMBDA_CONS=$LAMBDA_CONS
BETA_START_RATIO=$BETA_START_RATIO
CONSISTENCY_RAMP=$CONSISTENCY_RAMP
BOOTSTRAP_LR=$BOOTSTRAP_LR
SELECTOR_LR=$SELECTOR_LR
ENVIRONMENT_LR=$ENVIRONMENT_LR
CALIBRATOR_LR=$CALIBRATOR_LR
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $ROUND_ID training chain"
echo "candidate_id=$CANDIDATE_ID"
echo "final_model=$FINAL_MODEL"
