#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
SCRIPT_DIR="$ROOT_DIR/scripts/cel_stage1_last_layer"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

BOOTSTRAP_MODEL="cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b"
WARMUP_MODEL="cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b"
FINAL_MODEL="cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b"

cd "$ROOT_DIR"
mkdir -p "$RESULT_DIR/v26_stages"

for model_name in "$BOOTSTRAP_MODEL" "$WARMUP_MODEL" "$FINAL_MODEL"; do
  if [[ -e "$ROOT_DIR/saved_models/$model_name" ]]; then
    echo "refusing to overwrite existing v26 stage checkpoint: saved_models/$model_name" >&2
    exit 1
  fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START v26 bootstrap from Qwen3 base"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$SCRIPT_DIR/run_task_conditioned_v26_bootstrap.sh" \
  2>&1 | tee "$RESULT_DIR/task_conditioned_v26_bootstrap.stdout.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START v26 calibrator warmup from v26 bootstrap"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$SCRIPT_DIR/run_task_conditioned_v26_calibrator_warmup.sh" \
  2>&1 | tee "$RESULT_DIR/task_conditioned_v26_calibrator_warmup.stdout.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START v26 strict joint from v26 calibrator warmup"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$SCRIPT_DIR/run_task_conditioned_v26_joint.sh" \
  2>&1 | tee "$RESULT_DIR/task_conditioned_v26_selftrained_joint.stdout.log"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
python "$SCRIPT_DIR/audit_task_conditioned_v26_selftrained.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END v26 self-trained chain"
