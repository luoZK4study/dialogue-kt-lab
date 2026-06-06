#!/bin/bash
# Wave 2: New methods + best combinations (run after wave 1)
# Usage: bash scripts/batch_train/run_wave2.sh [gpu_id]
set -e
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
LOG_DIR=results/training_logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/wave2_gpu${GPU_ID}_$(date +%Y%m%d_%H%M).log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

train_one() {
    local desc="$1"; local model_name="$2"; shift 2
    log "=== $desc -> $model_name ==="
    python -m dialogue_kt.main train \
      --dataset mathdial --model_type lmkt \
      --base_model $BASE_MODEL --model_name $model_name \
      --epochs 2 --quantize 0 "$@" 2>&1 | tee -a $LOG
    local auc=$(python -c "
import re
with open('results/metrics_${model_name}.txt') as f:
    m = re.search(r'AUC: ([0-9.]+)', f.read())
    print(m.group(1) if m else 'FAIL')
" 2>/dev/null || echo "FAIL")
    log ">>> $desc AUC = $auc"
}

log "======== WAVE 2: New Standalone Methods ========"

# New paper-inspired methods
train_one "Evidence-R1 reasoning" "lmkt_evidence_r1_qwen3_1.7b" \
  --kt_method evidence_r1

train_one "HypER chain validation" "lmkt_hyper_validate_qwen3_1.7b" \
  --kt_method hyper_validate

train_one "IBCircuit turn selection" "lmkt_ib_turns_qwen3_1.7b" \
  --kt_method ib_turns

# Simple but untested baseline variant
train_one "Prompt with labels" "lmkt_prompt_labels_qwen3_1.7b" \
  --prompt_inc_labels 1

log "======== WAVE 2 DONE ========"
