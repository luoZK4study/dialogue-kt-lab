#!/bin/bash
# Focused training: loss methods + combinations only
# Usage: bash scripts/batch_train/run_loss_methods.sh [gpu_id]
set -e
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
LOG_DIR=results/training_logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/loss_methods_gpu${GPU_ID}_$(date +%Y%m%d_%H%M).log

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

log "======== LOSS METHODS + COMBINATIONS ========"

# New loss functions
train_one "Focal Loss (gamma=2.0)" "lmkt_focal_qwen3_1.7b" \
  --kt_method focal_loss --focal_gamma 2.0

train_one "Margin Loss (margin=0.1)" "lmkt_margin_qwen3_1.7b" \
  --kt_method margin_loss --rank_margin 0.1 --aux_loss_weight 0.1

# Rank_auc with stronger ranking weight
train_one "Rank AUC (weight=0.3)" "lmkt_rank_auc_strong_qwen3_1.7b" \
  --kt_method rank_auc --rank_loss_weight 0.3 --rank_margin 0.05

# Best combinations (prompt + rank_auc)
train_one "StateTable+RankAUC" "lmkt_state_rank_qwen3_1.7b" \
  --kt_prompt_method state_table --kt_loss_method rank_auc --kt_method base

train_one "SolutionContrast+RankAUC" "lmkt_solcon_rank_qwen3_1.7b" \
  --kt_prompt_method solution_contrast --kt_loss_method rank_auc --kt_method base

log "======== DONE ========"
