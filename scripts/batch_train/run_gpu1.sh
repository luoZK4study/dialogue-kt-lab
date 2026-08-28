#!/bin/bash
# GPU 1: Medium + Experimental methods
set -e
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
LOG_DIR=results/training_logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/batch_gpu1_$(date +%Y%m%d_%H%M).log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

train_one() {
    local method=$1; local model_name=$2; local extra_args=$3
    log "=== Training $method -> $model_name ==="
    python -m dialogue_kt.main train \
      --dataset mathdial --model_type lmkt \
      --base_model $BASE_MODEL --model_name $model_name \
      --epochs 2 --quantize 0 --kt_method $method $extra_args \
      2>&1 | tee -a $LOG
    local auc=$(python -c "
import re
with open('results/metrics_${model_name}.txt') as f:
    m = re.search(r'AUC: ([0-9.]+)', f.read())
    print(m.group(1) if m else 'FAIL')
" 2>/dev/null || echo "FAIL")
    log ">>> $method AUC = $auc"
}

log "======== GPU 1: Medium + Experimental Methods ========"
train_one "support_token" "lmkt_support_token_qwen3_1.7b" ""
train_one "hyper_chain" "lmkt_hyper_chain_qwen3_1.7b" ""
train_one "mil_noisy_and" "lmkt_mil_noisy_and_qwen3_1.7b" "--aux_loss_weight 0.2"
train_one "dual_view_consistency" "lmkt_dual_view_qwen3_1.7b" "--consistency_weight 0.15 --dual_view_type state_table"
log "======== GPU 1 DONE ========"
