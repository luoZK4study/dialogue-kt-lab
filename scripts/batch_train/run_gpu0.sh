#!/bin/bash
# GPU 0: High priority methods
set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
LOG_DIR=results/training_logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/batch_gpu0_$(date +%Y%m%d_%H%M).log

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

log "======== GPU 0: High Priority Methods ========"
train_one "rank_auc" "lmkt_rank_auc_qwen3_1.7b" "--rank_loss_weight 0.1 --rank_margin 0.05"
train_one "state_table" "lmkt_state_table_qwen3_1.7b" "--state_top_k 8"
train_one "solution_contrast" "lmkt_solution_contrast_qwen3_1.7b" ""
train_one "quito_mark" "lmkt_quito_mark_qwen3_1.7b" ""
train_one "dac_mark" "lmkt_dac_mark_qwen3_1.7b" ""
log "======== GPU 0 DONE ========"
