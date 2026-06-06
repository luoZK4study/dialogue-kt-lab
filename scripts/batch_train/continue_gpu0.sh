#!/bin/bash
# GPU 0: Continue with loss methods and combinations
set -e
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
LOG_DIR=results/training_logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/continue_gpu0_$(date +%Y%m%d_%H%M).log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

train_one() {
    local desc="$1"; local model_name="$2"; shift 2
    log "=== $desc -> $model_name ==="
    python -m dialogue_kt.main train --dataset mathdial --model_type lmkt --base_model $BASE_MODEL --model_name $model_name --epochs 2 --quantize 0 "$@" 2>&1 | tee -a $LOG
    local auc=$(python -c "import re; f=open('results/metrics_${model_name}.txt'); m=re.search(r'AUC: ([0-9.]+)', f.read()); print(m.group(1) if m else 'FAIL')" 2>/dev/null || echo "FAIL")
    log ">>> $desc AUC = $auc"
}

log "======== GPU 0: LOSS + COMBOS ========"
train_one "Margin Loss" "lmkt_margin_qwen3_1.7b" --kt_method margin_loss --rank_margin 0.1 --aux_loss_weight 0.1
train_one "StateTable+RankAUC" "lmkt_state_rank_qwen3_1.7b" --kt_prompt_method state_table --kt_loss_method rank_auc --kt_method base
train_one "Rank AUC strong" "lmkt_rank_auc_strong_qwen3_1.7b" --kt_method rank_auc --rank_loss_weight 0.3 --rank_margin 0.05
log "======== GPU 0 DONE ========"
