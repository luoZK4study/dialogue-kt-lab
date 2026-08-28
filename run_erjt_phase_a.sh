#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
cd /home/user4/dialogue-kt
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LOGDIR="results/erjt"
mkdir -p $LOGDIR/metrics $LOGDIR/kcs $LOGDIR/qual

COMMON="--dataset mathdial --model_type lmkt --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B --epochs 2 --quantize 0 --batch_size 1 --grad_accum_steps 32"

echo "===== Wave 1: A1 (Extractor) on GPU0 + A2 (Faithfulness) on GPU1 ====="
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train $COMMON --model_name erjt_a1 --erjt_mode extractor --erjt_alpha_p 0.2 2>&1 | tee $LOGDIR/train_erjt_a1.log &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train $COMMON --model_name erjt_a2 --erjt_mode faithfulness --erjt_alpha_f 0.1 --erjt_k_ratio 0.3 2>&1 | tee $LOGDIR/train_erjt_a2.log &
PID2=$!
echo "A1 PID: $PID1, A2 PID: $PID2"
wait $PID1 $PID2
echo "Wave 1 complete. A1 exit: $? (PID1), A2 exit: $? (PID2)"

echo "===== Wave 2: A3 (Disentangle) on GPU0 + A4 Round0 (Bootstrap) on GPU1 ====="
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train $COMMON --model_name erjt_a3 --erjt_mode disentangle --erjt_beta 0.05 --erjt_gamma 0.05 2>&1 | tee $LOGDIR/train_erjt_a3.log &
PID1=$!
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train $COMMON --model_name erjt_a4_r0 --epochs 1 --erjt_mode extractor --erjt_alpha_p 0.2 --erjt_bootstrap_round 0 2>&1 | tee $LOGDIR/train_erjt_a4_r0.log &
PID2=$!
echo "A3 PID: $PID1, A4-R0 PID: $PID2"
wait $PID1 $PID2
echo "Wave 2 complete."

echo "===== All Phase A training complete! ====="
echo "Logs in $LOGDIR/"
ls -la $LOGDIR/train_erjt_*.log
