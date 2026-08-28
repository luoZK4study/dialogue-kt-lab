#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
SSH_TARGET="user4@119.29.183.125"
SSH_OPTS="-i ~/.ssh/id_rsa_u4 -p 22049"
REMOTE_ROOT="/home/user4/dialogue-kt"

mkdir -p \
  "$ROOT_DIR/results/cel_stage1/metrics" \
  "$ROOT_DIR/results/cel_stage1/qual" \
  "$ROOT_DIR/results/cel_stage1/kcs"

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1/metrics/metrics_cel_*.txt" \
  "$ROOT_DIR/results/cel_stage1/metrics/" || true

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1/qual/qual_cel_*.csv" \
  "$ROOT_DIR/results/cel_stage1/qual/" || true

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1/kcs/kcs_cel_*.json" \
  "$ROOT_DIR/results/cel_stage1/kcs/" || true

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1/CEL_Stage1_DialogueKT_实验记录.md" \
  "$ROOT_DIR/results/cel_stage1/" || true

if ssh $SSH_OPTS "$SSH_TARGET" "test -f $REMOTE_ROOT/results/cel_stage1/final_metrics_table.md"; then
  rsync -avz -e "ssh $SSH_OPTS" \
    "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage1/final_metrics_table.md" \
    "$ROOT_DIR/results/cel_stage1/" || true
fi
