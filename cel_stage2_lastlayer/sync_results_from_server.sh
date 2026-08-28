#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lzk/code/dialogue-kt"
SSH_TARGET="user4@119.29.183.125"
SSH_OPTS="-i ~/.ssh/id_rsa_u4 -p 22049"
REMOTE_ROOT="/home/user4/dialogue-kt"

mkdir -p \
  "$ROOT_DIR/results/baseline_recert/metrics" \
  "$ROOT_DIR/results/baseline_recert/qual" \
  "$ROOT_DIR/results/baseline_recert/kcs" \
  "$ROOT_DIR/results/cel_stage2_lastlayer/metrics" \
  "$ROOT_DIR/results/cel_stage2_lastlayer/qual" \
  "$ROOT_DIR/results/cel_stage2_lastlayer/kcs"

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/baseline_recert/" \
  "$ROOT_DIR/results/baseline_recert/" || true

rsync -avz -e "ssh $SSH_OPTS" \
  "$SSH_TARGET:$REMOTE_ROOT/results/cel_stage2_lastlayer/" \
  "$ROOT_DIR/results/cel_stage2_lastlayer/" || true
