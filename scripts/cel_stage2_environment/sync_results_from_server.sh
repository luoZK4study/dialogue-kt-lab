#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${1:-3090}"
REMOTE_ROOT="${STAGE2_REMOTE_ROOT:-/home/user4/dialogue-kt}"

mkdir -p "$ROOT_DIR/results/cel_stage2_environment"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
  "$SSH_ALIAS:$REMOTE_ROOT/results/cel_stage2_environment/" \
  "$ROOT_DIR/results/cel_stage2_environment/"
echo "Synced Stage 2 results from $SSH_ALIAS:$REMOTE_ROOT"
