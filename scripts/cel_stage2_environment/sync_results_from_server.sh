#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${1:-3090}"
REMOTE_ROOT="${STAGE2_REMOTE_ROOT:-/home/user4/dialogue-kt}"

mkdir -p "$ROOT_DIR/results/a" "$ROOT_DIR/results/a_b"
for result_dir in a a_b; do
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$SSH_ALIAS:$REMOTE_ROOT/results/$result_dir/" \
    "$ROOT_DIR/results/$result_dir/"
done
echo "Synced A/A+B results from $SSH_ALIAS:$REMOTE_ROOT"
