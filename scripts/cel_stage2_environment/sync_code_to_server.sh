#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${1:-3090}"
REMOTE_ROOT="${STAGE2_REMOTE_ROOT:-/home/user4/dialogue-kt}"

FILES=(
  dialogue_kt/main.py
  dialogue_kt/training.py
  dialogue_kt/cel_methods.py
  dialogue_kt/data_loading.py
  scripts/cel/run_a_unified.sh
  scripts/cel/run_a_b_unified.sh
  scripts/cel_stage2_environment/check_stage2_tensor_contracts.py
  scripts/cel_stage2_environment/check_stage2_real_qwen_preflight.py
)

for rel_path in "${FILES[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_ALIAS" "mkdir -p '$REMOTE_ROOT/$(dirname "$rel_path")'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$ROOT_DIR/$rel_path" "$SSH_ALIAS:$REMOTE_ROOT/$rel_path"
done
echo "Synced ${#FILES[@]} CEL files to $SSH_ALIAS:$REMOTE_ROOT"
