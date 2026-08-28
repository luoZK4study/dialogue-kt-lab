#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="${1:-3090}"
REMOTE_ROOT="${STAGE2_REMOTE_ROOT:-/home/user4/dialogue-kt}"

FILES=(
  dialogue_kt/main.py
  dialogue_kt/training.py
  dialogue_kt/cel_methods.py
  scripts/cel_stage2_environment/check_stage2_tensor_contracts.py
  scripts/cel_stage2_environment/check_stage2_real_qwen_preflight.py
  scripts/cel_stage2_environment/run_stage2_debug_candidate.sh
  scripts/cel_stage2_environment/run_stage2_candidate.sh
  scripts/cel_stage2_environment/run_dual_path_candidate.sh
  scripts/cel_stage2_environment/derive_dual_path_round_decision.py
  scripts/cel_stage2_environment/run_dual_path_three_round_chain.sh
  scripts/cel_stage2_environment/run_validation_after_dual_path_chain.sh
  scripts/cel_stage2_environment/evaluate_dual_path_candidate.py
  scripts/cel_stage2_environment/audit_dual_path_candidate.py
  scripts/cel_stage2_environment/generate_dual_path_stage2_report.py
  scripts/cel_stage2_environment/launch_stage2_candidate_remote.sh
  scripts/cel_stage2_environment/audit_stage2_candidate.py
  scripts/cel_stage2_environment/generate_stage2_summary.py
)

for rel_path in "${FILES[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_ALIAS" "mkdir -p '$REMOTE_ROOT/$(dirname "$rel_path")'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$ROOT_DIR/$rel_path" "$SSH_ALIAS:$REMOTE_ROOT/$rel_path"
done
echo "Synced ${#FILES[@]} Stage 2 files to $SSH_ALIAS:$REMOTE_ROOT"
