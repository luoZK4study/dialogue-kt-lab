#!/usr/bin/env bash
set -euo pipefail

# Sequential formal supervisor.  It never starts a round until the previous
# round has a passing validation audit, and it delays all final tests until
# Round 3 has produced the validation-driven freeze decision.

ROOT_DIR="${DIALOGUE_KT_ROOT:-/home/user4/dialogue-kt}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_SEED="${STAGE2_MODEL_SEED:-1221}"

if [[ "$ROOT_DIR" != "/home/user4/dialogue-kt" ]]; then
  echo "refusing formal Stage 2 chain outside /home/user4/dialogue-kt: $ROOT_DIR" >&2
  exit 1
fi
if [[ "$MODEL_SEED" != "1221" ]]; then
  echo "formal Stage 2 chain is fixed to model/data seed 1221, got: $MODEL_SEED" >&2
  exit 1
fi

cd "$ROOT_DIR"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RESULTS_ROOT="$ROOT_DIR/results/cel_stage2_environment/dual_path"
DERIVER="scripts/cel_stage2_environment/derive_dual_path_round_decision.py"
RUNNER="scripts/cel_stage2_environment/run_dual_path_candidate.sh"
EVALUATOR="scripts/cel_stage2_environment/evaluate_dual_path_candidate.py"
AUDITOR="scripts/cel_stage2_environment/audit_dual_path_candidate.py"
REPORTER="scripts/cel_stage2_environment/generate_dual_path_stage2_report.py"
CONTRACT_CHECKER="scripts/cel_stage2_environment/check_stage2_tensor_contracts.py"

wait_for_round1_validation() {
  local audit_path="$RESULTS_ROOT/round1/audit/AUDIT.json"
  local watcher_pid_path="$RESULTS_ROOT/round1/post_training_validation.pid"
  while [[ ! -f "$audit_path" ]]; do
    if [[ -f "$watcher_pid_path" ]]; then
      local watcher_pid
      watcher_pid="$(<"$watcher_pid_path")"
      if ! kill -0 "$watcher_pid" 2>/dev/null; then
        echo "Round 1 validation watcher exited before writing $audit_path" >&2
        exit 1
      fi
    fi
    sleep 60
  done
}

run_new_round() {
  local round_id="$1"
  local env_path="$2"
  local candidate_id="stage2_dual_${round_id}_contextual_transformer_seed1221"
  local round_root="$RESULTS_ROOT/$round_id"
  mkdir -p "$round_root"
  if [[ ! -f "$env_path" ]]; then
    echo "missing generated configuration: $env_path" >&2
    exit 1
  fi
  set -a
  source "$env_path"
  set +a
  # Every derived round is a fresh raw-base candidate.  Do not inherit a
  # recovery-only phase override from the process that launched Round 1.
  export STAGE2_BASE_MODEL=/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B
  export STAGE2_MODEL_SEED=1221
  export STAGE2_ENV_HIDDEN_DIM=1024
  export STAGE2_ENV_NUM_LAYERS=4
  export STAGE2_ENV_NUM_HEADS=8
  export STAGE2_ENV_FFN_DIM=4096
  export STAGE2_ENV_DROP=0.10
  export STAGE2_ENV_OUTPUT_INIT_STD=0.01
  export STAGE2_LAMBDA_R=1.0
  export STAGE2_LAMBDA_M=1.0
  export STAGE2_START_PHASE=a_bootstrap
  if [[ "${STAGE2_CANDIDATE_ID:-}" != "$candidate_id" ]]; then
    echo "candidate mismatch in $env_path" >&2
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $round_id formal training chain"
  CUDA_VISIBLE_DEVICES="$GPU_ID" bash "$RUNNER" "$round_id" 2>&1 | tee "$round_root/launcher.stdout.log"
  if ! grep -Fq "END $round_id training chain" "$round_root/launcher.stdout.log"; then
    echo "$round_id launcher ended without a success marker" >&2
    exit 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $round_id validation"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVALUATOR" "$round_id" "$candidate_id" validation --cuda "$GPU_ID"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $round_id validation audit"
  python "$AUDITOR" "$round_id" "$candidate_id" --require-validation
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $round_id validation and audit"
}

run_final_test() {
  local round_id="$1"
  local candidate_id="stage2_dual_${round_id}_contextual_transformer_seed1221"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $round_id final test"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$EVALUATOR" "$round_id" "$candidate_id" test --cuda "$GPU_ID"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $round_id final audit"
  python "$AUDITOR" "$round_id" "$candidate_id" --require-validation --require-test
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $round_id final test and audit"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for existing Round 1 validation and audit"
wait_for_round1_validation
python "$DERIVER" round1 --target-round round2
run_new_round round2 "$RESULTS_ROOT/round2/next_round.env"
python "$DERIVER" round2 --target-round round3
run_new_round round3 "$RESULTS_ROOT/round3/next_round.env"
python "$DERIVER" round3 --finalize

for round_id in round1 round2 round3; do
  run_final_test "$round_id"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START final tensor-contract check"
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$CONTRACT_CHECKER" --cuda \
  | tee "$RESULTS_ROOT/tensor_contracts.final.stdout.log"
python "$REPORTER" --require-validation --require-test --require-contracts
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END formal three-round Stage 2 chain"
