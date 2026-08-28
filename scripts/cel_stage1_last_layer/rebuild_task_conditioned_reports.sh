#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
LOCK_FILE="$RESULT_DIR/.task_conditioned_reports.lock"
LOCK_TIMEOUT_SECS="${TASK_CONDITIONED_REPORT_LOCK_TIMEOUT_SECS:-600}"
PYTHON_BIN="${DIALOGUE_KT_PYTHON:-python3}"

mkdir -p "$RESULT_DIR"
exec 9>"$LOCK_FILE"
flock -w "$LOCK_TIMEOUT_SECS" 9

"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_status_summary.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_comparison_report.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_detailed_analysis.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/update_baseline_record.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_strict_full_train_report.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_briefs.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_candidate_config_diffs.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_experiment_audit.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_method_protocol.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_formal_experiment_summary.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_status_summary.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/cel_stage1_last_layer/generate_strict_full_train_report.py"
