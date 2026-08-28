#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

CANDIDATE_KEY="${1:-}"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_REVIEW_SYNC_TIMEOUT_SECS:-240}"
SKIP_REFRESH="${TASK_CONDITIONED_REVIEW_SKIP_REFRESH:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/review_formal_candidate.sh <candidate_key>

Purpose:
  Run the strict closeout path first, then print a compact review for one
  formal candidate from the authoritative markdown/json surfaces.

Environment:
  TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1      Skip SSH refresh and only read the current local snapshot
  TASK_CONDITIONED_REVIEW_SYNC_TIMEOUT_SECS   Sync timeout for the closeout refresh step (default: 240)

Examples:
  bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v21
  bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v22
  TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v21
EOF
}

if [[ -z "$CANDIDATE_KEY" ]]; then
  usage >&2
  exit 2
fi

formal_candidate_require_key "$CANDIDATE_KEY" || {
  usage >&2
  exit 2
}

MODEL_NAME="${FORMAL_MODEL_NAME[$CANDIDATE_KEY]}"
STRICT_REPORT="$RESULT_DIR/STRICT_FULL_TRAIN_REPORT.md"
AUDIT_JSON="$RESULT_DIR/formal_experiment_audit.json"
STATUS_MD="$RESULT_DIR/STATUS.md"

TASK_CONDITIONED_FINALIZE_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" \
TASK_CONDITIONED_FINALIZE_SKIP_REFRESH="$SKIP_REFRESH" \
TASK_CONDITIONED_FINALIZE_QUIET=1 \
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/finalize_formal_candidate.sh" "$CANDIDATE_KEY"

python3 - <<'PY' "$ROOT_DIR" "$CANDIDATE_KEY" "$MODEL_NAME" "$STRICT_REPORT" "$AUDIT_JSON" "$STATUS_MD"
import json
import re
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
sys.path.append(str(root_dir / "scripts" / "cel_stage1_last_layer"))
from formal_candidate_registry import FORMAL_CANDIDATES  # type: ignore

candidate_key = sys.argv[2]
model_name = sys.argv[3]
strict_report = Path(sys.argv[4]).read_text(encoding="utf-8", errors="ignore")
audit = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
status_md = Path(sys.argv[6]).read_text(encoding="utf-8", errors="ignore")

stage_match = re.search(r"- Stage: `([^`]+)`", status_md)
stage = stage_match.group(1) if stage_match else "unknown"
refresh_match = re.search(r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`", status_md)
refresh_event = refresh_match.group(1) if refresh_match else "none"

row = None
for entry in audit.get("rows") or []:
    if entry.get("model") == model_name:
        row = entry
        break

if row is None:
    raise SystemExit(f"candidate {model_name} not found in formal_experiment_audit.json")

print("== Strict Formal Review ==")
print(f"candidate_key={candidate_key}")
print(f"model_name={model_name}")
print(f"stage={stage}")
print(f"latest_refresh={refresh_event}")
print(f"audit_state={row.get('audit')}")
print(f"status={row.get('status')}")
print(f"artifacts_ok={row.get('artifacts_ok')}")
print(f"metrics_complete={row.get('metrics_complete')}")
print(f"log_ok={row.get('log_ok')}")
print(f"coverage_ok={row.get('coverage_ok')}")
print(f"beats_baseline={row.get('beats_baseline')}")

metrics = row.get("metrics") or {}
if metrics:
    print(f"overall_acc={metrics.get('overall_acc')}")
    print(f"overall_auc={metrics.get('overall_auc')}")
    print(f"final_acc={metrics.get('final_acc')}")
    print(f"final_auc={metrics.get('final_auc')}")
analysis = row.get("analysis") or {}
if analysis:
    if analysis.get("pred_true") is not None:
        print(f"pred_true={round(float(analysis.get('pred_true')), 4)}")
    if analysis.get("best_acc") is not None:
        print(f"best_acc={round(float(analysis.get('best_acc')), 4)}")
    if analysis.get("best_threshold") is not None:
        print(f"best_threshold={round(float(analysis.get('best_threshold')), 6)}")
    if analysis.get("delta_acc_vs_baseline") is not None:
        print(f"delta_acc_vs_baseline={round(float(analysis.get('delta_acc_vs_baseline')), 4)}")
    if analysis.get("delta_auc_vs_baseline") is not None:
        print(f"delta_auc_vs_baseline={round(float(analysis.get('delta_auc_vs_baseline')), 4)}")
    if analysis.get("delta_final_acc_vs_baseline") is not None:
        print(f"delta_final_acc_vs_baseline={round(float(analysis.get('delta_final_acc_vs_baseline')), 4)}")
    if analysis.get("delta_final_auc_vs_baseline") is not None:
        print(f"delta_final_auc_vs_baseline={round(float(analysis.get('delta_final_auc_vs_baseline')), 4)}")
    if analysis.get("delta_acc_vs_anchor") is not None:
        print(f"delta_acc_vs_anchor={round(float(analysis.get('delta_acc_vs_anchor')), 4)}")
    if analysis.get("delta_auc_vs_anchor") is not None:
        print(f"delta_auc_vs_anchor={round(float(analysis.get('delta_auc_vs_anchor')), 4)}")
print(f"audit_note={row.get('note')}")

audit_state = str(row.get("audit"))
beats_baseline = str(row.get("beats_baseline"))
formal_decision = "analysis_pending"
next_formal_action = "inspect_authoritative_surfaces"
method_change_allowed = "false"

if audit_state == "needs_rerun":
    formal_decision = "rerun_required"
    next_formal_action = f"rerun_same_scope:{candidate_key}"
elif audit_state == "recorded" and beats_baseline == "yes":
    formal_decision = "winner_ready_if_audit_clean"
    next_formal_action = f"freeze_winner:{candidate_key}"
    method_change_allowed = "false"
elif audit_state == "recorded":
    formal_decision = "recorded_non_winner"
    next_formal_action = "rerun_or_next_candidate_depends_on_remaining_queue"
elif audit_state in {"needs_docs", "missing_artifacts", "missing_log"}:
    formal_decision = "closeout_incomplete"
    next_formal_action = "finish_closeout_then_review"

if audit_state == "recorded":
    rows = audit.get("rows") or []
    remaining_reruns = [
        str(entry.get("label"))
        for entry in rows
        if entry.get("audit") == "needs_rerun" and entry.get("label") != row.get("label")
    ]
    if remaining_reruns:
        next_formal_action = "rerun_remaining_queue:" + ",".join(remaining_reruns)
    elif beats_baseline != "yes":
        next_formal_action = "design_next_formal_candidate"
        method_change_allowed = "true"

if audit_state == "needs_rerun":
    method_change_allowed = "false"
elif audit_state == "recorded" and beats_baseline == "yes":
    method_change_allowed = "false"

print(f"formal_decision={formal_decision}")
print(f"next_formal_action={next_formal_action}")
print(f"method_change_allowed={method_change_allowed}")

meta = FORMAL_CANDIDATES.get(model_name) or {}
if meta:
    print(f"single_variable={meta.get('single_variable_cn')}")
    print(f"hypothesis={meta.get('hypothesis_cn')}")
    if meta.get("config_reference_rel"):
        print(f"config_reference_rel={meta.get('config_reference_rel')}")
    if meta.get("config_reference_label_cn"):
        print(f"config_reference_label={meta.get('config_reference_label_cn')}")
    config_diff_keys = meta.get("config_diff_keys") or []
    if config_diff_keys:
        print(f"config_diff_keys={','.join(config_diff_keys)}")
    if meta.get("config_diff_rationale_cn"):
        print(f"config_diff_rationale={meta.get('config_diff_rationale_cn')}")
    if meta.get("implementation_guard_cn"):
        print(f"implementation_guard={meta.get('implementation_guard_cn')}")
    expected_signals = meta.get("expected_signals_cn") or []
    if expected_signals:
        print("expected_signals:")
        for signal in expected_signals:
            print(f"- {signal}")
    if meta.get("if_fail_cn"):
        print(f"if_not_winner={meta.get('if_fail_cn')}")

decision_lines = []
capture = False
for raw_line in strict_report.splitlines():
    if raw_line.strip() == "## Decision":
        capture = True
        continue
    if capture and raw_line.startswith("## "):
        break
    if capture and raw_line.startswith("- "):
        decision_lines.append(raw_line)

if decision_lines:
    print("decision_summary:")
    for line in decision_lines:
        print(line)
PY
