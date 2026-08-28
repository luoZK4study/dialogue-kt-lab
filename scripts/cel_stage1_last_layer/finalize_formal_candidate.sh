#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT_DIR/results/cel_stage1_last_layer"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"

CANDIDATE_KEY="${1:-}"
SYNC_TIMEOUT_SECS="${TASK_CONDITIONED_FINALIZE_SYNC_TIMEOUT_SECS:-240}"
SKIP_REFRESH="${TASK_CONDITIONED_FINALIZE_SKIP_REFRESH:-0}"
QUIET="${TASK_CONDITIONED_FINALIZE_QUIET:-0}"
ALLOW_INCOMPLETE="${TASK_CONDITIONED_FINALIZE_ALLOW_INCOMPLETE:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh <candidate_key>

Purpose:
  Run the strict post-run closeout sequence for one formal candidate:
  1. refresh/sync SSH artifacts once
  2. rebuild all authoritative markdown/json surfaces
  3. verify the candidate's current audit state and closeout readiness

Environment:
  TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1      Skip SSH refresh and only read the current local snapshot
  TASK_CONDITIONED_FINALIZE_SYNC_TIMEOUT_SECS   Sync timeout for the refresh step (default: 240)
  TASK_CONDITIONED_FINALIZE_QUIET=1            Suppress the human-facing closeout summary and only enforce the closeout step
  TASK_CONDITIONED_FINALIZE_ALLOW_INCOMPLETE=1 Allow incomplete closeout inspection without failing the command

Examples:
  bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v21
  TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v22
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
AUDIT_JSON="$RESULT_DIR/formal_experiment_audit.json"
STATUS_MD="$RESULT_DIR/STATUS.md"
STRICT_REPORT="$RESULT_DIR/STRICT_FULL_TRAIN_REPORT.md"
AUDIT_MD="$RESULT_DIR/FORMAL_EXPERIMENT_AUDIT.md"
RECORD_MD="$RESULT_DIR/CEL_Stage1_LastLayer_DialogueKT_实验记录.md"
TUNING_LOOP_MD="$RESULT_DIR/TASK_CONDITIONED_TUNING_LOOP.md"
COMPARISON_MD="$RESULT_DIR/COMPARISON.md"
ANALYSIS_MD="$RESULT_DIR/DETAILED_ANALYSIS.md"
BRIEFS_MD="$RESULT_DIR/FORMAL_CANDIDATE_BRIEFS.md"
CONFIG_DIFF_MD="$RESULT_DIR/FORMAL_CANDIDATE_CONFIG_DIFFS.md"

if [[ "$QUIET" != "1" ]]; then
  echo "== Strict Formal Finalize =="
  echo "candidate_key=$CANDIDATE_KEY"
  echo "model_name=$MODEL_NAME"
  echo "strict_report=$STRICT_REPORT"
  echo "formal_audit=$AUDIT_MD"
  echo "stage1_record=$RECORD_MD"
  echo
fi

if [[ "$SKIP_REFRESH" != "1" ]]; then
  TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS="$SYNC_TIMEOUT_SECS" \
    bash "$ROOT_DIR/scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh"
else
  bash "$ROOT_DIR/scripts/cel_stage1_last_layer/rebuild_task_conditioned_reports.sh"
fi

python3 - <<'PY' "$CANDIDATE_KEY" "$MODEL_NAME" "$AUDIT_JSON" "$STATUS_MD" "$STRICT_REPORT" "$AUDIT_MD" "$RECORD_MD" "$TUNING_LOOP_MD" "$COMPARISON_MD" "$ANALYSIS_MD" "$BRIEFS_MD" "$CONFIG_DIFF_MD" "$QUIET" "$ALLOW_INCOMPLETE"
import json
import re
import sys
from pathlib import Path

candidate_key = sys.argv[1]
model_name = sys.argv[2]
audit_json = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
status_md = Path(sys.argv[4]).read_text(encoding="utf-8", errors="ignore")
strict_report = Path(sys.argv[5]).read_text(encoding="utf-8", errors="ignore")
surface_paths = {
    "formal_audit": Path(sys.argv[6]),
    "stage1_record": Path(sys.argv[7]),
    "tuning_loop": Path(sys.argv[8]),
    "comparison": Path(sys.argv[9]),
    "detailed_analysis": Path(sys.argv[10]),
    "formal_candidate_briefs": Path(sys.argv[11]),
    "formal_candidate_config_diffs": Path(sys.argv[12]),
}
quiet = sys.argv[13] == "1"
allow_incomplete = sys.argv[14] == "1"

stage_match = re.search(r"- Stage: `([^`]+)`", status_md)
stage = stage_match.group(1) if stage_match else "unknown"
refresh_match = re.search(r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`", status_md)
refresh_event = refresh_match.group(1) if refresh_match else "none"

row = None
for entry in audit_json.get("rows") or []:
    if entry.get("model") == model_name:
        row = entry
        break

if row is None:
    raise SystemExit(f"candidate {model_name} not found in formal_experiment_audit.json")

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

surface_presence = {}
for label, path in surface_paths.items():
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    surface_presence[label] = model_name in text

missing_surfaces = row.get("missing_surfaces") or []
closeout_ready = row.get("audit") in {"recorded", "needs_rerun"}
if not quiet:
    print("closeout_stage=" + stage)
    print("latest_refresh=" + refresh_event)
    print("audit_state=" + str(row.get("audit")))
    print("status=" + str(row.get("status")))
    print("artifacts_ok=" + str(row.get("artifacts_ok")))
    print("metrics_complete=" + str(row.get("metrics_complete")))
    print("log_ok=" + str(row.get("log_ok")))
    print("coverage_ok=" + str(row.get("coverage_ok")))
    print("beats_baseline=" + str(row.get("beats_baseline")))
    metrics = row.get("metrics") or {}
    if metrics:
        print("overall_acc=" + str(metrics.get("overall_acc")))
        print("overall_auc=" + str(metrics.get("overall_auc")))
        print("final_acc=" + str(metrics.get("final_acc")))
        print("final_auc=" + str(metrics.get("final_auc")))
    analysis = row.get("analysis") or {}
    if analysis:
        if analysis.get("pred_true") is not None:
            print("pred_true=" + str(round(float(analysis.get("pred_true")), 4)))
        if analysis.get("best_acc") is not None:
            print("best_acc=" + str(round(float(analysis.get("best_acc")), 4)))
        if analysis.get("best_threshold") is not None:
            print("best_threshold=" + str(round(float(analysis.get("best_threshold")), 6)))
        if analysis.get("delta_acc_vs_baseline") is not None:
            print("delta_acc_vs_baseline=" + str(round(float(analysis.get("delta_acc_vs_baseline")), 4)))
        if analysis.get("delta_auc_vs_baseline") is not None:
            print("delta_auc_vs_baseline=" + str(round(float(analysis.get("delta_auc_vs_baseline")), 4)))
        if analysis.get("delta_final_acc_vs_baseline") is not None:
            print("delta_final_acc_vs_baseline=" + str(round(float(analysis.get("delta_final_acc_vs_baseline")), 4)))
        if analysis.get("delta_final_auc_vs_baseline") is not None:
            print("delta_final_auc_vs_baseline=" + str(round(float(analysis.get("delta_final_auc_vs_baseline")), 4)))
        if analysis.get("delta_acc_vs_anchor") is not None:
            print("delta_acc_vs_anchor=" + str(round(float(analysis.get("delta_acc_vs_anchor")), 4)))
        if analysis.get("delta_auc_vs_anchor") is not None:
            print("delta_auc_vs_anchor=" + str(round(float(analysis.get("delta_auc_vs_anchor")), 4)))
    print("audit_note=" + str(row.get("note")))

    print("closeout_surface_presence:")
    for label, present in surface_presence.items():
        print(f"- {label}={present}")

    if missing_surfaces:
        print("audit_missing_surfaces:")
        for label in missing_surfaces:
            print(f"- {label}")

    print("closeout_ready=" + str(closeout_ready))

    if decision_lines:
        print("decision_summary:")
        for line in decision_lines:
            print(line)

if not closeout_ready and not allow_incomplete:
    raise SystemExit(
        "strict formal closeout is incomplete; finish artifact sync / markdown coverage before review "
        f"(audit_state={row.get('audit')}, missing_surfaces={missing_surfaces})"
    )
PY
