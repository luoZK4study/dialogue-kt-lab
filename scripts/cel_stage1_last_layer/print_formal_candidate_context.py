#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_candidate_registry import (
    FORMAL_CANDIDATE_LAUNCH_KEYS,
    FORMAL_CANDIDATES,
    ROUND3_FAILURE_FOLLOWUPS_CN,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = ROOT / "results" / "cel_stage1_last_layer" / "formal_experiment_audit.json"
STATUS_JSON = ROOT / "results" / "cel_stage1_last_layer" / "task_conditioned_status.json"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def audit_rows(audit: dict | None) -> list[dict]:
    if not audit:
        return []
    return list(audit.get("rows") or [])


def derive_launch_allowed(candidate_key: str, rerun_queue: list[str]) -> bool:
    if not rerun_queue:
        return True
    return candidate_key in rerun_queue


def derive_next_formal_action(candidate_key: str, audit_state: str, beats_baseline: str, rerun_queue: list[str]) -> str:
    if audit_state == "needs_rerun":
        return f"rerun_same_scope:{candidate_key}"
    if audit_state == "recorded" and beats_baseline == "yes":
        return f"freeze_winner:{candidate_key}"
    if rerun_queue:
        return "rerun_remaining_queue:" + ",".join(rerun_queue)
    if audit_state == "recorded":
        return "design_next_formal_candidate"
    if audit_state in {"needs_docs", "missing_artifacts", "missing_log"}:
        return "finish_closeout_then_review"
    return "inspect_authoritative_surfaces"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_key")
    parser.add_argument("--audit-json", default=str(AUDIT_JSON))
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    args = parser.parse_args()

    model_name = FORMAL_CANDIDATE_LAUNCH_KEYS.get(args.candidate_key)
    if model_name is None:
        raise SystemExit(f"unknown candidate key: {args.candidate_key}")

    meta = FORMAL_CANDIDATES[model_name]
    audit = load_json(Path(args.audit_json))
    task_status = load_json(Path(args.status_json))
    rows = audit_rows(audit)
    current_row = next((row for row in rows if row.get("model") == model_name), None)
    rerun_queue = [str(row.get("label")) for row in rows if row.get("audit") == "needs_rerun"]
    recorded_non_winners = [
        str(row.get("label"))
        for row in rows
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]
    audit_state = str(current_row.get("audit")) if current_row else "pending"
    beats_baseline = str(current_row.get("beats_baseline")) if current_row else "--"
    launch_allowed = derive_launch_allowed(args.candidate_key, rerun_queue)
    next_formal_action = derive_next_formal_action(args.candidate_key, audit_state, beats_baseline, rerun_queue)

    print("formal_launch_context:")
    print(f"- candidate_key={args.candidate_key}")
    print(f"- model_name={model_name}")
    print(f"- next_action={task_status.get('next_action') if task_status else 'unknown'}")
    print(f"- audit_state={audit_state}")
    print(f"- beats_baseline={beats_baseline}")
    print(f"- launch_allowed={'yes' if launch_allowed else 'no'}")
    print(f"- next_formal_action={next_formal_action}")
    print("- strict_scope=SSH_train+val+test_only")
    print("- strict_scope_guard=no_ckpt_only_no_post_hoc_no_frozen_retraining")
    print(f"- start_point={meta.get('start_point_cn')}")
    print(f"- single_variable={meta.get('single_variable_cn')}")
    print(f"- hypothesis={meta.get('hypothesis_cn')}")
    if meta.get("implementation_guard_cn"):
        print(f"- implementation_guard={meta.get('implementation_guard_cn')}")
    print(f"- config_reference={meta.get('config_reference_rel')}")
    diff_keys = meta.get("config_diff_keys") or []
    print(f"- config_diff_keys={','.join(diff_keys) if diff_keys else 'none'}")
    print(f"- rerun_queue={','.join(rerun_queue) if rerun_queue else 'none'}")
    print(f"- recorded_non_winner_queue={','.join(recorded_non_winners) if recorded_non_winners else 'none'}")
    if current_row and current_row.get("note"):
        print(f"- audit_note={current_row.get('note')}")
    print("- expected_signals:")
    for signal in meta.get("expected_signals_cn") or []:
        print(f"  - {signal}")
    followups = ROUND3_FAILURE_FOLLOWUPS_CN.get(model_name) or []
    if followups:
        print("- rerun_requirements:")
        for line in followups:
            print(f"  - {line}")
    if meta.get("if_fail_cn"):
        print(f"- if_not_winner={meta.get('if_fail_cn')}")


if __name__ == "__main__":
    main()
