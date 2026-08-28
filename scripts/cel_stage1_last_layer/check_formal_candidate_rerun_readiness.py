#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATE_LAUNCH_KEYS, FORMAL_CANDIDATES


ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = ROOT / "results" / "cel_stage1_last_layer" / "formal_experiment_audit.json"
STATUS_JSON = ROOT / "results" / "cel_stage1_last_layer" / "task_conditioned_status.json"
CHECK_BCE = ROOT / "scripts" / "cel_stage1_last_layer" / "check_probability_bce_safety.py"
CHECK_SYNC = ROOT / "scripts" / "cel_stage1_last_layer" / "check_sync_manifest_coverage.py"


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


def run_check(script_path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or proc.stderr).strip()
    return proc.returncode == 0, output


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

    audit_state = str(current_row.get("audit")) if current_row else "pending"
    next_action = str(task_status.get("next_action")) if task_status else "unknown"
    queue_ok = (not rerun_queue) or (args.candidate_key in rerun_queue)
    candidate_requires_rerun = audit_state == "needs_rerun"
    candidate_is_recorded = audit_state == "recorded"
    bce_ok, bce_output = run_check(CHECK_BCE)
    sync_ok, sync_output = run_check(CHECK_SYNC)

    ready = queue_ok and audit_state not in {"recorded"} and bce_ok and sync_ok

    print("formal_launch_readiness:")
    print(f"- candidate_key={args.candidate_key}")
    print(f"- model_name={model_name}")
    print(f"- next_action={next_action}")
    print(f"- audit_state={audit_state}")
    print(f"- rerun_queue={','.join(rerun_queue) if rerun_queue else 'none'}")
    print(f"- queue_allows_this_candidate={'yes' if queue_ok else 'no'}")
    print(f"- candidate_requires_rerun={'yes' if candidate_requires_rerun else 'no'}")
    print(f"- candidate_is_recorded={'yes' if candidate_is_recorded else 'no'}")
    print(f"- probability_bce_safety_ok={'yes' if bce_ok else 'no'}")
    print(f"- sync_manifest_ok={'yes' if sync_ok else 'no'}")
    print(f"- strict_scope={meta.get('scope_en', 'SSH train + val + test')}")
    print("- strict_scope_guard=no_ckpt_only_no_post_hoc_no_frozen_retraining")
    if current_row and current_row.get("note"):
        print(f"- audit_note={current_row.get('note')}")
    print(f"- launch_ready={'yes' if ready else 'no'}")
    print("- readiness_checks:")
    print(f"  - BCE safety check: {bce_output or 'no output'}")
    print(f"  - Sync manifest check: {sync_output or 'no output'}")

    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
