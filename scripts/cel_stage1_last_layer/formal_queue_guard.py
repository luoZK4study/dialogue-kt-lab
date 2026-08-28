#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORMAL_AUDIT_JSON = ROOT / "results" / "cel_stage1_last_layer" / "formal_experiment_audit.json"


def load_audit(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"formal audit JSON missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse formal audit JSON at {path}: {exc}") from exc


def audit_rows(audit: dict) -> list[dict]:
    return list(audit.get("rows") or [])


def rerun_labels(audit: dict) -> list[str]:
    return [
        str(row.get("label") or row.get("model"))
        for row in audit_rows(audit)
        if row.get("audit") == "needs_rerun"
    ]


def recorded_non_winner_labels(audit: dict) -> list[str]:
    return [
        str(row.get("label") or row.get("model"))
        for row in audit_rows(audit)
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]


def assert_launch_allowed(candidate_label: str, audit: dict) -> None:
    reruns = rerun_labels(audit)
    if not reruns:
        return
    if candidate_label in reruns:
        return
    raise SystemExit(
        "strict formal queue violation: "
        f"candidate `{candidate_label}` cannot launch while rerun-first candidate(s) remain: {','.join(reruns)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-label", help="Formal candidate label to validate for launch, e.g. v21")
    parser.add_argument("--mode", choices=["summary", "assert-launch"], default="summary")
    parser.add_argument("--audit-json", default=str(FORMAL_AUDIT_JSON))
    args = parser.parse_args()

    audit = load_audit(Path(args.audit_json))
    reruns = rerun_labels(audit)
    recorded = recorded_non_winner_labels(audit)

    if args.mode == "assert-launch":
        if not args.candidate_label:
            raise SystemExit("--candidate-label is required for --mode assert-launch")
        assert_launch_allowed(args.candidate_label, audit)
        print(f"launch_allowed={args.candidate_label}")
        return

    rerun_text = ",".join(reruns) if reruns else "none"
    recorded_text = ",".join(recorded) if recorded else "none"
    print(f"needs_rerun={rerun_text}")
    print(f"recorded_non_winner={recorded_text}")


if __name__ == "__main__":
    main()
