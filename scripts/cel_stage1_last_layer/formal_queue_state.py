#!/usr/bin/env python3

from __future__ import annotations

from formal_candidate_registry import FORMAL_CANDIDATES


def launch_key_for_model(model_name: str | None) -> str | None:
    if not model_name:
        return None
    meta = FORMAL_CANDIDATES.get(model_name)
    if meta is None:
        return None
    return str(meta["launch_key"])


def audit_rows(formal_audit: dict | None) -> list[dict]:
    return list((formal_audit or {}).get("rows") or [])


def rerun_rows(formal_audit: dict | None) -> list[dict]:
    return [row for row in audit_rows(formal_audit) if row.get("audit") == "needs_rerun"]


def recorded_non_winner_rows(formal_audit: dict | None) -> list[dict]:
    return [
        row
        for row in audit_rows(formal_audit)
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]


def row_label_or_launch_key(row: dict) -> str | None:
    label = row.get("label")
    if label:
        return str(label)
    return launch_key_for_model(str(row.get("model")) if row.get("model") else None)


def preferred_explicit_launch_key(status_rows: list[dict] | None) -> str | None:
    rows = status_rows or []
    status_iterators = {
        "running": reversed(rows),
        "failed": reversed(rows),
        "pending": iter(rows),
        "done": reversed(rows),
    }
    for status in ("running", "failed", "pending", "done"):
        for row in status_iterators[status]:
            if row.get("status") != status:
                continue
            key = row_label_or_launch_key(row)
            if key:
                return key
    return None


def current_explicit_launch_key(formal_audit: dict | None, status_rows: list[dict] | None = None) -> str | None:
    for row in rerun_rows(formal_audit):
        key = row_label_or_launch_key(row)
        if key:
            return key
    return preferred_explicit_launch_key(status_rows)


def current_recorded_non_winner_key(formal_audit: dict | None, status_rows: list[dict] | None = None) -> str | None:
    for row in reversed(recorded_non_winner_rows(formal_audit)):
        key = row_label_or_launch_key(row)
        if key:
            return key
    rows = status_rows or []
    for row in reversed(rows):
        if row.get("status") != "done":
            continue
        key = row_label_or_launch_key(row)
        if key:
            return key
    return None
