#!/usr/bin/env python3

from __future__ import annotations

from formal_candidate_registry import FORMAL_CANDIDATE_LAUNCH_KEYS, FORMAL_CANDIDATES
from formal_queue_state import current_explicit_launch_key, current_recorded_non_winner_key


def label_to_model(audit_rows: list[dict], label: str) -> str | None:
    for row in audit_rows:
        if str(row.get("label")) == label:
            return str(row.get("model"))
    return None


def round3_rows(task_status: dict | None) -> list[dict]:
    if task_status is None:
        return []
    for round_info in task_status.get("rounds") or []:
        if str(round_info.get("title", "")).startswith("Round 3"):
            return list(round_info.get("models") or [])
    return []


def model_to_launch_key(model_name: str | None) -> str | None:
    if not model_name:
        return None
    meta = FORMAL_CANDIDATES.get(model_name)
    if meta is None:
        return None
    return str(meta["launch_key"])


def round3_status_for_launch_key(rows: list[dict], launch_key: str | None) -> str | None:
    if not launch_key:
        return None
    for row in rows:
        row_model = str(row.get("model")) if row.get("model") else None
        row_launch_key = model_to_launch_key(row_model)
        if row_launch_key == launch_key or str(row.get("label") or "") == launch_key:
            status = row.get("status")
            return str(status) if status else None
    return None


def all_round3_done(rows: list[dict]) -> bool:
    return bool(rows) and all(str(row.get("status") or "") == "done" for row in rows)


def build_current_formal_state(
    task_status: dict | None,
    audit: dict | None,
    *,
    stage: str = "unknown",
    sync_freshness: str = "unknown",
    recommended_poll_interval: str = "unknown",
) -> dict:
    next_action = str((task_status or {}).get("next_action") or "unknown")
    winner_found = bool((task_status or {}).get("winner_found"))
    audit_rows = list((audit or {}).get("rows") or [])
    round3 = round3_rows(task_status)
    explicit_launch_key = current_explicit_launch_key(audit, round3)
    recorded_non_winner_key = current_recorded_non_winner_key(audit, round3)
    rerun_queue = [str(row.get("label")) for row in audit_rows if row.get("audit") == "needs_rerun"]
    recorded_non_winners = [
        str(row.get("label"))
        for row in audit_rows
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]
    recorded_non_winner_model = FORMAL_CANDIDATE_LAUNCH_KEYS.get(recorded_non_winner_key) if recorded_non_winner_key else None

    state = {
        "stage": stage,
        "next_action": next_action,
        "winner_found": winner_found,
        "sync_freshness": sync_freshness,
        "recommended_poll_interval": recommended_poll_interval,
        "rerun_queue": rerun_queue,
        "recorded_non_winner_queue": recorded_non_winners,
        "current_formal_decision": "inspect_authoritative_surfaces",
        "current_rerun_target": None,
        "current_rerun_model": None,
        "current_active_target": None,
        "current_active_model": None,
        "current_completed_target": None,
        "current_completed_model": None,
        "current_recorded_non_winner_target": recorded_non_winner_key,
        "current_recorded_non_winner_model": recorded_non_winner_model,
        "suggested_launch_key": None,
        "suggested_finalize_key": None,
        "suggested_review_key": None,
    }

    if winner_found or next_action == "done":
        state["current_formal_decision"] = "winner_or_done"
        state["suggested_finalize_key"] = explicit_launch_key or recorded_non_winner_key
        state["suggested_review_key"] = explicit_launch_key or recorded_non_winner_key
        return state

    if next_action == "wait_round3":
        running_models = [
            str(row.get("model"))
            for row in round3
            if str(row.get("status")) == "running" and row.get("model")
        ]
        if len(running_models) == 1:
            running_model = running_models[0]
            running_launch_key = model_to_launch_key(running_model)
            state["current_active_model"] = running_model
            state["current_active_target"] = running_launch_key
            state["suggested_finalize_key"] = running_launch_key
            state["suggested_review_key"] = running_launch_key
        state["current_formal_decision"] = "monitor_active_full_train"
        return state

    if next_action == "launch_round3":
        first_launch_key = explicit_launch_key or next(iter(FORMAL_CANDIDATE_LAUNCH_KEYS.keys()))
        state["current_formal_decision"] = "launch_next_candidate"
        state["suggested_launch_key"] = first_launch_key
        state["suggested_finalize_key"] = first_launch_key
        state["suggested_review_key"] = first_launch_key
        return state

    if next_action == "manual_decide" and rerun_queue:
        rerun_label = rerun_queue[0]
        rerun_model = label_to_model(audit_rows, rerun_label)
        rerun_launch_key = FORMAL_CANDIDATES[rerun_model]["launch_key"] if rerun_model else rerun_label
        state["current_formal_decision"] = "freeze_recorded_then_rerun"
        state["current_rerun_target"] = rerun_label
        state["current_rerun_model"] = rerun_model
        state["suggested_launch_key"] = rerun_launch_key
        state["suggested_finalize_key"] = rerun_launch_key
        state["suggested_review_key"] = rerun_launch_key
        return state

    if next_action == "manual_decide" and explicit_launch_key is not None:
        explicit_status = round3_status_for_launch_key(round3, explicit_launch_key)
        if explicit_status == "pending":
            pending_model = FORMAL_CANDIDATE_LAUNCH_KEYS.get(explicit_launch_key)
            state["current_formal_decision"] = "launch_next_candidate"
            state["current_active_target"] = explicit_launch_key
            state["current_active_model"] = pending_model
            state["suggested_launch_key"] = explicit_launch_key
            state["suggested_finalize_key"] = explicit_launch_key
            state["suggested_review_key"] = explicit_launch_key
            return state
        if all_round3_done(round3) and recorded_non_winners:
            state["current_formal_decision"] = "design_next_formal_candidate"
            state["current_completed_target"] = recorded_non_winner_key
            state["current_completed_model"] = recorded_non_winner_model
            state["suggested_review_key"] = recorded_non_winner_key
            return state
        completed_model = FORMAL_CANDIDATE_LAUNCH_KEYS.get(explicit_launch_key)
        state["current_formal_decision"] = "review_recorded_or_completed_formal_candidate"
        state["current_completed_target"] = explicit_launch_key
        state["current_completed_model"] = completed_model
        state["suggested_finalize_key"] = explicit_launch_key
        state["suggested_review_key"] = explicit_launch_key
        return state

    return state
