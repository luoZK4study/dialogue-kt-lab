#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from current_formal_state import build_current_formal_state
from task_conditioned_failure_utils import monitor_due_state


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "cel_stage1_last_layer"
STATUS_JSON = RESULT_DIR / "task_conditioned_status.json"
AUDIT_JSON = RESULT_DIR / "formal_experiment_audit.json"
STATUS_MD = RESULT_DIR / "STATUS.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract(pattern: str, text: str, default: str = "unknown") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    parser.add_argument("--audit-json", default=str(AUDIT_JSON))
    parser.add_argument("--status-md", default=str(STATUS_MD))
    parser.add_argument("--ssh-alias", default="3090")
    parser.add_argument("--format", choices=("text", "json", "env"), default="text")
    args = parser.parse_args()

    task_status = load_json(Path(args.status_json))
    audit = load_json(Path(args.audit_json))
    status_md = read_text(Path(args.status_md))
    state = build_current_formal_state(
        task_status,
        audit,
        stage=extract(r"- Stage: `([^`]+)`", status_md),
        sync_freshness=extract(r"- Task-conditioned sync freshness: `([^`]+)`", status_md),
        recommended_poll_interval=extract(r"- Recommended task-conditioned poll interval: `([^`]+)`", status_md),
    )
    suggested_next_monitor_after = extract(
        r"- Suggested next task-conditioned monitor after: `([^`]+)`",
        status_md,
        default="",
    )
    authoritative_remote_refresh_event = extract(
        r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`",
        status_md,
        default="",
    )
    latest_refresh_attempt = extract(
        r"- Latest task-conditioned refresh attempt: `([^`]+)`",
        status_md,
        default="",
    )
    active_timing_summary = extract(
        r"- 当前[^：\n]*计时：([^\n]+)",
        status_md,
        default="",
    )
    state["suggested_next_monitor_after"] = suggested_next_monitor_after or None
    state["authoritative_remote_refresh_event"] = authoritative_remote_refresh_event or None
    state["latest_refresh_attempt"] = latest_refresh_attempt or None
    state["active_timing_summary"] = active_timing_summary or None
    due_state = monitor_due_state(state["suggested_next_monitor_after"])
    state["monitor_due_now"] = due_state["due_now"] if due_state is not None else None
    state["seconds_until_suggested_monitor"] = (
        due_state["seconds_until_due"] if due_state is not None else None
    )
    state["remaining_wait_text"] = due_state["remaining_wait_text"] if due_state is not None else None

    if args.format == "json":
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if args.format == "env":
        for key, value in state.items():
            if isinstance(value, list):
                print(f"{key}={','.join(value) if value else 'none'}")
            elif value is None:
                print(f"{key}=none")
            else:
                print(f"{key}={value}")
        return

    next_action = state["next_action"]
    winner_found = state["winner_found"]
    rerun_queue = state["rerun_queue"]
    recorded_non_winners = state["recorded_non_winner_queue"]
    sync_freshness = state["sync_freshness"]
    poll_interval = state["recommended_poll_interval"]
    stage = state["stage"]
    next_monitor_after = state["suggested_next_monitor_after"]
    authoritative_remote_refresh_event = state["authoritative_remote_refresh_event"]
    latest_refresh_attempt = state["latest_refresh_attempt"]
    active_timing_summary = state["active_timing_summary"]
    monitor_due_now = state["monitor_due_now"]
    seconds_until_suggested_monitor = state["seconds_until_suggested_monitor"]
    remaining_wait_text = state["remaining_wait_text"]

    print("== Current Formal Next Action ==")
    print(f"stage={stage}")
    print(f"next_action={next_action}")
    print(f"winner_found={winner_found}")
    print(f"sync_freshness={sync_freshness}")
    print(f"recommended_poll_interval={poll_interval}")
    if authoritative_remote_refresh_event:
        print(f"authoritative_remote_refresh_event={authoritative_remote_refresh_event}")
    if latest_refresh_attempt:
        print(f"latest_refresh_attempt={latest_refresh_attempt}")
    if next_monitor_after:
        print(f"suggested_next_monitor_after={next_monitor_after}")
    if active_timing_summary:
        print(f"active_timing_summary={active_timing_summary}")
    if monitor_due_now is not None:
        print(f"monitor_due_now={monitor_due_now}")
    if seconds_until_suggested_monitor is not None:
        print(f"seconds_until_suggested_monitor={seconds_until_suggested_monitor}")
    if remaining_wait_text is not None:
        print(f"remaining_wait_text={remaining_wait_text}")
    print(f"rerun_queue={','.join(rerun_queue) if rerun_queue else 'none'}")
    print(f"recorded_non_winner_queue={','.join(recorded_non_winners) if recorded_non_winners else 'none'}")
    print()

    current_formal_decision = state["current_formal_decision"]
    suggested_launch_key = state["suggested_launch_key"]

    if current_formal_decision == "winner_or_done":
        print("current_formal_decision=winner_or_done")
        print("1. Freeze the winner across all markdown surfaces.")
        print("2. One-shot alias wrapper for the current state:")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias} --dry-run")
        print("3. Run the current-state closeout wrappers if any winner surface is still pending:")
        print("   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh")
        print("   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh")
        print(f"   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        finalize_key = state["suggested_finalize_key"]
        review_key = state["suggested_review_key"]
        if finalize_key and review_key:
            print("4. If you need the explicit candidate-level closeout wrappers:")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {finalize_key}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {review_key}")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {finalize_key}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {review_key}")
        return

    if current_formal_decision == "monitor_active_full_train":
        print("current_formal_decision=monitor_active_full_train")
        current_active_target = state["current_active_target"]
        current_active_model = state["current_active_model"]
        if current_active_target:
            print(f"current_active_target={current_active_target}")
        if current_active_model:
            print(f"current_active_model={current_active_model}")
        if next_monitor_after and monitor_due_now is False and remaining_wait_text:
            print(
                f"0. Wait until around {next_monitor_after} before the next authoritative refresh "
                f"(remaining {remaining_wait_text})."
            )
        print("1. One-shot alias wrapper for the current state:")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias} --dry-run")
        print("2. Monitor one cycle:")
        print("   bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once")
        print("3. If you want the same monitor through your local SSH alias:")
        print(f"   bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh {args.ssh_alias} once")
        print(f"   TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh {args.ssh_alias} watch")
        print("4. If you need a bounded watch cycle in the current shell:")
        print("   TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch")
        print("5. After completion, close out through the current-state wrappers:")
        print("   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh")
        print("   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh")
        print("6. If you want the same closeout path through your local SSH alias:")
        print(f"   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        if current_active_target:
            print("7. If you need the explicit candidate-level closeout wrappers:")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {current_active_target}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {current_active_target}")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {current_active_target}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {current_active_target}")
        return

    if current_formal_decision == "launch_next_candidate":
        print("current_formal_decision=launch_next_candidate")
        print("1. One-shot alias wrapper for the current state:")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias} --dry-run")
        print("2. Print the recommended WSL alias template:")
        print("   bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh")
        print("3. Launch the current queued candidate through the current-state alias wrapper:")
        print(f"   bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        print("4. If you need the explicit candidate-level alias wrapper:")
        print(f"   bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {suggested_launch_key}")
        print("5. If alias mode is unavailable, use the current-state manual fallback:")
        print("   bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh")
        print("6. If you need the explicit candidate-level direct wrapper or manual fallback:")
        print(f"   bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {suggested_launch_key}")
        print(f"   bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {suggested_launch_key}")
        return

    if current_formal_decision == "freeze_recorded_then_rerun":
        rerun_label = state["current_rerun_target"]
        rerun_model = state["current_rerun_model"]
        rerun_launch_key = suggested_launch_key
        print("current_formal_decision=freeze_recorded_then_rerun")
        print(f"current_rerun_target={rerun_label}")
        if rerun_model:
            print(f"current_rerun_model={rerun_model}")
        print("1. One-shot alias wrapper for the current state:")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias} --dry-run")
        print("2. Freeze any recorded non-winner through the rebuilt authoritative surfaces.")
        print("3. Print the recommended WSL alias template:")
        print("   bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh")
        print("4. Relaunch the rerun-first candidate through the current-state alias wrapper:")
        print(f"   bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        print("5. If you need the explicit candidate-level alias wrapper:")
        print(f"   bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {rerun_launch_key}")
        print("6. If alias mode is unavailable, use the current-state manual fallback:")
        print("   bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh")
        print("7. If you need the explicit candidate-level direct wrapper or manual fallback:")
        print(f"   bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {rerun_launch_key}")
        print(f"   bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {rerun_launch_key}")
        print("8. After the SSH run lands, close out through the current-state wrappers:")
        print("   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh")
        print("   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh")
        print("9. If you need the explicit candidate-level closeout wrappers:")
        print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {rerun_launch_key}")
        print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {rerun_launch_key}")
        print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {rerun_launch_key}")
        print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {rerun_launch_key}")
        return

    if current_formal_decision == "design_next_formal_candidate":
        completed_target = state["current_completed_target"]
        completed_model = state["current_completed_model"]
        print("current_formal_decision=design_next_formal_candidate")
        if completed_target:
            print(f"latest_recorded_non_winner_target={completed_target}")
        if completed_model:
            print(f"latest_recorded_non_winner_model={completed_model}")
        print("1. No further alias automation should fire here; the completed formal queue is already frozen as non-winners.")
        print("2. Recheck the authoritative local/remote surfaces before editing the next candidate:")
        print("   python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py")
        print(f"   bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh {args.ssh_alias}")
        print("3. If you want an explicit closeout review of the latest recorded non-winner:")
        if completed_target:
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {completed_target}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {completed_target}")
        else:
            print("   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh")
            print(f"   bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh {args.ssh_alias}")
        print("4. Then scaffold the next one-variable strict formal candidate and rerun preflight before launch.")
        print("   python3 scripts/cel_stage1_last_layer/scaffold_formal_candidate.py --help")
        print("   bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh")
        return

    if current_formal_decision == "review_recorded_or_completed_formal_candidate":
        completed_target = state["current_completed_target"]
        completed_model = state["current_completed_model"]
        print("current_formal_decision=review_recorded_or_completed_formal_candidate")
        if completed_target:
            print(f"current_completed_target={completed_target}")
        if completed_model:
            print(f"current_completed_model={completed_model}")
        print("1. One-shot alias wrapper for the current state:")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias}")
        print(f"   bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh {args.ssh_alias} --dry-run")
        print("2. Finalize the current recorded/completed formal candidate through the current-state wrapper:")
        print("   bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh")
        print("3. Review the current recorded/completed formal candidate through the current-state wrapper:")
        print("   bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh")
        if completed_target:
            print("4. If you need the explicit candidate-level closeout wrappers:")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {completed_target}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {completed_target}")
            print(f"   bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {completed_target}")
            print(f"   bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh {args.ssh_alias} {completed_target}")
        print("5. After review, decide whether to freeze a winner or design the next one-variable formal candidate.")
        return

    print("current_formal_decision=inspect_authoritative_surfaces")
    print("1. Inspect STATUS / STRICT_FULL_TRAIN_REPORT / FORMAL_EXPERIMENT_AUDIT.")
    print("2. If the queue is fully recorded and still below baseline, design one new formal candidate with one main variable changed.")


if __name__ == "__main__":
    main()
