#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from current_formal_state import build_current_formal_state


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise SystemExit(f"missing required field in {label}: {pattern}")
    return match.group(1)


def extract_section(title: str, text: str, label: str) -> str:
    pattern = rf"## {re.escape(title)}\n\n(.*?)(?:\n## |\Z)"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise SystemExit(f"missing required section in {label}: {title}")
    return match.group(1)


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/cel_stage1_last_layer")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    status_json = load_json(result_dir / "task_conditioned_status.json")
    audit_json = load_json(result_dir / "formal_experiment_audit.json")
    status_md = read_text(result_dir / "STATUS.md")
    strict_report = read_text(result_dir / "STRICT_FULL_TRAIN_REPORT.md")
    protocol_md = read_text(result_dir / "FORMAL_METHOD_PROTOCOL.md")
    briefs_md = read_text(result_dir / "FORMAL_CANDIDATE_BRIEFS.md")
    tuning_loop_md = read_text(result_dir / "TASK_CONDITIONED_TUNING_LOOP.md")
    formal_loop_md = read_text(result_dir / "FORMAL_EXPERIMENT_LOOP.md")
    record_md = read_text(result_dir / "CEL_Stage1_LastLayer_DialogueKT_实验记录.md")
    runbook_md = read_text(result_dir / "STRICT_FULL_TRAIN_RUNBOOK.md")

    next_action = str(status_json.get("next_action"))
    winner_found = bool(status_json.get("winner_found"))
    expected_winner = bool_text(winner_found)

    audit_counts = audit_json.get("counts") or {}
    audit_summary = ", ".join(f"{key}={value}" for key, value in sorted(audit_counts.items())) or "unknown"
    audit_latest_refresh_event = str(audit_json.get("latest_refresh_event") or "none")
    audit_latest_successful_refresh_event = str(
        audit_json.get("latest_successful_refresh_event")
        or audit_json.get("latest_refresh_event")
        or "none"
    )
    rerun_queue = [
        str(row.get("label"))
        for row in (audit_json.get("rows") or [])
        if row.get("audit") == "needs_rerun"
    ]
    recorded_non_winners = [
        str(row.get("label"))
        for row in (audit_json.get("rows") or [])
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]

    status_stage = extract(r"- Stage: `([^`]+)`", status_md, "STATUS.md")
    status_sync_freshness = extract(r"- Task-conditioned sync freshness: `([^`]+)`", status_md, "STATUS.md")
    status_authoritative_remote_refresh = extract(
        r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    status_refresh_attempt = extract(
        r"- Latest task-conditioned refresh attempt: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    status_next_action = extract(r"- Task-conditioned next action: `([^`]+)`", status_md, "STATUS.md")
    status_winner = extract(r"- Task-conditioned winner found: `([^`]+)`", status_md, "STATUS.md")
    status_audit_summary = extract(r"- Formal experiment audit summary: `([^`]+)`", status_md, "STATUS.md")
    status_poll_interval = extract(r"- Recommended task-conditioned poll interval: `([^`]+)`", status_md, "STATUS.md")
    _status_controller_pid_alive = extract(
        r"- Task-conditioned controller pid/alive: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    _status_controller_cmd = extract(
        r"- Task-conditioned controller command: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    _status_controller_child_cmd = extract(
        r"- Task-conditioned controller child command: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    _status_controller_next_wake = extract(
        r"- Task-conditioned controller next wake estimate: `([^`]+)`",
        status_md,
        "STATUS.md",
    )
    status_next_monitor_after = extract(
        r"- Suggested next task-conditioned monitor after: `([^`]+)`",
        status_md,
        "STATUS.md",
    ) if "- Suggested next task-conditioned monitor after:" in status_md else ""
    status_formal_decision = extract(r"- Current formal decision: `([^`]+)`", status_md, "STATUS.md")
    status_rerun_target = extract(r"- Current formal rerun target: `([^`]+)`", status_md, "STATUS.md")
    status_recorded_non_winner = extract(r"- Current formal recorded non-winner: `([^`]+)`", status_md, "STATUS.md")
    status_recorded_non_winner_queue = extract(
        r"- Current formal recorded non-winner queue: `([^`]+)`",
        status_md,
        "STATUS.md",
    )

    report_stage = extract(r"- Stage: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_authoritative_remote_refresh = extract(
        r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_refresh_attempt = extract(
        r"- Latest task-conditioned refresh attempt: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_next_action = extract(r"- Next action: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_winner = extract(r"- Winner found: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_audit_summary = extract(r"- Formal experiment audit summary: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_poll_interval = extract(r"- Recommended local poll interval: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_controller_pid_alive = extract(
        r"- Task-conditioned controller pid/alive: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_controller_cmd = extract(
        r"- Task-conditioned controller command: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_controller_child_cmd = extract(
        r"- Task-conditioned controller child command: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_controller_next_wake = extract(
        r"- Task-conditioned controller next wake estimate: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    report_next_monitor_after = extract(
        r"- Suggested next local monitor after: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    ) if "- Suggested next local monitor after:" in strict_report else ""
    report_formal_decision = extract(r"- Current formal decision: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_rerun_target = extract(r"- Current formal rerun target: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_recorded_non_winner = extract(r"- Current formal recorded non-winner: `([^`]+)`", strict_report, "STRICT_FULL_TRAIN_REPORT.md")
    report_recorded_non_winner_queue = extract(
        r"- Current formal recorded non-winner queue: `([^`]+)`",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )

    audit_authoritative_remote_refresh = extract(
        r"- Latest authoritative task-conditioned remote refresh event: `([^`]+)`",
        briefs_md.replace("Formal Candidate Briefs", "Formal Candidate Briefs"),
        "FORMAL_CANDIDATE_BRIEFS.md",
    ) if "- Latest authoritative task-conditioned remote refresh event:" in briefs_md else None

    protocol_next_action = extract(r"- Current next action: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_winner = extract(r"- Winner found: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_authoritative_remote_refresh = extract(
        r"- Latest authoritative remote refresh event: `([^`]+)`",
        protocol_md,
        "FORMAL_METHOD_PROTOCOL.md",
    )
    protocol_refresh_attempt = extract(
        r"- Latest refresh attempt: `([^`]+)`",
        protocol_md,
        "FORMAL_METHOD_PROTOCOL.md",
    )
    protocol_audit_summary = extract(r"- Formal audit summary: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_poll_interval = extract(r"- Recommended local monitoring cadence: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_next_monitor_after = extract(
        r"- Suggested next local monitor after: `([^`]+)`",
        protocol_md,
        "FORMAL_METHOD_PROTOCOL.md",
    ) if "- Suggested next local monitor after:" in protocol_md else ""
    protocol_formal_decision = extract(r"- Current formal decision: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_rerun_target = extract(r"- Current formal rerun target: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_recorded_non_winner = extract(r"- Current formal recorded non-winner: `([^`]+)`", protocol_md, "FORMAL_METHOD_PROTOCOL.md")
    protocol_recorded_non_winner_queue = extract(
        r"- Current formal recorded non-winner queue: `([^`]+)`",
        protocol_md,
        "FORMAL_METHOD_PROTOCOL.md",
    )
    protocol_current_required_action = extract_section(
        "Current Required Action",
        protocol_md,
        "FORMAL_METHOD_PROTOCOL.md",
    )
    strict_report_decision = extract_section(
        "Decision",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    strict_report_operator_loop = extract_section(
        "Operator Loop",
        strict_report,
        "STRICT_FULL_TRAIN_REPORT.md",
    )
    formal_loop_current_queue = extract_section(
        "当前具体执行队列",
        formal_loop_md,
        "FORMAL_EXPERIMENT_LOOP.md",
    )
    record_current_status = extract_section(
        "当前状态",
        record_md,
        "CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
    )
    record_authoritative_remote_refresh = extract(
        r"- 最新 authoritative task-conditioned 远端真值摘要：`([^`]+)`",
        record_md,
        "CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
    )
    record_refresh_attempt = extract(
        r"- 最新 task-conditioned 刷新尝试：`([^`]+)`",
        record_md,
        "CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
    )
    record_next_steps = extract_section(
        "下一步",
        record_md,
        "CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
    )
    record_next_monitor_after = extract(
        r"- 建议下一次 task-conditioned 监控时间：`([^`]+)`",
        record_md,
        "CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
    ) if "- 建议下一次 task-conditioned 监控时间：" in record_md else ""

    current_state = build_current_formal_state(
        status_json,
        audit_json,
        stage=status_stage,
        sync_freshness=status_sync_freshness,
        recommended_poll_interval=status_poll_interval,
    )
    expected_formal_decision = str(current_state["current_formal_decision"])
    expected_rerun_target = str(current_state["current_rerun_target"] or "none")
    expected_recorded_non_winner = str(current_state["current_recorded_non_winner_target"] or "none")
    expected_recorded_non_winner_queue = ",".join(current_state.get("recorded_non_winner_queue") or []) or "none"

    mismatches: list[str] = []
    if status_next_action != next_action:
        mismatches.append(f"STATUS.md next action mismatch: {status_next_action} != {next_action}")
    if report_next_action != next_action:
        mismatches.append(f"STRICT_FULL_TRAIN_REPORT.md next action mismatch: {report_next_action} != {next_action}")
    if protocol_next_action != next_action:
        mismatches.append(f"FORMAL_METHOD_PROTOCOL.md next action mismatch: {protocol_next_action} != {next_action}")

    if status_winner != expected_winner:
        mismatches.append(f"STATUS.md winner mismatch: {status_winner} != {expected_winner}")
    if report_winner != expected_winner:
        mismatches.append(f"STRICT_FULL_TRAIN_REPORT.md winner mismatch: {report_winner} != {expected_winner}")
    if protocol_winner != expected_winner:
        mismatches.append(f"FORMAL_METHOD_PROTOCOL.md winner mismatch: {protocol_winner} != {expected_winner}")

    if status_stage != report_stage:
        mismatches.append(f"stage mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: {status_stage} != {report_stage}")

    if status_authoritative_remote_refresh != audit_latest_successful_refresh_event:
        mismatches.append(
            "STATUS.md authoritative remote refresh mismatch: "
            f"{status_authoritative_remote_refresh} != {audit_latest_successful_refresh_event}"
        )
    if report_authoritative_remote_refresh != audit_latest_successful_refresh_event:
        mismatches.append(
            "STRICT_FULL_TRAIN_REPORT.md authoritative remote refresh mismatch: "
            f"{report_authoritative_remote_refresh} != {audit_latest_successful_refresh_event}"
        )
    if protocol_authoritative_remote_refresh != audit_latest_successful_refresh_event:
        mismatches.append(
            "FORMAL_METHOD_PROTOCOL.md authoritative remote refresh mismatch: "
            f"{protocol_authoritative_remote_refresh} != {audit_latest_successful_refresh_event}"
        )
    if record_authoritative_remote_refresh != audit_latest_successful_refresh_event:
        mismatches.append(
            "CEL_Stage1_LastLayer_DialogueKT_实验记录.md authoritative remote refresh mismatch: "
            f"{record_authoritative_remote_refresh} != {audit_latest_successful_refresh_event}"
        )

    if status_refresh_attempt != audit_latest_refresh_event:
        mismatches.append(
            f"STATUS.md refresh attempt mismatch: {status_refresh_attempt} != {audit_latest_refresh_event}"
        )
    if report_refresh_attempt != audit_latest_refresh_event:
        mismatches.append(
            f"STRICT_FULL_TRAIN_REPORT.md refresh attempt mismatch: {report_refresh_attempt} != {audit_latest_refresh_event}"
        )
    if protocol_refresh_attempt != audit_latest_refresh_event:
        mismatches.append(
            f"FORMAL_METHOD_PROTOCOL.md refresh attempt mismatch: {protocol_refresh_attempt} != {audit_latest_refresh_event}"
        )
    if record_refresh_attempt != audit_latest_refresh_event:
        mismatches.append(
            "CEL_Stage1_LastLayer_DialogueKT_实验记录.md refresh attempt mismatch: "
            f"{record_refresh_attempt} != {audit_latest_refresh_event}"
        )

    if status_audit_summary != audit_summary:
        mismatches.append(f"STATUS.md audit summary mismatch: {status_audit_summary} != {audit_summary}")
    if report_audit_summary != audit_summary:
        mismatches.append(f"STRICT_FULL_TRAIN_REPORT.md audit summary mismatch: {report_audit_summary} != {audit_summary}")
    if protocol_audit_summary != audit_summary:
        mismatches.append(f"FORMAL_METHOD_PROTOCOL.md audit summary mismatch: {protocol_audit_summary} != {audit_summary}")

    if report_poll_interval != status_poll_interval:
        mismatches.append(
            f"poll interval mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: {status_poll_interval} != {report_poll_interval}"
        )
    if report_controller_pid_alive != _status_controller_pid_alive:
        mismatches.append(
            "controller pid/alive mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: "
            f"{_status_controller_pid_alive} != {report_controller_pid_alive}"
        )
    if report_controller_cmd != _status_controller_cmd:
        mismatches.append(
            "controller command mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: "
            f"{_status_controller_cmd} != {report_controller_cmd}"
        )
    if report_controller_child_cmd != _status_controller_child_cmd:
        mismatches.append(
            "controller child command mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: "
            f"{_status_controller_child_cmd} != {report_controller_child_cmd}"
        )
    if report_controller_next_wake != _status_controller_next_wake:
        mismatches.append(
            "controller next wake estimate mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: "
            f"{_status_controller_next_wake} != {report_controller_next_wake}"
        )
    if protocol_poll_interval != status_poll_interval:
        mismatches.append(
            f"FORMAL_METHOD_PROTOCOL.md poll interval mismatch: {protocol_poll_interval} != {status_poll_interval}"
        )
    if status_next_monitor_after != report_next_monitor_after:
        mismatches.append(
            "suggested next monitor mismatch between STATUS.md and STRICT_FULL_TRAIN_REPORT.md: "
            f"{status_next_monitor_after or 'none'} != {report_next_monitor_after or 'none'}"
        )
    if status_next_monitor_after != protocol_next_monitor_after:
        mismatches.append(
            "suggested next monitor mismatch between STATUS.md and FORMAL_METHOD_PROTOCOL.md: "
            f"{status_next_monitor_after or 'none'} != {protocol_next_monitor_after or 'none'}"
        )
    if status_next_monitor_after != record_next_monitor_after:
        mismatches.append(
            "suggested next monitor mismatch between STATUS.md and CEL_Stage1_LastLayer_DialogueKT_实验记录.md: "
            f"{status_next_monitor_after or 'none'} != {record_next_monitor_after or 'none'}"
        )

    if status_formal_decision != expected_formal_decision:
        mismatches.append(f"STATUS.md formal decision mismatch: {status_formal_decision} != {expected_formal_decision}")
    if report_formal_decision != expected_formal_decision:
        mismatches.append(
            f"STRICT_FULL_TRAIN_REPORT.md formal decision mismatch: {report_formal_decision} != {expected_formal_decision}"
        )
    if protocol_formal_decision != expected_formal_decision:
        mismatches.append(
            f"FORMAL_METHOD_PROTOCOL.md formal decision mismatch: {protocol_formal_decision} != {expected_formal_decision}"
        )

    if status_rerun_target != expected_rerun_target:
        mismatches.append(f"STATUS.md rerun target mismatch: {status_rerun_target} != {expected_rerun_target}")
    if report_rerun_target != expected_rerun_target:
        mismatches.append(
            f"STRICT_FULL_TRAIN_REPORT.md rerun target mismatch: {report_rerun_target} != {expected_rerun_target}"
        )
    if protocol_rerun_target != expected_rerun_target:
        mismatches.append(
            f"FORMAL_METHOD_PROTOCOL.md rerun target mismatch: {protocol_rerun_target} != {expected_rerun_target}"
        )

    if status_recorded_non_winner != expected_recorded_non_winner:
        mismatches.append(
            f"STATUS.md recorded non-winner mismatch: {status_recorded_non_winner} != {expected_recorded_non_winner}"
        )
    if report_recorded_non_winner != expected_recorded_non_winner:
        mismatches.append(
            f"STRICT_FULL_TRAIN_REPORT.md recorded non-winner mismatch: {report_recorded_non_winner} != {expected_recorded_non_winner}"
        )
    if protocol_recorded_non_winner != expected_recorded_non_winner:
        mismatches.append(
            f"FORMAL_METHOD_PROTOCOL.md recorded non-winner mismatch: {protocol_recorded_non_winner} != {expected_recorded_non_winner}"
        )
    if status_recorded_non_winner_queue.replace(" ", "") != expected_recorded_non_winner_queue:
        mismatches.append(
            "STATUS.md recorded non-winner queue mismatch: "
            f"{status_recorded_non_winner_queue} != {expected_recorded_non_winner_queue}"
        )
    if report_recorded_non_winner_queue.replace(" ", "") != expected_recorded_non_winner_queue:
        mismatches.append(
            "STRICT_FULL_TRAIN_REPORT.md recorded non-winner queue mismatch: "
            f"{report_recorded_non_winner_queue} != {expected_recorded_non_winner_queue}"
        )
    if protocol_recorded_non_winner_queue.replace(" ", "") != expected_recorded_non_winner_queue:
        mismatches.append(
            "FORMAL_METHOD_PROTOCOL.md recorded non-winner queue mismatch: "
            f"{protocol_recorded_non_winner_queue} != {expected_recorded_non_winner_queue}"
        )

    if expected_formal_decision == "monitor_active_full_train":
        for snippet in (
            "monitor_formal_candidate.sh once",
            "finalize_current_formal_candidate.sh",
            "review_current_formal_candidate.sh",
        ):
            if snippet not in protocol_current_required_action:
                mismatches.append(f"FORMAL_METHOD_PROTOCOL.md Current Required Action missing snippet: {snippet}")
        if "scaffold_formal_candidate.py" in protocol_current_required_action:
            mismatches.append(
                "FORMAL_METHOD_PROTOCOL.md Current Required Action should not suggest scaffold_formal_candidate.py during active monitoring"
            )
        active_target = str(current_state.get("current_active_target") or "")
        if active_target and active_target not in protocol_current_required_action:
            mismatches.append(
                f"FORMAL_METHOD_PROTOCOL.md Current Required Action missing active target label: {active_target}"
            )
        if active_target and active_target not in strict_report_decision:
            mismatches.append(
                f"STRICT_FULL_TRAIN_REPORT.md Decision section missing active target label: {active_target}"
            )
        if active_target:
            expected_finalize_cmd = (
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {active_target}"
            )
            expected_review_cmd = (
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {active_target}"
            )
            if expected_finalize_cmd not in strict_report_operator_loop:
                mismatches.append(
                    "STRICT_FULL_TRAIN_REPORT.md Operator Loop missing finalize command for active target: "
                    f"{active_target}"
                )
            if expected_review_cmd not in strict_report_operator_loop:
                mismatches.append(
                    "STRICT_FULL_TRAIN_REPORT.md Operator Loop missing review command for active target: "
                    f"{active_target}"
                )
        if status_sync_freshness == "fresh_remote" and status_poll_interval not in {"unknown", "not_applicable"}:
            if not status_next_monitor_after:
                mismatches.append("STATUS.md missing suggested next monitor timestamp during fresh active monitoring")
            if not report_next_monitor_after:
                mismatches.append(
                    "STRICT_FULL_TRAIN_REPORT.md missing suggested next monitor timestamp during fresh active monitoring"
                )
            if not protocol_next_monitor_after:
                mismatches.append(
                    "FORMAL_METHOD_PROTOCOL.md missing suggested next monitor timestamp during fresh active monitoring"
                )
            if not record_next_monitor_after:
                mismatches.append(
                    "CEL_Stage1_LastLayer_DialogueKT_实验记录.md missing suggested next monitor timestamp during fresh active monitoring"
                )
        if "当前不是启动新设计的时候" not in formal_loop_current_queue:
            mismatches.append(
                "FORMAL_EXPERIMENT_LOOP.md 当前具体执行队列 should explicitly block new-method design during active monitoring"
            )
        if active_target and active_target not in formal_loop_current_queue:
            mismatches.append(
                f"FORMAL_EXPERIMENT_LOOP.md 当前具体执行队列 missing active target label: {active_target}"
            )
        if active_target and active_target not in runbook_md:
            mismatches.append(
                f"STRICT_FULL_TRAIN_RUNBOOK.md missing active target label during active monitoring: {active_target}"
            )
        if "继续监控" not in record_current_status and "继续监控" not in record_next_steps:
            mismatches.append(
                "CEL_Stage1_LastLayer_DialogueKT_实验记录.md should explicitly tell the operator to keep monitoring during active training"
            )
        active_model = str(current_state.get("current_active_model") or "")
        if active_model and active_model not in record_current_status and active_model not in record_next_steps:
            mismatches.append(
                f"CEL_Stage1_LastLayer_DialogueKT_实验记录.md missing active model during active monitoring: {active_model}"
            )
        if active_model and active_model not in runbook_md:
            mismatches.append(
                f"STRICT_FULL_TRAIN_RUNBOOK.md missing active model during active monitoring: {active_model}"
            )

    if next_action == "manual_decide" and rerun_queue:
        rerun_target = rerun_queue[0]
        rerun_model = None
        for row in (audit_json.get("rows") or []):
            if row.get("label") == rerun_target:
                rerun_model = str(row.get("model"))
                break
        if rerun_model is None:
            mismatches.append(f"unable to resolve rerun target model for label {rerun_target}")

        for snippet in (
            "print_current_formal_next_action.py",
            "print_current_formal_manual_launch_steps.sh",
            "print_recommended_wsl_ssh_alias.sh",
            "start_current_formal_candidate_via_ssh_alias.sh 3090",
            "start_formal_candidate_via_ssh_alias.sh 3090",
            "finalize_current_formal_candidate.sh",
            "review_current_formal_candidate.sh",
        ):
            if snippet not in status_md:
                mismatches.append(f"STATUS.md missing snippet: {snippet}")
            if snippet not in strict_report and snippet != "print_current_formal_next_action.py":
                mismatches.append(f"STRICT_FULL_TRAIN_REPORT.md missing snippet: {snippet}")
            if snippet not in protocol_md:
                mismatches.append(f"FORMAL_METHOD_PROTOCOL.md missing snippet: {snippet}")
            if snippet not in briefs_md and snippet != "print_current_formal_next_action.py":
                mismatches.append(f"FORMAL_CANDIDATE_BRIEFS.md missing snippet: {snippet}")
            if snippet not in tuning_loop_md and snippet != "print_current_formal_next_action.py":
                mismatches.append(f"TASK_CONDITIONED_TUNING_LOOP.md missing snippet: {snippet}")
            if snippet not in formal_loop_md:
                mismatches.append(f"FORMAL_EXPERIMENT_LOOP.md missing snippet: {snippet}")
            if snippet not in record_md and snippet != "print_current_formal_next_action.py":
                mismatches.append(f"CEL_Stage1_LastLayer_DialogueKT_实验记录.md missing snippet: {snippet}")
            if snippet not in runbook_md:
                mismatches.append(f"STRICT_FULL_TRAIN_RUNBOOK.md missing snippet: {snippet}")

        if "Recommended WSL SSH Alias" not in protocol_md:
            mismatches.append("FORMAL_METHOD_PROTOCOL.md missing Recommended WSL SSH Alias section")
        if "print_manual_formal_launch_steps.sh" not in strict_report:
            mismatches.append("STRICT_FULL_TRAIN_REPORT.md missing manual fallback command")
        if "print_manual_formal_launch_steps.sh" not in briefs_md:
            mismatches.append("FORMAL_CANDIDATE_BRIEFS.md missing manual fallback command")
        if "print_manual_formal_launch_steps.sh" not in tuning_loop_md:
            mismatches.append("TASK_CONDITIONED_TUNING_LOOP.md missing manual fallback command")
        if "print_manual_formal_launch_steps.sh" not in formal_loop_md:
            mismatches.append("FORMAL_EXPERIMENT_LOOP.md missing manual fallback command")
        if "print_manual_formal_launch_steps.sh" not in record_md:
            mismatches.append("CEL_Stage1_LastLayer_DialogueKT_实验记录.md missing manual fallback command")
        if "print_manual_formal_launch_steps.sh" not in runbook_md:
            mismatches.append("STRICT_FULL_TRAIN_RUNBOOK.md missing manual fallback command")
        if rerun_target not in protocol_md:
            mismatches.append(f"FORMAL_METHOD_PROTOCOL.md missing rerun label: {rerun_target}")
        if rerun_target not in formal_loop_md:
            mismatches.append(f"FORMAL_EXPERIMENT_LOOP.md missing rerun label: {rerun_target}")
        if rerun_model and rerun_model not in strict_report:
            mismatches.append(f"STRICT_FULL_TRAIN_REPORT.md missing rerun model: {rerun_model}")
        if rerun_model and rerun_model not in tuning_loop_md:
            mismatches.append(f"TASK_CONDITIONED_TUNING_LOOP.md missing rerun model: {rerun_model}")

        for label in recorded_non_winners:
            if label not in protocol_md:
                mismatches.append(f"FORMAL_METHOD_PROTOCOL.md missing recorded non-winner label: {label}")
            if label not in formal_loop_md:
                mismatches.append(f"FORMAL_EXPERIMENT_LOOP.md missing recorded non-winner label: {label}")

    if mismatches:
        raise SystemExit("formal authoritative surface consistency check failed:\n" + "\n".join(mismatches))

    print(
        "formal authoritative surfaces agree on stage, next action, winner state, audit summary, and current rerun workflow"
    )


if __name__ == "__main__":
    main()
