#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from current_formal_state import build_current_formal_state
from formal_candidate_registry import (
    FORMAL_CANDIDATE_ORDER,
    FORMAL_CANDIDATES,
    ROUND3_FAILURE_FOLLOWUPS_EN,
    formal_candidate_log_path,
)
from formal_queue_state import current_explicit_launch_key, current_recorded_non_winner_key
from task_conditioned_failure_utils import (
    controller_next_cycle_estimate,
    epoch_cycle_note,
    extract_failure_evidence,
    format_monitor_timestamp,
    format_progress_en,
    progress_timing_note,
    latest_phase_progress_from_log as parse_phase_progress_from_log,
    recommend_round3_poll_interval,
    recommended_next_monitor_after,
    stability_milestone_note,
)


LOSS_RE = re.compile(r"^Loss:\s+([0-9.]+)$", re.MULTILINE)
OVERALL_RE = re.compile(
    r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)
PROGRESS_RE = re.compile(r"(Training|Validation|Validating|Testing):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
TIMESTAMPED_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+(.*)$")
CONTROLLER_PID_PATH = Path("results/cel_stage1_last_layer/task_conditioned_controller.pid")


def parse_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    loss_match = LOSS_RE.search(text)
    overall_match = OVERALL_RE.search(text)
    final_match = FINAL_RE.search(text)
    if not loss_match or not overall_match:
        return None
    return {
        "model": path.stem.replace("metrics_", "", 1),
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
    }


def read_task_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_formal_audit(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")


def round3_statuses(task_status: dict | None) -> dict[str, dict]:
    if task_status is None:
        return {}
    for round_info in task_status.get("rounds") or []:
        title = round_info.get("title") or ""
        if title.startswith("Round 3"):
            return {row.get("model"): row for row in (round_info.get("models") or []) if row.get("model")}
    return {}


def infer_stage(task_status: dict | None) -> str | None:
    if task_status is None:
        return None
    next_action = task_status.get("next_action")
    winner_found = task_status.get("winner_found")
    rounds = task_status.get("rounds") or []
    round1 = rounds[0] if len(rounds) > 0 else {}
    round2 = rounds[1] if len(rounds) > 1 else {}
    round2_models = round2.get("models") or []
    round2_has_started = any(model.get("status") in {"running", "done", "failed"} for model in round2_models)
    round2_has_running = any(model.get("status") == "running" for model in round2_models)

    if winner_found:
        return "task_conditioned_tuning_completed_with_winner"
    if next_action == "launch_round2":
        return "task_conditioned_legacy_diagnostic_state"
    if next_action == "launch_round3":
        return "task_conditioned_round3_pending_launch"
    if next_action == "wait_round1":
        if round1.get("all_done"):
            return "task_conditioned_legacy_diagnostic_state"
        if round2_has_running or round2_has_started:
            return "task_conditioned_legacy_diagnostic_state"
        return "task_conditioned_legacy_diagnostic_state"
    if next_action == "wait_round2":
        return "task_conditioned_legacy_diagnostic_state"
    if next_action == "wait_round3":
        return "task_conditioned_round3_running"
    if next_action == "manual_decide":
        return "task_conditioned_round3_done_waiting_decision"
    if next_action == "done":
        return "task_conditioned_tuning_done"
    return None


def latest_refresh_event(refresh_log_path: Path) -> str | None:
    events = [line.strip() for line in read_log_text(refresh_log_path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_complete_events = [line for line in events if "task_conditioned refresh complete" in line]
    return refresh_complete_events[-1] if refresh_complete_events else events[-1]


def latest_successful_refresh_event(refresh_log_path: Path) -> str | None:
    events = [line.strip() for line in read_log_text(refresh_log_path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_success_events = [
        line
        for line in events
        if "task_conditioned refresh complete" in line and "sync=ok" in line
    ]
    return refresh_success_events[-1] if refresh_success_events else None


def sync_freshness_label(refresh_event: str | None) -> str:
    if refresh_event is None:
        return "unknown"
    if "sync=ok" in refresh_event:
        return "fresh_remote"
    if "sync=failed" in refresh_event:
        return "stale_local_failed_refresh"
    if "sync=timeout" in refresh_event:
        return "stale_local_timed_out_refresh"
    return "unknown"


def controller_runtime_snapshot() -> dict[str, str]:
    snapshot = {
        "pid": "none",
        "alive": "no",
        "cmd": "none",
        "child_cmd": "none",
    }
    if not CONTROLLER_PID_PATH.exists():
        return snapshot
    pid_text = CONTROLLER_PID_PATH.read_text(encoding="utf-8").strip()
    if not pid_text:
        return snapshot
    snapshot["pid"] = pid_text
    try:
        ps_result = subprocess.run(
            ["ps", "-p", pid_text, "-o", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return snapshot
    cmd = ps_result.stdout.strip()
    if not cmd:
        return snapshot
    snapshot["alive"] = "yes"
    snapshot["cmd"] = cmd
    try:
        child_result = subprocess.run(
            ["pgrep", "-P", pid_text],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return snapshot
    child_pids = [line.strip() for line in child_result.stdout.splitlines() if line.strip()]
    if not child_pids:
        return snapshot
    try:
        child_ps_result = subprocess.run(
            ["ps", "-p", child_pids[0], "-o", "args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return snapshot
    child_cmd = child_ps_result.stdout.strip()
    if child_cmd:
        snapshot["child_cmd"] = child_cmd
    return snapshot


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def beat_flag(metrics: dict | None, baseline: dict | None) -> str:
    if metrics is None or baseline is None:
        return "--"
    acc_win = metrics["overall_acc"] > baseline["overall_acc"]
    auc_win = metrics["overall_auc"] > baseline["overall_auc"]
    if acc_win and auc_win:
        return "yes"
    if acc_win or auc_win:
        return "partial"
    return "no"


def final_pair_flag(metrics: dict | None, baseline: dict | None) -> str:
    if metrics is None or baseline is None:
        return "--"
    if any(
        value is None
        for value in (
            metrics.get("final_acc"),
            metrics.get("final_auc"),
            baseline.get("final_acc"),
            baseline.get("final_auc"),
        )
    ):
        return "--"
    final_acc_win = metrics["final_acc"] > baseline["final_acc"]
    final_auc_win = metrics["final_auc"] > baseline["final_auc"]
    if final_acc_win and final_auc_win:
        return "yes"
    if final_acc_win or final_auc_win:
        return "partial"
    return "no"


def build_analysis(
    status_row: dict | None,
    metrics: dict | None,
    baseline: dict | None,
    anchor: dict | None,
) -> dict | None:
    analysis = {
        "pred_true": status_row.get("pred_true") if status_row else None,
        "best_acc": status_row.get("best_acc") if status_row else None,
        "best_threshold": status_row.get("best_threshold") if status_row else None,
        "delta_acc_vs_baseline": None,
        "delta_auc_vs_baseline": None,
        "delta_final_acc_vs_baseline": None,
        "delta_final_auc_vs_baseline": None,
        "delta_acc_vs_anchor": None,
        "delta_auc_vs_anchor": None,
    }
    if metrics is not None and baseline is not None:
        analysis["delta_acc_vs_baseline"] = metrics["overall_acc"] - baseline["overall_acc"]
        analysis["delta_auc_vs_baseline"] = metrics["overall_auc"] - baseline["overall_auc"]
        if metrics.get("final_acc") is not None and baseline.get("final_acc") is not None:
            analysis["delta_final_acc_vs_baseline"] = metrics["final_acc"] - baseline["final_acc"]
        if metrics.get("final_auc") is not None and baseline.get("final_auc") is not None:
            analysis["delta_final_auc_vs_baseline"] = metrics["final_auc"] - baseline["final_auc"]
    if metrics is not None and anchor is not None:
        analysis["delta_acc_vs_anchor"] = metrics["overall_acc"] - anchor["overall_acc"]
        analysis["delta_auc_vs_anchor"] = metrics["overall_auc"] - anchor["overall_auc"]
    if any(value is not None for value in analysis.values()):
        return analysis
    return None


def build_candidate_rows(
    task_status: dict | None,
    metrics_dir: Path,
    baseline: dict | None,
    anchor: dict | None,
) -> list[dict]:
    statuses = round3_statuses(task_status)
    rows: list[dict] = []
    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        status_row = statuses.get(model_name, {})
        metrics = parse_metrics(metrics_dir / f"metrics_{model_name}.txt")
        progress = parse_phase_progress_from_log(formal_candidate_log_path(Path("."), model_name))
        rows.append(
            {
                "model": model_name,
                "label": meta["label"],
                "start_point": meta["start_point_en"],
                "method": meta["method_en"],
                "status": status_row.get("status", "pending"),
                "failure_reason": status_row.get("failure_reason"),
                "metrics": metrics,
                "analysis": build_analysis(status_row, metrics, baseline, anchor),
                "progress": progress,
            }
        )
    return rows


def recommended_poll_interval(rows: list[dict], task_status: dict | None) -> tuple[int | None, str] | None:
    if task_status is None:
        return None
    next_action = task_status.get("next_action")
    if next_action in {"manual_decide", "done"}:
        return None, "not applicable while no active strict full-train monitoring is required"
    running_rows = [row for row in rows if row.get("status") == "running"]
    for row in running_rows or rows:
        interval = recommend_round3_poll_interval(row["progress"], next_action)
        if interval is not None and interval[1] != "default training cadence":
            return interval
    return recommend_round3_poll_interval(None, next_action)


def poll_interval_label(poll_interval: tuple[int | None, str] | None) -> str:
    if poll_interval is None:
        return "unknown"
    interval_secs, _reason = poll_interval
    if interval_secs is None:
        return "not_applicable"
    return f"{interval_secs}s"


def status_text(row: dict) -> str:
    status = row["status"]
    if status == "failed" and row["failure_reason"]:
        status = f"{status} ({row['failure_reason']})"
    progress = row["progress"]
    if progress is not None and row["status"] == "running":
        epoch_text = f", epoch {progress['epoch']}" if progress["epoch"] is not None else ""
        status += f"; {progress['phase']}{epoch_text}; {progress['progress']}"
    return status


def build_failure_focus(rows: list[dict]) -> list[str]:
    failed_rows = [row for row in rows if row["status"] == "failed"]
    if not failed_rows:
        return []

    lines = ["## Failure Focus", ""]
    for row in failed_rows:
        lines.append(f"### `{row['label']}` / `{row['model']}`")
        lines.append("")
        log_path = formal_candidate_log_path(Path("."), row["model"])
        evidence = extract_failure_evidence(log_path)
        notes = []
        progress_text = format_progress_en(evidence["progress"])
        if progress_text is not None:
            notes.append(f"Latest synced failure point: {progress_text}.")
        if evidence["assertion_line"] is not None:
            notes.append(f'Key assertion from the synced log: "{evidence["assertion_line"]}"')
        if evidence.get("assertion_site") is not None:
            notes.append(f"Assertion site: `{evidence['assertion_site']}`")
        if evidence["runtime_line"] is not None:
            notes.append(f"Key runtime error from the synced log: `{evidence['runtime_line']}`")
        if not notes:
            notes.append(f"Current failure type: `{row.get('failure_reason', 'failed')}`. Keep the strict full-train scope fixed when rerunning.")
        notes.extend(ROUND3_FAILURE_FOLLOWUPS_EN.get(row["model"], []))
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return lines


def build_current_decision(
    rows: list[dict],
    baseline: dict | None,
    anchor: dict | None,
    current_formal_state: dict,
) -> list[str]:
    if baseline is None:
        return ["- Baseline metrics are missing, so strict winner judgment is currently unavailable."]

    if str(current_formal_state.get("current_formal_decision") or "") == "monitor_active_full_train":
        active_target = str(current_formal_state.get("current_active_target") or "unknown")
        active_model = str(current_formal_state.get("current_active_model") or "unknown")
        lines = [
            f"- An active strict full-train is still running: `{active_target}` / `{active_model}`.",
            "- No strict winner judgment is allowed until this SSH-side `train + val + test` finishes and the closeout audit is rebuilt.",
        ]
        done = [row for row in rows if row["metrics"] is not None]
        if done:
            best = max(done, key=lambda row: (row["metrics"]["overall_auc"], row["metrics"]["overall_acc"]))
            lines.append(
                f"- Best completed Round 3 candidate so far remains `{best['model']}` with Overall Acc / AUC **{best['metrics']['overall_acc']:.2f} / {best['metrics']['overall_auc']:.2f}**, but it is only a recorded non-winner under the current baseline gate."
            )
        lines.append(
            "- Next step: keep monitoring the active run, then execute unified `finalize -> review` closeout after its metrics land."
        )
        return lines

    audit_ready_winners = [
        row for row in rows
        if row["status"] == "done" and row["metrics"] is not None and beat_flag(row["metrics"], baseline) == "yes"
    ]
    if audit_ready_winners:
        best = max(audit_ready_winners, key=lambda row: (row["metrics"]["overall_acc"], row["metrics"]["overall_auc"]))
        return [
            f"- A-module reference recorded from historical checkpoint `{best['model']}` with Overall Acc / AUC **{best['metrics']['overall_acc']:.2f} / {best['metrics']['overall_auc']:.2f}**.",
            "- Next step: keep the A-module result frozen, then implement and verify the explicit `h_r/h_m` dual-path objective, tensor contracts, and audit path before registering a formal Stage 2 candidate.",
        ]

    done = [row for row in rows if row["metrics"] is not None]
    failed = [row for row in rows if row["status"] == "failed"]
    if done:
        best = max(done, key=lambda row: (row["metrics"]["overall_auc"], row["metrics"]["overall_acc"]))
        raw_metric_win = beat_flag(best["metrics"], baseline) == "yes"
        lines = [
            f"- Full-train results exist, but no candidate is audit-ready as a strict winner yet. Best completed Round 3 candidate so far: `{best['model']}` with Overall Acc / AUC **{best['metrics']['overall_acc']:.2f} / {best['metrics']['overall_auc']:.2f}**."
        ]
        if anchor is not None:
            anchor_acc = anchor["overall_acc"]
            anchor_auc = anchor["overall_auc"]
            lines.append(
                f"- Relative to the current anchor `v1`, that candidate changes Acc / AUC by **{best['metrics']['overall_acc'] - anchor_acc:+.2f} / {best['metrics']['overall_auc'] - anchor_auc:+.2f}**."
            )
        if raw_metric_win:
            lines.append("- Its raw metrics already clear the baseline gate, but strict winner declaration is still blocked until `FORMAL_EXPERIMENT_AUDIT.md` is clean.")
        elif failed:
            failed_names = " / ".join(f"`{row['model']}`" for row in failed)
            lines.append("- The completed candidate should now be treated as a recorded non-winner under the strict baseline gate.")
            lines.append(
                f"- Next step: keep that completed result frozen, sync the `dialogue_kt/training.py` fix to SSH, and rerun {failed_names} first before designing any new end-to-end candidate."
            )
        else:
            lines.append("- All completed strict full-train candidates are now recorded non-winners under the baseline gate.")
            lines.append("- Next step: keep the Round 3 queue frozen, review the latest landed candidate, and design the next one-variable end-to-end formal candidate.")
        return lines

    running = [row for row in rows if row["status"] == "running"]
    if running:
        names = " / ".join(f"`{row['model']}`" for row in running)
        return [
            f"- No full-train metrics are synced yet; currently still waiting on {names}.",
            "- Next step: continue SSH monitoring and only analyze once full metrics files land.",
        ]

    failed = [row for row in rows if row["status"] == "failed"]
    if failed:
        names = " / ".join(f"`{row['model']}`" for row in failed)
        return [
            f"- The current strict full-train candidates are failed or incomplete: {names}.",
            "- Next step: sync the updated `dialogue_kt/training.py`, keep the full-train scope unchanged, and rerun on SSH only after the failure path is understood.",
        ]

    return ["- Round 3 strict full-train candidates are not yet fully launched."]


def build_completed_candidate_analysis(rows: list[dict]) -> list[str]:
    completed_rows = [row for row in rows if row["metrics"] is not None]
    if not completed_rows:
        return []

    lines = ["## Completed Candidate Analysis", ""]
    for row in completed_rows:
        metrics = row["metrics"]
        analysis = row.get("analysis") or {}
        lines.append(f"### `{row['label']}` / `{row['model']}`")
        lines.append("")
        lines.append(
            f"- Synced Overall Acc / AUC: **{metrics['overall_acc']:.2f} / {metrics['overall_auc']:.2f}**; "
            f"Final Acc / AUC: **{fmt_num(metrics['final_acc'])} / {fmt_num(metrics['final_auc'])}**."
        )
        if analysis.get("pred_true") is not None:
            lines.append(f"- `Pred True` at threshold `0.5`: **{analysis['pred_true']:.2f}%**.")
        if analysis.get("best_acc") is not None or analysis.get("best_threshold") is not None:
            lines.append(
                f"- Best-threshold diagnostic: best Acc **{fmt_num(analysis.get('best_acc'))}** at threshold **{fmt_num(analysis.get('best_threshold'), 3)}**."
            )
        if analysis.get("delta_acc_vs_baseline") is not None or analysis.get("delta_auc_vs_baseline") is not None:
            lines.append(
                f"- Delta vs baseline Overall Acc / AUC: **{analysis.get('delta_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_auc_vs_baseline', 0.0):+.2f}**."
            )
        if analysis.get("delta_final_acc_vs_baseline") is not None or analysis.get("delta_final_auc_vs_baseline") is not None:
            lines.append(
                f"- Delta vs baseline Final Acc / AUC: **{analysis.get('delta_final_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_final_auc_vs_baseline', 0.0):+.2f}**."
            )
        if analysis.get("delta_acc_vs_anchor") is not None or analysis.get("delta_auc_vs_anchor") is not None:
            lines.append(
                f"- Delta vs anchor `v1` Overall Acc / AUC: **{analysis.get('delta_acc_vs_anchor', 0.0):+.2f} / {analysis.get('delta_auc_vs_anchor', 0.0):+.2f}**."
            )
        lines.append("")
    return lines


def build_operator_commands(
    task_status: dict | None,
    rows: list[dict],
    sync_freshness: str | None,
    formal_audit: dict | None,
    baseline: dict | None,
    current_formal_state: dict,
) -> list[str]:
    if task_status is None:
        return []

    next_action = task_status.get("next_action")
    running = [row for row in rows if row["status"] == "running"]
    failed = [row for row in rows if row["status"] == "failed"]
    explicit_launch_key = current_explicit_launch_key(formal_audit, rows) or "v21"
    recorded_non_winner_key = current_recorded_non_winner_key(formal_audit, rows) or "v22"
    lines = ["## Operator Commands", ""]

    if next_action == "wait_round3" and running:
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
                "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
                "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {current_explicit_launch_key(formal_audit, rows) or 'v21'}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {current_explicit_launch_key(formal_audit, rows) or 'v21'}",
                "```",
            ]
        )
        if sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
            lines.extend(
                [
                    "",
                    "- Current reports are still based on an older synced snapshot; do not treat local progress movement as fresh SSH truth until refresh returns `sync=ok`.",
                ]
            )
        if failed:
            failed_names = " / ".join(f"`{row['model']}`" for row in failed)
            lines.extend(
                [
                    "",
                    f"- Keep the failed candidate(s) {failed_names} out of automatic rerun until the failure path is understood.",
                ]
            )
        return lines

    if next_action == "manual_decide":
        if failed:
            lines.extend(
                [
                    "```bash",
                    "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                    "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                    "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
                    "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
                    "bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
                    "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                    f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                    f"bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}",
                    "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                    f"bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}",
                    "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                    "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                    "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                    "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                    f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                    f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                    f"TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {recorded_non_winner_key}",
                    f"TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}",
                    "```",
                    "",
                    f"- Freeze any completed non-winner after rebuild, then rerun `{explicit_launch_key}` first with the synced `dialogue_kt/training.py` fix.",
                    "- If this environment cannot open SSH directly, use the manual fallback command above and run the same strict candidate from another reachable terminal.",
                ]
            )
            return lines
        if str(current_formal_state.get("current_formal_decision") or "") == "design_next_formal_candidate":
            lines.extend(
                [
                    "```bash",
                    "python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py",
                    "bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090",
                    f"bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {recorded_non_winner_key}",
                    f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}",
                    "python3 scripts/cel_stage1_last_layer/scaffold_formal_candidate.py --help",
                    "bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh",
                    "```",
                    "",
                    "- The current formal queue is complete and frozen as recorded non-winners; stop auto-closeout here and prepare the next one-variable candidate.",
                ]
            )
            return lines
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                "```",
                "",
                "- After rebuild, check `FORMAL_EXPERIMENT_AUDIT.md` first, then judge whether any completed strict full-train candidate is a formal winner.",
            ]
        )
        return lines

    if next_action == "launch_round3":
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
                "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                "```",
            ]
        )
        return lines

    if failed and not running:
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
                "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                f"bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                "```",
                "",
                "- Rerun only after confirming the failure fix is already synced and the scope remains strict full-train.",
            ]
        )
        return lines

    return []


def build_operator_loop(
    task_status: dict | None,
    rows: list[dict],
    sync_freshness: str | None,
    poll_interval: tuple[int, str] | None,
    formal_audit: dict | None,
    baseline: dict | None,
    current_formal_state: dict,
) -> list[str]:
    next_action = task_status.get("next_action") if task_status else None
    failed = [row for row in rows if row["status"] == "failed"]
    explicit_launch_key = current_explicit_launch_key(formal_audit, rows) or "v21"
    recorded_non_winner_key = current_recorded_non_winner_key(formal_audit, rows) or "v22"
    active_target_key = str(
        current_formal_state.get("current_active_target")
        or current_formal_state.get("suggested_finalize_key")
        or explicit_launch_key
    )
    lines = ["## Operator Loop", ""]
    lines.extend(
        [
            "1. Launch or rerun only through the SSH strict full-train path; local runs do not count as formal evidence.",
            "2. Monitor with the unified monitor entrypoint and adjust cadence based on live phase.",
            "3. After each landed SSH run, use the unified finalize entrypoint to sync, rebuild, and verify closeout readiness.",
            "4. Check `FORMAL_EXPERIMENT_AUDIT.md` before declaring a result recorded or successful.",
            "5. Review the candidate from the rebuilt authoritative surfaces before judging winner status.",
            "6. If the method still misses baseline on either metric, record the result and change only one main variable in the next formal design.",
        ]
    )
    lines.extend(["", "Recommended commands:", ""])

    if next_action == "wait_round3":
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
                "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
                "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {active_target_key}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {active_target_key}",
                "```",
            ]
        )
    elif next_action == "manual_decide" and failed:
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
                "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                f"bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {recorded_non_winner_key}",
                f"TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}",
                f"TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {recorded_non_winner_key}",
                f"TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}",
                "```",
            ]
        )
    elif next_action == "launch_round3":
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
                "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}",
                "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "```bash",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}",
                "```",
            ]
        )

    if poll_interval is not None:
        interval_secs, reason = poll_interval
        if interval_secs is None:
            lines.extend(["", f"- Current recommended local monitoring cadence: `not_applicable` ({reason})."])
        else:
            lines.extend(["", f"- Current recommended local monitoring cadence: `{interval_secs}s` ({reason})."])
    else:
        lines.extend(["", "- Default local monitoring cadence comes from the shared adaptive policy; near-finish or `Validation / Testing` tightens automatically."])

    if sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
        lines.append("- Current local reports are still snapshot-based; wait for `sync=ok` before treating any movement as fresh SSH truth.")

    return lines


def build_report(
    stage: str | None,
    sync_event: str | None,
    authoritative_sync_event: str | None,
    sync_freshness: str | None,
    baseline: dict | None,
    anchor: dict | None,
    task_status: dict | None,
    formal_audit: dict | None,
    rows: list[dict],
) -> str:
    next_action = task_status.get("next_action") if task_status else None
    winner_found = task_status.get("winner_found") if task_status else False
    poll_interval = recommended_poll_interval(rows, task_status)
    controller_runtime = controller_runtime_snapshot()
    controller_log_path = Path("results/cel_stage1_last_layer/task_conditioned_controller.log")
    controller_events = [
        line.strip()
        for line in read_log_text(controller_log_path).splitlines()
        if TIMESTAMPED_EVENT_RE.match(line.strip())
    ]
    controller_event = controller_events[-1] if controller_events else None
    controller_next_wake = None
    if controller_runtime["alive"] == "yes":
        controller_next_wake = format_monitor_timestamp(controller_next_cycle_estimate(controller_event))
    current_formal_state = build_current_formal_state(
        task_status,
        formal_audit,
        stage=stage or "unknown",
        sync_freshness=sync_freshness or "unknown",
        recommended_poll_interval=poll_interval_label(poll_interval),
    )

    lines = [
        "# Strict Full-Train Report",
        "",
        "> Terminology note: historical candidate identifiers are preserved for closeout and audit. Method documents refer to the audited successful design as the **A module**.",
        "",
        "- Scope: only SSH-side complete `train + val + test` candidates are eligible.",
        "- Excluded from winner judgment: calibrator-only, frozen retraining, fixed-bias eval, validation-fit bias, and any other post-hoc diagnostic flow.",
        "",
        "## Current State",
        "",
        f"- Stage: `{stage or 'unknown'}`",
        f"- Next action: `{next_action or 'unknown'}`",
        f"- Winner found: `{winner_found}`",
        f"- Current formal decision: `{current_formal_state['current_formal_decision']}`",
        f"- Current formal rerun target: `{current_formal_state['current_rerun_target'] or 'none'}`",
        f"- Current formal recorded non-winner: `{current_formal_state['current_recorded_non_winner_target'] or 'none'}`",
        f"- Current formal recorded non-winner queue: `{', '.join(current_formal_state['recorded_non_winner_queue']) if current_formal_state['recorded_non_winner_queue'] else 'none'}`",
        f"- Task-conditioned sync freshness: `{sync_freshness or 'unknown'}`",
        f"- Latest authoritative task-conditioned remote refresh event: `{authoritative_sync_event or 'none'}`",
        f"- Latest task-conditioned refresh attempt: `{sync_event or 'none'}`",
        f"- Task-conditioned controller pid/alive: `{controller_runtime['pid']} / {controller_runtime['alive']}`",
        f"- Task-conditioned controller command: `{controller_runtime['cmd']}`",
        f"- Task-conditioned controller child command: `{controller_runtime['child_cmd']}`",
        f"- Task-conditioned controller next wake estimate: `{controller_next_wake or 'none'}`",
    ]
    if sync_freshness == "fresh_remote":
        lines.append("- Current local report is backed by a fresh SSH sync.")
    elif sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
        lines.append("- Current local report was rebuilt from older synced artifacts; do not treat progress movement as new remote truth.")
    if poll_interval is not None:
        interval_secs, reason = poll_interval
        if interval_secs is None:
            lines.append(f"- Recommended local poll interval: `not_applicable` ({reason})")
        else:
            lines.append(f"- Recommended local poll interval: `{interval_secs}s` ({reason})")
            next_monitor_after = recommended_next_monitor_after(sync_event, poll_interval)
            next_monitor_after_text = format_monitor_timestamp(next_monitor_after)
            if next_monitor_after_text is not None and sync_freshness == "fresh_remote":
                lines.append(
                    f"- Suggested next local monitor after: `{next_monitor_after_text}` (latest authoritative refresh + {interval_secs}s)"
                )
    if formal_audit is not None:
        counts = formal_audit.get("counts") or {}
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        if summary:
            lines.append(f"- Formal experiment audit summary: `{summary}`")
    active_row = next((row for row in rows if row["status"] == "running"), None)
    if active_row is not None:
        milestone_note = stability_milestone_note(FORMAL_CANDIDATES.get(active_row["model"]), active_row.get("progress"), language="en")
        if milestone_note:
            lines.append(f"- {milestone_note}")
        epoch_note = epoch_cycle_note(FORMAL_CANDIDATES.get(active_row["model"]), active_row.get("progress"), language="en")
        if epoch_note:
            lines.append(f"- {epoch_note}")
        timing_note = progress_timing_note(active_row.get("progress"), language="en")
        if timing_note:
            lines.append(f"- {timing_note}")

    if baseline is not None:
        lines.extend(
            [
                f"- Baseline: `{baseline['model']}`",
                f"- Baseline Overall Acc / AUC: **{baseline['overall_acc']:.2f} / {baseline['overall_auc']:.2f}**",
                f"- Baseline Final Acc / AUC: **{fmt_num(baseline.get('final_acc'))} / {fmt_num(baseline.get('final_auc'))}**",
            ]
        )
    if anchor is not None:
        lines.extend(
            [
                f"- Historical selector/ranking comparison anchor: `{anchor['model']}`",
                f"- Comparison-anchor Overall Acc / AUC: **{anchor['overall_acc']:.2f} / {anchor['overall_auc']:.2f}**",
            ]
        )

    lines.extend(["", "## Formal Round 3 Candidates", ""])
    lines.extend(
        [
            "| Candidate | Start Point | Method | Status | Overall Acc | Delta Overall Acc | Overall AUC | Delta Overall AUC | Final Acc | Delta Final Acc | Final AUC | Delta Final AUC | Overall Gate | Final Pair vs Baseline |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for row in rows:
        metrics = row["metrics"]
        delta_acc = "--"
        delta_auc = "--"
        delta_final_acc = "--"
        delta_final_auc = "--"
        if metrics is not None and baseline is not None:
            delta_acc = f"{metrics['overall_acc'] - baseline['overall_acc']:+.2f}"
            delta_auc = f"{metrics['overall_auc'] - baseline['overall_auc']:+.2f}"
            if metrics.get("final_acc") is not None and baseline.get("final_acc") is not None:
                delta_final_acc = f"{metrics['final_acc'] - baseline['final_acc']:+.2f}"
            if metrics.get("final_auc") is not None and baseline.get("final_auc") is not None:
                delta_final_auc = f"{metrics['final_auc'] - baseline['final_auc']:+.2f}"
        final_acc = fmt_num(metrics["final_acc"]) if metrics is not None else "--"
        final_auc = fmt_num(metrics["final_auc"]) if metrics is not None else "--"
        lines.append(
            f"| `{row['label']}` / `{row['model']}` | {row['start_point']} | {row['method']} | {status_text(row)} | "
            f"{fmt_num(metrics['overall_acc']) if metrics is not None else '--'} | {delta_acc} | "
            f"{fmt_num(metrics['overall_auc']) if metrics is not None else '--'} | {delta_auc} | {final_acc} | {delta_final_acc} | {final_auc} | {delta_final_auc} | {beat_flag(metrics, baseline)} | {final_pair_flag(metrics, baseline)} |"
        )

    lines.extend(["", "## Decision", ""])
    lines.extend(build_current_decision(rows, baseline, anchor, current_formal_state))
    completed_candidate_analysis = build_completed_candidate_analysis(rows)
    if completed_candidate_analysis:
        lines.extend([""])
        lines.extend(completed_candidate_analysis)
    operator_commands = build_operator_commands(
        task_status,
        rows,
        sync_freshness,
        formal_audit,
        baseline,
        current_formal_state,
    )
    if operator_commands:
        lines.extend([""])
        lines.extend(operator_commands)
    operator_loop = build_operator_loop(
        task_status,
        rows,
        sync_freshness,
        poll_interval,
        formal_audit,
        baseline,
        current_formal_state,
    )
    if operator_loop:
        lines.extend([""])
        lines.extend(operator_loop)
    failure_focus = build_failure_focus(rows)
    if failure_focus:
        lines.extend([""])
        lines.extend(failure_focus)
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "- A new method only counts as successful when one strict full-train candidate finishes on SSH and achieves both `Overall Acc > baseline` and `Overall AUC > baseline`.",
            "- Every completed candidate must also report Final Acc / AUC and their two deltas versus baseline; Final-turn comparison is a required diagnostic and audit surface, not a replacement for the Overall winner gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--anchor_metrics", default="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--refresh_log", default="results/cel_stage1_last_layer/task_conditioned_refresh.log")
    parser.add_argument("--formal_audit_json", default="results/cel_stage1_last_layer/formal_experiment_audit.json")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md")
    args = parser.parse_args()

    task_status = read_task_status(Path(args.task_status))
    stage = infer_stage(task_status)
    sync_event = latest_refresh_event(Path(args.refresh_log))
    authoritative_sync_event = latest_successful_refresh_event(Path(args.refresh_log)) or sync_event
    sync_freshness = sync_freshness_label(sync_event)
    baseline = parse_metrics(Path(args.baseline_metrics))
    anchor = parse_metrics(Path(args.anchor_metrics))
    formal_audit = read_formal_audit(Path(args.formal_audit_json))
    rows = build_candidate_rows(task_status, Path(args.metrics_dir), baseline, anchor)
    Path(args.out).write_text(
        build_report(stage, sync_event, authoritative_sync_event, sync_freshness, baseline, anchor, task_status, formal_audit, rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
