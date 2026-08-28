#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from current_formal_state import build_current_formal_state
from formal_candidate_registry import FORMAL_CANDIDATES, build_formal_log_path_map
from task_conditioned_failure_utils import (
    controller_next_cycle_estimate,
    epoch_cycle_note,
    format_monitor_timestamp,
    latest_phase_progress_from_log as parse_phase_progress_from_log,
    progress_timing_note,
    recommend_round3_poll_interval,
    recommended_next_monitor_after,
    stability_milestone_note,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "results" / "baseline"
CEL_DIR = ROOT / "results" / "cel_stage1_last_layer"
STATUS_PATH = CEL_DIR / "STATUS.md"
TASK_STATUS_PATH = CEL_DIR / "task_conditioned_status.json"
STEP_LOG_DIR = CEL_DIR / "step_logs"
LOOP_EVENT_PATH = CEL_DIR / "loop_stdout.log"
LOOP_PROGRESS_PATH = CEL_DIR / "loop.log"
FOLLOWUP_EVENT_PATH = CEL_DIR / "followup.log"
FOLLOWUP_PROGRESS_PATH = CEL_DIR / "followup_stdout.log"
WATCH_LOG_PATH = CEL_DIR / "task_conditioned_watch.log"
REFRESH_LOG_PATH = CEL_DIR / "task_conditioned_refresh.log"
FORMAL_AUDIT_JSON_PATH = CEL_DIR / "formal_experiment_audit.json"
CONTROLLER_PID_PATH = CEL_DIR / "task_conditioned_controller.pid"
ROUND3_LOG_FILES = build_formal_log_path_map(ROOT)

BASELINE_MODEL = "lmkt_qwen3_1.7b_recert_20260620"
ROUND1_MODELS = [
    "cel_mlp_lastlayer_v1_qwen3_1.7b",
    "cel_adapter_lastlayer_v1_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
]
ROUND2_MODELS = [
    "cel_adapter_lastlayer_v2_qwen3_1.7b",
]
FOLLOWUP_SUCCESS_MARKERS = [
    "existing CEL result already beats baseline; no followup runs needed",
    "selector-only round beats baseline; followup finished",
    "early prediction-token batch1 beats baseline; followup finished",
    "early task-conditioned prediction-token batch beats baseline; followup finished",
    "early vector-predshift batch1 beats baseline; followup finished",
    "early task-conditioned vector-predshift batch beats baseline; followup finished",
    "early pre-lm-head vector-predshift batch1 beats baseline; followup finished",
    "early task-conditioned pre-lm-head vector-predshift batch beats baseline; followup finished",
    "direct pre-lm-head selector-only batch1 beats baseline; followup finished",
    "direct task-conditioned pre-lm-head selector-only batch beats baseline; followup finished",
    "scalar-gate followup beats baseline; followup finished",
    "vector-shift batch1 beats baseline; followup finished",
    "vector-shift batch2 beats baseline; followup finished",
    "conservative scalar batch beats baseline; followup finished",
    "conservative low-lr phase3 beats baseline; followup finished",
    "phase4 stabilization batch beats baseline; followup finished",
    "phase5 warm-start batch1 beats baseline; followup finished",
    "phase5 warm-start batch2 beats baseline; followup finished",
    "phase6 prediction-token batch1 beats baseline; followup finished",
    "phase6 task-conditioned prediction-token batch beats baseline; followup finished",
    "task-conditioned vector-shift followup beats baseline; followup finished",
    "ultralow-gamma vector-shift followup beats baseline; followup finished",
    "followup loop found a CEL model that beats baseline",
]
FOLLOWUP_FAILURE_MARKERS = [
    "followup loop completed; no CEL model beat baseline yet",
    "baseline metrics missing; exiting followup loop",
]
DIAGNOSTIC_ONLY_MODELS = {
    "cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v19_fullv1_cal_bias_only_tinylr_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v20_fullv1_valfit_bias_qwen3_1.7b",
}
PROGRESS_RE = re.compile(r"(Training|Validation|Validating|Testing):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
EVENT_RE = re.compile(r"^\[[0-9:\- ]+\]\s+(START|END|SKIP|FAIL|ABORT)\s+(\S+)")
TASK_CONTROLLER_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+(.*)$")
TIMESTAMPED_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+(.*)$")


def extract_metric_block(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    overall = re.search(r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)", text, re.S)
    final_turn = re.search(r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)", text, re.S)
    loss = re.search(r"Loss:\s+([0-9.]+)", text)
    diag = re.search(r"CEL Diagnostics:\s+(.*)", text)
    return {
        "path": path,
        "loss": float(loss.group(1)) if loss else None,
        "overall_auc": float(overall.group(2)) if overall else None,
        "overall_acc": float(overall.group(1)) if overall else None,
        "final_acc": float(final_turn.group(1)) if final_turn else None,
        "final_auc": float(final_turn.group(2)) if final_turn else None,
        "diag": diag.group(1).strip() if diag else None,
    }


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")


def read_task_status() -> dict | None:
    if not TASK_STATUS_PATH.exists():
        return None
    return json.loads(TASK_STATUS_PATH.read_text(encoding="utf-8"))


def read_formal_audit() -> dict | None:
    if not FORMAL_AUDIT_JSON_PATH.exists():
        return None
    return json.loads(FORMAL_AUDIT_JSON_PATH.read_text(encoding="utf-8"))


def latest_controller_event() -> str | None:
    controller_log = CEL_DIR / "task_conditioned_controller.log"
    events = [
        line.strip()
        for line in read_log_text(controller_log).splitlines()
        if TASK_CONTROLLER_EVENT_RE.match(line.strip())
    ]
    return events[-1] if events else None


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


def normalize_controller_event(controller_event: str | None, task_status: dict | None) -> str | None:
    if controller_event is None or task_status is None:
        return controller_event
    if not task_status.get("winner_found") and "winner found:" in controller_event:
        return f"{controller_event} [stale historical event under old criterion]"
    return controller_event


def latest_loop_event() -> str | None:
    for path in (LOOP_EVENT_PATH, LOOP_PROGRESS_PATH):
        events = [line.strip() for line in read_log_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
        if events:
            return events[-1]
    return None


def latest_followup_event() -> str | None:
    for path in (FOLLOWUP_EVENT_PATH, FOLLOWUP_PROGRESS_PATH):
        events = [line.strip() for line in read_log_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
        if events:
            return events[-1]
    return None


def latest_watch_event() -> str | None:
    events = [line.strip() for line in read_log_text(WATCH_LOG_PATH).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    sync_complete_events = [line for line in events if "sync complete" in line]
    return sync_complete_events[-1] if sync_complete_events else events[-1]


def latest_refresh_event() -> str | None:
    events = [line.strip() for line in read_log_text(REFRESH_LOG_PATH).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_complete_events = [line for line in events if "task_conditioned refresh complete" in line]
    return refresh_complete_events[-1] if refresh_complete_events else events[-1]


def latest_successful_refresh_event() -> str | None:
    events = [line.strip() for line in read_log_text(REFRESH_LOG_PATH).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_success_events = [
        line
        for line in events
        if "task_conditioned refresh complete" in line and "sync=ok" in line
    ]
    return refresh_success_events[-1] if refresh_success_events else None


def event_timestamp(event: str | None) -> datetime | None:
    if event is None:
        return None
    match = TIMESTAMPED_EVENT_RE.match(event)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def latest_task_sync_event(watch_event: str | None, refresh_event: str | None) -> str | None:
    if watch_event is None:
        return refresh_event
    if refresh_event is None:
        return watch_event
    watch_ts = event_timestamp(watch_event)
    refresh_ts = event_timestamp(refresh_event)
    if watch_ts is None:
        return refresh_event
    if refresh_ts is None:
        return watch_event
    return refresh_event if refresh_ts >= watch_ts else watch_event


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


def active_steps_from_log(path: Path) -> list[str]:
    active: list[str] = []
    for raw_line in read_log_text(path).splitlines():
        match = EVENT_RE.match(raw_line.strip())
        if not match:
            continue
        action, step = match.groups()
        if action == "START":
            if step not in active:
                active.append(step)
        else:
            active = [item for item in active if item != step]
    return active


def latest_phase_progress_from_log(path: Path) -> dict | None:
    return parse_phase_progress_from_log(path)


def recommended_poll_interval(round3_rows: list[dict], cel_progress: dict | None, task_status: dict | None) -> tuple[int | None, str] | None:
    if task_status is None:
        return None
    next_action = task_status.get("next_action")
    if next_action in {"manual_decide", "done"}:
        return None, "not applicable while no active strict full-train monitoring is required"
    if not round3_rows:
        return None
    if cel_progress is not None:
        interval = recommend_round3_poll_interval(cel_progress, next_action)
        if interval is not None and interval[1] != "default training cadence":
            return interval
    for row in round3_rows:
        interval = recommend_round3_poll_interval(row.get("progress"), next_action)
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


def current_formal_workflow(task_status: dict | None, current_state: dict) -> list[str]:
    next_action = task_status.get("next_action") if task_status else None
    current_formal_decision = current_state["current_formal_decision"]
    explicit_launch_key = (
        current_state.get("suggested_launch_key")
        or current_state.get("current_completed_target")
        or current_state.get("current_active_target")
        or "v21"
    )
    recorded_non_winner_key = current_state.get("current_recorded_non_winner_target") or "v22"
    lines = ["## Current Formal Workflow"]

    if current_formal_decision == "winner_or_done":
        winner = (task_status or {}).get("winner") or {}
        winner_model = winner.get("model") or "cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b"
        winner_meta = FORMAL_CANDIDATES.get(winner_model) or {}
        winner_key = winner_meta.get("launch_key") or "v26"
        lines.extend(
            [
                f"- Current priority: preserve the A-module reference from historical checkpoint `{winner_model}`; do not launch another Stage 1 candidate.",
                "- Current local state helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
                f"- Verified local-snapshot review in this WSL checkout: `TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {winner_key}`",
                "- Stage 2 first-round legacy single-path experiments are complete and audited, but remain historical pilots; the next work is the explicit `h_r/h_m` dual-path implementation, tensor contracts, and audit path.",
                "- Copied-workspace note: the active Stage 1 orchestration resolves the local repository root from each script location; SSH target and remote repository settings remain unchanged.",
            ]
        )
        return lines

    if current_formal_decision == "freeze_recorded_then_rerun":
        lines.extend(
            [
                "- Current priority: freeze completed non-winners, then rerun the failed strict full-train candidate first.",
                "- Current local helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
                "- Recommended alias bootstrap: `bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`",
                "- Start current queued candidate: `bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`",
                f"- Explicit start for current rerun target: `bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}`",
                f"- Direct start: `bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}`",
                "- Manual SSH fallback for current queued candidate: `bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
                f"- Explicit manual SSH fallback: `bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}`",
                "- Monitor via alias: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
                "- Monitor: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
                "- Finalize current rerun target after it lands via alias: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
                "- Finalize current rerun target after it lands: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
                "- Review current rerun target after it lands via alias: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
                "- Review current rerun target after it lands: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
                f"- Finalize recorded non-winner via alias snapshot: `TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}`",
                f"- Finalize recorded non-winner: `TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {recorded_non_winner_key}`",
                f"- Review failed candidate via alias: `bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}`",
                f"- Review failed candidate: `bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {explicit_launch_key}`",
                f"- Review recorded non-winner from alias local snapshot: `TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}`",
                f"- Review recorded non-winner from local snapshot: `TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {recorded_non_winner_key}`",
                "- Local snapshot only: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`",
            ]
        )
        return lines

    if current_formal_decision == "monitor_active_full_train" or next_action == "wait_round3":
        lines.extend(
            [
                "- Current priority: keep monitoring the active strict full-train candidate(s) and refresh reports after each landed sync.",
                "- Runtime health snapshot: `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`",
                "- One-shot current-state alias action: `bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`",
                "- One-shot current-state alias action dry-run: `bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`",
                "- Monitor once: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once`",
                "- Monitor once via alias: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once`",
                "- Monitor loop: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
                "- Monitor loop via alias: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
                "- Background full-loop controller: `bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090`",
                "- Finalize current active candidate after completion: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
                "- Finalize current active candidate after completion via alias: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
                "- Review current active candidate: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
                "- Review current active candidate via alias: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
            ]
        )
        return lines

    if current_formal_decision == "launch_next_candidate":
        lines.extend(
            [
                "- Current priority: launch the next strict full-train candidate on SSH after preflight passes.",
                "- Current local helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
                "- Recommended alias bootstrap: `bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`",
                "- Start current queued candidate: `bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`",
                f"- Explicit candidate start: `bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}`",
                f"- Direct start: `bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}`",
                "- Manual SSH fallback for current queued candidate: `bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
                f"- Explicit manual SSH fallback: `bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}`",
            ]
        )
        return lines

    if current_formal_decision == "design_next_formal_candidate":
        lines.extend(
            [
                "- Current priority: the completed formal queue is already frozen as recorded non-winners; stop auto-closeout and design the next one-variable strict full-train candidate.",
                "- Current local helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
                "- Runtime health snapshot: `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`",
                f"- Review the latest recorded non-winner explicitly: `bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {recorded_non_winner_key}`",
                f"- Review the latest recorded non-winner via alias: `bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {recorded_non_winner_key}`",
                "- Scaffold the next candidate draft: `python3 scripts/cel_stage1_last_layer/scaffold_formal_candidate.py --help`",
                "- Re-run guarded checks before launch: `bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh`",
            ]
        )
        return lines

    lines.extend(
        [
            "- Current priority: use the strict start/monitor/finalize/review workflow instead of piecing together low-level commands manually.",
            "- Current local helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
            "- Recommended alias bootstrap: `bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`",
            "- Start current queued candidate: `bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`",
            f"- Explicit candidate start: `bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_launch_key}`",
            f"- Direct start: `bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_launch_key}`",
            "- Manual SSH fallback for current queued candidate: `bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
            f"- Explicit manual SSH fallback: `bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_launch_key}`",
            "- Monitor via alias: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
            "- Monitor: `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
            "- Finalize current candidate via alias: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
            "- Finalize current candidate: `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
            "- Review current candidate via alias: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
            "- Review current candidate: `bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
        ]
    )
    return lines


def formal_loop_summary(task_status: dict | None, formal_audit: dict | None, poll_interval: tuple[int, str] | None) -> list[str]:
    next_action = task_status.get("next_action") if task_status else None
    audit_rows = (formal_audit or {}).get("rows") or []
    has_rerun = any(row.get("audit") == "needs_rerun" for row in audit_rows)
    recorded_non_winners = [
        row for row in audit_rows
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]
    recorded_winners = [
        row for row in audit_rows
        if row.get("audit") == "recorded" and row.get("beats_baseline") == "yes"
    ]
    lines = ["## Formal Loop Summary"]
    lines.extend(
        [
            "- Formal method results only count after one SSH-side complete `train + val + test` run.",
            "- Any calibrator-only, frozen retraining, fixed-bias eval, validation-fit bias, or other post-hoc ckpt adjustment stays diagnostic-only.",
            "- Every completed formal SSH run must also pass the `finalize -> audit -> review` closeout path before it is treated as recorded and analyzed.",
            "- If the current environment cannot open SSH itself, print the strict manual launch fallback and run the same registered candidate from another reachable terminal.",
        ]
    )

    if recorded_winners:
        lines.append("- A recorded strict formal winner already exists in the audit-ready queue; keep all conclusion surfaces frozen, stop launching new formal candidates, and only reopen the loop for an explicit strict rerun.")
    elif has_rerun:
        lines.append("- Because a strict formal candidate still has `needs_rerun`, do not design a new method yet; freeze recorded non-winners first and rerun the failed candidate under the same full-train scope.")
    elif next_action == "manual_decide" and recorded_non_winners:
        lines.append("- Current formal queue has completed recorded non-winners but no remaining rerun obligation; only now is it valid to design the next one-variable strict formal candidate.")
    elif next_action == "wait_round3":
        lines.append("- Current loop state is still mid-run; do not judge winner status until the SSH run lands full metrics and the audit is rebuilt.")
    elif next_action == "launch_round3":
        lines.append("- Current loop state is pre-launch; finish preflight, then launch the next strict formal candidate on SSH rather than running local-only experiments.")

    if poll_interval is not None:
        interval_secs, reason = poll_interval
        if interval_secs is None:
            lines.append(f"- Current recommended local monitoring cadence: `not_applicable` ({reason}). Early long training can stay slower, stable mid-run stays moderate, and near-finish or `Validation / Testing` tightens automatically.")
        else:
            lines.append(f"- Current recommended local monitoring cadence: `{interval_secs}s` ({reason}). Early long training can stay slower, stable mid-run stays moderate, and near-finish or `Validation / Testing` tightens automatically.")
    else:
        lines.append("- Default local monitoring cadence comes from the shared adaptive policy; near-finish or `Validation / Testing` tightens automatically.")

    lines.extend(
        [
            "- After each completed SSH run, process results in this order: sync `metrics / qual / stdout log` -> rebuild `task_conditioned_status.json` and markdown surfaces -> check `FORMAL_EXPERIMENT_AUDIT.md` -> only then judge baseline win/loss.",
            "- If the candidate still fails to beat baseline on both Overall Acc and Overall AUC, record the result and change only one main design variable for the next formal candidate.",
        ]
    )
    return lines


def step_progress(step_name: str) -> dict | None:
    return latest_phase_progress_from_log(STEP_LOG_DIR / f"{step_name}.log")


def model_role(model_name: str) -> str:
    return "diagnostic" if model_name in DIAGNOSTIC_ONLY_MODELS else "formal"


def classify_followup_event(followup_event: str | None) -> str:
    if not followup_event:
        return "none"
    if "waiting for primary loop to finish" in followup_event:
        return "waiting"
    if any(marker in followup_event for marker in FOLLOWUP_SUCCESS_MARKERS):
        return "success"
    if any(marker in followup_event for marker in FOLLOWUP_FAILURE_MARKERS):
        return "failure"
    return "running"


def round_models(task_status: dict | None, round_prefix: str) -> list[dict]:
    if task_status is None:
        return []
    for round_info in task_status.get("rounds") or []:
        title = round_info.get("title") or ""
        if title.startswith(round_prefix):
            return list(round_info.get("models") or [])
    return []


def round3_status_rows(task_status: dict | None, cel_metrics: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for model_entry in round_models(task_status, "Round 3"):
        model_name = model_entry.get("model")
        if not model_name:
            continue
        log_path = ROUND3_LOG_FILES.get(model_name)
        rows.append(
            {
                "model": model_name,
                "status": model_entry.get("status", "unknown"),
                "failure_reason": model_entry.get("failure_reason"),
                "progress": latest_phase_progress_from_log(log_path) if log_path is not None else None,
                "metrics": cel_metrics.get(model_name),
            }
        )
    return rows


def infer_stage(
    baseline_metrics: dict | None,
    cel_metrics: dict[str, dict],
    last_event: str | None,
    followup_state: str,
    baseline_progress: dict | None,
) -> str:
    if followup_state == "waiting":
        if baseline_metrics is None:
            if baseline_progress and baseline_progress["phase"] == "testing":
                return "baseline_testing + followup_waiting"
            if baseline_progress and baseline_progress["phase"] == "validating":
                return "baseline_validating + followup_waiting"
            return "baseline_running + followup_waiting"
        return "stage1_last_layer_primary_running + followup_waiting"
    if followup_state == "running":
        return "stage1_last_layer_followup_running"
    if followup_state == "success":
        return "stage1_last_layer_completed_with_followup_win"
    if followup_state == "failure":
        return "stage1_last_layer_followup_completed_no_win"
    if last_event and "experiment loop finished" in last_event:
        return "stage1_last_layer_primary_finished"
    if baseline_metrics is None:
        if baseline_progress and baseline_progress["phase"] == "testing":
            return "baseline_testing"
        if baseline_progress and baseline_progress["phase"] == "validating":
            return "baseline_validating"
        return "baseline_running"
    round1_ready = sum(1 for model in ROUND1_MODELS if model in cel_metrics)
    round2_ready = sum(1 for model in ROUND2_MODELS if model in cel_metrics)
    if round1_ready < len(ROUND1_MODELS):
        return f"stage1_last_layer_round1_running ({round1_ready}/{len(ROUND1_MODELS)})"
    if round2_ready == 0:
        return "stage1_last_layer_awaiting_round1_decision_or_adapter_v2"
    return f"stage1_last_layer_round2_running_or_finished ({round2_ready}/{len(ROUND2_MODELS)})"


def current_cel_progress(last_event: str | None, followup_state: str) -> dict | None:
    if followup_state == "running":
        return latest_phase_progress_from_log(FOLLOWUP_PROGRESS_PATH) or latest_phase_progress_from_log(FOLLOWUP_EVENT_PATH)
    if last_event and re.search(r"START cel_", last_event):
        return latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    return None


def first_active_step_progress(active_steps: list[str], fallback_path: Path | None = None) -> dict | None:
    for step_name in active_steps:
        progress = step_progress(step_name)
        if progress is not None:
            return progress
    if fallback_path is not None:
        return latest_phase_progress_from_log(fallback_path)
    return None


def build_table(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return "_No metrics synced yet._"
    out = [
        "| Model | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Diagnostics |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, metrics in rows:
        loss = f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "--"
        overall_acc = f"{metrics['overall_acc']:.2f}" if metrics["overall_acc"] is not None else "--"
        overall_auc = f"{metrics['overall_auc']:.2f}" if metrics["overall_auc"] is not None else "--"
        final_acc = f"{metrics['final_acc']:.2f}" if metrics["final_acc"] is not None else "--"
        final_auc = f"{metrics['final_auc']:.2f}" if metrics["final_auc"] is not None else "--"
        diag = metrics["diag"] or "--"
        out.append(
            f"| `{model}` | {model_role(model)} | {loss} | {overall_acc} | {overall_auc} | {final_acc} | {final_auc} | {diag} |"
        )
    return "\n".join(out)


def main() -> None:
    baseline_metrics = extract_metric_block(BASELINE_DIR / "metrics" / f"metrics_{BASELINE_MODEL}.txt")
    cel_metrics = {}
    for metrics_file in sorted((CEL_DIR / "metrics").glob("metrics_*.txt")):
        model = metrics_file.stem.replace("metrics_", "", 1)
        metrics = extract_metric_block(metrics_file)
        if metrics is not None:
            cel_metrics[model] = metrics

    last_event = latest_loop_event()
    followup_event = latest_followup_event()
    watch_event = latest_watch_event()
    refresh_event = latest_refresh_event()
    authoritative_refresh_event = latest_successful_refresh_event() or refresh_event
    sync_event = latest_task_sync_event(watch_event, refresh_event)
    controller_event = latest_controller_event()
    controller_runtime = controller_runtime_snapshot()
    task_status = read_task_status()
    formal_audit = read_formal_audit()
    controller_event = normalize_controller_event(controller_event, task_status)
    controller_next_wake = None
    if controller_runtime["alive"] == "yes":
        controller_next_wake = format_monitor_timestamp(controller_next_cycle_estimate(controller_event))
    baseline_progress = latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    followup_state = classify_followup_event(followup_event)
    active_loop_steps = active_steps_from_log(LOOP_EVENT_PATH)
    active_followup_steps = active_steps_from_log(FOLLOWUP_EVENT_PATH)
    cel_progress = current_cel_progress(last_event, followup_state)
    if cel_progress is None and (active_followup_steps or active_loop_steps):
        fallback_path = FOLLOWUP_PROGRESS_PATH if active_followup_steps else LOOP_PROGRESS_PATH
        cel_progress = first_active_step_progress(active_followup_steps + active_loop_steps, fallback_path=fallback_path)
    stage = infer_stage(baseline_metrics, cel_metrics, last_event, followup_state, baseline_progress)
    round3_rows: list[dict] = []

    if task_status is not None:
        next_action = task_status.get("next_action")
        winner_found = task_status.get("winner_found")
        rounds = task_status.get("rounds") or []
        round1 = rounds[0] if len(rounds) > 0 else {}
        round2 = rounds[1] if len(rounds) > 1 else {}
        round2_models = round2.get("models") or []
        round2_has_started = any(model.get("status") in {"running", "done", "failed"} for model in round2_models)
        round2_has_running = any(model.get("status") == "running" for model in round2_models)
        if winner_found:
            stage = "task_conditioned_tuning_completed_with_winner"
        elif next_action == "launch_round2":
            stage = "task_conditioned_legacy_diagnostic_state"
        elif next_action == "launch_round3":
            stage = "task_conditioned_round3_pending_launch"
        elif next_action == "wait_round1":
            if round1.get("all_done"):
                stage = "task_conditioned_legacy_diagnostic_state"
            elif round2_has_running or round2_has_started:
                stage = "task_conditioned_legacy_diagnostic_state"
            else:
                stage = "task_conditioned_legacy_diagnostic_state"
        elif next_action == "wait_round2":
            stage = "task_conditioned_legacy_diagnostic_state"
        elif next_action == "wait_round3":
            stage = "task_conditioned_round3_running"
        elif next_action == "manual_decide":
            stage = "task_conditioned_round3_done_waiting_decision"
        elif next_action == "done":
            stage = "task_conditioned_tuning_done"
        round3_rows = round3_status_rows(task_status, cel_metrics)
        if cel_progress is None:
            for row in round3_rows:
                if row["status"] == "running" and row["progress"] is not None:
                    cel_progress = row["progress"]
                    break
    poll_interval = recommended_poll_interval(round3_rows, cel_progress, task_status)

    display_loop_event = last_event
    display_followup_event = followup_event
    if task_status is not None:
        display_loop_event = None
        display_followup_event = None

    formal_best = None
    diagnostic_best = None
    if cel_metrics:
        formal_best = max(
            (
                (model, metrics)
                for model, metrics in cel_metrics.items()
                if metrics["overall_auc"] is not None and model_role(model) == "formal"
            ),
            key=lambda item: (item[1]["overall_auc"], item[1]["overall_acc"]),
            default=None,
        )
        diagnostic_best = max(
            (
                (model, metrics)
                for model, metrics in cel_metrics.items()
                if metrics["overall_auc"] is not None and model_role(model) == "diagnostic"
            ),
            key=lambda item: (item[1]["overall_auc"], item[1]["overall_acc"]),
            default=None,
        )

    lines = [
        "# STATUS",
        "",
        "> Terminology note: model and candidate identifiers below retain their historical names for provenance. The audited current method is referred to as the **A module** in method documents. Stage 2 first-round records use a legacy single-path environment-residual flow and are historical pilots, not the current `h_r/h_m` dual-path target method.",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Stage: `{stage}`",
        f"- Latest loop event: `{display_loop_event or 'none'}`",
        f"- Latest follow-up event: `{display_followup_event or 'none'}`",
        f"- Task-conditioned sync freshness: `{sync_freshness_label(refresh_event)}`",
        f"- Latest authoritative task-conditioned remote refresh event: `{authoritative_refresh_event or 'none'}`",
        f"- Latest task-conditioned sync event: `{sync_event or 'none'}`",
        f"- Latest task-conditioned refresh attempt: `{refresh_event or 'none'}`",
        f"- Latest task-conditioned watcher event: `{watch_event or 'none'}`",
        f"- Latest task-conditioned controller event: `{controller_event or 'none'}`",
        f"- Task-conditioned controller pid/alive: `{controller_runtime['pid']} / {controller_runtime['alive']}`",
        f"- Task-conditioned controller command: `{controller_runtime['cmd']}`",
        f"- Task-conditioned controller child command: `{controller_runtime['child_cmd']}`",
        f"- Task-conditioned controller next wake estimate: `{controller_next_wake or 'none'}`",
        "- Snapshot provenance: `fresh_remote` means the latest recorded refresh succeeded; it does not prove that the copied checkout has revalidated the remote state on the report-generation date.",
        "- Authoritative task-conditioned snapshot should be read from the latest successful remote refresh above; watcher/controller events are process-log breadcrumbs and may lag behind that recorded snapshot.",
    ]
    if task_status is not None:
        lines.append(f"- Task-conditioned next action: `{task_status.get('next_action')}`")
        lines.append(f"- Task-conditioned winner found: `{task_status.get('winner_found')}`")
    if formal_audit is not None:
        counts = formal_audit.get("counts") or {}
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        if summary:
            lines.append(f"- Formal experiment audit summary: `{summary}`")
    if poll_interval is not None:
        interval_secs, reason = poll_interval
        if interval_secs is None:
            lines.append(f"- Recommended task-conditioned poll interval: `not_applicable` ({reason})")
        else:
            lines.append(f"- Recommended task-conditioned poll interval: `{interval_secs}s` ({reason})")
            next_monitor_after = recommended_next_monitor_after(refresh_event, poll_interval)
            next_monitor_after_text = format_monitor_timestamp(next_monitor_after)
            if next_monitor_after_text is not None and sync_freshness_label(refresh_event) == "fresh_remote":
                lines.append(
                    f"- Suggested next task-conditioned monitor after: `{next_monitor_after_text}` (latest authoritative refresh + {interval_secs}s)"
                )
    current_formal_state = build_current_formal_state(
        task_status,
        formal_audit,
        stage=stage or "unknown",
        sync_freshness=sync_freshness_label(refresh_event),
        recommended_poll_interval=poll_interval_label(poll_interval),
    )
    active_model = None
    active_progress = None
    if task_status is not None:
        lines.append(f"- Current formal decision: `{current_formal_state['current_formal_decision']}`")
        lines.append(
            f"- Current formal rerun target: `{current_formal_state['current_rerun_target'] or 'none'}`"
        )
        lines.append(
            f"- Current formal recorded non-winner: `{current_formal_state['current_recorded_non_winner_target'] or 'none'}`"
        )
        lines.append(
            f"- Current formal recorded non-winner queue: `{', '.join(current_formal_state['recorded_non_winner_queue']) if current_formal_state['recorded_non_winner_queue'] else 'none'}`"
        )
        active_model = current_formal_state.get("current_active_model")
        if active_model:
            for row in round3_rows:
                if row.get("model") == active_model:
                    active_progress = row.get("progress")
                    break
            milestone_note = stability_milestone_note(FORMAL_CANDIDATES.get(active_model), active_progress, language="cn")
            if milestone_note:
                lines.append(f"- Formal stability milestone: {milestone_note}")
            epoch_note = epoch_cycle_note(FORMAL_CANDIDATES.get(active_model), active_progress, language="cn")
            if epoch_note:
                lines.append(f"- Formal epoch-cycle note: {epoch_note}")
    if active_loop_steps:
        lines.append(f"- Active loop steps: `{', '.join(active_loop_steps)}`")
    if active_followup_steps:
        lines.append(f"- Active follow-up steps: `{', '.join(active_followup_steps)}`")
    active_step_progress = []
    loop_stdout_progress = latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    followup_stdout_progress = latest_phase_progress_from_log(FOLLOWUP_PROGRESS_PATH) or latest_phase_progress_from_log(FOLLOWUP_EVENT_PATH)
    for step_name in active_loop_steps + active_followup_steps:
        progress = step_progress(step_name)
        if progress is None:
            if step_name in active_loop_steps and step_name == active_loop_steps[-1]:
                progress = loop_stdout_progress
            elif step_name in active_followup_steps and step_name == active_followup_steps[-1]:
                progress = followup_stdout_progress
        if progress is None:
            continue
        label = step_name
        phase = progress["phase"]
        epoch_text = f", epoch {progress['epoch']}" if progress["epoch"] is not None else ""
        active_step_progress.append(
            f"- `{label}`: `{phase}`{epoch_text}, `{progress['progress']}`"
        )
    lines.extend(["", "## Baseline"])
    if baseline_metrics is None:
        lines.append("- Baseline metrics: pending")
        if baseline_progress:
            lines.append(f"- Current phase: `{baseline_progress['phase']}`")
            if baseline_progress["epoch"] is not None:
                lines.append(f"- Current epoch: `{baseline_progress['epoch']}`")
            lines.append(f"- Latest {baseline_progress['phase']} progress: `{baseline_progress['progress']}`")
    else:
        lines.extend([
            f"- Model: `{BASELINE_MODEL}`",
            f"- Loss: `{baseline_metrics['loss']:.4f}`",
            f"- Overall Acc: `{baseline_metrics['overall_acc']:.2f}`",
            f"- Overall AUC: `{baseline_metrics['overall_auc']:.2f}`",
            f"- Final Acc: `{baseline_metrics['final_acc']:.2f}`",
            f"- Final AUC: `{baseline_metrics['final_auc']:.2f}`",
        ])

    lines.extend(["", "## CEL Stage1 Last-Layer"])
    if cel_progress:
        lines.append(f"- Live phase: `{cel_progress['phase']}`")
        if cel_progress["epoch"] is not None:
            lines.append(f"- Live epoch: `{cel_progress['epoch']}`")
        lines.append(f"- Live {cel_progress['phase']} progress: `{cel_progress['progress']}`")
        timing_note = progress_timing_note(cel_progress, language="cn")
        if timing_note:
            lines.append(f"- {timing_note}")
    if active_step_progress:
        lines.append("- Active step progress:")
        lines.extend(active_step_progress)
    if round3_rows:
        lines.extend(["", "## Task-Conditioned Round 3"])
        for row in round3_rows:
            status = row["status"]
            if status == "failed" and row["failure_reason"]:
                status = f"{status} ({row['failure_reason']})"
            line = f"- `{row['model']}`: `{status}`"
            progress = row["progress"]
            if progress is not None:
                epoch_text = f", epoch `{progress['epoch']}`" if progress["epoch"] is not None else ""
                line += f", `{progress['phase']}`{epoch_text}, `{progress['progress']}`"
            metrics = row["metrics"]
            if metrics is not None and metrics["overall_acc"] is not None and metrics["overall_auc"] is not None:
                line += f", Overall Acc/AUC `{metrics['overall_acc']:.2f}/{metrics['overall_auc']:.2f}`"
            if metrics is not None and metrics["final_acc"] is not None and metrics["final_auc"] is not None:
                line += f", Final Acc/AUC `{metrics['final_acc']:.2f}/{metrics['final_auc']:.2f}`"
            lines.append(line)
    if round3_rows:
        lines.extend([""])
        lines.extend(current_formal_workflow(task_status, current_formal_state))
        lines.extend([""])
        lines.extend(formal_loop_summary(task_status, formal_audit, poll_interval))
        lines.extend(["", "## Synced CEL Metrics"])
    lines.append(build_table([(model, cel_metrics[model]) for model in sorted(cel_metrics)]))
    lines.extend(["", "## Best So Far"])
    if formal_best is None and diagnostic_best is None:
        lines.append("- No CEL metrics available yet")
    else:
        if formal_best is not None:
            model, metrics = formal_best
            baseline_auc = baseline_metrics["overall_auc"] if baseline_metrics else None
            baseline_acc = baseline_metrics["overall_acc"] if baseline_metrics else None
            delta_acc = ""
            delta_auc = ""
            if baseline_acc is not None and metrics["overall_acc"] is not None:
                delta_acc = f", delta Acc `{metrics['overall_acc'] - baseline_acc:+.2f}`"
            if baseline_auc is not None and metrics["overall_auc"] is not None:
                delta_auc = f", delta AUC `{metrics['overall_auc'] - baseline_auc:+.2f}`"
            lines.append(
                f"- A-module reference (historical checkpoint): `{model}` with Overall Acc `{metrics['overall_acc']:.2f}` and Overall AUC `{metrics['overall_auc']:.2f}`{delta_acc}{delta_auc}"
            )
        if diagnostic_best is not None:
            model, metrics = diagnostic_best
            lines.append(
                f"- Diagnostic-only best: `{model}` with Overall Acc `{metrics['overall_acc']:.2f}` and Overall AUC `{metrics['overall_auc']:.2f}`"
            )

    lines.extend([
        "",
        "## Key Files",
        "- Formal loop: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`",
        "- Formal method protocol: `results/cel_stage1_last_layer/FORMAL_METHOD_PROTOCOL.md`",
        "- Baseline record: `results/baseline/Baseline_DialogueKT_实验记录.md`",
        "- Stage1 last-layer record: `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`",
        "- Strict full-train report: `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`",
        "- Formal experiment audit: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`",
        "- Formal candidate briefs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`",
        "- Formal candidate config diffs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md`",
        "- Strict full-train runbook: `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md`",
        "- Formal next-action helper: `scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
        "- Formal start/monitor/finalize/review scripts: `scripts/cel_stage1_last_layer/`",
        "- Comparison report: `results/cel_stage1_last_layer/COMPARISON.md`",
        "- Retained candidate stdout, refresh log, status JSON, and audit files: `results/cel_stage1_last_layer/`",
        "- Historical controller/watcher process logs, pid/lock files, and temporary scaffolds were removed after Stage 1 closeout",
        "- Step logs: `results/cel_stage1_last_layer/step_logs`",
    ])

    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
