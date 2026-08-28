#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from current_formal_state import build_current_formal_state
from formal_candidate_registry import (
    FORMAL_CANDIDATE_ORDER,
    FORMAL_CANDIDATES,
    ROUND3_FAILURE_FOLLOWUPS_CN,
)
from task_conditioned_failure_utils import (
    epoch_cycle_note,
    format_monitor_timestamp,
    format_progress_cn,
    latest_phase_progress_from_log,
    recommend_round3_poll_interval,
    recommended_next_monitor_after,
    stability_milestone_note,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "cel_stage1_last_layer"
STATUS_JSON = RESULT_DIR / "task_conditioned_status.json"
AUDIT_JSON = RESULT_DIR / "formal_experiment_audit.json"
REFRESH_LOG = RESULT_DIR / "task_conditioned_refresh.log"
OUTPUT_MD = RESULT_DIR / "FORMAL_METHOD_PROTOCOL.md"


TIMESTAMPED_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+(.*)$")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def latest_refresh_event(path: Path) -> str | None:
    if not path.exists():
        return None
    events = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if TIMESTAMPED_EVENT_RE.match(line)
    ]
    if not events:
        return None
    refresh_complete = [line for line in events if "task_conditioned refresh complete" in line]
    return refresh_complete[-1] if refresh_complete else events[-1]


def latest_successful_refresh_event(path: Path) -> str | None:
    if not path.exists():
        return None
    events = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if TIMESTAMPED_EVENT_RE.match(line)
    ]
    if not events:
        return None
    refresh_success = [
        line
        for line in events
        if "task_conditioned refresh complete" in line and "sync=ok" in line
    ]
    return refresh_success[-1] if refresh_success else None


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


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def round3_status_map(task_status: dict | None) -> dict[str, dict]:
    if task_status is None:
        return {}
    for round_info in task_status.get("rounds") or []:
        if str(round_info.get("title") or "").startswith("Round 3"):
            return {
                str(row.get("model")): row
                for row in (round_info.get("models") or [])
                if row.get("model")
            }
    return {}


def model_name_from_label(label: str) -> str | None:
    for model_name, meta in FORMAL_CANDIDATES.items():
        if meta.get("label") == label:
            return model_name
    return None


def candidate_rows(task_status: dict | None, audit: dict | None) -> list[dict]:
    status_map = round3_status_map(task_status)
    audit_map = {
        str(row.get("model")): row
        for row in (audit or {}).get("rows") or []
        if row.get("model")
    }
    rows: list[dict] = []
    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        status_row = status_map.get(model_name, {})
        audit_row = audit_map.get(model_name, {})
        rows.append(
            {
                "model": model_name,
                "meta": meta,
                "task_status": status_row,
                "audit_row": audit_row,
                "progress": latest_phase_progress_from_log(ROOT / meta["stdout_log_rel"]),
            }
        )
    return rows


def next_use_text(row: dict, rerun_queue: list[str], current_formal_decision: str) -> str:
    meta = row["meta"]
    audit_state = str((row.get("audit_row") or {}).get("audit") or "pending")
    beats_baseline = str((row.get("audit_row") or {}).get("beats_baseline") or "--")
    task_state = str((row.get("task_status") or {}).get("status") or "")
    if current_formal_decision == "monitor_active_full_train":
        if task_state == "running":
            return "monitor active run"
        if audit_state == "recorded":
            return "freeze / analyze after active run"
    if audit_state == "needs_rerun":
        return "rerun first"
    if audit_state == "recorded" and beats_baseline == "yes":
        return "freeze winner"
    if audit_state == "recorded":
        if rerun_queue:
            return "freeze / analyze"
        return "design next one-variable candidate only after review"
    if meta.get("label") in rerun_queue:
        return "rerun first"
    return "pending"


def poll_interval_text(rows: list[dict], next_action: str | None) -> tuple[str, str]:
    if next_action in {"manual_decide", "done"}:
        return "not_applicable", "当前没有 active strict full-train 需要监控"
    if next_action not in {"wait_round3", "launch_round3"}:
        return "unknown", "当前 formal 状态尚未形成稳定监控节奏"
    running_rows = [row for row in rows if str((row.get("task_status") or {}).get("status") or "") == "running"]
    for row in running_rows or rows:
        interval = recommend_round3_poll_interval(row.get("progress"), next_action)
        if interval is not None and interval[1] != "default training cadence":
            seconds, reason = interval
            if seconds is None:
                return "not_applicable", reason
            return f"{seconds}s", reason
    interval = recommend_round3_poll_interval(None, next_action)
    if interval is None:
        return "unknown", "当前 formal 状态尚未形成稳定监控节奏"
    seconds, reason = interval
    if seconds is None:
        return "not_applicable", reason
    return f"{seconds}s", reason


def poll_interval_tuple_from_label(label: str, reason: str) -> tuple[int | None, str]:
    if label == "not_applicable":
        return None, reason
    if label.endswith("s") and label[:-1].isdigit():
        return int(label[:-1]), reason
    return None, reason


def build_current_action_lines(
    rows: list[dict],
    rerun_queue: list[str],
    recorded_non_winners: list[str],
    current_formal_state: dict,
) -> list[str]:
    current_formal_decision = str(current_formal_state.get("current_formal_decision") or "unknown")
    current_active_target = current_formal_state.get("current_active_target")
    current_active_model = current_formal_state.get("current_active_model")
    suggested_launch_key = current_formal_state.get("suggested_launch_key")
    recorded_winners = [
        row
        for row in rows
        if (row.get("audit_row") or {}).get("audit") == "recorded"
        and (row.get("audit_row") or {}).get("beats_baseline") == "yes"
    ]

    if current_formal_decision == "launch_next_candidate":
        target_label = str(suggested_launch_key or current_active_target or "unknown")
        target_model = model_name_from_label(target_label)
        if target_model is None:
            return [
                f"- 当前 formal queue 已切到 launch_next_candidate，但本地 registry 无法解析 `{target_label}`；先修复 registry 再继续。",
            ]
        meta = FORMAL_CANDIDATES[target_model]
        model_pattern = meta["model_pattern"]
        metrics_rel = meta["metrics_rel"]
        stdout_rel = meta["stdout_log_rel"]
        gpu_id = meta["default_gpu_id"]
        lines = [
            f"- 当前 strict formal queue 已允许启动下一轮候选：`{target_label}` / `{target_model}`。",
            "- 这一步应继续遵守“只改一个主变量”的 formal 规则；本轮不要再并行追加其他新候选。",
            "- 启动前建议先读取当前权威下一步：",
            "",
            "```bash",
            "python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py",
            "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
            "```",
            "",
            "- 本地准备命令：",
            "",
            "```bash",
            "bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh",
            "bash scripts/cel_stage1_last_layer/sync_code_to_server.sh",
            "bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh 3090",
            "```",
            "",
            "- 推荐的当前状态一键 alias 启动入口：",
            "",
            "```bash",
            "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
            "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
            f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
            "```",
            "",
            "- 若当前终端不能直接 SSH，则在可联网终端执行下面这组远端命令：",
            "",
            "```bash",
            "ssh -i ~/.ssh/id_rsa_u4 -p 22049 user4@119.29.183.125",
            "cd /home/user4/dialogue-kt",
            f"ps -ef | grep -E \"{model_pattern}\" | grep -v grep",
            f"test -e \"{metrics_rel}\" && echo \"metrics_present\" || echo \"metrics_missing\"",
            f"TASK_CONDITIONED_GPU_ID=\"{gpu_id}\" TASK_CONDITIONED_FORCE_RERUN=\"0\" bash \"scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh\" \"{target_label}\"",
            f"tail -n 40 \"{stdout_rel}\"",
            "```",
            "",
            "- 启动后回到统一监控/落盘/回看入口：",
            "",
            "```bash",
            "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
            "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
            "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
            "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
            "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
            "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
            "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
            "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
            f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {target_label}",
            f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
            f"bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {target_label}",
            f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
            "```",
        ]
        if recorded_non_winners:
            lines.append(
                f"- 当前已记录但未获胜的候选：`{', '.join(recorded_non_winners)}`；它们现在只用于冻结、对比和指导下一轮单变量设计。"
            )
        return lines

    if current_formal_decision == "monitor_active_full_train":
        target_label = str(current_active_target or "unknown")
        target_model = str(current_active_model or "unknown")
        lines = [
            f"- 当前 formal queue 正在执行 active strict full-train：`{target_label}` / `{target_model}`。",
            "- 当前不允许跳过这轮运行去设计下一轮 formal candidate；先继续监控并等待 SSH 侧完整 `train + val + test` 落地。",
            "- 推荐先读取当前权威下一步：",
            "",
            "```bash",
            "python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py",
            "bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090",
            "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090",
            "bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run",
            "```",
            "",
            "- 统一监控入口：",
            "",
            "```bash",
            "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once",
            "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once",
            "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
            "TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
            "bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090",
            "```",
            "",
            "- 训练完成后，再按统一 closeout 顺序执行：",
            "",
            "```bash",
            "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
            "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
            "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
            "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
            f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {target_label}",
            f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
            f"bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {target_label}",
            f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
            "```",
        ]
        if recorded_non_winners:
            lines.append(
                f"- 当前已记录但未获胜的候选：`{', '.join(recorded_non_winners)}`；它们现在只能冻结和分析，不能覆盖 active run 的优先级。"
            )
        return lines

    if rerun_queue:
        target_label = rerun_queue[0]
        target_model = model_name_from_label(target_label)
        if target_model is None:
            return [
                "- 当前 formal queue 仍有 `needs_rerun` 候选，但本地 registry 无法解析对应模型；先修复 registry 再继续。",
            ]
        meta = FORMAL_CANDIDATES[target_model]
        metrics_rel = meta["metrics_rel"]
        stdout_rel = meta["stdout_log_rel"]
        model_pattern = meta["model_pattern"]
        gpu_id = meta["default_gpu_id"]
        lines = [
            f"- 当前 formal queue 仍有 `needs_rerun`：`{target_label}`。在它完成同口径 SSH full-train 前，不允许设计新的 formal candidate。",
        ]
        if recorded_non_winners:
            lines.append(
                f"- 当前已记录但未获胜的候选：`{', '.join(recorded_non_winners)}`；这些结果只允许冻结、分析、回看，不允许抢先覆盖 rerun 优先级。"
            )
        lines.extend(
            [
                "- 当前本地可先运行统一 helper，直接读取权威状态给出的正式下一步：",
                "",
                "```bash",
                "python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py",
                "```",
                "",
                "- 本地准备命令：",
                "",
                "```bash",
                "bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh",
                "bash scripts/cel_stage1_last_layer/sync_code_to_server.sh",
                "bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
                f"bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {target_label}",
                "```",
                "",
                "- 若本地 WSL `ssh` 已经配置好 alias，例如 `3090`，优先用自动解析当前 formal queue 的一键入口：",
                "",
                "```bash",
                "bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090",
                "```",
                "",
                "- 若你要显式指定当前 rerun 目标，也可以使用候选级 alias 入口：",
                "",
                "```bash",
                f"bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
                "```",
                "",
                "- 若当前终端不能直接 SSH，则在可联网终端执行下面这组远端命令：",
                "",
                "```bash",
                "ssh -i ~/.ssh/id_rsa_u4 -p 22049 user4@119.29.183.125",
                "cd /home/user4/dialogue-kt",
                f"ps -ef | grep -E \"{model_pattern}\" | grep -v grep",
                f"test -e \"{metrics_rel}\" && echo \"metrics_present\" || echo \"metrics_missing\"",
                f"TASK_CONDITIONED_GPU_ID=\"{gpu_id}\" TASK_CONDITIONED_FORCE_RERUN=\"0\" bash \"scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh\" \"{target_label}\"",
                f"tail -n 40 \"{stdout_rel}\"",
                "```",
                "",
                "- 启动后继续沿用统一监控/落盘/回看入口：",
                "",
                "```bash",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch",
                "bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
                "bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {target_label}",
                f"bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {target_label}",
                f"bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {target_label}",
                "```",
            ]
        )
        return lines

    if current_formal_decision == "winner_or_done" and recorded_winners:
        winner_row = max(
            recorded_winners,
            key=lambda row: (
                ((row.get("audit_row") or {}).get("metrics") or {}).get("overall_acc", float("-inf")),
                ((row.get("audit_row") or {}).get("metrics") or {}).get("overall_auc", float("-inf")),
            ),
        )
        winner_label = winner_row["meta"]["label"]
        return [
            f"- 当前 A 模块参考结果已固定（历史 candidate key：`{winner_label}`），formal queue 已进入 done 状态，不再设计新的 Stage 1 candidate。",
            f"- 如需复核结论，优先使用 `review_formal_candidate.sh {winner_label}`、对应 alias 包装器，或统一 `finalize -> review` 快照入口。",
            "- 当前工作重点是实现 `h_r/h_m` 双路径前向、完整训练目标、tensor contracts 与 audit；在这些工作完成前不注册新的正式 Stage 2 candidate。",
        ]

    if current_formal_decision == "design_next_formal_candidate" and recorded_non_winners:
        latest_recorded = recorded_non_winners[-1]
        return [
            "- 当前不存在 rerun 队列，Round 3 已完成候选也都已经冻结为 recorded non-winner。",
            f"- 最新落地的 recorded non-winner 是 `{latest_recorded}`；当前不应再让 controller 自动反复 finalize/review 旧候选。",
            "- 先用 `print_current_formal_next_action.py` 和 `show_current_formal_runtime_health.sh` 复核权威状态，再通过 `scaffold_formal_candidate.py` 设计下一轮只改一个主变量的新 formal candidate。",
            f"- 如需显式回看最新落地候选，可使用 `review_formal_candidate.sh {latest_recorded}` 或 alias 包装器。",
        ]

    if recorded_non_winners:
        return [
            "- 当前不存在 rerun 队列，已完成 formal candidate 也都只是 recorded non-winner。",
            "- 先用 `review_formal_candidate.sh` 回看现有结果，再通过 `scaffold_formal_candidate.py` 设计下一轮只改一个主变量的新 formal candidate。",
        ]

    return [
        "- 当前 formal queue 还没有形成可执行的 rerun/recorded 结论；先刷新本地状态并检查 `FORMAL_EXPERIMENT_AUDIT.md`。",
    ]


def build_candidate_detail_lines(row: dict) -> list[str]:
    meta = row["meta"]
    audit_row = row.get("audit_row") or {}
    task_row = row.get("task_status") or {}
    metrics = audit_row.get("metrics") or {}
    analysis = audit_row.get("analysis") or {}
    progress = row.get("progress")

    lines = [
        f"### `{meta['label']}` / `{row['model']}`",
        "",
        f"- 起点：{meta['start_point_cn']}",
        f"- 方法：{meta['method_cn']}",
        *([f"- 实现口径核对：{meta['implementation_guard_cn']}"] if meta.get("implementation_guard_cn") else []),
        f"- 当前状态：`{audit_row.get('status') or task_row.get('status') or 'pending'}`",
        f"- Audit 状态：`{audit_row.get('audit') or 'pending'}`",
    ]
    if progress is not None:
        lines.append(f"- 当前日志进度：{format_progress_cn(progress)}")
    epoch_note = epoch_cycle_note(meta, progress, language="cn")
    if epoch_note is not None:
        lines.append(f"- 多 epoch 监控说明：{epoch_note}")
    milestone_note = stability_milestone_note(meta, progress, language="cn")
    if milestone_note is not None:
        lines.append(f"- {milestone_note}")
    if metrics:
        lines.append(
            f"- 当前已同步 Overall Acc / AUC：**{fmt_num(metrics.get('overall_acc'))} / {fmt_num(metrics.get('overall_auc'))}**；"
            f"Final Acc / AUC：**{fmt_num(metrics.get('final_acc'))} / {fmt_num(metrics.get('final_auc'))}**。"
        )
    if analysis.get("pred_true") is not None:
        lines.append(f"- `Pred True`：**{analysis['pred_true']:.2f}%**。")
    if analysis.get("best_acc") is not None or analysis.get("best_threshold") is not None:
        lines.append(
            f"- best-threshold 诊断：best Acc **{fmt_num(analysis.get('best_acc'))}**，threshold **{fmt_num(analysis.get('best_threshold'), 3)}**。"
        )
    if analysis.get("delta_acc_vs_baseline") is not None or analysis.get("delta_auc_vs_baseline") is not None:
        lines.append(
            f"- 相对 baseline 的 Overall Acc / AUC 变化：**{analysis.get('delta_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_auc_vs_baseline', 0.0):+.2f}**。"
        )
    if analysis.get("delta_final_acc_vs_baseline") is not None or analysis.get("delta_final_auc_vs_baseline") is not None:
        lines.append(
            f"- 相对 baseline 的 Final Acc / AUC 变化：**{analysis.get('delta_final_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_final_auc_vs_baseline', 0.0):+.2f}**。"
        )
    note_text = localized_audit_note(audit_row)
    if note_text:
        lines.append(f"- 当前结论：{note_text}")
    for followup in ROUND3_FAILURE_FOLLOWUPS_CN.get(row["model"], []):
        lines.append(f"- {followup}")
    lines.append(f"- 若未双超时的下一轮解释：{meta['if_fail_cn']}")
    return lines


def localized_audit_note(audit_row: dict) -> str | None:
    audit_state = str(audit_row.get("audit") or "")
    beats_baseline = str(audit_row.get("beats_baseline") or "--")
    if audit_state == "needs_rerun":
        return "该 full-train 候选已失败；先保持方法 scope 不变，理解失败原因后按同一 strict full-train 口径重跑。"
    if audit_state == "recorded" and beats_baseline == "yes":
        return "该 full-train 候选的 artifacts 与 markdown coverage 已完整，且结果已双超 baseline；可以进入 winner 固化流程。"
    if audit_state == "recorded":
        return "该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。"
    note = audit_row.get("note")
    return str(note) if note else None


def render_protocol(
    task_status: dict | None,
    audit: dict | None,
    refresh_event: str | None,
    authoritative_refresh_event: str | None,
    output_path: Path,
) -> str:
    baseline = (task_status or {}).get("baseline") or {}
    anchor = (task_status or {}).get("v1") or {}
    next_action = str((task_status or {}).get("next_action") or "unknown")
    winner_found = bool((task_status or {}).get("winner_found"))
    sync_freshness = str((audit or {}).get("sync_freshness") or sync_freshness_label(refresh_event))
    counts = (audit or {}).get("counts") or {}
    rerun_queue = [str(row.get("label")) for row in (audit or {}).get("rows") or [] if row.get("audit") == "needs_rerun"]
    recorded_non_winners = [
        str(row.get("label"))
        for row in (audit or {}).get("rows") or []
        if row.get("audit") == "recorded" and row.get("beats_baseline") != "yes"
    ]
    rows = candidate_rows(task_status, audit)
    poll_interval, poll_reason = poll_interval_text(rows, next_action)
    next_monitor_after = recommended_next_monitor_after(
        refresh_event,
        poll_interval_tuple_from_label(poll_interval, poll_reason),
    )
    next_monitor_after_text = format_monitor_timestamp(next_monitor_after)
    current_formal_state = build_current_formal_state(
        task_status,
        audit,
        sync_freshness=sync_freshness,
        recommended_poll_interval=poll_interval,
    )

    lines = [
        "# Formal Method Protocol",
        "",
        "> Terminology note: this historical protocol retains candidate identifiers for auditability. The audited current method is called the **A module** in method documents; candidate numbers are not method names.",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Scope: `results/cel_stage1_last_layer/` 当前 strict formal `task_conditioned` loop。",
        "- 目的：把“什么算正式方法、当前该跑什么、如何监控、如何落盘、每轮实验后必须记录什么”集中到一个摘要入口。",
        "",
        "## Core Rules",
        "",
        "- 只有 SSH 服务器上完整 `train + val + test` 的结果才算正式方法结果。",
        "- 任何 calibrator-only、frozen retraining、fixed-bias eval、validation-fit bias、以及从已训练 ckpt 出发的 post-hoc 调整都只算诊断证据。",
        "- formal winner 只能在 `Overall Acc > baseline` 且 `Overall AUC > baseline` 时成立，而且必须先通过 `finalize -> audit -> review`。",
        "- 若 formal queue 中仍存在 `needs_rerun` 候选，禁止跳过去直接设计新方法。",
        "- 每轮新的 formal candidate 只允许改一个主变量，并且必须能解释为什么有希望同时改善 Acc 与 AUC。",
        "",
        "## Method Admissibility Checklist",
        "",
        "在把某个新方案登记为正式方法前，先逐条确认：",
        "",
        "1. 该候选会在 SSH 上独立完成一轮新的 `train + val + test`，而不是只基于已有 ckpt 做补训、后处理或校准。",
        "2. 如果使用 warm-start，它必须被明确写进候选方法定义本身，并且该候选仍需完成自己的 SSH-side full-train；单纯复用旧 ckpt 的后续结果不计入正式方法。",
        "3. 候选只改一个主变量，其余训练口径、比较基线和结果解读规则保持稳定。",
        "4. 候选在启动前已经写清：模型名、起点、唯一主改动、为什么有机会同时改善 Acc 与 AUC、预期观察信号、以及失败后下一轮只改一个变量的解释。",
        "5. 候选脚本、formal registry、`FORMAL_CANDIDATE_BRIEFS.md`、`FORMAL_CANDIDATE_CONFIG_DIFFS.md`、`preflight_strict_full_train.sh` 必须一致；若 config diff 未声明就发生漂移，该候选不应启动。",
        "",
        "## Current Gate",
        "",
        f"- Baseline gate: `Overall Acc > {fmt_num(baseline.get('overall_acc'))}` and `Overall AUC > {fmt_num(baseline.get('overall_auc'))}`.",
        f"- Historical selector/ranking comparison anchor: `{anchor.get('model', 'unknown')}` with Overall Acc / AUC **{fmt_num(anchor.get('overall_acc'))} / {fmt_num(anchor.get('overall_auc'))}**.",
        f"- Current next action: `{next_action}`",
        f"- Winner found: `{winner_found}`",
        f"- Current formal decision: `{current_formal_state['current_formal_decision']}`",
        f"- Current formal rerun target: `{current_formal_state['current_rerun_target'] or 'none'}`",
        f"- Current formal recorded non-winner: `{current_formal_state['current_recorded_non_winner_target'] or 'none'}`",
        f"- Current formal recorded non-winner queue: `{', '.join(current_formal_state['recorded_non_winner_queue']) if current_formal_state['recorded_non_winner_queue'] else 'none'}`",
        f"- Task-conditioned sync freshness: `{sync_freshness}`",
        f"- Latest authoritative remote refresh event: `{authoritative_refresh_event or 'none'}`",
        f"- Latest refresh attempt: `{refresh_event or 'none'}`",
        f"- Formal audit summary: `{', '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'unknown'}`",
        f"- Recommended local monitoring cadence: `{poll_interval}` ({poll_reason})",
    ]
    if next_monitor_after_text is not None and sync_freshness == "fresh_remote":
        lines.append(
            f"- Suggested next local monitor after: `{next_monitor_after_text}` (latest authoritative refresh + {poll_interval})"
        )

    if sync_freshness != "fresh_remote":
        lines.append("- 当前本地 markdown 仍可能只是基于旧快照重建；只要 `sync!=ok`，就不能把本地进度变化当成新的远端真值。")

    lines.extend(["", "## Current Queue", ""])
    lines.extend(
        [
            "| Candidate | Start Point | Status | Audit | Current Result | Next Use |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        meta = row["meta"]
        audit_row = row.get("audit_row") or {}
        metrics = audit_row.get("metrics") or {}
        result = "--"
        if metrics:
            result = f"{fmt_num(metrics.get('overall_acc'))}/{fmt_num(metrics.get('overall_auc'))}"
        lines.append(
            f"| `{meta['label']}` | {meta['start_point_en']} | {audit_row.get('status') or (row.get('task_status') or {}).get('status') or 'pending'} | "
            f"`{audit_row.get('audit') or 'pending'}` | {result} | {next_use_text(row, rerun_queue, str(current_formal_state['current_formal_decision']))} |"
        )

    lines.extend(["", "## Current Required Action", ""])
    lines.extend(build_current_action_lines(rows, rerun_queue, recorded_non_winners, current_formal_state))

    lines.extend(
        [
            "",
            "## Runtime Source Priority",
            "",
            "运行期判断一律按下面顺序取权威来源，不要从 `.claude/` 或手工笔记里读取易过期的 live 数字：",
            "",
            "1. `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
            "2. `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`",
            "3. `results/cel_stage1_last_layer/STATUS.md`",
            "4. `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`",
            "5. `results/cel_stage1_last_layer/task_conditioned_refresh.log`",
            "",
            "补充规则：",
            "",
            "- 若最新 refresh 不是 `sync=ok`，只能把本地 surface 当作旧快照重建结果，不能当成新的远端真值。",
            "- 若 `current_formal_decision=monitor_active_full_train`，默认只继续 refresh / monitor，不得并行启动新候选。",
            "- 若当前候选是多 epoch full-train，`epoch 1 validation -> epoch 2 training` 的切换应视为正常流程，不表示 run 重启。",
        ]
    )

    lines.extend(
        [
            "",
            "## Recommended WSL SSH Alias",
            "",
            "- 推荐先在 WSL 下固定一个本地 alias，例如 `3090`，再通过 alias wrapper 启动 strict formal run。",
            "- 仓库内置模板脚本：`bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`。",
            "- 推荐 `~/.ssh/config` 配置块：",
            "",
            "```sshconfig",
            "Host 3090",
            "    HostName 119.29.183.125",
            "    Port 22049",
            "    User user4",
            "    PreferredAuthentications publickey",
            "    IdentityFile /home/luo/.ssh/id_rsa_u4",
            "    IdentitiesOnly yes",
            "```",
            "",
            "- 推荐权限：`chmod 700 ~/.ssh`、`chmod 600 ~/.ssh/config`、`chmod 600 ~/.ssh/id_rsa_u4`。",
            "- 自动化 SSH / rsync 命令会额外附加 `BatchMode=yes` 与 `ConnectTimeout=15`，避免 formal loop 卡在密码提示或长时间无响应的连接上。",
        ]
    )

    lines.extend(
        [
            "",
            "## Formal Loop",
            "",
            "1. 先读 `STATUS.md`、`STRICT_FULL_TRAIN_REPORT.md`、`FORMAL_EXPERIMENT_AUDIT.md`，确认当前是 rerun、继续监控，还是已经允许设计下一轮 formal candidate。",
            "2. 若仍有 `needs_rerun`，先保持方法范围不变，修复代码后按同一 strict full-train 口径在 SSH 上重跑。",
            "3. 若 rerun 队列已清空且当前结果仍未双超 baseline，使用 `scaffold_formal_candidate.py` 生成下一轮脚手架，并确保只改一个主变量。",
            "4. 本地只做代码实现、文档更新、`py_compile`、shell 语法检查、`preflight_strict_full_train.sh`；本地训练结果不计入正式结论。",
            "5. 所有正式实验都必须通过 `start_current_formal_candidate_via_ssh_alias.sh`、`start_formal_candidate.sh` 或其 manual SSH fallback 走 SSH full-train 启动。",
            "6. SSH run 落地后，必须先执行 `finalize_formal_candidate.sh`（或对应 alias wrapper），再检查 `FORMAL_EXPERIMENT_AUDIT.md`，最后才允许 `review_formal_candidate.sh`（或对应 alias wrapper）判断 winner / non-winner / rerun。",
            "7. 只有在 audit 干净、结果完整落盘、且双超 baseline 时，才允许把某个候选升级为新的正式方法结论。",
        ]
    )

    lines.extend(
        [
            "",
            "## Monitoring Cadence",
            "",
            "- 共享自适应轮询规则：训练早期长时 full-train 用 `900s`，稳定训练中期通常 `600s`，`Validation / Testing`、`>=95%` 进度或短剩余 ETA 时通常收紧到 `300s`，而 `<=5m` 的极短剩余 ETA 会进一步收紧到 `120s`。",
            "- 当前 cadence 的解释性来源以 `STATUS.md` 和 `STRICT_FULL_TRAIN_REPORT.md` 里的 recommended poll interval 为准。",
            "- 若最新 authoritative refresh 不是 `sync=ok`，说明本地只是旧快照 rebuild；此时继续看日志可以，但不能把本地状态变化当成新的远端结论。",
            "- 手动 monitor、后台 controller、finalize、review 现在共享同一 refresh 锁；相邻请求会串行化执行，避免同一时间窗口重复 sync/rebuild。",
            "- controller 现在会先读取本地 authoritative surface；若当前仍是 `fresh_remote` 且未到 `Suggested next local monitor after`，它会直接继续休眠，而不是先做一次多余 refresh。",
            "- 若需要在本地长期挂起完整 formal loop，优先使用后台 controller：`bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090`。",
            "- 若需要确认“后台 controller 现在到底在执行什么”，优先看 `STATUS.md` 里的 `Task-conditioned controller pid/alive`、`Task-conditioned controller command`、`Task-conditioned controller child command`，或直接运行 `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`。",
            "- 推荐监控命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once`、`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`。",
            "- 若当前在 WSL 里依赖 SSH alias，推荐命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once`、`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`。",
            "",
            "### Cadence Decision Checklist",
            "",
            "每次决定要不要加密监控前，先看这四件事：",
            "",
            "1. `Live phase` 是 `training`、`validating` 还是 `testing`。",
            "2. `Live epoch / step / ETA` 是否持续推进。",
            "3. 最新 authoritative refresh 是否是 `sync=ok`。",
            "4. 当前报告给出的 `Recommended poll interval` 与 `Suggested next local monitor after`。",
            "",
            "执行规则：",
            "",
            "- 训练早期且 ETA 仍长时，优先接受较慢 cadence，不必人为缩短。",
            "- 中期稳定训练默认接受 `600s` 左右的 cadence，避免无意义高频 SSH 同步。",
            "- 一旦转入 `Validation / Testing`、`95%+` 进度或 ETA 明显缩短，主动收紧 cadence。",
            "- 若 ETA 已短到约 `<=5m`，再进一步收紧到 `120s`，以便尽快接住 closeout。",
        ]
    )

    lines.extend(
        [
            "",
            "## Closeout And Recording",
            "",
            "1. 同步远端 `metrics / qual / stdout log`。",
            "2. 重建 `task_conditioned_status.json` 与所有 markdown surfaces。",
            "3. 检查 `FORMAL_EXPERIMENT_AUDIT.md`，确认 artifacts 与 markdown coverage 都完整。",
            "4. 再通过 `review_formal_candidate.sh` 或对应 alias wrapper 读取 authoritative surfaces 做 formal judgment。",
            "",
            "每个完成的 formal candidate 必须记录：",
            "",
            "- 模型名",
            "- 起点",
            "- 方法主改动",
            "- Overall Acc / AUC",
            "- Final Acc / AUC",
            "- `Pred True`",
            "- Overall Acc / AUC 与 Final Acc / AUC 相对 baseline 的四项 delta",
            "- 是否双超 baseline",
            "- 若失败：失败类型、失败位置或关键报错、以及为什么修复后仍保持 strict full-train 口径",
            "- 若未双超：下一轮为什么只改一个新的主变量",
            "",
            "### Per-Candidate Recording Checklist",
            "",
            "每个 SSH full-train 候选结束后，至少补齐下面这些字段，缺一项都不算正式闭环完成：",
            "",
            "1. `Overall Acc / AUC`",
            "2. `Final Acc / AUC`",
            "3. `Pred True`",
            "4. best-threshold diagnostic",
            "5. Overall Acc / AUC 与 Final Acc / AUC 的四项 delta vs baseline",
            "6. delta vs historical selector/ranking comparison anchor `v1`",
            "7. 当前审计状态：`recorded`、`pending` 或 `needs_rerun`",
            "8. 若失败，写清失败类型、失败位置、修复动作、以及为什么修复后仍保持 strict full-train 口径",
            "9. 若未双超，写清下一轮只改一个变量的理由",
            "",
            "必须覆盖的 markdown surfaces：",
            "",
            "- `results/cel_stage1_last_layer/STATUS.md`",
            "- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`",
            "- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`",
            "- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`",
            "- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md`",
            "- `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`",
            "- `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`",
            "- `results/cel_stage1_last_layer/COMPARISON.md`",
            "- `results/cel_stage1_last_layer/DETAILED_ANALYSIS.md`",
        ]
    )

    lines.extend(["", "## Current Candidate Notes", ""])
    for row in rows:
        lines.extend(build_candidate_detail_lines(row))
        lines.append("")

    lines.extend(
        [
            "## Key References",
            "",
            "- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`",
            "- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md`",
            "- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`",
            "- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`",
            "- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`",
            "- `scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
            "- `scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh`",
            "- `scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/start_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh`",
            "- `scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/print_current_formal_next_action.py`",
            "- `scripts/cel_stage1_last_layer/monitor_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh`",
            "- `scripts/cel_stage1_last_layer/finalize_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/review_formal_candidate.sh`",
            "- `scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh`",
        ]
    )

    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", default=str(STATUS_JSON))
    parser.add_argument("--audit-json", default=str(AUDIT_JSON))
    parser.add_argument("--refresh-log", default=str(REFRESH_LOG))
    parser.add_argument("--output", default=str(OUTPUT_MD))
    args = parser.parse_args()

    task_status = load_json(Path(args.status_json))
    audit = load_json(Path(args.audit_json))
    refresh_event = latest_refresh_event(Path(args.refresh_log))
    authoritative_refresh_event = latest_successful_refresh_event(Path(args.refresh_log)) or refresh_event
    render_protocol(task_status, audit, refresh_event, authoritative_refresh_event, Path(args.output))


if __name__ == "__main__":
    main()
