#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from formal_candidate_registry import (
    FORMAL_CANDIDATES,
    ROUND3_FAILURE_FOLLOWUPS_CN,
    build_formal_log_path_map,
)
from formal_queue_state import launch_key_for_model
from task_conditioned_failure_utils import (
    epoch_cycle_note,
    extract_failure_evidence,
    format_monitor_timestamp,
    format_progress_cn,
    latest_phase_progress_from_log,
    progress_timing_note,
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
DIAG_RE = re.compile(r"CEL Diagnostics:\s+(.*)")
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
ROUND3_LOG_FILES = {
    model_name: str(path)
    for model_name, path in build_formal_log_path_map(Path(".")).items()
}


def parse_metrics(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    loss_match = LOSS_RE.search(text)
    overall_match = OVERALL_RE.search(text)
    final_match = FINAL_RE.search(text)
    if not loss_match or not overall_match:
        return None
    diag_match = DIAG_RE.search(text)
    return {
        "model": path.stem.replace("metrics_", "", 1),
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
        "diag": diag_match.group(1).strip() if diag_match else None,
    }


def model_role(model_name: str) -> str:
    return "diagnostic" if model_name in DIAGNOSTIC_ONLY_MODELS else "formal"


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


def round3_rows(task_status: dict | None) -> list[dict]:
    if task_status is None:
        return []
    rows: list[dict] = []
    for round_info in task_status.get("rounds") or []:
        title = round_info.get("title") or ""
        if not title.startswith("Round 3"):
            continue
        for row in round_info.get("models") or []:
            model = row.get("model")
            if not model:
                continue
            log_path = Path(ROUND3_LOG_FILES[model]) if model in ROUND3_LOG_FILES else None
            rows.append(
                {
                    "model": model,
                    "status": row.get("status", "unknown"),
                    "failure_reason": row.get("failure_reason"),
                    "progress": latest_phase_progress_from_log(log_path) if log_path is not None else None,
                }
            )
    return rows


def audit_row_map(formal_audit: dict | None) -> dict[str, dict]:
    if not formal_audit:
        return {}
    return {
        row["model"]: row
        for row in (formal_audit.get("rows") or [])
        if row.get("model")
    }


def current_next_step_text(stage: str | None, task_status: dict | None, round3: list[dict]) -> str:
    next_action = task_status.get("next_action") if task_status else None
    winner_found = task_status.get("winner_found") if task_status else False
    winner = task_status.get("winner") if task_status else None
    winner_model = winner.get("model") if isinstance(winner, dict) else None
    running = [row for row in round3 if row.get("status") == "running"]
    failed = [row for row in round3 if row.get("status") == "failed"]
    done = [row for row in round3 if row.get("status") == "done"]
    running_names = " / ".join(f"`{row['model']}`" for row in running)
    failed_names = " / ".join(f"`{row['model']}`" for row in failed)
    if winner_found:
        if winner_model:
            return (
                f"- 当前正式下一步：A 模块历史 checkpoint `{winner_model}` 已完成严格 closeout；"
                "停止新增 Stage 1 candidate，转入目标 A+B 双路径实现与 contract 验证。"
            )
        return "- 当前正式下一步：严格 winner 已完成 closeout；停止新增 formal candidate，冻结 Stage 1 结论并转入实验总结与结果分析。"
    if next_action == "wait_round3" or stage == "task_conditioned_round3_running":
        if running and failed:
            return f"- 当前正式下一步：继续监控仍在运行的 {running_names}，并把已失败的 {failed_names} 作为代码修复后待重跑的完整联合训练候选。"
        if running:
            return f"- 当前正式下一步：继续监控仍在运行的 {running_names} 完整联合训练，并在指标落盘后刷新本地分析文档。"
        return "- 当前正式下一步：继续监控已经启动的完整联合训练，并在指标落盘后刷新本地分析文档。"
    if next_action == "manual_decide" or stage == "task_conditioned_round3_done_waiting_decision":
        if failed and not running:
            if done:
                return f"- 当前正式下一步：确认已完成候选未能双超 baseline，并优先为已失败的 {failed_names} 准备修复后的 SSH 全量重跑；推荐先打印 WSL alias 模板，再使用 alias 启动入口。"
            return f"- 当前正式下一步：分析现有完整训练结果，并为已失败的 {failed_names} 准备修复后的 SSH 全量重跑；推荐先打印 WSL alias 模板，再使用 alias 启动入口。"
        return "- 当前正式下一步：分析当前 strict full-train 队列中已完成候选的结果，判断是否同时超过 baseline Overall Acc 与 Overall AUC。"
    if next_action == "launch_round3":
        return "- 当前正式下一步：在 SSH 上启动当前 formal queue 允许的 strict full-train 候选；在当前队列下应先打印 WSL alias 模板，再用 alias 入口启动 `v21`。"
    if round3:
        if running and failed:
            return f"- 当前正式下一步：继续围绕 {running_names} 的严格 end-to-end 结果更新正式结论，并修复后重跑 {failed_names}。"
        if running:
            return f"- 当前正式下一步：继续围绕 {running_names} 的严格 end-to-end 结果更新正式结论。"
        if failed:
            return f"- 当前正式下一步：修复并重跑已失败的 {failed_names}，仍按严格 end-to-end 口径更新正式结论。"
        return "- 当前正式下一步：继续围绕完整联合训练的严格 end-to-end 结果更新正式结论。"
    return "- 当前正式下一步：在 SSH 上运行当前 formal queue 允许的 strict full-train 候选。"


def build_next_steps(stage: str | None, task_status: dict | None, round3: list[dict]) -> list[str]:
    next_action = task_status.get("next_action") if task_status else None
    winner_found = task_status.get("winner_found") if task_status else False
    winner = task_status.get("winner") if task_status else None
    winner_model = winner.get("model") if isinstance(winner, dict) else None
    running = [row for row in round3 if row.get("status") == "running"]
    failed = [row for row in round3 if row.get("status") == "failed"]
    done = [row for row in round3 if row.get("status") == "done"]
    running_names = " / ".join(f"`{row['model']}`" for row in running)
    failed_names = " / ".join(f"`{row['model']}`" for row in failed)
    if winner_found:
        winner_text = f"A 模块历史 checkpoint `{winner_model}`" if winner_model else "当前 A 模块参考结果"
        return [
            f"1. 冻结 {winner_text} 的 strict winner 结论，不再启动新的 formal candidate。",
            "2. 复核 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / TASK_CONDITIONED_TUNING_LOOP.md / CEL_Stage1_LastLayer_DialogueKT_实验记录.md` 的 winner 文案是否一致。",
            "3. 将 Stage 2 首轮旧单路径结果保留为历史 pilot，不作为当前 A+B 机制的正式验证。",
            "4. 实现 `a -> h_r/h_n/h_nb/h_m`、共享后续网络双路径前向、`L_r + L_m + L_cons` 三项核心损失与 tensor contracts。",
            "5. 在代码与 audit 路径验证完成前，不注册新的正式 Stage 2 candidate，也不启动 Stage 3。",
        ]
    if next_action == "wait_round3" or stage == "task_conditioned_round3_running":
        if running and failed:
            return [
                f"1. 在 SSH 上继续监控 {running_names} 的训练进度，并同步最新日志回本地。",
                f"2. 同步更新后的 `dialogue_kt/training.py` 到 SSH，使用新的数值稳定 BCE 路径后再重跑 {failed_names}。",
                "3. 结果落盘后同步远端 `metrics / qual / logs` 回本地。",
                "4. 刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
                "5. 刷新后先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否已经覆盖已完成 formal candidate，再决定是否可以正式判 winner。",
            ]
        return [
            "1. 在 SSH 上继续监控当前 active strict full-train candidate 的训练进度，并同步最新日志回本地。",
            "2. 结果落盘后同步远端 `metrics / qual / logs` 回本地。",
            "3. 刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
            "4. 刷新后先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否干净。",
            "5. 仅根据完整训练结果决定是否进入下一轮结构改动。",
        ]
    if next_action == "manual_decide" or stage == "task_conditioned_round3_done_waiting_decision":
        if failed and done:
            explicit_failed_key = None
            for row in failed:
                explicit_failed_key = launch_key_for_model(row.get("model"))
                if explicit_failed_key:
                    break
            explicit_failed_key = explicit_failed_key or "v21"
            return [
                "1. 确认 `FORMAL_EXPERIMENT_AUDIT.md` 已完整记录当前已完成 formal candidate，并据此冻结 `v22` 未双超 baseline 的结论。",
                f"2. 同步更新后的 `dialogue_kt/training.py` 到 SSH，并按同一 strict full-train 配置优先重跑 {failed_names}；推荐先运行 `bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`，再使用 `bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`；若要显式指定当前 rerun 目标，可使用 `bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_failed_key}`；若当前终端不能直接打开 SSH，则先运行 `bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`；若仍要显式指定候选级步骤，再运行 `bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_failed_key}`，并按其输出落到共享远端入口 `bash scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh {explicit_failed_key}`。",
                "3. 重跑后优先使用 `bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh` 做统一 closeout，再同步新的 `metrics / qual / logs` 回本地。",
                "4. 再用 `bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh` 回看，并刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
                f"5. 只有在 `{explicit_failed_key}` 修复重跑后仍未双超 baseline，才进入下一轮新的完整联合训练设计。",
            ]
        return [
            "1. 同步并核对当前 strict full-train 队列已完成候选的 `metrics / qual / logs` 是否完整。",
            "2. 刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
            "3. 先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否已经对已完成 formal candidate 给出完整 coverage。",
            "4. 对比 baseline 的 Overall Acc / Overall AUC，判断是否出现严格 winner。",
            "5. 若仍未双超 baseline，则继续设计新的完整联合训练候选。",
        ]
    return [
        "1. 在 SSH 上检查当前 formal queue 候选是否已经启动或完成。",
        "2. 同步远端 `metrics / qual / logs` 回本地。",
        "3. 刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
        "4. 刷新后先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否干净。",
        "5. 仅根据完整训练结果决定是否进入下一轮结构改动。",
    ]


def build_round3_failure_analysis(round3: list[dict]) -> list[str]:
    failed_rows = [row for row in round3 if row.get("status") == "failed"]
    if not failed_rows:
        return []

    lines = ["## 当前失败分析", ""]
    for row in failed_rows:
        lines.append(f"### `{row['model']}`")
        lines.append("")
        log_path = Path(ROUND3_LOG_FILES.get(row["model"], "")) if row["model"] in ROUND3_LOG_FILES else None
        evidence = extract_failure_evidence(log_path) if log_path is not None else {
            "progress": None,
            "assertion_line": None,
            "runtime_line": None,
        }
        notes = []
        progress_text = format_progress_cn(evidence.get("progress"))
        if progress_text is not None:
            notes.append(f"最新已同步失败点：{progress_text}。")
        if evidence.get("assertion_line") is not None:
            notes.append(f'日志关键断言："{evidence["assertion_line"]}"')
        if evidence.get("assertion_site") is not None:
            notes.append(f"断言位置：`{evidence['assertion_site']}`")
        if evidence.get("runtime_line") is not None:
            notes.append(f"日志关键报错：`{evidence['runtime_line']}`")
        if not notes:
            notes.append(f"当前失败类型：`{row.get('failure_reason', 'failed')}`；需要在不改变 strict full-train 口径的前提下修复并重跑。")
        notes.extend(ROUND3_FAILURE_FOLLOWUPS_CN.get(row["model"], []))
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return lines


def build_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Diagnostics |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        final_acc = f"{row['final_acc']:.2f}" if row["final_acc"] is not None else "--"
        final_auc = f"{row['final_auc']:.2f}" if row["final_auc"] is not None else "--"
        lines.append(
            f"| `{row['model']}` | {model_role(row['model'])} | {row['loss']:.4f} | {row['overall_acc']:.2f} | "
            f"{row['overall_auc']:.2f} | {final_acc} | {final_auc} | {row['diag'] or '--'} |"
        )
    return lines


def build_round3_formal_table(round3: list[dict], baseline: dict | None) -> list[str]:
    lines = [
        "| Model | Start Point | Method | Scope | Status | Success Gate |",
        "|---|---|---|---|---|---|",
    ]
    if baseline is None:
        success_gate = "beat baseline Acc/AUC on full result"
    else:
        success_gate = f"> {baseline['overall_acc']:.2f} Acc and > {baseline['overall_auc']:.2f} AUC"

    for row in round3:
        notes = FORMAL_CANDIDATES.get(row["model"], {})
        status = row["status"]
        if status == "failed" and row.get("failure_reason"):
            status = f"{status} ({row['failure_reason']})"
        progress = row.get("progress")
        if progress is not None:
            epoch_text = f", epoch {progress['epoch']}" if progress["epoch"] is not None else ""
            status = f"{status}; {progress['phase']}{epoch_text}; {progress['progress']}"
        lines.append(
            f"| `{row['model']}` | {notes.get('start_point_en', '--')} | {notes.get('method_en', '--')} | "
            f"{notes.get('scope_en', '--')} | {status} | {success_gate} |"
        )
    return lines


def recommended_poll_interval(round3: list[dict], task_status: dict | None) -> tuple[int, str] | None:
    if task_status is None or not round3:
        return None
    next_action = task_status.get("next_action")
    running_rows = [row for row in round3 if row.get("status") == "running"]
    for row in running_rows or round3:
        interval = recommend_round3_poll_interval(row.get("progress"), next_action)
        if interval is not None and interval[1] != "default training cadence":
            return interval
    return recommend_round3_poll_interval(None, next_action)


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def build_completed_formal_analysis(round3: list[dict], audit_map: dict[str, dict]) -> list[str]:
    completed_rows = [row for row in round3 if row.get("status") == "done"]
    if not completed_rows:
        return []

    lines = ["## 正式候选已完成结果分析", ""]
    for row in completed_rows:
        audit_row = audit_map.get(row["model"]) or {}
        metrics = audit_row.get("metrics") or {}
        analysis = audit_row.get("analysis") or {}
        audit_state = audit_row.get("audit", "unknown")
        beats_baseline = audit_row.get("beats_baseline", "--")
        meta = FORMAL_CANDIDATES.get(row["model"], {})
        lines.append(f"### `{row['model']}`")
        lines.append("")
        lines.append(f"- Audit 状态：`{audit_state}`")
        lines.append(f"- 是否双超 baseline：`{beats_baseline}`")
        if meta:
            lines.append(f"- 起点：{meta.get('start_point_cn', '--')}")
            lines.append(f"- 方法主改动：{meta.get('single_variable_cn', '--')}")
            if meta.get("implementation_guard_cn"):
                lines.append(f"- 实现口径核对：{meta.get('implementation_guard_cn')}")
        if metrics:
            lines.append(
                f"- Overall Acc / AUC：**{fmt_num(metrics.get('overall_acc'))} / {fmt_num(metrics.get('overall_auc'))}**；"
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
        if analysis.get("delta_acc_vs_anchor") is not None or analysis.get("delta_auc_vs_anchor") is not None:
            lines.append(
                f"- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**{analysis.get('delta_acc_vs_anchor', 0.0):+.2f} / {analysis.get('delta_auc_vs_anchor', 0.0):+.2f}**。"
            )
        if audit_state == "recorded" and beats_baseline == "yes":
            lines.append("- 当前结论：该候选已完成完整 SSH full-train，且当前记录口径下已满足双超 baseline；当前应冻结为严格 winner，并作为 Stage 1 的正式方法结论。")
        elif audit_state == "recorded":
            lines.append("- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。")
        elif audit_state == "needs_docs":
            lines.append("- 当前结论：该候选结果已落地，但 markdown closeout 还不完整；补齐记录与分析前不能当作正式闭环完成。")
        elif audit_state == "missing_artifacts":
            lines.append("- 当前结论：该候选被标记为 done，但正式 artifact 仍不完整；在补齐同步结果前不能进入正式 winner 判断。")
        elif audit_state == "missing_log":
            lines.append("- 当前结论：该候选指标已同步，但 stdout log 缺失；在补齐日志前不能进入正式 winner 判断。")
        note = audit_row.get("note")
        if note and audit_state not in {"recorded", "needs_docs", "missing_artifacts", "missing_log"}:
            lines.append(f"- 当前 closeout 说明：{note}")
        if beats_baseline != "yes" and meta.get("if_fail_cn"):
            lines.append(f"- 若未双超时的下一轮解释：{meta.get('if_fail_cn')}")
        lines.append("")
    return lines


def build_record(
    rows: list[dict],
    baseline: dict | None,
    stage: str | None,
    task_status: dict | None,
    sync_event: str | None,
    authoritative_sync_event: str | None,
    sync_freshness: str | None,
    formal_audit: dict | None,
) -> str:
    formal_rows = [row for row in rows if model_role(row["model"]) == "formal"]
    diagnostic_rows = [row for row in rows if model_role(row["model"]) == "diagnostic"]
    best_formal = max(formal_rows, key=lambda row: (row["overall_auc"], row["overall_acc"])) if formal_rows else None
    best_diagnostic = max(diagnostic_rows, key=lambda row: (row["overall_auc"], row["overall_acc"])) if diagnostic_rows else None
    v1 = next((row for row in rows if row["model"] == "cel_task_conditioned_lastlayer_v1_qwen3_1.7b"), None)
    winner = task_status.get("winner") if task_status else None
    winner_model = winner.get("model") if isinstance(winner, dict) else None
    round3 = round3_rows(task_status)
    audit_map = audit_row_map(formal_audit)
    poll_interval = recommended_poll_interval(round3, task_status)
    lines = [
        "# CEL Stage1 Last-Layer 实验记录",
        "",
        "> 术语说明：本文保留历史 candidate 与 checkpoint 名称用于实验追溯；当前方法统一称为 **A 模块**，不再以历史候选编号作为方法名称。",
        "",
        "## 当前任务定义",
        "",
        "- 当前阶段仍属于 `Stage 1 last-layer`，不是 Stage 2。",
        "- 当前正式评价口径：只有完整训练完成后得到的指标才算正式结果。",
        "- 冻结补训、calibrator-only、fixed-bias eval、validation-fit bias 全部只记为诊断证据。",
        "",
        "## 当前状态",
        "",
        f"- 当前阶段：`{stage or 'unknown'}`",
    ]

    if baseline is not None:
        lines.extend(
            [
                f"- Baseline：`{baseline['model']}`",
                f"- Baseline Overall Acc / AUC：**{baseline['overall_acc']:.2f} / {baseline['overall_auc']:.2f}**",
                (
                    f"- Baseline Final Acc / AUC：**{baseline['final_acc']:.2f} / {baseline['final_auc']:.2f}**"
                    if baseline.get("final_acc") is not None and baseline.get("final_auc") is not None
                    else "- Baseline Final Acc / AUC：--"
                ),
            ]
        )
    if v1 is not None:
        anchor_label = "Stage 1 历史 selector/ranking 比较锚点"
        lines.append(f"- {anchor_label}：`{v1['model']}`，Overall Acc / AUC **{v1['overall_acc']:.2f} / {v1['overall_auc']:.2f}**")
        if v1.get("final_acc") is not None and v1.get("final_auc") is not None:
            lines.append(f"- {anchor_label} Final Acc / AUC：**{v1['final_acc']:.2f} / {v1['final_auc']:.2f}**")
    if isinstance(winner, dict) and winner_model:
        lines.append(
            f"- 当前 A 模块参考结果：历史 checkpoint `{winner_model}`，Overall Acc / AUC **{winner['overall_acc']:.2f} / {winner['overall_auc']:.2f}**"
        )
    if sync_event is not None:
        lines.append(f"- task-conditioned sync freshness：`{sync_freshness or 'unknown'}`")
        lines.append(f"- 最新 authoritative task-conditioned 远端真值摘要：`{authoritative_sync_event or 'none'}`")
        lines.append(f"- 最新 task-conditioned 刷新尝试：`{sync_event}`")
        if sync_freshness == "fresh_remote":
            lines.append(
                "- 当前本地汇总基于上方最近一次 `sync=ok` 事件；其时效性以事件时间戳为准，"
                "不能仅凭 `fresh_remote` 判断为当前实时远端状态。"
            )
        elif sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
            lines.append("- 当前本地汇总只是基于上一轮已同步快照重建；在下一次 `sync=ok` 前不要把进度变化当成新的远端真值。")
    if poll_interval is not None:
        interval_secs, reason = poll_interval
        lines.append(f"- 当前建议本地监控间隔：`{interval_secs}s` ({reason})。")
        next_monitor_after = recommended_next_monitor_after(sync_event, poll_interval)
        next_monitor_after_text = format_monitor_timestamp(next_monitor_after)
        if next_monitor_after_text is not None and sync_freshness == "fresh_remote":
            lines.append(
                f"- 建议下一次 task-conditioned 监控时间：`{next_monitor_after_text}`（latest authoritative refresh + {interval_secs}s）。"
            )
    lines.append(current_next_step_text(stage, task_status, round3))
    if round3:
        for row in round3:
            status = row["status"]
            if status == "failed" and row["failure_reason"]:
                status = f"{status} ({row['failure_reason']})"
            progress = row["progress"]
            line = f"- Round 3 `{row['model']}`：`{status}`"
            if progress is not None:
                epoch_text = f"，epoch {progress['epoch']}" if progress["epoch"] is not None else ""
                line += f"，当前 `{progress['phase']}`{epoch_text}，进度 `{progress['progress']}`"
            lines.append(line)
            timing_note = progress_timing_note(progress, language="cn")
            if timing_note is not None and row.get("status") == "running":
                lines.append(f"- {timing_note}")
            epoch_note = epoch_cycle_note(FORMAL_CANDIDATES.get(row["model"]), progress, language="cn")
            if epoch_note is not None and row.get("status") == "running":
                lines.append(f"- 多 epoch 监控说明：{epoch_note}")
        running_milestone_notes = []
        for row in round3:
            if row.get("status") != "running":
                continue
            milestone_note = stability_milestone_note(FORMAL_CANDIDATES.get(row["model"]), row.get("progress"), language="cn")
            if milestone_note:
                running_milestone_notes.append(milestone_note)
        for note in running_milestone_notes:
            lines.append(f"- {note}")
    if round3:
        round3_labels = []
        for row in round3:
            meta = FORMAL_CANDIDATES.get(row["model"], {})
            round3_labels.append(str(meta.get("label") or row["model"]))
        round3_label_text = " / ".join(round3_labels) if round3_labels else "Round 3 formal candidates"
        lines.extend(
            [
                "",
                "## 当前严格正式候选",
                "",
                (
                    f"- 当前 Round 3 strict formal queue 已全部完成，A 模块历史 checkpoint `{winner_model}` 已通过 audit；"
                    "controller / watcher 不再自动启动任何新候选，除非后续明确需要 strict full-train 复验。"
                    if winner_model
                    else f"- 当前 controller / watcher 只允许围绕 `{round3_label_text}` 这类完整训练候选工作，不再自动启动任何 diagnostic round；当前 formal queue 以本地权威 `current_formal_decision` 为准，active full-train 存在时只继续监控，训练完成后再统一 `finalize + review`。"
                ),
                "- formal winner 的唯一判定标准：SSH 上完整 `train + val + test` 结束后，Overall Acc 和 Overall AUC 都严格高于 baseline。",
                "- Final Acc / AUC 不改变上述 winner gate，但必须与 baseline 的两项 Final delta 一起完整落盘并通过 audit。",
                "",
            ]
        )
        lines.extend(build_round3_formal_table(round3, baseline))
        running_rows = [row for row in round3 if row.get("status") == "running"]
        for row in running_rows:
            meta = FORMAL_CANDIDATES.get(row["model"], {})
            if meta.get("implementation_guard_cn"):
                lines.extend(
                    [
                        "",
                        f"- 当前 active 候选 `{row['model']}` 的实现口径核对：{meta['implementation_guard_cn']}",
                    ]
                )
        failure_analysis = build_round3_failure_analysis(round3)
        if failure_analysis:
            lines.extend([""])
            lines.extend(failure_analysis)
        completed_formal_analysis = build_completed_formal_analysis(round3, audit_map)
        if completed_formal_analysis:
            lines.extend([""])
            lines.extend(completed_formal_analysis)
    lines.extend(["", "## 已同步结果", ""])

    if rows:
        lines.extend(build_table(sorted(rows, key=lambda row: (model_role(row["model"]) != "formal", -row["overall_auc"], -row["overall_acc"]))))
    else:
        lines.append("_No metrics synced yet._")

    lines.extend(["", "## 当前结论", ""])
    if best_formal is not None and baseline is not None:
        lines.append(
            f"- 当前 A 模块参考结果来自历史 checkpoint `{best_formal['model']}`，Overall Acc / AUC **{best_formal['overall_acc']:.2f} / {best_formal['overall_auc']:.2f}**，"
            f"相对 baseline 为 **{best_formal['overall_acc'] - baseline['overall_acc']:+.2f} / {best_formal['overall_auc'] - baseline['overall_auc']:+.2f}**。"
        )
    elif best_formal is not None:
        lines.append(
            f"- 当前 A 模块参考结果来自历史 checkpoint `{best_formal['model']}`，Overall Acc / AUC **{best_formal['overall_acc']:.2f} / {best_formal['overall_auc']:.2f}**。"
        )
    else:
        lines.append("- 目前还没有可用的正式结果。")

    if best_diagnostic is not None:
        lines.append(
            f"- 当前诊断最强结果是 `{best_diagnostic['model']}`，Overall Acc / AUC **{best_diagnostic['overall_acc']:.2f} / {best_diagnostic['overall_auc']:.2f}**，但它不计入正式 winner。"
        )

    lines.extend(
        [
            "- 当前不能把 `v15` 或 `v20` 当作正式方法分数。",
            (
                f"- 当前 A 模块参考结果已冻结，历史 checkpoint 为 `{winner_model}`；诊断轮次仍不计入正式方法，若未来需要复验必须重新走完整 SSH strict full-train workflow。"
                if winner_model
                else "- 只有完整训练候选全部完成，或失败候选完成修复后重跑，才允许更新正式 winner。"
            ),
            "",
            "## 下一步",
            "",
        ]
    )
    lines.extend(build_next_steps(stage, task_status, round3))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    parser.add_argument("--refresh_log", default="results/cel_stage1_last_layer/task_conditioned_refresh.log")
    parser.add_argument("--formal_audit_json", default="results/cel_stage1_last_layer/formal_experiment_audit.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.metrics_dir).glob("metrics_*.txt")):
        row = parse_metrics(path)
        if row is not None:
            rows.append(row)

    baseline = None
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.exists():
        baseline = parse_metrics(baseline_path)

    task_status = read_task_status(Path(args.task_status))
    formal_audit = read_formal_audit(Path(args.formal_audit_json))
    stage = infer_stage(task_status)
    sync_event = latest_refresh_event(Path(args.refresh_log))
    authoritative_sync_event = latest_successful_refresh_event(Path(args.refresh_log)) or sync_event
    sync_freshness = sync_freshness_label(sync_event)
    Path(args.record).write_text(
        build_record(
            rows,
            baseline,
            stage,
            task_status,
            sync_event,
            authoritative_sync_event,
            sync_freshness,
            formal_audit,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
