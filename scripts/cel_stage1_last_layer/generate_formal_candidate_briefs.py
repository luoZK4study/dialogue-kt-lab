#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from formal_candidate_registry import (
    FORMAL_CANDIDATE_ORDER,
    FORMAL_CANDIDATES,
    ROUND3_FAILURE_FOLLOWUPS_CN,
)
from task_conditioned_failure_utils import (
    epoch_cycle_note,
    format_progress_cn,
    latest_phase_progress_from_log,
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
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def round3_status_map(task_status: dict | None) -> dict[str, dict]:
    if not task_status:
        return {}
    for round_info in task_status.get("rounds", []):
        if str(round_info.get("title", "")).startswith("Round 3"):
            return {
                row["model"]: row
                for row in (round_info.get("models") or [])
                if row.get("model")
            }
    return {}


def audit_row_map(audit: dict | None) -> dict[str, dict]:
    if not audit:
        return {}
    return {row["model"]: row for row in (audit.get("rows") or []) if row.get("model")}


def status_text(status_row: dict | None) -> str:
    if not status_row:
        return "pending"
    status = status_row.get("status", "unknown")
    reason = status_row.get("failure_reason")
    if status == "failed" and reason:
        return f"{status} ({reason})"
    return status


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def build_analysis(status_row: dict | None, metrics: dict | None, baseline: dict | None, anchor: dict | None) -> dict | None:
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
    if metrics and baseline:
        analysis["delta_acc_vs_baseline"] = metrics["overall_acc"] - baseline["overall_acc"]
        analysis["delta_auc_vs_baseline"] = metrics["overall_auc"] - baseline["overall_auc"]
        if metrics.get("final_acc") is not None and baseline.get("final_acc") is not None:
            analysis["delta_final_acc_vs_baseline"] = metrics["final_acc"] - baseline["final_acc"]
        if metrics.get("final_auc") is not None and baseline.get("final_auc") is not None:
            analysis["delta_final_auc_vs_baseline"] = metrics["final_auc"] - baseline["final_auc"]
    if metrics and anchor:
        analysis["delta_acc_vs_anchor"] = metrics["overall_acc"] - anchor["overall_acc"]
        analysis["delta_auc_vs_anchor"] = metrics["overall_auc"] - anchor["overall_auc"]
    if any(value is not None for value in analysis.values()):
        return analysis
    return None


def current_decision_text(model_name: str, status_row: dict | None, audit_row: dict | None, baseline: dict | None, metrics: dict | None) -> str:
    audit_state = audit_row.get("audit") if audit_row else None
    if audit_state == "needs_rerun":
        return "- 当前结论：先保持方法范围不变，优先按同一 strict full-train 口径重跑该候选。"
    if audit_state == "recorded":
        if baseline and metrics and metrics["overall_acc"] > baseline["overall_acc"] and metrics["overall_auc"] > baseline["overall_auc"]:
            return "- 当前结论：该候选在已同步结果中满足双超 baseline，且已完成 recorded closeout；当前应保持 strict winner 结论冻结，而不是继续启动新 run。"
        return "- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。"
    if status_row and status_row.get("status") == "running":
        return "- 当前结论：继续按统一监控节奏观察，等完整 SSH 结果落盘后再做方法判断。"
    return "- 当前结论：该候选尚未形成新的正式结论，应继续遵守完整 SSH full-train -> 同步 -> 审计 -> 判断 的顺序。"


def recommended_command_lines(meta: dict, audit_row: dict | None, status_row: dict | None, all_audit_rows: list[dict]) -> list[str]:
    launch_key = meta["launch_key"]
    audit_state = str((audit_row or {}).get("audit", "pending"))
    beats_baseline = str((audit_row or {}).get("beats_baseline", "--"))
    status = str((status_row or {}).get("status", "pending"))
    rerun_queue = [
        str(row.get("label"))
        for row in all_audit_rows
        if row.get("audit") == "needs_rerun"
    ]

    lines = ["- 推荐命令："]

    if audit_state == "needs_rerun":
        lines.extend(
            [
                "  - 推荐 alias 模板：`bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`",
                "  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`",
                "  - 当前 formal queue 一键 alias 启动：`bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`",
                f"  - alias 启动：`bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - 直接启动：`bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {launch_key}`",
                "  - 当前 formal queue 一键手工 SSH fallback：`bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
                f"  - 手工 SSH fallback：`bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {launch_key}`",
                f"  - 共享远端启动入口：`bash scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh {launch_key}`",
                "  - alias 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
                "  - 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
                "  - 当前 formal queue 一键 alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
                "  - 当前 formal queue 一键 alias 回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
                f"  - alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - alias 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - 完成后落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {launch_key}`",
                f"  - 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {launch_key}`",
            ]
        )
        return lines

    if status == "running":
        lines.extend(
            [
                "  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`",
                "  - alias 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
                "  - 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
                "  - 当前 formal queue 一键 alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
                "  - 当前 formal queue 一键 alias 回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
                f"  - alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - alias 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - 完成后落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {launch_key}`",
                f"  - 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {launch_key}`",
            ]
        )
        return lines

    if audit_state == "recorded":
        lines.extend(
            [
                "  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`",
                "  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`",
                f"  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {launch_key}`",
                f"  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {launch_key}`",
                f"  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                f"  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
                "  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`",
            ]
        )
        if beats_baseline == "yes":
            lines.append(f"  - 当前候选已是 recorded strict winner；优先保持结论面冻结状态，而不是重新启动 `{launch_key}`。")
        elif rerun_queue:
            lines.append(
                f"  - 当前存在更高优先级 rerun 队列：`{','.join(rerun_queue)}`；在该队列清空前，不应重新启动 `{launch_key}`。"
            )
        else:
            lines.append(f"  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `{launch_key}`。")
        return lines

    lines.extend(
        [
            "  - 推荐 alias 模板：`bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`",
            "  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`",
            "  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`",
            "  - 当前 formal queue 一键 alias 启动：`bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`",
            f"  - alias 启动：`bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
            f"  - 直接启动：`bash scripts/cel_stage1_last_layer/start_formal_candidate.sh {launch_key}`",
            "  - 当前 formal queue 一键手工 SSH fallback：`bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`",
            f"  - 手工 SSH fallback：`bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {launch_key}`",
            f"  - 共享远端启动入口：`bash scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh {launch_key}`",
            "  - alias 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`",
            "  - 监控：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`",
            "  - 当前 formal queue 一键 alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`",
            "  - 当前 formal queue 一键落盘：`bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`",
            "  - 当前 formal queue 一键 alias 回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`",
            "  - 当前 formal queue 一键回看：`bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`",
            f"  - alias 落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
            f"  - alias 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {launch_key}`",
            f"  - 完成后落盘：`bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh {launch_key}`",
            f"  - 回看：`bash scripts/cel_stage1_last_layer/review_formal_candidate.sh {launch_key}`",
        ]
    )
    return lines


def build_report(baseline: dict | None, anchor: dict | None, task_status: dict | None, audit: dict | None) -> str:
    round3_map = round3_status_map(task_status)
    audit_map = audit_row_map(audit)
    audit_rows = (audit or {}).get("rows") or []

    lines = [
        "# Formal Candidate Briefs",
        "",
        "- 目的：把当前 strict full-train 正式候选的“唯一主改动 / 目标假设 / 期望观察 / 失败后如何解释”集中记录在一处。",
        "- 口径：只有 SSH 上完整 `train + val + test` 的结果才允许进入正式方法判断；任何 ckpt 后处理或冻结补训都只算诊断。",
        "",
        "## Current Gate",
        "",
    ]

    if baseline is not None:
        lines.append(
            f"- Formal winner gate: `Overall Acc > {baseline['overall_acc']:.2f}` and `Overall AUC > {baseline['overall_auc']:.2f}`."
        )
    if anchor is not None:
        lines.append(
            f"- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC **{anchor['overall_acc']:.2f} / {anchor['overall_auc']:.2f}**."
        )
    next_action = task_status.get("next_action") if task_status else "unknown"
    lines.append(f"- Current loop next action: `{next_action}`.")
    lines.append("- Rule: if any formal candidate is still `needs_rerun`, do not design a new method yet.")

    lines.extend(["", "## Current Queue", ""])
    lines.extend(
        [
            "| Candidate | Status | Audit | Current Result | Next Use |",
            "|---|---|---|---|---|",
        ]
    )
    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        status_row = round3_map.get(model_name)
        audit_row = audit_map.get(model_name)
        metrics = (audit_row or {}).get("metrics") or {}
        result_text = "--"
        if metrics:
            result_text = f"{metrics.get('overall_acc'):.2f}/{metrics.get('overall_auc'):.2f}"
        if (audit_row or {}).get("audit") == "needs_rerun":
            next_use = "rerun first"
        elif (status_row or {}).get("status") == "running":
            next_use = "monitor active run"
        else:
            next_use = "freeze / analyze"
        lines.append(
            f"| `{meta['label']}` | {status_text(status_row)} | `{(audit_row or {}).get('audit', 'pending')}` | {result_text} | {next_use} |"
        )

    lines.extend(["", "## Candidate Briefs", ""])
    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        status_row = round3_map.get(model_name)
        audit_row = audit_map.get(model_name)
        metrics = (audit_row or {}).get("metrics") or {}
        analysis = build_analysis(status_row, metrics if metrics else None, baseline, anchor)
        lines.extend(
            [
                f"### `{meta['label']}` / `{model_name}`",
                "",
                f"- 当前状态：`{status_text(status_row)}`",
                f"- Audit 状态：`{(audit_row or {}).get('audit', 'pending')}`",
                f"- 起点：{meta['start_point_cn']}",
                f"- 方法：{meta['method_cn']}",
                f"- 唯一主改动：{meta['single_variable_cn']}",
                f"- 目标假设：{meta['hypothesis_cn']}",
            ]
        )
        if meta.get("implementation_guard_cn"):
            lines.append(f"- 实现口径核对：{meta['implementation_guard_cn']}")
        lines.append("- 期望观察：")
        for signal in meta["expected_signals_cn"]:
            lines.append(f"  - {signal}")
        if metrics:
            lines.append(
                f"- 当前已同步结果：Overall Acc / AUC **{metrics.get('overall_acc'):.2f} / {metrics.get('overall_auc'):.2f}**，"
                f"Final Acc / AUC **{fmt_num(metrics.get('final_acc'))} / {fmt_num(metrics.get('final_auc'))}**。"
            )
        if analysis and analysis.get("pred_true") is not None:
            lines.append(f"- 当前 `Pred True`：**{analysis['pred_true']:.2f}%**。")
        if analysis and (analysis.get("best_acc") is not None or analysis.get("best_threshold") is not None):
            lines.append(
                f"- 当前 best-threshold 诊断：best Acc **{fmt_num(analysis.get('best_acc'))}**，threshold **{fmt_num(analysis.get('best_threshold'), 3)}**。"
            )
        if analysis and (analysis.get("delta_acc_vs_baseline") is not None or analysis.get("delta_auc_vs_baseline") is not None):
            lines.append(
                f"- 相对 baseline 的 Overall Acc / AUC 变化：**{analysis.get('delta_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_auc_vs_baseline', 0.0):+.2f}**。"
            )
        if analysis and (analysis.get("delta_final_acc_vs_baseline") is not None or analysis.get("delta_final_auc_vs_baseline") is not None):
            lines.append(
                f"- 相对 baseline 的 Final Acc / AUC 变化：**{analysis.get('delta_final_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_final_auc_vs_baseline', 0.0):+.2f}**。"
            )
        if analysis and (analysis.get("delta_acc_vs_anchor") is not None or analysis.get("delta_auc_vs_anchor") is not None):
            lines.append(
                f"- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**{analysis.get('delta_acc_vs_anchor', 0.0):+.2f} / {analysis.get('delta_auc_vs_anchor', 0.0):+.2f}**。"
            )
        progress = latest_phase_progress_from_log(Path(meta["stdout_log_rel"]))
        if progress is not None:
            lines.append(f"- 当前日志进度：{format_progress_cn(progress)}")
        epoch_note = epoch_cycle_note(meta, progress, language="cn")
        if epoch_note is not None:
            lines.append(f"- 多 epoch 监控说明：{epoch_note}")
        milestone_note = stability_milestone_note(meta, progress, language="cn")
        if milestone_note is not None:
            lines.append(f"- {milestone_note}")
        lines.append(f"- 若未双超时的下一轮解释：{meta['if_fail_cn']}")
        for followup in ROUND3_FAILURE_FOLLOWUPS_CN.get(model_name, []):
            lines.append(f"- 当前失败修复说明：{followup}")
        lines.append(current_decision_text(model_name, status_row, audit_row, baseline, metrics if metrics else None))
        lines.extend(recommended_command_lines(meta, audit_row, status_row, audit_rows))
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--anchor_metrics", default="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt")
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    parser.add_argument("--formal_audit_json", default="results/cel_stage1_last_layer/formal_experiment_audit.json")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md")
    args = parser.parse_args()

    baseline = parse_metrics(Path(args.baseline_metrics))
    anchor = parse_metrics(Path(args.anchor_metrics))
    task_status = load_json(Path(args.task_status))
    audit = load_json(Path(args.formal_audit_json))
    Path(args.out).write_text(build_report(baseline, anchor, task_status, audit), encoding="utf-8")


if __name__ == "__main__":
    main()
