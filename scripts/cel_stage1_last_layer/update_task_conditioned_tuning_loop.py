#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from formal_candidate_registry import (
    FORMAL_CANDIDATES,
    FORMAL_CANDIDATE_MODELS,
    FORMAL_CANDIDATE_LOG_FILES,
    ROUND3_FAILURE_FOLLOWUPS_CN,
    build_formal_launch_script_lines,
    build_round3_description_cn,
)
from formal_queue_state import launch_key_for_model
from task_conditioned_failure_utils import (
    extract_failure_evidence,
    format_progress_cn,
    read_log_excerpt,
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


ROUND_GROUPS = [
    {
        "title": "Round 1: Diagnostic Post-hoc Calibration",
        "description": [
            "这一轮只用于诊断 `v1` 的概率偏移问题，不计入“完整方法最终指标”。",
            "",
            "1. `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`",
            "   - baseline LoRA + `v1` selector",
            "   - 冻结主体，只训练 bias calibrator",
            "2. `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`",
            "   - baseline LoRA + `v1` selector",
            "   - 冻结主体，只训练 affine calibrator",
            "3. `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b`",
            "   - 从完整 `task_conditioned v1` checkpoint 起步",
            "   - 冻结主体，只训练 bias calibrator",
            "4. `cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b`",
            "   - 从完整 `task_conditioned v1` checkpoint 起步",
            "   - 冻结主体，只训练 affine calibrator",
            "",
            "设计意图：",
            "",
            "- 判断 `v1` 的主要短板是否真的是概率偏保守",
            "- 估计“只调输出刻度”最多能带来多少 Acc 收益",
            "- 这些结果只作为诊断/上限，不作为最终正式方法分数",
        ],
        "models": [
            "cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b",
        ],
    },
    {
        "title": "Round 2: Diagnostic Frozen / Partial Retraining",
        "description": [
            "这一轮仍不属于严格 end-to-end 完整方法训练，只作为补充诊断。",
            "",
            "1. `cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b`",
            "   - 从完整 `task_conditioned v1` checkpoint 起步",
            "   - 冻结 backbone，仅联训 selector + bias calibrator",
            "2. `cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b`",
            "   - 从完整 `task_conditioned v1` checkpoint 起步",
            "   - 冻结 backbone，仅联训 selector + affine calibrator",
            "3. `cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b`",
            "   - `v13` 的显存修正版",
            "4. `cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b`",
            "   - 固定 bias 的纯诊断评估，不训练",
            "",
            "设计意图：",
            "",
            "- 判断冻结 backbone 后，selector/calibrator 微调是否还能稳定改进",
            "- 这些结果也不计入最终正式方法分数",
        ],
        "models": [
            "cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b",
        ],
    },
    {
        "title": "Round 3: Valid End-to-End Full Training",
        "description": build_round3_description_cn(),
        "models": list(FORMAL_CANDIDATE_MODELS),
    },
]

VALID_END_TO_END_MODELS = set(FORMAL_CANDIDATE_MODELS)
WINNER_REQUIRED_SURFACES = [
    {
        "label": "strict report",
        "path": "results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
    },
    {
        "label": "stage1 record",
        "path": "results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
    },
    {
        "label": "tuning loop",
        "path": "results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
    },
    {
        "label": "comparison",
        "path": "results/cel_stage1_last_layer/COMPARISON.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
    },
    {
        "label": "detailed analysis",
        "path": "results/cel_stage1_last_layer/DETAILED_ANALYSIS.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
    },
]

MODEL_LOG_FILES = {
    "cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b": "task_conditioned_v11_cal_bias_only.stdout.log",
    "cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b": "task_conditioned_v12_cal_affine_only.stdout.log",
    "cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b": "task_conditioned_v13_selector_cal_bias.stdout.log",
    "cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b": "task_conditioned_v14_selector_cal_affine.stdout.log",
    "cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b": "task_conditioned_v15_fullv1_cal_bias_only.stdout.log",
    "cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b": "task_conditioned_v16_fullv1_cal_affine_only.stdout.log",
    "cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b": "task_conditioned_v17_selector_cal_bias_memfix.stdout.log",
    "cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b": "task_conditioned_v18_fullv1_fixed_bias_eval.stdout.log",
    "cel_task_conditioned_lastlayer_v19_fullv1_cal_bias_only_tinylr_qwen3_1.7b": "task_conditioned_v19_fullv1_cal_bias_only_tinylr.stdout.log",
    "cel_task_conditioned_lastlayer_v20_fullv1_valfit_bias_qwen3_1.7b": "task_conditioned_v20_fullv1_valfit_bias.stdout.log",
    **FORMAL_CANDIDATE_LOG_FILES,
}

TERMINAL_STATUSES = {"done", "failed"}


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


def parse_probs(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = (row.get("Correct") or "").strip().lower()
            prob_text = (row.get("Prob") or "").strip()
            if correct not in {"true", "false"} or prob_text in {"", "--"}:
                continue
            rows.append((1 if correct == "true" else 0, float(prob_text)))
    return rows


def acc_at_threshold(rows: list[tuple[int, float]], threshold: float) -> float:
    return 100.0 * sum(((prob >= threshold) == bool(label)) for label, prob in rows) / len(rows)


def best_threshold_acc(rows: list[tuple[int, float]]) -> tuple[float, float]:
    thresholds = sorted(set(prob for _, prob in rows))
    best_acc = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        acc = acc_at_threshold(rows, threshold)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_acc, best_threshold


def pred_true_pct(rows: list[tuple[int, float]], threshold: float = 0.5) -> float:
    return 100.0 * sum(prob >= threshold for _, prob in rows) / len(rows)


def inspect_log_status(model_name: str, qual_dir: Path) -> dict:
    log_name = MODEL_LOG_FILES.get(model_name)
    if not log_name:
        return {"started": False, "failed": False, "failure_reason": None}
    log_path = qual_dir.parent / log_name
    text = read_log_excerpt(log_path)
    if not text:
        return {"started": False, "failed": False, "failure_reason": None}
    started = any(marker in text for marker in ("Arguments:", "Epoch 1", "Training:", "Testing:", "Validating:", "Validation:"))
    failure_reason = extract_failure_evidence(log_path)["failure_reason"]
    return {
        "started": started,
        "failed": started and failure_reason is not None,
        "failure_reason": failure_reason,
    }


def build_candidate_row(name: str, metrics_dir: Path, qual_dir: Path) -> dict:
    metrics_path = metrics_dir / f"metrics_{name}.txt"
    qual_path = qual_dir / f"qual_{name}.csv"
    metrics = parse_metrics(metrics_path)
    probs = parse_probs(qual_path)
    if metrics is None:
        log_status = inspect_log_status(name, qual_dir)
        if log_status["failed"]:
            return {
                "model": name,
                "status": "failed",
                "failure_reason": log_status["failure_reason"],
            }
        if log_status["started"]:
            return {
                "model": name,
                "status": "running",
            }
        return {
            "model": name,
            "status": "pending",
        }
    best_acc = None
    best_threshold = None
    pred_true = None
    if probs:
        best_acc, best_threshold = best_threshold_acc(probs)
        pred_true = pred_true_pct(probs)
    return {
        "model": name,
        "status": "done",
        "overall_acc": metrics["overall_acc"],
        "overall_auc": metrics["overall_auc"],
        "final_acc": metrics["final_acc"],
        "final_auc": metrics["final_auc"],
        "pred_true": pred_true,
        "best_acc": best_acc,
        "best_threshold": best_threshold,
    }


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def is_valid_end_to_end_model(model_name: str) -> bool:
    return model_name in VALID_END_TO_END_MODELS


def log_path_for_model(model_name: str, qual_dir: Path) -> Path | None:
    log_name = MODEL_LOG_FILES.get(model_name)
    if not log_name:
        return None
    return qual_dir.parent / log_name


def metric_tokens(row: dict, fields: tuple[str, ...]) -> list[str]:
    tokens: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        tokens.append(f"{value:.2f}")
    return tokens


def winner_artifacts_ready(row: dict, metrics_dir: Path, qual_dir: Path) -> bool:
    metrics_path = metrics_dir / f"metrics_{row['model']}.txt"
    qual_path = qual_dir / f"qual_{row['model']}.csv"
    log_path = log_path_for_model(row["model"], qual_dir)
    ready = (
        metrics_path.exists()
        and qual_path.exists()
        and log_path is not None
        and log_path.exists()
    )
    meta = FORMAL_CANDIDATES.get(row["model"]) or {}
    if meta.get("launch_key") == "v26":
        audit_path = qual_dir.parent / "V26_SELFTRAINED_AUDIT.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        ready = ready and bool(audit.get("audit_pass"))
    return ready


def has_complete_metric_set(row: dict) -> bool:
    return all(
        row.get(field) is not None
        for field in ("overall_acc", "overall_auc", "final_acc", "final_auc")
    )


def winner_markdown_coverage_ready(row: dict) -> bool:
    model = row["model"]
    for surface in WINNER_REQUIRED_SURFACES:
        text = read_text(Path(surface["path"]))
        if not text or model not in text:
            return False
        tokens = metric_tokens(row, surface["metric_fields"])
        if any(token not in text for token in tokens):
            return False
    return True


def is_audit_ready_winner(row: dict, baseline_acc: float, baseline_auc: float, metrics_dir: Path, qual_dir: Path) -> bool:
    if row["status"] != "done":
        return False
    if not is_valid_end_to_end_model(row["model"]):
        return False
    if not has_complete_metric_set(row):
        return False
    if row["overall_acc"] <= baseline_acc or row["overall_auc"] <= baseline_auc:
        return False
    if not winner_artifacts_ready(row, metrics_dir, qual_dir):
        return False
    return winner_markdown_coverage_ready(row)


def best_done(rows: list[dict], baseline_acc: float, baseline_auc: float) -> dict | None:
    done = [row for row in rows if row["status"] == "done" and is_valid_end_to_end_model(row["model"])]
    if not done:
        return None
    return max(
        done,
        key=lambda row: (
            row["overall_acc"] > baseline_acc and row["overall_auc"] > baseline_auc,
            row["overall_acc"] > baseline_acc,
            row["overall_auc"],
            row["overall_acc"],
        ),
    )


def best_non_winner(rows: list[dict], baseline_acc: float, baseline_auc: float) -> dict | None:
    done = [row for row in rows if row["status"] == "done" and is_valid_end_to_end_model(row["model"])]
    if not done:
        return None
    return max(
        done,
        key=lambda row: (
            row["overall_auc"] > baseline_auc,
            row["overall_acc"] > baseline_acc,
            row["overall_auc"],
            row["overall_acc"],
        ),
    )


def improves_over_anchor(row: dict, anchor_acc: float, anchor_auc: float) -> bool:
    if row["status"] != "done":
        return False
    if not is_valid_end_to_end_model(row["model"]):
        return False
    return row["overall_acc"] > anchor_acc or row["overall_auc"] > anchor_auc


def build_round_table(rows: list[dict], baseline: dict) -> list[str]:
    baseline_acc = baseline["overall_acc"]
    baseline_auc = baseline["overall_auc"]
    baseline_final_acc = baseline.get("final_acc")
    baseline_final_auc = baseline.get("final_auc")
    lines = [
        "| Model | Status | Overall Acc | Overall AUC | Final Acc | Final AUC | Pred True | Best Acc | Best Threshold | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Overall Gate | Final Pair |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if row["status"] != "done":
            status = row["status"]
            if status == "failed" and row.get("failure_reason"):
                status = f"failed ({row['failure_reason']})"
            lines.append(f"| `{row['model']}` | {status} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        display_status = "done" if is_valid_end_to_end_model(row["model"]) else "diagnostic"
        delta_final_acc = "--"
        delta_final_auc = "--"
        final_pair = "--"
        if row.get("final_acc") is not None and baseline_final_acc is not None:
            delta_final_acc = f"{row['final_acc'] - baseline_final_acc:+.2f}"
        if row.get("final_auc") is not None and baseline_final_auc is not None:
            delta_final_auc = f"{row['final_auc'] - baseline_final_auc:+.2f}"
        if row.get("final_acc") is not None and row.get("final_auc") is not None and baseline_final_acc is not None and baseline_final_auc is not None:
            final_acc_win = row["final_acc"] > baseline_final_acc
            final_auc_win = row["final_auc"] > baseline_final_auc
            final_pair = "yes" if final_acc_win and final_auc_win else "partial" if final_acc_win or final_auc_win else "no"
        overall_gate = "yes" if row["overall_acc"] > baseline_acc and row["overall_auc"] > baseline_auc else "no"
        lines.append(
            f"| `{row['model']}` | {display_status} | {fmt_num(row['overall_acc'])} | {fmt_num(row['overall_auc'])} | "
            f"{fmt_num(row['final_acc'])} | {fmt_num(row['final_auc'])} | {fmt_num(row['pred_true'])}% | "
            f"{fmt_num(row['best_acc'])} | {fmt_num(row['best_threshold'], 3)} | "
            f"{row['overall_acc'] - baseline_acc:+.2f} | {row['overall_auc'] - baseline_auc:+.2f} | "
            f"{delta_final_acc} | {delta_final_auc} | {overall_gate} | {final_pair} |"
        )
    return lines


def summarize_round_decision(
    rows: list[dict],
    baseline_acc: float,
    baseline_auc: float,
    success_line: str,
    failure_hint: str,
    pending_hint: str,
) -> list[str]:
    winners = [
        row for row in rows
        if row["status"] == "done"
        and is_valid_end_to_end_model(row["model"])
        and has_complete_metric_set(row)
        and row["overall_acc"] > baseline_acc
        and row["overall_auc"] > baseline_auc
    ]
    if winners:
        best = max(winners, key=lambda row: (row["overall_acc"], row["overall_auc"]))
        return [
            f"- {success_line}：`{best['model']}` (Acc **{best['overall_acc']:.2f}**, AUC **{best['overall_auc']:.2f}**)。"
        ]

    done = [row for row in rows if row["status"] == "done" and is_valid_end_to_end_model(row["model"])]
    failed = [row for row in rows if row["status"] == "failed"]
    if done:
        best = best_done(rows, baseline_acc, baseline_auc)
        assert best is not None
        raw_metric_win = best["overall_acc"] > baseline_acc and best["overall_auc"] > baseline_auc
        lines = [
            f"- 本轮已有结果，但尚未形成 audit-ready 的严格 winner。当前最好候选：`{best['model']}` "
            f"(Acc **{best['overall_acc']:.2f}**, AUC **{best['overall_auc']:.2f}**)。"
        ]
        if raw_metric_win:
            lines.append("- 该候选的原始指标已经双超 baseline，但在 `FORMAL_EXPERIMENT_AUDIT.md` 干净前仍不能宣布为正式 winner。")
        if failed:
            failed_desc = "、".join(
                f"`{row['model']}` ({row.get('failure_reason', 'failed')})"
                for row in failed
            )
            lines.append(f"- 已失败候选：{failed_desc}。")
        lines.append(f"- {failure_hint}")
        return lines

    running = [row for row in rows if row["status"] == "running"]
    if running:
        running_names = "、".join(f"`{row['model']}`" for row in running)
        lines = [f"- 当前仍在运行：{running_names}；结果落盘后自动刷新。"]
        if failed:
            failed_desc = "、".join(
                f"`{row['model']}` ({row.get('failure_reason', 'failed')})"
                for row in failed
            )
            lines.append(f"- 已失败候选：{failed_desc}。")
        return lines

    if failed:
        failed_desc = "、".join(
            f"`{row['model']}` ({row.get('failure_reason', 'failed')})"
            for row in failed
        )
        return [
            f"- 已失败候选：{failed_desc}。",
            f"- {pending_hint}",
        ]

    return [f"- {pending_hint}"]


def build_round3_failure_analysis(rows: list[dict]) -> list[str]:
    failed_rows = [row for row in rows if row["status"] == "failed"]
    if not failed_rows:
        return []

    lines = ["## Active Round 3 Failure Analysis", ""]
    for row in failed_rows:
        lines.append(f"### `{row['model']}`")
        lines.append("")
        log_name = MODEL_LOG_FILES.get(row["model"])
        evidence = extract_failure_evidence((Path("results/cel_stage1_last_layer") / log_name) if log_name else Path())
        notes = []
        progress_text = format_progress_cn(evidence["progress"])
        if progress_text is not None:
            notes.append(f"最新已同步失败点：{progress_text}。")
        if evidence["assertion_line"] is not None:
            notes.append(f'日志关键断言："{evidence["assertion_line"]}"')
        if evidence.get("assertion_site") is not None:
            notes.append(f"断言位置：`{evidence['assertion_site']}`")
        if evidence["runtime_line"] is not None:
            notes.append(f"日志关键报错：`{evidence['runtime_line']}`")
        if not notes:
            notes.append(f"当前失败类型：`{row.get('failure_reason', 'failed')}`；需要在保持 strict full-train 口径不变的前提下修复并重跑。")
        notes.extend(ROUND3_FAILURE_FOLLOWUPS_CN.get(row["model"], []))
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return lines


def build_markdown(
    baseline: dict,
    baseline_pred_true: float | None,
    v1: dict,
    v1_best_acc: float | None,
    v1_best_threshold: float | None,
    round_rows: list[list[dict]],
    metrics_dir: Path,
    qual_dir: Path,
) -> str:
    baseline_acc = baseline["overall_acc"]
    baseline_auc = baseline["overall_auc"]

    all_rows = [row for rows in round_rows for row in rows]
    all_winners = [
        row for row in all_rows
        if is_audit_ready_winner(row, baseline_acc, baseline_auc, metrics_dir, qual_dir)
    ]
    best_winner = max(all_winners, key=lambda row: (row["overall_acc"], row["overall_auc"])) if all_winners else None
    improving_rows = [row for row in all_rows if improves_over_anchor(row, v1["overall_acc"], v1["overall_auc"])]
    global_best = best_non_winner(improving_rows, baseline_acc, baseline_auc) if improving_rows else None
    round3_rows = round_rows[2]
    round3_running = [row for row in round3_rows if row["status"] == "running"]
    round3_failed = [row for row in round3_rows if row["status"] == "failed"]
    round3_pending = [row for row in round3_rows if row["status"] == "pending"]
    running_round3_names = "、".join(f"`{row['model']}`" for row in round3_running)
    failed_round3_names = "、".join(f"`{row['model']}`" for row in round3_failed)

    if best_winner is not None:
        objective_summary = (
            "本记录保留历史 task-conditioned 调优结果；独立初始化的 A 模块历史 candidate "
            "已经完成 strict full-train、正式测试、来源/参数审计与 closeout。"
        )
        formal_status_lines = [
            "- 旧 `v26 fullv1` 结果已撤回，原因是最终测试 checkpoint 来自 calibrator-only warmup。",
            f"- 当前 strict formal winner：`{best_winner['model']}`。",
            f"- Overall Acc / AUC：**{best_winner['overall_acc']:.2f} / {best_winner['overall_auc']:.2f}**；"
            f"Final Acc / AUC：**{fmt_num(best_winner.get('final_acc'))} / {fmt_num(best_winner.get('final_auc'))}**。",
            "- A 模块从 Qwen3-1.7B 原始基座建立自己的 bootstrap，后续阶段只加载同一 candidate 上一阶段的 checkpoint；baseline 与 v1 仅用于比较。",
        ]
    else:
        objective_summary = (
            "本记录保留历史 task-conditioned 调优结果；当前正在按独立初始化口径训练 A 模块历史 candidate，"
            "该 candidate 不加载 baseline 或 v1 checkpoint。"
        )
        formal_status_lines = [
            "- 旧 `v26 fullv1` 结果已撤回，原因是最终测试 checkpoint 来自 calibrator-only warmup。",
            "- A 模块历史 candidate 必须完成 bootstrap -> calibrator warmup -> strict joint，并通过来源与参数变化审计后才可参与 winner 判断。",
        ]

    lines = [
        "# Task-Conditioned Tuning Loop",
        "",
        "## 目标",
        "",
        objective_summary,
        "",
        f"- Baseline Overall AUC: **{baseline_auc:.2f}**",
        f"- Baseline Overall Acc: **{baseline_acc:.2f}**",
        f"- Baseline Final AUC: **{fmt_num(baseline.get('final_auc'))}**",
        f"- Baseline Final Acc: **{fmt_num(baseline.get('final_acc'))}**",
        "",
        "历史 selector/ranking anchor（comparison-only，不是 v26 checkpoint 起点）：",
        "",
        "- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`",
        f"- Overall AUC: **{v1['overall_auc']:.2f}**",
        f"- Overall Acc: **{v1['overall_acc']:.2f}**",
        f"- Final AUC: **{fmt_num(v1.get('final_auc'))}**",
        f"- Final Acc: **{fmt_num(v1.get('final_acc'))}**",
        "",
        "当前正式状态：",
        "",
        *formal_status_lines,
        "",
        "## 当前分析",
        "",
        "当前 `v1` 的关键现象不是“排序不够”，而是“概率偏保守”：",
        "",
        f"- `Pred True`: **{fmt_num(v1['pred_true'])}%**",
        f"- baseline `Pred True`: **{fmt_num(baseline_pred_true)}%**",
        f"- `v1` 的最佳 Acc 阈值约为 **{fmt_num(v1_best_threshold, 3)}**",
        f"- `v1` 的最佳理论 Acc 约为 **{fmt_num(v1_best_acc)}**",
        "- baseline 的最佳 Acc 阈值约为 **0.488**",
        "- 离线 logit 扫描表明：如果只做保守 bias 平移，`v1` 的 `0.5` 阈值下 Acc 理论上可到约 **69.67**，而 affine 没有额外增益",
        "",
        "这说明：",
        "",
        "- 当前 `task_conditioned v1` 的排序能力已经优于 baseline",
        "- 但输出分数整体偏低，导致固定 `0.5` 阈值下 Recall/Acc 被压住",
        "- 因此历史上先做了一批 calibration / frozen 微调诊断，但这些结果不再计入正式方法胜负判断",
        "",
    ]

    for idx, (group, rows) in enumerate(zip(ROUND_GROUPS, round_rows)):
        lines.extend([f"## {group['title']}", ""])
        lines.extend(group["description"])
        lines.extend(["", f"### {group['title']} Results", ""])
        lines.extend(build_round_table(rows, baseline))
        lines.extend([""])

        if idx == 0:
            lines.extend(["### Round 1 Decision", ""])
            lines.extend([
                "- 这一轮只提供诊断证据，不计入正式 end-to-end 方法结果。",
                "- `v15` 说明：如果只做后续 bias 校准，`v1` 的 Acc 确实可以被拉回 baseline 之上；但这种做法不属于当前严格口径下的最终方法。",
            ])
        elif idx == 1:
            lines.extend(["### Round 2 Decision", ""])
            lines.extend([
                "- 这一轮也只提供诊断证据，不计入正式 end-to-end 方法结果。",
                "- 当前最重要的信息是：冻结/部分微调路线没有形成可直接汇报的完整方法闭环。",
            ])
        else:
            lines.extend(["### Round 3 Decision", ""])
            lines.extend(
                summarize_round_decision(
                    rows,
                    baseline_acc,
                    baseline_auc,
                    success_line="Round 3 已经找到符合严格口径的 end-to-end 候选",
                    failure_hint="若 Round 3 仍失败，下一步应继续设计新的完整联合训练方案，而不是回退到后处理校准。",
                    pending_hint="Round 3 是当前正式方法主线；只看这里的结果是否能形成新的有效 winner。",
                )
            )
        lines.extend([""])

    failure_analysis = build_round3_failure_analysis(round3_rows)
    if failure_analysis:
        lines.extend(failure_analysis)

    lines.extend(["## Global Decision", ""])
    if all_winners:
        lines.append(
            f"- 当前已有符合严格口径的模型同时超过 baseline 的 Acc/AUC：`{best_winner['model']}` "
            f"(Acc **{best_winner['overall_acc']:.2f}**, AUC **{best_winner['overall_auc']:.2f}**)。"
        )
        lines.append("- `task_conditioned` Stage 1 调优目标已完成；停止新增 formal candidate，后续只做结果总结、误差分析和必要的 strict 复验。")
    elif global_best is not None:
        lines.append(
            f"- 当前最值得继续参考的严格口径候选是：`{global_best['model']}` "
            f"(Acc **{global_best['overall_acc']:.2f}**, AUC **{global_best['overall_auc']:.2f}**)，但尚未同时超过 baseline。"
        )
        lines.append("- tuning loop 继续保持开启，优先完成当前 end-to-end 轮次，再决定是否要进入更强结构改动。")
    else:
        lines.append(
            f"- 当前严格口径下的最强锚点仍是：`{v1['model']}` "
            f"(Acc **{v1['overall_acc']:.2f}**, AUC **{v1['overall_auc']:.2f}**)；历史诊断轮次尚未产生可替代它的有效 end-to-end 候选。"
        )
        lines.append("- tuning loop 继续保持开启，优先完成当前 end-to-end 轮次，再决定是否要进入更强结构改动。")

    next_section_title = "Winner Closeout" if all_winners else "Prepared Next Round"
    lines.extend(["", f"## {next_section_title}", ""])
    lines.append("- 历史上的 `v11 / v12 / v15 / v16 / v19 / v20` 全部降级为诊断或后处理结果，不再作为正式 winner 判定依据。")
    if all_winners and best_winner is not None:
        lines.append(f"- 当前 strict formal winner 已固定为 `{best_winner['model']}`；当前 `task_conditioned` 调优停止新增 formal candidate，也不再自动重启任何旧候选。")
    elif round3_running and round3_failed:
        lines.append(f"- 当前正式主线先继续监控仍在运行的 {running_round3_names}，并在修复后重跑 {failed_round3_names}。")
    elif round3_running:
        lines.append(f"- 当前正式主线只看仍在运行的完整联合训练：{running_round3_names}。")
    elif round3_failed:
        lines.append(f"- 当前正式主线是修复并重跑已失败的完整联合训练候选：{failed_round3_names}。")
    else:
        lines.append("- 当前正式主线只看 strict full-train formal queue；若存在 rerun obligation，则必须先处理当前 queue 允许的候选。")
    explicit_formal_key = None
    for row in round3_rows:
        if row.get("status") == "failed":
            explicit_formal_key = launch_key_for_model(row.get("model"))
            if explicit_formal_key:
                break
    if explicit_formal_key is None and best_winner is not None:
        explicit_formal_key = launch_key_for_model(best_winner.get("model"))
    if explicit_formal_key is None:
        for row in round3_rows:
            if row.get("status") == "running":
                explicit_formal_key = launch_key_for_model(row.get("model"))
                if explicit_formal_key:
                    break
    if explicit_formal_key is None and round3_rows:
        explicit_formal_key = launch_key_for_model(round3_rows[-1].get("model"))
    explicit_formal_key = explicit_formal_key or "v21"
    if all_winners and best_winner is not None:
        lines.extend([
            f"- 如需回看当前 winner，可直接使用：`scripts/cel_stage1_last_layer/review_formal_candidate.sh {explicit_formal_key}`。",
            f"- 若当前在 WSL alias 路径下回看，也可用：`scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_formal_key}`。",
            f"- 如需再次确认 winner closeout，可使用：`scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 {explicit_formal_key}`。",
            "- Stage 2 首轮旧单路径 environment-residual 实验已完成并审计，但只保留为历史 pilot；当前工作重点是实现 `h_r/h_m` 双路径目标、tensor contracts 与新 audit 路径。",
            "- 已准备好的底层候选脚本：",
        ])
    elif not round3_running and not round3_failed and not round3_pending and round3_rows:
        lines.extend([
            "- 当前 Round 3 formal queue 已全部完成且都只是 recorded non-winner；现在不应继续自动启动旧候选。",
            "- 推荐先打印本地 WSL alias 模板：`scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`。",
            "- 先读取当前权威 formal 下一步：`scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`。",
            f"- 显式回看最新落地的 recorded non-winner：`scripts/cel_stage1_last_layer/review_formal_candidate.sh {explicit_formal_key}`。",
            f"- 若当前在 WSL alias 路径下回看，也可用：`scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 {explicit_formal_key}`。",
            "- 下一步应通过 `scripts/cel_stage1_last_layer/scaffold_formal_candidate.py` 设计新的单变量 strict formal candidate，再重新做 preflight / sync / SSH launch。",
            "- 已准备好的底层候选脚本：",
        ])
    else:
        lines.extend([
            "- 若当前存在 traceback 失败，先同步更新后的 `dialogue_kt/training.py`，再按同一 strict full-train 配置在 SSH 上重跑。",
            "- 推荐先打印本地 WSL alias 模板：`scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`。",
            "- 已准备好的当前 formal queue 一键 alias 动作入口：`scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`。",
            "- 若当前环境不能 SSH、但想先验证它会执行哪条命令，可使用：`scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`。",
            "- 已准备好的当前 formal queue 一键 alias 启动入口：`scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090`。",
            f"- 已准备好的显式 formal alias 启动入口：`scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 {explicit_formal_key}`。",
            "- 已准备好的 alias 监控入口：`scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once`。",
            f"- 仍可直接使用正式启动入口：`scripts/cel_stage1_last_layer/start_formal_candidate.sh {explicit_formal_key}`。",
            "- 若当前执行环境不能直接打开 SSH，则先打印当前 formal queue 一键手工 fallback：`scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`。",
            f"- 若要显式指定当前 rerun 目标，也可打印候选级手工 fallback：`scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh {explicit_formal_key}`。",
            f"- 手工 SSH fallback 与自动 launcher 现在共享同一个远端启动入口：`scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh {explicit_formal_key}`。",
            "- 训练完成后可直接使用当前 formal queue 一键落盘入口：`scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`。",
            "- 训练完成后也可直接使用当前 formal queue 一键 alias 落盘入口：`scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090`。",
            "- 训练完成后可直接使用当前 formal queue 一键回看入口：`scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`。",
            "- 训练完成后也可直接使用当前 formal queue 一键 alias 回看入口：`scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090`。",
            "- 已准备好的底层候选脚本：",
        ])
    lines.extend(build_formal_launch_script_lines())
    lines.extend([
        "- 诊断脚本仍保留，但仅作参考，不参与正式 winner 判定。",
        "- 设计意图：直接验证“把 bias calibrator 当成方法一部分、完整训练后是否能双超 baseline”。",
    ])

    lines.extend(["", "## Strict Full-Train Execution Loop", ""])
    lines.extend([
        "1. 只有当代码改动、启动脚本和记录口径都同步完成后，才允许启动新的 `task_conditioned` 正式候选。",
        "2. 正式候选必须在 SSH 服务器上完成完整 `train + val + test`；冻结补训、fixed-bias eval、validation-fit bias 都只算诊断。",
        "3. 训练运行后，优先使用 `scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh` 做统一 closeout：sync `metrics / qual / stdout log`、刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。",
        "4. closeout 完成后先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否已经对已完成 formal candidate 给出完整 artifact + markdown coverage。",
        "5. 只有完整结果全部落盘且 audit 干净后，才通过 `review_current_formal_candidate.sh` 判断是否双超 baseline；中途的 val 感觉更好、threshold 更好，都不能提前宣布 winner。",
        "6. 若本轮失败或未双超，记录 Overall/Final 四项指标、相对 baseline 的四项 delta、失败类型、`Pred True`，以及下一轮只改一个主要变量的原因。",
    ])

    lines.extend(["", "## Monitoring Cadence", ""])
    if round3_running and round3_failed:
        lines.extend([
            f"- 当前对仍在运行的 {running_round3_names} 采用共享自适应监控策略：长时训练早期可放宽到 **900s**，稳定训练中期通常为 **600s**，进入 `Validation / Testing`、达到 **95%+** 进度或剩余 ETA 已较短时通常收紧到 **300s**，而 `<=5m` 的极短剩余 ETA 会进一步收紧到 **120s**。",
            f"- {failed_round3_names} 在代码修复完成前不自动重启，避免重复写入同类 traceback。",
            "- 推荐命令：`TASK_CONDITIONED_CONTROLLER_SLEEP_SECS=600 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh`。",
            "- 若当前在 WSL 里依赖 SSH alias，也可用：`TASK_CONDITIONED_CONTROLLER_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh 3090`。",
            "- 若需要在本地长期挂起完整 formal loop，推荐后台入口：`bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090`。",
            "- 只做本地同步观察时，推荐命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`。",
            "- 若当前在 WSL 里依赖 SSH alias，推荐命令：`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`。",
            "- 实际当前建议间隔以 `STATUS.md` 或 `STRICT_FULL_TRAIN_REPORT.md` 里实时生成的 Recommended poll interval 为准。",
        ])
    elif all_winners and not round3_running:
        lines.extend([
            "- 当前没有 active strict full-train run；winner 已完成 closeout，本地 controller / watcher 无需持续轮询，推荐监控间隔为 `not_applicable`。",
            "- 如需核对 winner 工件或文档一致性，优先使用一次性的 `review` / `finalize` / `sync` 命令，而不是重新启动 controller。",
            "- 只有未来明确发起新的 strict SSH full-train 复验时，才恢复共享自适应监控策略：训练早期 **900s**，稳定中期 **600s**，进入 `Validation / Testing`、达到 **95%+** 进度或剩余 ETA 很短时收紧到 **300s**，而 `<=5m` 时进一步收紧到 **120s**。",
        ])
    else:
        lines.extend([
            "- 当前若重新进入 active `task_conditioned` strict full-train run，本地监控间隔仍采用共享自适应策略：长时训练早期可放宽到 **900s**，稳定训练中期通常为 **600s**，进入 `Validation / Testing`、达到 **95%+** 进度或剩余 ETA 已较短时通常收紧到 **300s**，而 `<=5m` 的极短剩余 ETA 会进一步收紧到 **120s**。",
            "- 推荐命令：`TASK_CONDITIONED_CONTROLLER_SLEEP_SECS=600 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh`。",
            "- 若当前在 WSL 里依赖 SSH alias，也可用：`TASK_CONDITIONED_CONTROLLER_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh 3090`。",
            "- 若需要在本地长期挂起完整 formal loop，推荐后台入口：`bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090`。",
            "- 只做本地同步观察时，推荐命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`。",
            "- 若当前在 WSL 里依赖 SSH alias，推荐命令：`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`。",
            "- 实际当前建议间隔以 `STATUS.md` 或 `STRICT_FULL_TRAIN_REPORT.md` 里实时生成的 Recommended poll interval 为准。",
        ])

    lines.extend(["", "## 记录要求", ""])
    lines.extend([
        "- 每轮结束后补充：",
        "  - 模型名",
        "  - Overall Acc / AUC",
        "  - Final Acc / AUC",
        "  - `Pred True`",
        "  - Overall Acc / AUC 与 Final Acc / AUC 相对 baseline 的四项差值",
        "  - 是否同时超过 baseline 的 Overall Acc 和 Overall AUC",
        "  - Final Acc / AUC 相对 baseline 的差值，以及是否同时超过 baseline",
        "  - 若未超过，写明下一轮调整原因",
        "- 对已完成 formal SSH run，额外确认 `FORMAL_EXPERIMENT_AUDIT.md` 已显示 artifact 完整且主要 markdown surfaces 已覆盖。",
    ])
    return "\n".join(lines) + "\n"


def build_state(
    baseline: dict,
    v1: dict,
    v1_best_acc: float | None,
    v1_best_threshold: float | None,
    round_rows: list[list[dict]],
    metrics_dir: Path,
    qual_dir: Path,
) -> dict:
    baseline_acc = baseline["overall_acc"]
    baseline_auc = baseline["overall_auc"]
    all_rows = [row for rows in round_rows for row in rows]
    winners = [
        row for row in all_rows
        if is_audit_ready_winner(row, baseline_acc, baseline_auc, metrics_dir, qual_dir)
    ]
    improving_rows = [row for row in all_rows if improves_over_anchor(row, v1["overall_acc"], v1["overall_auc"])]
    best = best_non_winner(improving_rows, baseline_acc, baseline_auc) if improving_rows else None

    round1_rows = round_rows[0]
    round2_rows = round_rows[1]
    round3_rows = round_rows[2]
    round3_started = any(row["status"] in {"running", "done", "failed"} for row in round3_rows)
    round3_running = any(row["status"] == "running" for row in round3_rows)
    round3_pending = any(row["status"] == "pending" for row in round3_rows)
    round3_complete = all(row["status"] in TERMINAL_STATUSES for row in round3_rows)

    next_action = "manual_decide"
    if winners:
        next_action = "done"
    elif round3_pending and not round3_running:
        next_action = "launch_round3"
    elif round3_started and not round3_complete:
        next_action = "wait_round3"
    elif round3_complete:
        next_action = "manual_decide"

    return {
        "baseline": {
            "overall_acc": baseline_acc,
            "overall_auc": baseline_auc,
            "final_acc": baseline.get("final_acc"),
            "final_auc": baseline.get("final_auc"),
        },
        "v1": {
            "model": v1["model"],
            "overall_acc": v1["overall_acc"],
            "overall_auc": v1["overall_auc"],
            "final_acc": v1.get("final_acc"),
            "final_auc": v1.get("final_auc"),
            "best_acc": v1_best_acc,
            "best_threshold": v1_best_threshold,
        },
        "rounds": [
            {
                "title": group["title"],
                "models": rows,
                "all_done": all(row["status"] in TERMINAL_STATUSES for row in rows),
            }
            for group, rows in zip(ROUND_GROUPS, round_rows)
        ],
        "winner_found": bool(winners),
        "winner": max(winners, key=lambda row: (row["overall_acc"], row["overall_auc"])) if winners else None,
        "best_candidate": best or {
            "model": v1["model"],
            "status": "anchor",
            "overall_acc": v1["overall_acc"],
            "overall_auc": v1["overall_auc"],
            "best_acc": v1_best_acc,
            "best_threshold": v1_best_threshold,
        },
        "next_action": next_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--baseline_qual", default="results/baseline/qual/qual_lmkt_qwen3_1.7b_recert_20260620.csv")
    parser.add_argument("--v1_metrics", default="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt")
    parser.add_argument("--v1_qual", default="results/cel_stage1_last_layer/qual/qual_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.csv")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--qual_dir", default="results/cel_stage1_last_layer/qual")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md")
    parser.add_argument("--state_out", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    args = parser.parse_args()

    baseline = parse_metrics(Path(args.baseline_metrics))
    v1 = parse_metrics(Path(args.v1_metrics))
    if baseline is None or v1 is None:
        raise SystemExit("baseline or v1 metrics missing")

    baseline_probs = parse_probs(Path(args.baseline_qual))
    v1_probs = parse_probs(Path(args.v1_qual))
    baseline_pred_true = pred_true_pct(baseline_probs) if baseline_probs else None
    v1_pred_true = pred_true_pct(v1_probs) if v1_probs else None
    v1_best_acc, v1_best_threshold = best_threshold_acc(v1_probs) if v1_probs else (None, None)
    v1["pred_true"] = v1_pred_true

    metrics_dir = Path(args.metrics_dir)
    qual_dir = Path(args.qual_dir)
    round_rows = [
        [build_candidate_row(name, metrics_dir, qual_dir) for name in group["models"]]
        for group in ROUND_GROUPS
    ]

    Path(args.out).write_text(
        build_markdown(baseline, baseline_pred_true, v1, v1_best_acc, v1_best_threshold, round_rows, metrics_dir, qual_dir),
        encoding="utf-8",
    )
    Path(args.state_out).write_text(
        json.dumps(
            build_state(baseline, v1, v1_best_acc, v1_best_threshold, round_rows, metrics_dir, qual_dir),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
