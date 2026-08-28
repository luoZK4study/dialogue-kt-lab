#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


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


def build_candidate_row(name: str, metrics_dir: Path, qual_dir: Path) -> dict:
    metrics_path = metrics_dir / f"metrics_{name}.txt"
    qual_path = qual_dir / f"qual_{name}.csv"
    metrics = parse_metrics(metrics_path)
    probs = parse_probs(qual_path)
    if metrics is None:
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


def build_markdown(
    baseline: dict,
    baseline_pred_true: float | None,
    v1: dict,
    v1_best_acc: float | None,
    v1_best_threshold: float | None,
    candidates: list[dict],
) -> str:
    baseline_acc = baseline["overall_acc"]
    baseline_auc = baseline["overall_auc"]
    lines = [
        "# Task-Conditioned Tuning Loop",
        "",
        "## 目标",
        "",
        "基于当前固定方案 `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`，继续调优，目标是在同一模型族中同时超过当前 baseline：",
        "",
        f"- Baseline Overall AUC: **{baseline_auc:.2f}**",
        f"- Baseline Overall Acc: **{baseline_acc:.2f}**",
        "",
        "当前起点：",
        "",
        f"- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`",
        f"- Overall AUC: **{v1['overall_auc']:.2f}**",
        f"- Overall Acc: **{v1['overall_acc']:.2f}**",
        "",
        "## 当前分析",
        "",
        "当前 `v1` 的关键现象不是“排序不够”，而是“概率偏保守”：",
        "",
        f"- `Pred True`: **{fmt_num(v1['pred_true'])}%**",
        f"- baseline `Pred True`: **{fmt_num(baseline_pred_true)}%**",
        f"- `v1` 的最佳 Acc 阈值约为 **{fmt_num(v1_best_threshold, 3)}**",
        "- baseline 的最佳 Acc 阈值约为 **0.488**",
        "",
        "这说明：",
        "",
        "- 当前 `task_conditioned v1` 的排序能力已经优于 baseline",
        "- 但输出分数整体偏低，导致固定 `0.5` 阈值下 Recall/Acc 被压住",
        "",
        "## Round 1: Output Calibration",
        "",
        "候选：",
        "",
        "1. `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`",
        "   - 固定 `task_conditioned v1` selector 行为",
        "   - 只训练一个 logit bias 校准项",
        "2. `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`",
        "   - 固定 `task_conditioned v1` selector 行为",
        "   - 只训练一个正向 affine logit 校准器",
        "",
        "设计意图：",
        "",
        "- 尽量保留 `v1` 已有的 AUC 优势",
        "- 优先把 `Pred True` 从 33.8% 往 baseline 40.8% 附近拉回",
        "- 如果 Acc 能超过 **69.22**，就证明当前短板主要是 calibration，不是 ranking",
        "",
        "当前状态：",
        "",
        "- `2026-06-20` 已在 SSH 服务器上启动：",
        "  - `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`",
        "- 运行方式：",
        "  - 都基于 `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` 的 selector warm-start",
        "  - 都固定 backbone/selector，只训练输出校准器",
        "  - `v11` 训练 1 个参数（bias）",
        "  - `v12` 训练 2 个参数（positive affine scale + bias）",
        "",
        "## Round 1 Results",
        "",
        "| Model | Status | Acc | AUC | Final Acc | Final AUC | Pred True | Best Acc | Best Threshold | Delta Acc | Delta AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in candidates:
        if row["status"] != "done":
            lines.append(f"| `{row['model']}` | pending | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| `{row['model']}` | done | {fmt_num(row['overall_acc'])} | {fmt_num(row['overall_auc'])} | "
            f"{fmt_num(row['final_acc'])} | {fmt_num(row['final_auc'])} | {fmt_num(row['pred_true'])}% | "
            f"{fmt_num(row['best_acc'])} | {fmt_num(row['best_threshold'], 3)} | "
            f"{row['overall_acc'] - baseline_acc:+.2f} | {row['overall_auc'] - baseline_auc:+.2f} |"
        )

    winners = [
        row for row in candidates
        if row["status"] == "done"
        and row["overall_acc"] > baseline_acc
        and row["overall_auc"] > baseline_auc
    ]

    lines.extend(["", "## Round 1 Decision", ""])
    if winners:
        best = max(winners, key=lambda row: (row["overall_acc"], row["overall_auc"]))
        lines.append(
            f"- 当前已有模型同时超过 baseline 的 Acc/AUC：`{best['model']}` "
            f"(Acc **{best['overall_acc']:.2f}**, AUC **{best['overall_auc']:.2f}**)。"
        )
        lines.append("- 下一步应切换到误差分析与复验，而不是继续盲目搜索。")
    elif any(row["status"] == "done" for row in candidates):
        best = max(
            (row for row in candidates if row["status"] == "done"),
            key=lambda row: (row["overall_acc"] > baseline_acc, row["overall_auc"], row["overall_acc"]),
        )
        lines.append(
            f"- Round 1 已有结果，但尚未同时超过 baseline Acc/AUC。当前最好候选：`{best['model']}` "
            f"(Acc **{best['overall_acc']:.2f}**, AUC **{best['overall_auc']:.2f}**)。"
        )
        if best["overall_auc"] > baseline_auc and best["overall_acc"] <= baseline_acc:
            lines.append("- 这说明 calibration 方向有效但还不够，下一轮优先考虑“轻微解冻 selector + calibrator 联训”或更靠近 0.5 的 bias 调整。")
        else:
            lines.append("- 这说明单纯 calibrator-only 还不足，下一轮应考虑结构性修正，而不是继续只做后处理式校准。")
    else:
        lines.append("- Round 1 结果尚未落盘，等待 SSH 训练完成后自动刷新。")

    lines.extend(["", "## Prepared Next Round", ""])
    lines.extend([
        "- 如果 Round 1 仍不能同时超过 baseline 的 Acc/AUC，下一轮优先测试“冻结 backbone，仅联训 selector + calibrator”。",
        "- 已准备好的候选脚本：",
        "  1. `scripts/cel_stage1_last_layer/run_task_conditioned_v13_selector_cal_bias.sh`",
        "     - `bias calibrator` + selector warm-start 联训",
        "  2. `scripts/cel_stage1_last_layer/run_task_conditioned_v14_selector_cal_affine.sh`",
        "     - `affine calibrator` + selector warm-start 联训",
        "- 设计意图：保留 `v1` 的排序结构，只允许 selector 小幅上调分数刻度与正类召回，不直接解冻 backbone。",
    ])

    lines.extend(["", "## 记录要求", ""])
    lines.extend([
        "- 每轮结束后补充：",
        "  - 模型名",
        "  - Overall Acc / AUC",
        "  - Final Acc / AUC",
        "  - `Pred True`",
        "  - 和 baseline 的差值",
        "  - 是否同时超过 baseline 的 Acc 和 AUC",
        "  - 若未超过，写明下一轮调整原因",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--baseline_qual", default="results/baseline/qual/qual_lmkt_qwen3_1.7b_recert_20260620.csv")
    parser.add_argument("--v1_metrics", default="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt")
    parser.add_argument("--v1_qual", default="results/cel_stage1_last_layer/qual/qual_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.csv")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--qual_dir", default="results/cel_stage1_last_layer/qual")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md")
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
    candidates = [
        build_candidate_row("cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b", metrics_dir, qual_dir),
        build_candidate_row("cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b", metrics_dir, qual_dir),
    ]
    Path(args.out).write_text(
        build_markdown(baseline, baseline_pred_true, v1, v1_best_acc, v1_best_threshold, candidates),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
