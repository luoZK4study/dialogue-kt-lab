#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
PHASE_RE = re.compile(r"- Current phase: `([^`]+)`")
TRAIN_PROGRESS_RE = re.compile(r"- Latest training progress: `([^`]+)`")
TEST_PROGRESS_RE = re.compile(r"- Latest testing progress: `([^`]+)`")
VAL_PROGRESS_RE = re.compile(r"- Latest validating progress: `([^`]+)`")
EPOCH_RE = re.compile(r"- Current epoch: `([^`]+)`")


def parse_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    loss_match = LOSS_RE.search(text)
    overall_match = OVERALL_RE.search(text)
    final_match = FINAL_RE.search(text)
    if not loss_match or not overall_match or not final_match:
        return None
    return {
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "overall_prec": float(overall_match.group(3)),
        "overall_rec": float(overall_match.group(4)),
        "overall_f1": float(overall_match.group(5)),
        "final_acc": float(final_match.group(1)),
        "final_auc": float(final_match.group(2)),
        "final_prec": float(final_match.group(3)),
        "final_rec": float(final_match.group(4)),
        "final_f1": float(final_match.group(5)),
    }


def parse_progress(status_path: Path) -> tuple[str | None, str | None, str | None]:
    if not status_path.exists():
        return None, None, None
    text = status_path.read_text(encoding="utf-8")
    phase_match = PHASE_RE.search(text)
    epoch_match = EPOCH_RE.search(text)
    train_match = TRAIN_PROGRESS_RE.search(text)
    test_match = TEST_PROGRESS_RE.search(text)
    val_match = VAL_PROGRESS_RE.search(text)
    if test_match:
        return phase_match.group(1) if phase_match else "testing", epoch_match.group(1) if epoch_match else None, test_match.group(1)
    if val_match:
        return phase_match.group(1) if phase_match else "validating", epoch_match.group(1) if epoch_match else None, val_match.group(1)
    if train_match:
        return phase_match.group(1) if phase_match else "training", epoch_match.group(1) if epoch_match else None, train_match.group(1)
    return phase_match.group(1) if phase_match else None, epoch_match.group(1) if epoch_match else None, None


def build_record(metrics: dict | None, phase: str | None, epoch: str | None, progress: str | None) -> str:
    lines = [
        "# Baseline 实验记录",
        "",
        "## 目标",
        "",
        "重新训练并认证 `Qwen3-1.7B + LoRA` 在当前代码与当前数据切分下的 baseline 性能，作为新一轮 CEL 设计的唯一比较基准。",
        "",
        "## 计划运行",
        "",
        "- 模型名：`lmkt_qwen3_1.7b_recert_20260620`",
        "- 结果目录：`results/baseline/`",
        "- 训练脚本：`scripts/cel_stage1_last_layer/run_baseline.sh`",
        "",
        "## 状态",
        "",
    ]
    if metrics is None:
        if progress:
            if phase == "testing":
                phase_text = "测试"
            elif phase == "validating":
                phase_text = "validation"
            else:
                phase_text = "训练"
            epoch_text = f"（epoch {epoch}）" if epoch else ""
            if phase == "testing":
                lines.append(f"- 2026-06-20：服务器已完成 baseline 训练与 validation，当前正在做{phase_text}{epoch_text}，最新同步进度约为 `{progress}`。")
            elif phase == "validating":
                lines.append(f"- 2026-06-20：服务器正在做 baseline {phase_text}{epoch_text}，最新同步进度约为 `{progress}`。")
            else:
                lines.append(f"- 2026-06-20：服务器{phase_text}进行中{epoch_text}，最新同步进度约为 `{progress}`。")
        else:
            lines.append("- 2026-06-20：脚本已创建，等待服务器正式运行结果。")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "- 2026-06-20：baseline 已完成当前认证，指标已同步到本地。",
            "",
            "## 结果",
            "",
            f"- Loss: **{metrics['loss']:.4f}**",
            f"- Overall: Acc **{metrics['overall_acc']:.2f}**, AUC **{metrics['overall_auc']:.2f}**, Prec **{metrics['overall_prec']:.2f}**, Rec **{metrics['overall_rec']:.2f}**, F1 **{metrics['overall_f1']:.2f}**",
            f"- Final Turn: Acc **{metrics['final_acc']:.2f}**, AUC **{metrics['final_auc']:.2f}**, Prec **{metrics['final_prec']:.2f}**, Rec **{metrics['final_rec']:.2f}**, F1 **{metrics['final_f1']:.2f}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="results/baseline/Baseline_DialogueKT_实验记录.md")
    parser.add_argument("--metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--status", default="results/cel_stage1_last_layer/STATUS.md")
    args = parser.parse_args()

    record_path = Path(args.record)
    metrics = parse_metrics(Path(args.metrics))
    phase, epoch, progress = parse_progress(Path(args.status))
    record_path.write_text(build_record(metrics, phase, epoch, progress), encoding="utf-8")


if __name__ == "__main__":
    main()
