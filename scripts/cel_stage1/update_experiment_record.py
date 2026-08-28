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

BASELINE_AUC = 75.99
PAPER_BASELINE_AUC = 76.71

MODEL_LABELS = {
    "cel_mlp_qwen3_1.7b": "MLP",
    "cel_mlp_qwen3_1.7b_debug": "MLP",
    "cel_adapter_qwen3_1.7b": "Adapter",
    "cel_adapter_qwen3_1.7b_debug": "Adapter",
    "cel_task_conditioned_qwen3_1.7b": "Task-conditioned",
    "cel_task_conditioned_qwen3_1.7b_debug": "Task-conditioned",
}

RESULT_BLOCK_RE = re.compile(
    r"## 4\. 正式训练结果\n.*?\n## 5\. 结论\n.*?(?=\n## 6\.)",
    re.DOTALL,
)


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    loss_match = LOSS_RE.search(text)
    overall_match = OVERALL_RE.search(text)
    final_match = FINAL_RE.search(text)
    if not loss_match or not overall_match or not final_match:
        raise ValueError(f"Cannot parse metrics file: {path}")
    return {
        "model_name": path.stem.removeprefix("metrics_"),
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


def build_results_table(rows: list[dict]) -> str:
    lines = [
        "| Method | Loss | Overall Acc | Overall AUC | Overall Prec | Overall Rec | Overall F1 | Final Acc | Final AUC | Final Prec | Final Rec | Final F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {loss:.4f} | {overall_acc:.2f} | {overall_auc:.2f} | {overall_prec:.2f} | {overall_rec:.2f} | {overall_f1:.2f} | {final_acc:.2f} | {final_auc:.2f} | {final_prec:.2f} | {final_rec:.2f} | {final_f1:.2f} |".format(
                label=MODEL_LABELS[row["model_name"]],
                loss=row["loss"],
                overall_acc=row["overall_acc"],
                overall_auc=row["overall_auc"],
                overall_prec=row["overall_prec"],
                overall_rec=row["overall_rec"],
                overall_f1=row["overall_f1"],
                final_acc=row["final_acc"],
                final_auc=row["final_auc"],
                final_prec=row["final_prec"],
                final_rec=row["final_rec"],
                final_f1=row["final_f1"],
            )
        )
    return "\n".join(lines)


def build_conclusion(rows: list[dict]) -> str:
    best_overall = max(rows, key=lambda row: row["overall_auc"])
    best_final = max(rows, key=lambda row: row["final_auc"])

    overall_delta = best_overall["overall_auc"] - BASELINE_AUC
    paper_delta = best_overall["overall_auc"] - PAPER_BASELINE_AUC

    if overall_delta > 0:
        overall_line = (
            f"- 最佳 Overall AUC 来自 **{MODEL_LABELS[best_overall['model_name']]}**，"
            f"为 **{best_overall['overall_auc']:.2f}**，较 Qwen3-1.7B baseline `75.99` "
            f"提升 **{overall_delta:.2f}**。"
        )
    else:
        overall_line = (
            f"- 最佳 Overall AUC 来自 **{MODEL_LABELS[best_overall['model_name']]}**，"
            f"为 **{best_overall['overall_auc']:.2f}**，较 Qwen3-1.7B baseline `75.99` "
            f"下降 **{abs(overall_delta):.2f}**。"
        )

    if paper_delta > 0:
        paper_line = (
            f"- 该结果高于论文 baseline `76.71`，超出 **{paper_delta:.2f}**。"
        )
    else:
        paper_line = (
            f"- 该结果未超过论文 baseline `76.71`，仍相差 **{abs(paper_delta):.2f}**。"
        )

    final_line = (
        f"- 最佳 Final Turn AUC 来自 **{MODEL_LABELS[best_final['model_name']]}**，"
        f"为 **{best_final['final_auc']:.2f}**。"
    )

    ranking = ", ".join(
        f"{MODEL_LABELS[row['model_name']]} ({row['overall_auc']:.2f})"
        for row in sorted(rows, key=lambda row: row["overall_auc"], reverse=True)
    )
    ranking_line = f"- 按 Overall AUC 排序：{ranking}。"

    return "\n".join([overall_line, paper_line, final_line, ranking_line])


def build_result_section(rows: list[dict]) -> str:
    table = build_results_table(rows)
    conclusion = build_conclusion(rows)
    return (
        "## 4. 正式训练结果\n\n"
        f"{table}\n\n"
        "三种方法的正式训练命令与配置已固定在 `scripts/cel_stage1/` 下，结果文件已归档到：\n\n"
        "- `results/cel_stage1/metrics/`\n"
        "- `results/cel_stage1/qual/`\n"
        "- `results/cel_stage1/kcs/`\n\n"
        "## 5. 结论\n\n"
        f"{conclusion}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="results/cel_stage1/CEL_Stage1_DialogueKT_实验记录.md")
    parser.add_argument(
        "--metrics",
        nargs=3,
        required=True,
        help="Three formal metrics files in MLP / Adapter / Task-conditioned order.",
    )
    args = parser.parse_args()

    rows = [parse_metrics(Path(path)) for path in args.metrics]
    record_path = Path(args.record)
    record_text = record_path.read_text(encoding="utf-8")
    updated_block = build_result_section(rows)

    if not RESULT_BLOCK_RE.search(record_text):
        raise ValueError("Cannot locate formal-result section in experiment record")

    record_text = RESULT_BLOCK_RE.sub(updated_block, record_text)
    record_path.write_text(record_text, encoding="utf-8")


if __name__ == "__main__":
    main()
