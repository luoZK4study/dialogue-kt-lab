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
DIAG_RE = re.compile(r"CEL Diagnostics:\s+(.*)")
STAGE_RE = re.compile(r"- Stage: `([^`]+)`")
FOLLOWUP_RE = re.compile(r"- Latest follow-up event: `([^`]+)`")


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
        "overall_prec": float(overall_match.group(3)),
        "overall_rec": float(overall_match.group(4)),
        "overall_f1": float(overall_match.group(5)),
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
        "diag": diag_match.group(1).strip() if diag_match else None,
    }


def parse_status(status_path: Path) -> tuple[str | None, str | None]:
    if not status_path.exists():
        return None, None
    text = status_path.read_text(encoding="utf-8")
    stage_match = STAGE_RE.search(text)
    followup_match = FOLLOWUP_RE.search(text)
    return (
        stage_match.group(1) if stage_match else None,
        followup_match.group(1) if followup_match else None,
    )


def build_metrics_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | Loss | Overall AUC | Final AUC | Diagnostics |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        final_auc = f"{row['final_auc']:.2f}" if row["final_auc"] is not None else "--"
        diag = row["diag"] or "--"
        lines.append(
            f"| `{row['model']}` | {row['loss']:.4f} | {row['overall_auc']:.2f} | {final_auc} | {diag} |"
        )
    return lines


def build_record(rows: list[dict], baseline_auc: float | None, stage: str | None, followup_event: str | None) -> str:
    lines = [
        "# CEL Stage1 Last-Layer 实验记录",
        "",
        "## 目标",
        "",
        "在当前 `baseline` 之后，固定 A 模块在最后一层表示上的实现方式，并选出 Stage 1 的暂定方案：",
        "",
        "1. `mlp`",
        "2. `adapter`",
        "3. `task_conditioned`",
        "",
        "当前这轮实验归类为 **Stage 1 last-layer**，不是 Stage 2。Stage 2 将专门保留给 environment generator。",
        "",
        "## 设计变更",
        "",
        "- 注入层改为最后一层：`--cel_layer_idx -1`",
        "- selector 输出从 `sigmoid gate` 改为 `tanh delta`",
        "- selector 末层零初始化，训练起点与 baseline 等价",
        "- 默认关闭额外 norm：`--cel_use_norm 0`",
        "- `task_conditioned` 强制走 unpacked prompt，避免多 KC 平均 gate",
        "- CEL 训练从 baseline checkpoint 开始：`--pt_model_name lmkt_qwen3_1.7b_recert_20260620`",
        "- 同族 follow-up 新增 selector warm-start：`--cel_selector_init_model_name <model_name>`",
        "",
        "## 首轮计划",
        "",
        "- `cel_mlp_lastlayer_v1_qwen3_1.7b`",
        "- `cel_adapter_lastlayer_v1_qwen3_1.7b`",
        "- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`",
        "- 调度方式：`MLP@GPU0 + Adapter@GPU1` 并行，`Task-conditioned` 在 `Adapter` 完成后复用 `GPU1`。",
        "",
        "## 条件化第二轮",
        "",
        "如果首轮最佳 AUC 仍未超过 baseline，则自动追加：",
        "",
        "- `cel_adapter_lastlayer_v2_qwen3_1.7b`",
        "  - 现改为 `adapter + pre_lm_head + vector_shift + prediction_token_shift + selector-only` 的保守 early candidate，避免先浪费在 joint scalar-gate 上。",
        "",
        "## 自动 follow-up",
        "",
        "如果 primary loop 结束后最佳 CEL 仍未超过 baseline，则 follow-up loop 继续自动验证：",
        "",
        "- 并行策略：",
        "  - `MLP selector-only` 与 `Adapter selector-only` 并行",
        "  - 第一批 `vector_shift`：`adapter` 与 `task_conditioned` 并行",
        "  - 第二批 `vector_shift`：`adapter highgamma` 与 `adapter ultralowgamma` 并行",
        "- `selector-only`:",
        "  - `cel_mlp_lastlayer_v2_selector_only_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v3_selector_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v2_selector_only_qwen3_1.7b`",
        "- early prediction-token shift:",
        "  - `cel_mlp_lastlayer_v6_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v16_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v7_predshift_selector_only_qwen3_1.7b`",
        "- early vector-shift + prediction-token shift:",
        "  - `cel_mlp_lastlayer_v7_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v17_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v8_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "- early pre-lm-head vector-shift + prediction-token shift:",
        "  - `cel_mlp_lastlayer_v8_prelm_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v18_prelm_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v9_prelm_vector_shift_predshift_selector_only_qwen3_1.7b`",
        "- diagnostics-guided adapter scalar-gate:",
        "  - `cel_adapter_lastlayer_v4_selector_only_lowgamma_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v5_selector_only_highgamma_qwen3_1.7b`",
        "- vector-shift:",
        "  - `cel_adapter_lastlayer_v6_vector_shift_selector_only_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v3_vector_shift_selector_only_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v7_vector_shift_selector_only_highgamma_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v8_vector_shift_selector_only_ultralowgamma_qwen3_1.7b`",
        "- conservative low-lr phase3:",
        "  - `cel_mlp_lastlayer_v3_selector_only_tinylr_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v9_selector_only_tinylr_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v10_vector_shift_selector_only_tinylr_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v4_vector_shift_selector_only_tinylr_qwen3_1.7b`",
        "- phase4 stabilization batch:",
        "  - `cel_adapter_lastlayer_v11_vector_shift_selector_only_tinylr_norm_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v12_selector_only_tinylr_nowd_qwen3_1.7b`",
        "- phase5 warm-start batch:",
        "  - `cel_mlp_lastlayer_v4_vector_shift_selector_only_tinylr_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v13_vector_shift_selector_only_tinylr_nowd_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v14_vector_shift_selector_only_epoch3_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v5_vector_shift_selector_only_tinylr_nowd_qwen3_1.7b`",
        "- phase6 prediction-token shift batch:",
        "  - `cel_mlp_lastlayer_v5_predshift_selector_only_tinylr_qwen3_1.7b`",
        "  - `cel_adapter_lastlayer_v15_predshift_selector_only_tinylr_nowd_qwen3_1.7b`",
        "  - `cel_task_conditioned_lastlayer_v6_predshift_selector_only_tinylr_nowd_qwen3_1.7b`",
        "",
        "## 状态",
        "",
    ]

    if not rows:
        stage_text = stage or "unknown"
        lines.append(f"- 当前阶段：`{stage_text}`")
        if followup_event:
            lines.append(f"- 最新 follow-up 事件：`{followup_event}`")
        if baseline_auc is None:
            lines.append("- 2026-06-20：baseline 仍在运行，等待首个 Stage 1 last-layer 指标同步。")
        else:
            lines.append(
                f"- 2026-06-20：baseline 已完成（Overall AUC **{baseline_auc:.2f}**），primary loop 正在继续推进首个 Stage 1 last-layer 指标。"
            )
        return "\n".join(lines) + "\n"

    lines.append(f"- 当前阶段：`{stage or 'unknown'}`")
    if followup_event:
        lines.append(f"- 最新 follow-up 事件：`{followup_event}`")
    lines.extend(["", "## 已同步结果", ""])
    lines.extend(build_metrics_table(rows))

    best_row = max(rows, key=lambda row: row["overall_auc"])
    lines.extend(["", "## 当前结论", ""])
    if baseline_auc is None:
        lines.append(
            f"- 当前最佳 CEL 模型为 `{best_row['model']}`，Overall AUC **{best_row['overall_auc']:.2f}**；baseline 指标尚未同步，暂不能比较增益。"
        )
    else:
        delta = best_row["overall_auc"] - baseline_auc
        delta_text = f"高于 baseline **{delta:.2f}**" if delta > 0 else f"低于 baseline **{abs(delta):.2f}**"
        lines.append(
            f"- 当前最佳 CEL 模型为 `{best_row['model']}`，Overall AUC **{best_row['overall_auc']:.2f}**，{delta_text}。"
        )
    lines.append(
        "- 当前固定的 Stage 1 暂定方案：`cel_task_conditioned_lastlayer_v1_qwen3_1.7b`；后续 Stage 2 默认以它作为固定 A 模块起点。"
    )
    if best_row["diag"]:
        lines.append(f"- 该模型最新 diagnostics：`{best_row['diag']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--status", default="results/cel_stage1_last_layer/STATUS.md")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = []
    for path in sorted(metrics_dir.glob("metrics_*.txt")):
        row = parse_metrics(path)
        if row is not None:
            rows.append(row)

    baseline_auc = None
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.exists():
        baseline_row = parse_metrics(baseline_path)
        baseline_auc = baseline_row["overall_auc"] if baseline_row else None

    stage, followup_event = parse_status(Path(args.status))
    Path(args.record).write_text(build_record(rows, baseline_auc, stage, followup_event), encoding="utf-8")


if __name__ == "__main__":
    main()
