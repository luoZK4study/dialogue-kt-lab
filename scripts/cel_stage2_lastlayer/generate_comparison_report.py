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
        "diag": diag_match.group(1).strip() if diag_match else "--",
    }


def infer_family(model_name: str) -> str:
    if "task_conditioned" in model_name:
        return "task_conditioned"
    if "adapter" in model_name:
        return "adapter"
    if "mlp" in model_name:
        return "mlp"
    return "other"


def format_model_row(row: dict, baseline_auc: float | None) -> str:
    delta = "--"
    if baseline_auc is not None:
        delta = f"{row['overall_auc'] - baseline_auc:+.2f}"
    final_auc = f"{row['final_auc']:.2f}" if row["final_auc"] is not None else "--"
    return (
        f"| `{row['model']}` | {infer_family(row['model'])} | {row['loss']:.4f} | "
        f"{row['overall_auc']:.2f} | {final_auc} | {delta} | {row['diag']} |"
    )


def build_report(stage: str | None, baseline: dict | None, rows: list[dict]) -> str:
    lines = [
        "# CEL Stage1 Last-Layer Comparison",
        "",
        f"- Current stage: `{stage or 'unknown'}`",
        "",
        "## Baseline",
        "",
    ]
    if baseline is None:
        lines.append("- Baseline metrics: pending")
    else:
        lines.extend(
            [
                f"- Model: `{baseline['model']}`",
                f"- Loss: **{baseline['loss']:.4f}**",
                f"- Overall AUC: **{baseline['overall_auc']:.2f}**",
                f"- Final AUC: **{baseline['final_auc']:.2f}**" if baseline["final_auc"] is not None else "- Final AUC: --",
            ]
        )

    lines.extend(["", "## Best By Family", ""])
    if not rows:
        lines.append("- No CEL metrics synced yet")
    else:
        families = {}
        for row in rows:
            family = infer_family(row["model"])
            prev = families.get(family)
            if prev is None or row["overall_auc"] > prev["overall_auc"]:
                families[family] = row
        for family in ["mlp", "adapter", "task_conditioned", "other"]:
            row = families.get(family)
            if row is None:
                continue
            delta = ""
            if baseline is not None:
                delta = f", delta vs baseline **{row['overall_auc'] - baseline['overall_auc']:+.2f}**"
            lines.append(f"- `{family}`: `{row['model']}` with Overall AUC **{row['overall_auc']:.2f}**{delta}")

    lines.extend(["", "## All CEL Results", ""])
    if not rows:
        lines.append("_No CEL metrics synced yet._")
    else:
        lines.extend(
            [
                "| Model | Family | Loss | Overall AUC | Final AUC | Delta vs Baseline | Diagnostics |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        baseline_auc = baseline["overall_auc"] if baseline else None
        for row in sorted(rows, key=lambda item: item["overall_auc"], reverse=True):
            lines.append(format_model_row(row, baseline_auc))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="results/cel_stage1_last_layer/STATUS.md")
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/COMPARISON.md")
    args = parser.parse_args()

    status_text = Path(args.status).read_text(encoding="utf-8") if Path(args.status).exists() else ""
    stage_match = STAGE_RE.search(status_text)
    stage = stage_match.group(1) if stage_match else None

    baseline = None
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.exists():
        baseline = parse_metrics(baseline_path)

    rows = []
    for path in sorted(Path(args.metrics_dir).glob("metrics_*.txt")):
        row = parse_metrics(path)
        if row is not None:
            rows.append(row)

    Path(args.out).write_text(build_report(stage, baseline, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
