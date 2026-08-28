#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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


def model_role(model_name: str) -> str:
    return "diagnostic" if model_name in DIAGNOSTIC_ONLY_MODELS else "formal"


def load_task_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def format_round_status(row: dict) -> str:
    status = row.get("status", "unknown")
    reason = row.get("failure_reason")
    if status == "failed" and reason:
        return f"{status} ({reason})"
    return status


def beats_formal_gate(row: dict, baseline: dict | None) -> bool:
    if baseline is None:
        return False
    return row["overall_acc"] > baseline["overall_acc"] and row["overall_auc"] > baseline["overall_auc"]


def build_round3_state_lines(task_status: dict | None) -> list[str]:
    if not task_status:
        return []

    lines = ["## Round 3 Formal State", ""]
    winner_found = bool(task_status.get("winner_found"))
    best_candidate = task_status.get("best_candidate") or {}
    round3 = []
    for round_info in task_status.get("rounds", []):
        if "Round 3" in (round_info.get("title") or ""):
            round3 = round_info.get("models", [])
            break

    lines.append(f"- Winner found: `{winner_found}`")
    if best_candidate.get("model"):
        best_label = "Current strict formal winner" if winner_found else "Current best formal candidate"
        lines.append(
            f"- {best_label}: `{best_candidate['model']}` with Overall Acc **{best_candidate['overall_acc']:.2f}** "
            f"and Overall AUC **{best_candidate['overall_auc']:.2f}**."
        )
    if not winner_found and round3:
        lines.append(
            "- Round 3 strict full-train is still pending completion, so the synced metrics table below only reflects older landed artifacts."
        )
        for row in round3:
            lines.append(f"- `{row['model']}`: `{format_round_status(row)}`")
    return lines + [""]


def format_model_row(row: dict, baseline: dict | None) -> str:
    delta_acc = "--"
    delta_auc = "--"
    delta_final_acc = "--"
    delta_final_auc = "--"
    gate = "--"
    role = model_role(row["model"])
    if baseline is not None:
        delta_acc = f"{row['overall_acc'] - baseline['overall_acc']:+.2f}"
        delta_auc = f"{row['overall_auc'] - baseline['overall_auc']:+.2f}"
        if row["final_acc"] is not None and baseline["final_acc"] is not None:
            delta_final_acc = f"{row['final_acc'] - baseline['final_acc']:+.2f}"
        if row["final_auc"] is not None and baseline["final_auc"] is not None:
            delta_final_auc = f"{row['final_auc'] - baseline['final_auc']:+.2f}"
        if role == "diagnostic":
            gate = "diagnostic"
        else:
            gate = "yes" if beats_formal_gate(row, baseline) else "no"
    final_acc = f"{row['final_acc']:.2f}" if row["final_acc"] is not None else "--"
    final_auc = f"{row['final_auc']:.2f}" if row["final_auc"] is not None else "--"
    return (
        f"| `{row['model']}` | {infer_family(row['model'])} | {role} | {row['loss']:.4f} | "
        f"{row['overall_acc']:.2f} | {row['overall_auc']:.2f} | {final_acc} | {final_auc} | "
        f"{delta_acc} | {delta_auc} | {delta_final_acc} | {delta_final_auc} | {gate} | {row['diag']} |"
    )


def build_report(stage: str | None, baseline: dict | None, rows: list[dict], task_status: dict | None) -> str:
    lines = [
        "# CEL Stage1 Last-Layer Comparison",
        "",
        "> Terminology note: the model identifier below is a historical artifact name. The current method name is **A module**.",
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
                f"- Overall Acc: **{baseline['overall_acc']:.2f}**",
                f"- Overall AUC: **{baseline['overall_auc']:.2f}**",
                f"- Final Acc: **{baseline['final_acc']:.2f}**" if baseline["final_acc"] is not None else "- Final Acc: --",
                f"- Final AUC: **{baseline['final_auc']:.2f}**" if baseline["final_auc"] is not None else "- Final AUC: --",
            ]
        )

    lines.extend(["", *build_round3_state_lines(task_status)])
    lines.extend(["", "## Formal Winner Gate", ""])
    if baseline is None:
        lines.append("- Baseline metrics are missing, so the strict formal gate cannot be evaluated yet.")
    else:
        lines.append(
            f"- A formal winner must satisfy both `Overall Acc > {baseline['overall_acc']:.2f}` and `Overall AUC > {baseline['overall_auc']:.2f}`."
        )
        formal_rows = [row for row in rows if model_role(row["model"]) == "formal"]
        winning_formals = [row for row in formal_rows if beats_formal_gate(row, baseline)]
        if winning_formals:
            best_winner = max(winning_formals, key=lambda row: (row["overall_acc"], row["overall_auc"]))
            lines.append(
                f"- Current strict formal winner in synced artifacts: `{best_winner['model']}` with Overall Acc / AUC **{best_winner['overall_acc']:.2f} / {best_winner['overall_auc']:.2f}**."
            )
        elif formal_rows:
            best_formal = max(formal_rows, key=lambda row: (row["overall_auc"], row["overall_acc"]))
            lines.append(
                f"- No synced formal model currently clears both gates. Current formal AUC anchor is `{best_formal['model']}` at Overall Acc / AUC **{best_formal['overall_acc']:.2f} / {best_formal['overall_auc']:.2f}**."
            )
        diagnostic_rows = [row for row in rows if model_role(row["model"]) == "diagnostic"]
        if diagnostic_rows:
            best_diag = max(diagnostic_rows, key=lambda row: (row["overall_auc"], row["overall_acc"]))
            lines.append(
                f"- Best diagnostic-only result is `{best_diag['model']}` at Overall Acc / AUC **{best_diag['overall_acc']:.2f} / {best_diag['overall_auc']:.2f}**, but it remains excluded from formal winner judgment."
            )

    lines.extend(["", "## Best Formal Result By Family", ""])
    if not rows:
        lines.append("- No CEL metrics synced yet")
    else:
        families = {}
        for row in rows:
            if model_role(row["model"]) == "diagnostic":
                continue
            family = infer_family(row["model"])
            prev = families.get(family)
            if prev is None or row["overall_auc"] > prev["overall_auc"]:
                families[family] = row
        for family in ["mlp", "adapter", "task_conditioned", "other"]:
            row = families.get(family)
            if row is None:
                continue
            delta = ""
            gate = ""
            if baseline is not None:
                delta = (
                    f", delta Acc / AUC **{row['overall_acc'] - baseline['overall_acc']:+.2f} / "
                    f"{row['overall_auc'] - baseline['overall_auc']:+.2f}**"
                )
                gate = ", clears strict gate" if beats_formal_gate(row, baseline) else ", does not clear strict gate"
            lines.append(
                f"- `{family}`: `{row['model']}` with Overall Acc / AUC **{row['overall_acc']:.2f} / {row['overall_auc']:.2f}**{delta}{gate}"
            )

    diagnostic_rows = [row for row in rows if model_role(row["model"]) == "diagnostic"]
    if diagnostic_rows:
        best_diag = max(diagnostic_rows, key=lambda row: (row["overall_auc"], row["overall_acc"]))
        lines.extend(["", "## Best Diagnostic Result", ""])
        lines.append(
            f"- `{best_diag['model']}` with Overall Acc **{best_diag['overall_acc']:.2f}** and Overall AUC **{best_diag['overall_auc']:.2f}**."
        )
        lines.append("- 该结果只用于说明后处理/冻结补训上限，不计入正式方法比较。")

    lines.extend(["", "## All CEL Results", ""])
    if not rows:
        lines.append("_No CEL metrics synced yet._")
    else:
        lines.extend(
            [
                "| Model | Family | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Formal Gate | Diagnostics |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (model_role(item["model"]) != "formal", -item["overall_auc"], -item["overall_acc"]),
        ):
            lines.append(format_model_row(row, baseline))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="results/cel_stage1_last_layer/STATUS.md")
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
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
    task_status = load_task_status(Path(args.task_status))

    rows = []
    for path in sorted(Path(args.metrics_dir).glob("metrics_*.txt")):
        row = parse_metrics(path)
        if row is not None:
            rows.append(row)

    Path(args.out).write_text(build_report(stage, baseline, rows, task_status), encoding="utf-8")


if __name__ == "__main__":
    main()
