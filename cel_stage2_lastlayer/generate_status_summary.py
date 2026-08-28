#!/usr/bin/env python3

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "results" / "baseline_recert"
CEL_DIR = ROOT / "results" / "cel_stage2_lastlayer"
STATUS_PATH = CEL_DIR / "STATUS.md"

BASELINE_MODEL = "lmkt_qwen3_1.7b_recert_20260620"
ROUND1_MODELS = [
    "cel_mlp_lastlayer_v1_qwen3_1.7b",
    "cel_adapter_lastlayer_v1_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
]
ROUND2_MODELS = [
    "cel_adapter_lastlayer_v2_qwen3_1.7b",
]


def extract_metric_block(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    overall = re.search(r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)", text, re.S)
    final_turn = re.search(r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)", text, re.S)
    loss = re.search(r"Loss:\s+([0-9.]+)", text)
    diag = re.search(r"CEL Diagnostics:\s+(.*)", text)
    return {
        "path": path,
        "loss": float(loss.group(1)) if loss else None,
        "overall_auc": float(overall.group(2)) if overall else None,
        "overall_acc": float(overall.group(1)) if overall else None,
        "final_auc": float(final_turn.group(2)) if final_turn else None,
        "diag": diag.group(1).strip() if diag else None,
    }


def latest_loop_event() -> str | None:
    loop_log = CEL_DIR / "loop.log"
    if not loop_log.exists():
        return None
    events = [line.strip() for line in loop_log.read_text(encoding="utf-8", errors="ignore").splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    return events[-1] if events else None


def latest_training_progress() -> str | None:
    loop_log = CEL_DIR / "loop.log"
    if not loop_log.exists():
        return None
    text = loop_log.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    matches = list(re.finditer(r"Training:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)", text))
    if not matches:
        return None
    last = matches[-1]
    pct, step, total = last.groups()
    return f"{step}/{total} ({pct}%)"


def infer_stage(baseline_metrics: dict | None, cel_metrics: dict[str, dict], last_event: str | None) -> str:
    if last_event and "experiment loop finished" in last_event:
        return "completed"
    if baseline_metrics is None:
        return "baseline_recert_running"
    round1_ready = sum(1 for model in ROUND1_MODELS if model in cel_metrics)
    round2_ready = sum(1 for model in ROUND2_MODELS if model in cel_metrics)
    if round1_ready < len(ROUND1_MODELS):
        return f"cel_round1_running ({round1_ready}/{len(ROUND1_MODELS)})"
    if round2_ready == 0:
        return "awaiting_round1_decision_or_adapter_v2"
    return f"cel_round2_running_or_finished ({round2_ready}/{len(ROUND2_MODELS)})"


def build_table(rows: list[tuple[str, dict]]) -> str:
    if not rows:
        return "_No metrics synced yet._"
    out = ["| Model | Loss | Overall AUC | Final AUC | Diagnostics |", "|---|---:|---:|---:|---|"]
    for model, metrics in rows:
        loss = f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "--"
        overall_auc = f"{metrics['overall_auc']:.2f}" if metrics["overall_auc"] is not None else "--"
        final_auc = f"{metrics['final_auc']:.2f}" if metrics["final_auc"] is not None else "--"
        diag = metrics["diag"] or "--"
        out.append(f"| `{model}` | {loss} | {overall_auc} | {final_auc} | {diag} |")
    return "\n".join(out)


def main() -> None:
    baseline_metrics = extract_metric_block(BASELINE_DIR / "metrics" / f"metrics_{BASELINE_MODEL}.txt")
    cel_metrics = {}
    for metrics_file in sorted((CEL_DIR / "metrics").glob("metrics_*.txt")):
        model = metrics_file.stem.replace("metrics_", "", 1)
        metrics = extract_metric_block(metrics_file)
        if metrics is not None:
            cel_metrics[model] = metrics

    last_event = latest_loop_event()
    progress = latest_training_progress()
    stage = infer_stage(baseline_metrics, cel_metrics, last_event)

    best_model = None
    if cel_metrics:
        best_model = max(
            ((model, metrics) for model, metrics in cel_metrics.items() if metrics["overall_auc"] is not None),
            key=lambda item: item[1]["overall_auc"],
            default=None,
        )

    lines = [
        "# STATUS",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Stage: `{stage}`",
        f"- Latest loop event: `{last_event or 'none'}`",
        "",
        "## Baseline Recert",
    ]
    if baseline_metrics is None:
        lines.append("- Baseline metrics: pending")
        if progress:
            lines.append(f"- Latest training progress: `{progress}`")
    else:
        lines.extend([
            f"- Model: `{BASELINE_MODEL}`",
            f"- Loss: `{baseline_metrics['loss']:.4f}`",
            f"- Overall AUC: `{baseline_metrics['overall_auc']:.2f}`",
            f"- Final AUC: `{baseline_metrics['final_auc']:.2f}`",
        ])

    lines.extend(["", "## CEL Stage2", build_table([(model, cel_metrics[model]) for model in sorted(cel_metrics)])])
    lines.extend(["", "## Best So Far"])
    if best_model is None:
        lines.append("- No CEL metrics available yet")
    else:
        model, metrics = best_model
        baseline_auc = baseline_metrics["overall_auc"] if baseline_metrics else None
        delta_str = ""
        if baseline_auc is not None and metrics["overall_auc"] is not None:
            delta = metrics["overall_auc"] - baseline_auc
            delta_str = f", delta vs baseline `{delta:+.2f}`"
        lines.append(f"- `{model}` with Overall AUC `{metrics['overall_auc']:.2f}`{delta_str}")

    lines.extend([
        "",
        "## Key Files",
        f"- Baseline record: `{BASELINE_DIR / 'Baseline_Recert_DialogueKT_实验记录.md'}`",
        f"- Stage2 record: `{CEL_DIR / 'CEL_Stage2_LastLayer_DialogueKT_实验记录.md'}`",
        f"- Loop log: `{CEL_DIR / 'loop.log'}`",
        f"- Follow-up log: `{CEL_DIR / 'followup.log'}`",
    ])

    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
