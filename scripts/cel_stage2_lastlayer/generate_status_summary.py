#!/usr/bin/env python3

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "results" / "baseline"
CEL_DIR = ROOT / "results" / "cel_stage1_last_layer"
STATUS_PATH = CEL_DIR / "STATUS.md"
STEP_LOG_DIR = CEL_DIR / "step_logs"
LOOP_EVENT_PATH = CEL_DIR / "loop_stdout.log"
LOOP_PROGRESS_PATH = CEL_DIR / "loop.log"
FOLLOWUP_EVENT_PATH = CEL_DIR / "followup.log"
FOLLOWUP_PROGRESS_PATH = CEL_DIR / "followup_stdout.log"

BASELINE_MODEL = "lmkt_qwen3_1.7b_recert_20260620"
ROUND1_MODELS = [
    "cel_mlp_lastlayer_v1_qwen3_1.7b",
    "cel_adapter_lastlayer_v1_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
]
ROUND2_MODELS = [
    "cel_adapter_lastlayer_v2_qwen3_1.7b",
]
FOLLOWUP_SUCCESS_MARKERS = [
    "existing CEL result already beats baseline; no followup runs needed",
    "selector-only round beats baseline; followup finished",
    "early prediction-token batch1 beats baseline; followup finished",
    "early task-conditioned prediction-token batch beats baseline; followup finished",
    "early vector-predshift batch1 beats baseline; followup finished",
    "early task-conditioned vector-predshift batch beats baseline; followup finished",
    "early pre-lm-head vector-predshift batch1 beats baseline; followup finished",
    "early task-conditioned pre-lm-head vector-predshift batch beats baseline; followup finished",
    "direct pre-lm-head selector-only batch1 beats baseline; followup finished",
    "direct task-conditioned pre-lm-head selector-only batch beats baseline; followup finished",
    "scalar-gate followup beats baseline; followup finished",
    "vector-shift batch1 beats baseline; followup finished",
    "vector-shift batch2 beats baseline; followup finished",
    "conservative scalar batch beats baseline; followup finished",
    "conservative low-lr phase3 beats baseline; followup finished",
    "phase4 stabilization batch beats baseline; followup finished",
    "phase5 warm-start batch1 beats baseline; followup finished",
    "phase5 warm-start batch2 beats baseline; followup finished",
    "phase6 prediction-token batch1 beats baseline; followup finished",
    "phase6 task-conditioned prediction-token batch beats baseline; followup finished",
    "task-conditioned vector-shift followup beats baseline; followup finished",
    "ultralow-gamma vector-shift followup beats baseline; followup finished",
    "followup loop found a CEL model that beats baseline",
]
FOLLOWUP_FAILURE_MARKERS = [
    "followup loop completed; no CEL model beat baseline yet",
    "baseline metrics missing; exiting followup loop",
]
PROGRESS_RE = re.compile(r"(Training|Validation|Validating|Testing):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
EVENT_RE = re.compile(r"^\[[0-9:\- ]+\]\s+(START|END|SKIP|FAIL|ABORT)\s+(\S+)")


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


def read_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")


def latest_loop_event() -> str | None:
    for path in (LOOP_EVENT_PATH, LOOP_PROGRESS_PATH):
        events = [line.strip() for line in read_log_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
        if events:
            return events[-1]
    return None


def latest_followup_event() -> str | None:
    for path in (FOLLOWUP_EVENT_PATH, FOLLOWUP_PROGRESS_PATH):
        events = [line.strip() for line in read_log_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
        if events:
            return events[-1]
    return None


def active_steps_from_log(path: Path) -> list[str]:
    active: list[str] = []
    for raw_line in read_log_text(path).splitlines():
        match = EVENT_RE.match(raw_line.strip())
        if not match:
            continue
        action, step = match.groups()
        if action == "START":
            if step not in active:
                active.append(step)
        else:
            active = [item for item in active if item != step]
    return active


def latest_phase_progress_from_log(path: Path) -> dict | None:
    text = read_log_text(path)
    matches = list(PROGRESS_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    phase, pct, step, total = last.groups()
    epoch_matches = list(EPOCH_RE.finditer(text[: last.start()]))
    epoch = int(epoch_matches[-1].group(1)) if epoch_matches else None
    return {
        "phase": phase.lower(),
        "progress": f"{step}/{total} ({pct}%)",
        "epoch": epoch,
    }


def step_progress(step_name: str) -> dict | None:
    return latest_phase_progress_from_log(STEP_LOG_DIR / f"{step_name}.log")


def classify_followup_event(followup_event: str | None) -> str:
    if not followup_event:
        return "none"
    if "waiting for primary loop to finish" in followup_event:
        return "waiting"
    if any(marker in followup_event for marker in FOLLOWUP_SUCCESS_MARKERS):
        return "success"
    if any(marker in followup_event for marker in FOLLOWUP_FAILURE_MARKERS):
        return "failure"
    return "running"


def infer_stage(
    baseline_metrics: dict | None,
    cel_metrics: dict[str, dict],
    last_event: str | None,
    followup_state: str,
    baseline_progress: dict | None,
) -> str:
    if followup_state == "waiting":
        if baseline_metrics is None:
            if baseline_progress and baseline_progress["phase"] == "testing":
                return "baseline_testing + followup_waiting"
            if baseline_progress and baseline_progress["phase"] == "validating":
                return "baseline_validating + followup_waiting"
            return "baseline_running + followup_waiting"
        return "stage1_last_layer_primary_running + followup_waiting"
    if followup_state == "running":
        return "stage1_last_layer_followup_running"
    if followup_state == "success":
        return "stage1_last_layer_completed_with_followup_win"
    if followup_state == "failure":
        return "stage1_last_layer_followup_completed_no_win"
    if last_event and "experiment loop finished" in last_event:
        return "stage1_last_layer_primary_finished"
    if baseline_metrics is None:
        if baseline_progress and baseline_progress["phase"] == "testing":
            return "baseline_testing"
        if baseline_progress and baseline_progress["phase"] == "validating":
            return "baseline_validating"
        return "baseline_running"
    round1_ready = sum(1 for model in ROUND1_MODELS if model in cel_metrics)
    round2_ready = sum(1 for model in ROUND2_MODELS if model in cel_metrics)
    if round1_ready < len(ROUND1_MODELS):
        return f"stage1_last_layer_round1_running ({round1_ready}/{len(ROUND1_MODELS)})"
    if round2_ready == 0:
        return "stage1_last_layer_awaiting_round1_decision_or_adapter_v2"
    return f"stage1_last_layer_round2_running_or_finished ({round2_ready}/{len(ROUND2_MODELS)})"


def current_cel_progress(last_event: str | None, followup_state: str) -> dict | None:
    if followup_state == "running":
        return latest_phase_progress_from_log(FOLLOWUP_PROGRESS_PATH) or latest_phase_progress_from_log(FOLLOWUP_EVENT_PATH)
    if last_event and re.search(r"START cel_", last_event):
        return latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    return None


def first_active_step_progress(active_steps: list[str], fallback_path: Path | None = None) -> dict | None:
    for step_name in active_steps:
        progress = step_progress(step_name)
        if progress is not None:
            return progress
    if fallback_path is not None:
        return latest_phase_progress_from_log(fallback_path)
    return None


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
    followup_event = latest_followup_event()
    baseline_progress = latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    followup_state = classify_followup_event(followup_event)
    active_loop_steps = active_steps_from_log(LOOP_EVENT_PATH)
    active_followup_steps = active_steps_from_log(FOLLOWUP_EVENT_PATH)
    cel_progress = current_cel_progress(last_event, followup_state)
    if cel_progress is None and (active_followup_steps or active_loop_steps):
        fallback_path = FOLLOWUP_PROGRESS_PATH if active_followup_steps else LOOP_PROGRESS_PATH
        cel_progress = first_active_step_progress(active_followup_steps + active_loop_steps, fallback_path=fallback_path)
    stage = infer_stage(baseline_metrics, cel_metrics, last_event, followup_state, baseline_progress)

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
        f"- Latest follow-up event: `{followup_event or 'none'}`",
    ]
    if active_loop_steps:
        lines.append(f"- Active loop steps: `{', '.join(active_loop_steps)}`")
    if active_followup_steps:
        lines.append(f"- Active follow-up steps: `{', '.join(active_followup_steps)}`")
    active_step_progress = []
    loop_stdout_progress = latest_phase_progress_from_log(LOOP_PROGRESS_PATH) or latest_phase_progress_from_log(LOOP_EVENT_PATH)
    followup_stdout_progress = latest_phase_progress_from_log(FOLLOWUP_PROGRESS_PATH) or latest_phase_progress_from_log(FOLLOWUP_EVENT_PATH)
    for step_name in active_loop_steps + active_followup_steps:
        progress = step_progress(step_name)
        if progress is None:
            if step_name in active_loop_steps and step_name == active_loop_steps[-1]:
                progress = loop_stdout_progress
            elif step_name in active_followup_steps and step_name == active_followup_steps[-1]:
                progress = followup_stdout_progress
        if progress is None:
            continue
        label = step_name
        phase = progress["phase"]
        epoch_text = f", epoch {progress['epoch']}" if progress["epoch"] is not None else ""
        active_step_progress.append(
            f"- `{label}`: `{phase}`{epoch_text}, `{progress['progress']}`"
        )
    lines.extend(["", "## Baseline"])
    if baseline_metrics is None:
        lines.append("- Baseline metrics: pending")
        if baseline_progress:
            lines.append(f"- Current phase: `{baseline_progress['phase']}`")
            if baseline_progress["epoch"] is not None:
                lines.append(f"- Current epoch: `{baseline_progress['epoch']}`")
            lines.append(f"- Latest {baseline_progress['phase']} progress: `{baseline_progress['progress']}`")
    else:
        lines.extend([
            f"- Model: `{BASELINE_MODEL}`",
            f"- Loss: `{baseline_metrics['loss']:.4f}`",
            f"- Overall AUC: `{baseline_metrics['overall_auc']:.2f}`",
            f"- Final AUC: `{baseline_metrics['final_auc']:.2f}`",
        ])

    lines.extend(["", "## CEL Stage1 Last-Layer"])
    if cel_progress:
        lines.append(f"- Live phase: `{cel_progress['phase']}`")
        if cel_progress["epoch"] is not None:
            lines.append(f"- Live epoch: `{cel_progress['epoch']}`")
        lines.append(f"- Live {cel_progress['phase']} progress: `{cel_progress['progress']}`")
    if active_step_progress:
        lines.append("- Active step progress:")
        lines.extend(active_step_progress)
    lines.append(build_table([(model, cel_metrics[model]) for model in sorted(cel_metrics)]))
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
        f"- Baseline record: `{BASELINE_DIR / 'Baseline_DialogueKT_实验记录.md'}`",
        f"- Stage1 last-layer record: `{CEL_DIR / 'CEL_Stage1_LastLayer_DialogueKT_实验记录.md'}`",
        f"- Comparison report: `{CEL_DIR / 'COMPARISON.md'}`",
        f"- Loop log: `{CEL_DIR / 'loop.log'}`",
        f"- Follow-up log: `{CEL_DIR / 'followup.log'}`",
        f"- Supervisor log: `{CEL_DIR / 'supervisor.log'}`",
        f"- Step logs: `{STEP_LOG_DIR}`",
    ])

    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
