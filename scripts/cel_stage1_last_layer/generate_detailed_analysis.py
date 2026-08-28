#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import pstdev


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
GATE_ABS_MEAN_RE = re.compile(r"gate_abs_mean:\s*([0-9.]+)")
SHIFT_ABS_MEAN_RE = re.compile(r"shift_abs_mean:\s*([0-9.]+)")
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


@dataclass(frozen=True)
class EvalRow:
    dialogue_id: str
    turn: int
    label: int
    prob: float
    is_final: bool


@dataclass(frozen=True)
class Summary:
    count: int
    pred_pos_frac: float
    pos_mean: float
    neg_mean: float
    gap: float
    std: float
    auc: float


@dataclass(frozen=True)
class ModelAnalysis:
    model: str
    metrics_acc: float
    metrics_auc: float
    metrics_final_acc: float | None
    metrics_final_auc: float | None
    diag: str | None
    role: str
    overall: Summary
    final: Summary
    non_final_auc: float
    abs_better: int | None
    abs_worse: int | None
    abs_same: int | None
    note: str


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
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
        "diag": diag_match.group(1).strip() if diag_match else None,
    }


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


def parse_eval_rows(path: Path) -> list[EvalRow]:
    raw_rows: list[tuple[str, int, int, float]] = []
    max_turn_by_dialogue: dict[str, int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = (row.get("Correct") or "").strip().lower()
            prob_text = (row.get("Prob") or "").strip()
            if correct not in {"true", "false"} or prob_text in {"", "--"}:
                continue
            try:
                turn = int(row["Turn"])
                prob = float(prob_text)
            except (KeyError, ValueError):
                continue
            dialogue_id = row["Dialogue ID"]
            label = 1 if correct == "true" else 0
            raw_rows.append((dialogue_id, turn, label, prob))
            if turn > max_turn_by_dialogue[dialogue_id]:
                max_turn_by_dialogue[dialogue_id] = turn

    eval_rows = [
        EvalRow(
            dialogue_id=dialogue_id,
            turn=turn,
            label=label,
            prob=prob,
            is_final=(turn == max_turn_by_dialogue[dialogue_id]),
        )
        for dialogue_id, turn, label, prob in raw_rows
    ]
    eval_rows.sort(key=lambda row: (int(row.dialogue_id), row.turn))
    return eval_rows


def compute_auc(rows: list[EvalRow]) -> float:
    if not rows:
        return float("nan")
    scored = sorted((row.prob, row.label) for row in rows)
    n_pos = sum(label for _, label in scored)
    n_neg = len(scored) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank = 1
    pos_rank_sum = 0.0
    idx = 0
    while idx < len(scored):
        j = idx
        while j < len(scored) and scored[j][0] == scored[idx][0]:
            j += 1
        avg_rank = (rank + (rank + (j - idx) - 1)) / 2.0
        pos_in_group = sum(label for _, label in scored[idx:j])
        pos_rank_sum += pos_in_group * avg_rank
        rank += j - idx
        idx = j

    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize(rows: list[EvalRow]) -> Summary:
    probs = [row.prob for row in rows]
    pos_probs = [row.prob for row in rows if row.label == 1]
    neg_probs = [row.prob for row in rows if row.label == 0]
    return Summary(
        count=len(rows),
        pred_pos_frac=sum(prob >= 0.5 for prob in probs) / len(probs),
        pos_mean=sum(pos_probs) / len(pos_probs),
        neg_mean=sum(neg_probs) / len(neg_probs),
        gap=(sum(pos_probs) / len(pos_probs)) - (sum(neg_probs) / len(neg_probs)),
        std=pstdev(probs),
        auc=compute_auc(rows),
    )


def compare_abs_error(candidate_rows: list[EvalRow], baseline_rows: list[EvalRow]) -> tuple[int, int, int]:
    baseline_map = {(row.dialogue_id, row.turn): row for row in baseline_rows}
    better = worse = same = 0
    for row in candidate_rows:
        base = baseline_map.get((row.dialogue_id, row.turn))
        if base is None:
            continue
        base_err = abs(base.prob - base.label)
        cand_err = abs(row.prob - row.label)
        if cand_err + 1e-12 < base_err:
            better += 1
        elif base_err + 1e-12 < cand_err:
            worse += 1
        else:
            same += 1
    return better, worse, same


def extract_signal_abs(diag: str | None) -> float | None:
    if not diag:
        return None
    gate_match = GATE_ABS_MEAN_RE.search(diag)
    if gate_match:
        return float(gate_match.group(1))
    shift_match = SHIFT_ABS_MEAN_RE.search(diag)
    if shift_match:
        return float(shift_match.group(1))
    return None


def build_note(
    model: str,
    metrics_auc: float,
    overall: Summary,
    final: Summary,
    non_final_auc: float,
    baseline: ModelAnalysis,
    abs_better: int | None,
    abs_worse: int | None,
    diag: str | None,
) -> str:
    notes: list[str] = []
    baseline_overall = baseline.overall
    baseline_final = baseline.final

    if overall.gap > baseline_overall.gap and metrics_auc < baseline.metrics_auc:
        notes.append("better mean separation, weaker ranking")
    if final.auc > baseline_final.auc and non_final_auc < baseline.non_final_auc:
        notes.append("helps final, hurts non-final")
    if overall.pred_pos_frac > baseline_overall.pred_pos_frac + 0.08:
        notes.append("over-predicts True")
    elif overall.pred_pos_frac < baseline_overall.pred_pos_frac - 0.08:
        notes.append("under-predicts True")
    if abs_better is not None and abs_worse is not None and abs_better > abs_worse and metrics_auc < baseline.metrics_auc:
        notes.append("more rows improve, ranking still below baseline")
    signal_abs = extract_signal_abs(diag)
    if signal_abs is not None and signal_abs < 1e-4:
        notes.append("near-identity diagnostics")
    if "task_conditioned" in model and overall.pred_pos_frac < 0.30:
        notes.append("strong positive-class suppression")
    return "; ".join(notes) if notes else "--"


def format_pct(frac: float) -> str:
    return f"{frac * 100:.1f}%"


def formal_gate_status(row: ModelAnalysis, baseline: ModelAnalysis) -> str:
    return "yes" if row.metrics_acc > baseline.metrics_acc and row.metrics_auc > baseline.metrics_auc else "no"


def build_report(baseline: ModelAnalysis, rows: list[ModelAnalysis], task_status: dict | None) -> str:
    lines = [
        "# CEL Stage1 Last-Layer Detailed Analysis",
        "",
        "> Terminology note: historical model identifiers remain unchanged for prediction-file alignment. The audited method is called the **A module** in current method documents.",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Baseline Probability Profile",
        "",
        f"- Model: `{baseline.model}`",
        f"- Overall Acc / AUC: **{baseline.metrics_acc:.2f} / {baseline.metrics_auc:.2f}**",
        f"- Final Acc / AUC: **{baseline.metrics_final_acc:.2f} / {baseline.metrics_final_auc:.2f}**"
        if baseline.metrics_final_acc is not None and baseline.metrics_final_auc is not None
        else "- Final Acc / AUC: --",
        f"- Overall: AUC **{baseline.metrics_auc:.2f}**, pred-True **{format_pct(baseline.overall.pred_pos_frac)}**, "
        f"pos/neg mean **{baseline.overall.pos_mean:.4f}/{baseline.overall.neg_mean:.4f}**, gap **{baseline.overall.gap:.4f}**, std **{baseline.overall.std:.4f}**",
        f"- Final: AUC **{baseline.final.auc * 100:.2f}**, pred-True **{format_pct(baseline.final.pred_pos_frac)}**, "
        f"pos/neg mean **{baseline.final.pos_mean:.4f}/{baseline.final.neg_mean:.4f}**, gap **{baseline.final.gap:.4f}**",
        f"- Non-final AUC: **{baseline.non_final_auc:.2f}**",
        "",
        "## Current Formal State",
        "",
    ]

    if task_status:
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
        lines.append(
            f"- Strict formal winner gate: `Overall Acc > {baseline.metrics_acc:.2f}` and `Overall AUC > {baseline.metrics_auc:.2f}`."
        )
        if not winner_found and round3:
            lines.append(
                "- Round 3 strict full-train is still pending completion, so the probability analysis below only covers already-synced eval artifacts."
            )
            for row in round3:
                lines.append(f"- `{row['model']}`: `{format_round_status(row)}`")
    else:
        lines.append("- Task-conditioned formal-state file not available.")

    lines.extend(
        [
            "",
        "## CEL vs Baseline",
        "",
        ]
    )

    if not rows:
        lines.append("_No CEL qual files available yet._")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Model | Type | Overall Acc | Overall AUC | Delta Overall Acc | Delta Overall AUC | Formal Gate | Non-final AUC | Final Acc | Delta Final Acc | Final AUC | Delta Final AUC | Pred True | Gap | Final Gap | Better/Worse | Notes |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for row in sorted(rows, key=lambda item: (item.role != "formal", -item.metrics_auc)):
        delta_acc = row.metrics_acc - baseline.metrics_acc
        delta_auc = row.metrics_auc - baseline.metrics_auc
        delta_final_acc = "--"
        delta_final_auc = "--"
        if row.metrics_final_acc is not None and baseline.metrics_final_acc is not None:
            delta_final_acc = f"{row.metrics_final_acc - baseline.metrics_final_acc:+.2f}"
        if row.metrics_final_auc is not None and baseline.metrics_final_auc is not None:
            delta_final_auc = f"{row.metrics_final_auc - baseline.metrics_final_auc:+.2f}"
        better_worse = "--"
        if row.abs_better is not None and row.abs_worse is not None and row.abs_same is not None:
            better_worse = f"{row.abs_better}/{row.abs_worse}/{row.abs_same}"
        gate = "diagnostic" if row.role == "diagnostic" else formal_gate_status(row, baseline)
        final_acc = f"{row.metrics_final_acc:.2f}" if row.metrics_final_acc is not None else "--"
        final_auc = f"{row.metrics_final_auc:.2f}" if row.metrics_final_auc is not None else "--"
        lines.append(
            f"| `{row.model}` | {row.role} | {row.metrics_acc:.2f} | {row.metrics_auc:.2f} | {delta_acc:+.2f} | {delta_auc:+.2f} | {gate} | {row.non_final_auc:.2f} | "
            f"{final_acc} | {delta_final_acc} | {final_auc} | {delta_final_auc} | {format_pct(row.overall.pred_pos_frac)} | {row.overall.gap:.4f} | "
            f"{row.final.gap:.4f} | {better_worse} | {row.note} |"
        )

    formal_rows = [row for row in rows if row.role == "formal"]
    diagnostic_rows = [row for row in rows if row.role == "diagnostic"]
    best_formal_auc = max(formal_rows, key=lambda item: item.metrics_auc) if formal_rows else None
    formal_winners = [row for row in formal_rows if formal_gate_status(row, baseline) == "yes"]
    best_diag_auc = max(diagnostic_rows, key=lambda item: item.metrics_auc) if diagnostic_rows else None
    best_gap = max(rows, key=lambda item: item.overall.gap)
    lines.extend(
        [
            "",
            "## Current Read",
            "",
        ]
    )
    if formal_winners:
        best_formal_winner = max(formal_winners, key=lambda item: (item.metrics_acc, item.metrics_auc))
        lines.append(
            f"- Current strict formal winner in synced artifacts: `{best_formal_winner.model}` with Overall Acc / AUC **{best_formal_winner.metrics_acc:.2f} / {best_formal_winner.metrics_auc:.2f}**."
        )
    elif best_formal_auc is not None:
        lines.append(
            f"- No synced formal model currently clears both gates. Best formal AUC anchor remains `{best_formal_auc.model}` at Overall Acc / AUC **{best_formal_auc.metrics_acc:.2f} / {best_formal_auc.metrics_auc:.2f}**."
        )
    if best_diag_auc is not None:
        lines.append(
            f"- Best diagnostic-only result so far: `{best_diag_auc.model}` at Overall Acc / AUC **{best_diag_auc.metrics_acc:.2f} / {best_diag_auc.metrics_auc:.2f}**."
        )
        lines.append("- That diagnostic result remains excluded from formal winner judgment because it is not a complete strict full-train method result.")
    lines.extend(
        [
            f"- Largest overall probability gap so far: `{best_gap.model}` at **{best_gap.overall.gap:.4f}**.",
        ]
    )
    reference_model = best_formal_auc.model if best_formal_auc is not None else best_gap.model
    if best_gap.model != reference_model:
        lines.append(
            "- The model with the widest positive/negative mean separation is not currently the best-AUC model, "
            "so ranking quality and class bias still matter more than average probability spread alone."
        )

    return "\n".join(lines) + "\n"


def load_baseline(baseline_metrics: Path, baseline_qual: Path) -> ModelAnalysis | None:
    metrics = parse_metrics(baseline_metrics)
    if metrics is None or not baseline_qual.exists():
        return None
    rows = parse_eval_rows(baseline_qual)
    overall = summarize(rows)
    final_rows = [row for row in rows if row.is_final]
    non_final_rows = [row for row in rows if not row.is_final]
    final = summarize(final_rows)
    return ModelAnalysis(
        model=metrics["model"],
        metrics_acc=metrics["overall_acc"],
        metrics_auc=metrics["overall_auc"],
        metrics_final_acc=metrics["final_acc"],
        metrics_final_auc=metrics["final_auc"],
        diag=metrics["diag"],
        role="formal",
        overall=overall,
        final=final,
        non_final_auc=compute_auc(non_final_rows) * 100.0,
        abs_better=None,
        abs_worse=None,
        abs_same=None,
        note="--",
    )


def load_model_analysis(metrics_path: Path, qual_dir: Path, baseline: ModelAnalysis, baseline_rows: list[EvalRow]) -> ModelAnalysis | None:
    metrics = parse_metrics(metrics_path)
    if metrics is None:
        return None
    qual_path = qual_dir / f"qual_{metrics['model']}.csv"
    if not qual_path.exists():
        return None
    rows = parse_eval_rows(qual_path)
    overall = summarize(rows)
    final_rows = [row for row in rows if row.is_final]
    non_final_rows = [row for row in rows if not row.is_final]
    final = summarize(final_rows)
    abs_better, abs_worse, abs_same = compare_abs_error(rows, baseline_rows)
    non_final_auc = compute_auc(non_final_rows) * 100.0
    note = build_note(
        model=metrics["model"],
        metrics_auc=metrics["overall_auc"],
        overall=overall,
        final=final,
        non_final_auc=non_final_auc,
        baseline=baseline,
        abs_better=abs_better,
        abs_worse=abs_worse,
        diag=metrics["diag"],
    )
    return ModelAnalysis(
        model=metrics["model"],
        metrics_acc=metrics["overall_acc"],
        metrics_auc=metrics["overall_auc"],
        metrics_final_acc=metrics["final_acc"],
        metrics_final_auc=metrics["final_auc"],
        diag=metrics["diag"],
        role=model_role(metrics["model"]),
        overall=overall,
        final=final,
        non_final_auc=non_final_auc,
        abs_better=abs_better,
        abs_worse=abs_worse,
        abs_same=abs_same,
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--baseline_qual", default="results/baseline/qual/qual_lmkt_qwen3_1.7b_recert_20260620.csv")
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--qual_dir", default="results/cel_stage1_last_layer/qual")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/DETAILED_ANALYSIS.md")
    args = parser.parse_args()

    baseline = load_baseline(Path(args.baseline_metrics), Path(args.baseline_qual))
    if baseline is None:
        Path(args.out).write_text(
            "# CEL Stage1 Last-Layer Detailed Analysis\n\n_Baseline metrics or qual file not available yet._\n",
            encoding="utf-8",
        )
        return

    baseline_rows = parse_eval_rows(Path(args.baseline_qual))
    analyses: list[ModelAnalysis] = []
    metrics_dir = Path(args.metrics_dir)
    qual_dir = Path(args.qual_dir)
    for metrics_path in sorted(metrics_dir.glob("metrics_*.txt")):
        analysis = load_model_analysis(metrics_path, qual_dir, baseline, baseline_rows)
        if analysis is not None:
            analyses.append(analysis)

    task_status = load_task_status(Path(args.task_status))
    Path(args.out).write_text(build_report(baseline, analyses, task_status), encoding="utf-8")


if __name__ == "__main__":
    main()
