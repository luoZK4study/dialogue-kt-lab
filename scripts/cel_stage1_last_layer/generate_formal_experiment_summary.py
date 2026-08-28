#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATE_ORDER, FORMAL_CANDIDATES


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_metric(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


def fmt_threshold(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.4f}"


def fmt_pct(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}%"


def metric_text(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def choose_analysis(row: dict, baseline: dict, anchor: dict) -> str:
    status = str(row.get("status") or "unknown")
    audit_state = str(row.get("audit") or "unknown")
    beats_baseline = str(row.get("beats_baseline") or "--")
    metrics = row.get("metrics") or {}
    analysis = row.get("analysis") or {}

    if status in {"running", "pending"}:
        return "实验仍在进行，当前还不能做正式 winner 判断；需要等待完整 `train + val + test` 落盘后再 closeout。"
    if audit_state == "needs_rerun":
        return "该候选尚未形成可记录的完整 formal 结果，必须先按同一 strict full-train 口径完成重跑或补齐 closeout。"
    if beats_baseline == "yes":
        return "该候选已经在完整 SSH full-train 口径下同时超过 baseline 的 Overall Acc 和 Overall AUC，可视为 formal winner。"

    overall_acc = metrics.get("overall_acc")
    overall_auc = metrics.get("overall_auc")
    baseline_acc = baseline.get("overall_acc")
    baseline_auc = baseline.get("overall_auc")
    anchor_auc = anchor.get("overall_auc")
    pred_true = analysis.get("pred_true")

    fragments: list[str] = []
    if overall_auc is not None and baseline_auc is not None and overall_auc >= baseline_auc:
        if overall_acc is not None and baseline_acc is not None and overall_acc < baseline_acc:
            fragments.append("AUC 仍高于或接近 baseline，但固定阈值下的 Acc 仍未过线")
    elif overall_auc is not None and baseline_auc is not None and overall_auc < baseline_auc:
        if overall_acc is not None and baseline_acc is not None and overall_acc < baseline_acc:
            fragments.append("AUC 与 Acc 都未能同时保住，说明排序与阈值表现都没有完成 formal gate")
        else:
            fragments.append("Acc 没有形成稳定优势，同时 AUC 也落到 baseline 以下")

    if anchor_auc is not None and overall_auc is not None and overall_auc < anchor_auc:
        fragments.append("相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住")
    if pred_true is not None:
        fragments.append(f"`Pred True` 为 {pred_true:.2f}%")

    if not fragments:
        return "该候选已完整落盘，但没有同时超过 baseline 的 Overall Acc 和 Overall AUC，因此只能记录为 formal non-winner。"
    return "；".join(fragments) + "。"


def build_summary(result_dir: Path) -> str:
    status = load_json(result_dir / "task_conditioned_status.json")
    audit = load_json(result_dir / "formal_experiment_audit.json")

    baseline = status.get("baseline") or {}
    anchor = status.get("v1") or {}
    rows = {
        str(row.get("model")): row
        for row in (audit.get("rows") or [])
        if row.get("model")
    }
    counts = audit.get("counts") or {}

    active_rows = [
        row for row in (audit.get("rows") or [])
        if str(row.get("status")) == "running"
    ]
    active_text = ", ".join(str(row.get("label")) for row in active_rows) or "none"

    lines: list[str] = []
    lines.append("# Formal Experiment Summary")
    lines.append("")
    lines.append("> Terminology note: candidate identifiers in this historical table are retained for provenance. The recorded successful design is referred to as the **A module** outside the experiment ledger.")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("- Scope: strict formal `task_conditioned` candidates only; diagnostic or post-hoc calibration runs are excluded from method judgment.")
    lines.append(f"- Current next action: `{status.get('next_action')}`")
    lines.append(f"- Winner found: `{status.get('winner_found')}`")
    lines.append(f"- Active formal candidate(s): `{active_text}`")
    lines.append(
        f"- Baseline gate: `Overall Acc > {metric_text(baseline.get('overall_acc'))}` and `Overall AUC > {metric_text(baseline.get('overall_auc'))}`"
    )
    lines.append(
        f"- Historical selector/ranking comparison anchor: `{anchor.get('model', '--')}` with Overall Acc / AUC `{metric_text(anchor.get('overall_acc'))} / {metric_text(anchor.get('overall_auc'))}`"
    )
    if counts:
        counts_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- Audit counts: `{counts_text}`")

    lines.append("")
    lines.append("## Candidate Table")
    lines.append("")
    lines.append("| Candidate | Start Point | Single Variable | Status | Audit | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Pred True | Beats Baseline |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        row = rows.get(model_name, {})
        metrics = row.get("metrics") or {}
        analysis = row.get("analysis") or {}
        lines.append(
            "| "
            + f"`{meta['label']}` / `{model_name}`"
            + f" | {meta.get('start_point_cn', '--')}"
            + f" | {meta.get('single_variable_cn', '--')}"
            + f" | {row.get('status', 'missing')}"
            + f" | `{row.get('audit', 'missing')}`"
            + f" | {fmt_metric(metrics.get('overall_acc'))}"
            + f" | {fmt_metric(metrics.get('overall_auc'))}"
            + f" | {fmt_metric(metrics.get('final_acc'))}"
            + f" | {fmt_metric(metrics.get('final_auc'))}"
            + f" | {fmt_metric(analysis.get('delta_acc_vs_baseline'))}"
            + f" | {fmt_metric(analysis.get('delta_auc_vs_baseline'))}"
            + f" | {fmt_metric(analysis.get('delta_final_acc_vs_baseline'))}"
            + f" | {fmt_metric(analysis.get('delta_final_auc_vs_baseline'))}"
            + f" | {fmt_pct(analysis.get('pred_true'))}"
            + f" | {row.get('beats_baseline', '--')}"
            + " |"
        )

    lines.append("")
    lines.append("## Per-Candidate Analysis")
    lines.append("")

    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        row = rows.get(model_name, {})
        metrics = row.get("metrics") or {}
        analysis = row.get("analysis") or {}

        lines.append(f"### `{meta['label']}` / `{model_name}`")
        lines.append("")
        lines.append(f"- 起点：{meta.get('start_point_cn', '--')}")
        lines.append(f"- 方法：{meta.get('method_cn', '--')}")
        lines.append(f"- 单变量：{meta.get('single_variable_cn', '--')}")
        lines.append(f"- 假设：{meta.get('hypothesis_cn', '--')}")
        lines.append(f"- 当前状态：`{row.get('status', 'missing')}` / audit=`{row.get('audit', 'missing')}` / beats_baseline=`{row.get('beats_baseline', '--')}`")
        lines.append(
            "- 当前结果："
            + f"Overall Acc / AUC `{fmt_metric(metrics.get('overall_acc'))} / {fmt_metric(metrics.get('overall_auc'))}`；"
            + f"Final Acc / AUC `{fmt_metric(metrics.get('final_acc'))} / {fmt_metric(metrics.get('final_auc'))}`；"
            + f"`Pred True` `{fmt_pct(analysis.get('pred_true'))}`；"
            + f"best Acc `{fmt_metric(analysis.get('best_acc'))}` @ threshold `{fmt_threshold(analysis.get('best_threshold'))}`"
        )
        lines.append(
            "- 相对 baseline："
            + f"delta Acc `{fmt_metric(analysis.get('delta_acc_vs_baseline'))}`；"
            + f"delta AUC `{fmt_metric(analysis.get('delta_auc_vs_baseline'))}`；"
            + f"delta Final Acc `{fmt_metric(analysis.get('delta_final_acc_vs_baseline'))}`；"
            + f"delta Final AUC `{fmt_metric(analysis.get('delta_final_auc_vs_baseline'))}`"
        )
        lines.append(
            "- 相对 anchor `v1`："
            + f"delta Acc `{fmt_metric(analysis.get('delta_acc_vs_anchor'))}`；"
            + f"delta AUC `{fmt_metric(analysis.get('delta_auc_vs_anchor'))}`"
        )
        lines.append(f"- 分析：{choose_analysis(row, baseline, anchor)}")
        note = row.get("note")
        if note:
            lines.append(f"- Closeout note: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/cel_stage1_last_layer")
    parser.add_argument(
        "--out",
        default="results/cel_stage1_last_layer/FORMAL_EXPERIMENT_SUMMARY.md",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    summary = build_summary(result_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
