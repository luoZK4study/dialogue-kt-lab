#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATE_ORDER, FORMAL_CANDIDATES


LOSS_RE = re.compile(r"^Loss:\s+([0-9.]+)$", re.MULTILINE)
OVERALL_RE = re.compile(
    r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)

REQUIRED_SURFACES = [
    {
        "label": "status",
        "path": "results/cel_stage1_last_layer/STATUS.md",
        "metric_fields": (),
        "analysis_fields": (),
    },
    {
        "label": "strict report",
        "path": "results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": (
            "pred_true",
            "best_acc",
            "best_threshold",
            "delta_acc_vs_baseline",
            "delta_auc_vs_baseline",
            "delta_final_acc_vs_baseline",
            "delta_final_auc_vs_baseline",
            "delta_acc_vs_anchor",
            "delta_auc_vs_anchor",
        ),
    },
    {
        "label": "stage1 record",
        "path": "results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": ("delta_final_acc_vs_baseline", "delta_final_auc_vs_baseline"),
    },
    {
        "label": "tuning loop",
        "path": "results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": ("delta_final_acc_vs_baseline", "delta_final_auc_vs_baseline"),
    },
    {
        "label": "comparison",
        "path": "results/cel_stage1_last_layer/COMPARISON.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": (
            "delta_acc_vs_baseline",
            "delta_auc_vs_baseline",
            "delta_final_acc_vs_baseline",
            "delta_final_auc_vs_baseline",
        ),
    },
    {
        "label": "detailed analysis",
        "path": "results/cel_stage1_last_layer/DETAILED_ANALYSIS.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": (
            "delta_acc_vs_baseline",
            "delta_auc_vs_baseline",
            "delta_final_acc_vs_baseline",
            "delta_final_auc_vs_baseline",
        ),
    },
    {
        "label": "formal candidate briefs",
        "path": "results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md",
        "metric_fields": ("overall_acc", "overall_auc", "final_acc", "final_auc"),
        "analysis_fields": (
            "pred_true",
            "best_acc",
            "best_threshold",
            "delta_acc_vs_baseline",
            "delta_auc_vs_baseline",
            "delta_final_acc_vs_baseline",
            "delta_final_auc_vs_baseline",
            "delta_acc_vs_anchor",
            "delta_auc_vs_anchor",
        ),
    },
    {
        "label": "formal candidate config diffs",
        "path": "results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md",
        "metric_fields": (),
        "analysis_fields": (),
    },
]
REFRESH_COMPLETE_RE = re.compile(r"task_conditioned refresh complete :: sync=([a-z_]+)")


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
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "final_acc": float(final_match.group(1)) if final_match else None,
        "final_auc": float(final_match.group(2)) if final_match else None,
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def round3_status_map(task_status: dict | None) -> dict[str, dict]:
    if not task_status:
        return {}
    for round_info in task_status.get("rounds", []):
        if "Round 3" in (round_info.get("title") or ""):
            return {
                row["model"]: row
                for row in (round_info.get("models") or [])
                if row.get("model")
            }
    return {}


def build_analysis(status_row: dict | None, metrics: dict | None, baseline: dict | None, anchor: dict | None) -> dict | None:
    analysis = {
        "pred_true": status_row.get("pred_true") if status_row else None,
        "best_acc": status_row.get("best_acc") if status_row else None,
        "best_threshold": status_row.get("best_threshold") if status_row else None,
        "delta_acc_vs_baseline": None,
        "delta_auc_vs_baseline": None,
        "delta_final_acc_vs_baseline": None,
        "delta_final_auc_vs_baseline": None,
        "delta_acc_vs_anchor": None,
        "delta_auc_vs_anchor": None,
    }
    if metrics is not None and baseline is not None:
        analysis["delta_acc_vs_baseline"] = metrics["overall_acc"] - baseline["overall_acc"]
        analysis["delta_auc_vs_baseline"] = metrics["overall_auc"] - baseline["overall_auc"]
        if metrics.get("final_acc") is not None and baseline.get("final_acc") is not None:
            analysis["delta_final_acc_vs_baseline"] = metrics["final_acc"] - baseline["final_acc"]
        if metrics.get("final_auc") is not None and baseline.get("final_auc") is not None:
            analysis["delta_final_auc_vs_baseline"] = metrics["final_auc"] - baseline["final_auc"]
    if metrics is not None and anchor is not None:
        analysis["delta_acc_vs_anchor"] = metrics["overall_acc"] - anchor["overall_acc"]
        analysis["delta_auc_vs_anchor"] = metrics["overall_auc"] - anchor["overall_auc"]
    if any(value is not None for value in analysis.values()):
        return analysis
    return None


def has_full_result_artifacts(model_name: str, metrics_dir: Path, qual_dir: Path) -> bool:
    return (
        (metrics_dir / f"metrics_{model_name}.txt").exists()
        and (qual_dir / f"qual_{model_name}.csv").exists()
    )


def has_complete_metric_set(metrics: dict | None) -> bool:
    return bool(
        metrics
        and all(
            metrics.get(field) is not None
            for field in ("overall_acc", "overall_auc", "final_acc", "final_auc")
        )
    )


def has_log_artifact(model_name: str) -> bool:
    return Path(FORMAL_CANDIDATES[model_name]["stdout_log_rel"]).exists()


def has_method_specific_audit(model_name: str) -> bool:
    meta = FORMAL_CANDIDATES[model_name]
    if meta.get("launch_key") != "v26":
        return True
    audit = load_json(Path("results/cel_stage1_last_layer/V26_SELFTRAINED_AUDIT.json"))
    return bool(audit and audit.get("audit_pass"))


def latest_refresh_event(path: Path) -> str | None:
    events = [line.strip() for line in read_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_complete = [line for line in events if "task_conditioned refresh complete" in line]
    return refresh_complete[-1] if refresh_complete else events[-1]


def latest_successful_refresh_event(path: Path) -> str | None:
    events = [line.strip() for line in read_text(path).splitlines() if re.match(r"^\[\d{4}-\d{2}-\d{2}", line)]
    if not events:
        return None
    refresh_success = [
        line
        for line in events
        if "task_conditioned refresh complete" in line and "sync=ok" in line
    ]
    return refresh_success[-1] if refresh_success else None


def sync_freshness_label(refresh_event: str | None) -> str:
    if refresh_event is None:
        return "unknown"
    match = REFRESH_COMPLETE_RE.search(refresh_event)
    if not match:
        return "unknown"
    sync = match.group(1)
    if sync == "ok":
        return "fresh_remote"
    if sync == "failed":
        return "stale_local_failed_refresh"
    if sync == "timeout":
        return "stale_local_timed_out_refresh"
    return "unknown"


def surface_metric_tokens(metrics: dict | None, fields: tuple[str, ...]) -> list[str]:
    if metrics is None:
        return []
    tokens: list[str] = []
    for field in fields:
        value = metrics.get(field)
        if value is None:
            continue
        tokens.append(f"{value:.2f}")
    return tokens


def surface_analysis_tokens(analysis: dict | None, fields: tuple[str, ...]) -> list[str]:
    if analysis is None:
        return []
    tokens: list[str] = []
    for field in fields:
        value = analysis.get(field)
        if value is None:
            continue
        digits = 3 if field == "best_threshold" else 2
        suffix = "%" if field == "pred_true" else ""
        tokens.append(f"{value:.{digits}f}{suffix}")
    return tokens


def candidate_surface_coverage(model_name: str, metrics: dict | None, analysis: dict | None, status: str | None) -> tuple[bool, list[str]]:
    missing: list[str] = []
    require_metric_coverage = status == "done" and metrics is not None
    require_failure_coverage = status == "failed"
    for surface in REQUIRED_SURFACES:
        path = Path(surface["path"])
        label = surface["label"]
        if not path.exists():
            missing.append(label)
            continue
        text = read_text(path)
        if model_name not in text:
            missing.append(label)
            continue
        if require_metric_coverage:
            tokens = surface_metric_tokens(metrics, surface["metric_fields"])
            if any(token not in text for token in tokens):
                missing.append(label)
                continue
            analysis_tokens = surface_analysis_tokens(analysis, surface["analysis_fields"])
            if any(token not in text for token in analysis_tokens):
                missing.append(label)
        elif require_failure_coverage and label in {
            "status",
            "strict report",
            "stage1 record",
            "tuning loop",
            "formal candidate briefs",
        }:
            if "failed" not in text and "traceback" not in text and "rerun" not in text:
                missing.append(label)
    return not missing, missing


def format_status(status: str | None, failure_reason: str | None) -> str:
    if status == "failed" and failure_reason:
        return f"{status} ({failure_reason})"
    return status or "unknown"


def audit_row(
    model_name: str,
    baseline: dict | None,
    anchor: dict | None,
    task_status_map: dict[str, dict],
    metrics_dir: Path,
    qual_dir: Path,
) -> dict:
    status_row = task_status_map.get(model_name, {})
    metrics = parse_metrics(metrics_dir / f"metrics_{model_name}.txt")
    analysis = build_analysis(status_row, metrics, baseline, anchor)
    artifacts_ok = has_full_result_artifacts(model_name, metrics_dir, qual_dir)
    metrics_complete = has_complete_metric_set(metrics)
    log_ok = has_log_artifact(model_name)
    status = status_row.get("status")
    coverage_ok, missing_surfaces = candidate_surface_coverage(model_name, metrics, analysis, status)
    method_audit_ok = has_method_specific_audit(model_name)
    is_complete = status == "done" and artifacts_ok and metrics_complete and log_ok and method_audit_ok
    beats = "--"
    if metrics is not None and baseline is not None:
        acc_win = metrics["overall_acc"] > baseline["overall_acc"]
        auc_win = metrics["overall_auc"] > baseline["overall_auc"]
        beats = "yes" if acc_win and auc_win else "no"
    audit = "pending"
    note = "Waiting on SSH full-train completion."
    if status == "failed":
        audit = "needs_rerun"
        note = "Full-train candidate failed; keep scope fixed and rerun only after the failure is understood."
    elif is_complete and coverage_ok:
        audit = "recorded"
        note = "Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces."
    elif is_complete and not coverage_ok:
        audit = "needs_docs"
        note = f"Full-train artifacts are present, but markdown result coverage is incomplete: {', '.join(missing_surfaces)}."
    elif status == "done" and not artifacts_ok:
        audit = "missing_artifacts"
        note = "Run is marked done, but synced metrics/qual artifacts are incomplete."
    elif status == "done" and not metrics_complete:
        audit = "incomplete_metrics"
        note = "Run is marked done, but its metrics do not contain the required Overall and Final Acc/AUC values."
    elif status == "done" and artifacts_ok and not log_ok:
        audit = "missing_log"
        note = "Run is marked done, but the synced stdout log is missing."
    elif status == "done" and artifacts_ok and log_ok and not method_audit_ok:
        audit = "method_audit_failed"
        note = "Run artifacts exist, but the v26 self-trained provenance/parameter-change audit is missing or failed."

    return {
        "model": model_name,
        "label": FORMAL_CANDIDATES[model_name]["label"],
        "status": format_status(status, status_row.get("failure_reason")),
        "artifacts_ok": artifacts_ok,
        "metrics_complete": metrics_complete,
        "log_ok": log_ok,
        "coverage_ok": coverage_ok,
        "missing_surfaces": missing_surfaces,
        "audit": audit,
        "beats": beats,
        "metrics": metrics,
        "analysis": analysis,
        "note": note,
    }


def build_report(
    baseline: dict | None,
    task_status: dict | None,
    rows: list[dict],
    refresh_event: str | None,
    authoritative_refresh_event: str | None,
    sync_freshness: str,
) -> str:
    lines = [
        "# Formal Experiment Audit",
        "",
        "- Scope: only strict SSH-side full `train + val + test` candidates are audited here.",
        "- Purpose: verify that every completed formal candidate has synced artifacts and has been reflected across the required markdown surfaces before any winner claim.",
        "",
        "## Audit Rule",
        "",
        "1. A formal candidate is only completion-ready after synced `metrics`, `qual`, and `stdout log` all exist, and its metrics contain Overall Acc/AUC plus Final Acc/AUC.",
        "2. A completed formal candidate must also be visible in `STATUS.md`, `STRICT_FULL_TRAIN_REPORT.md`, `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`, `TASK_CONDITIONED_TUNING_LOOP.md`, `COMPARISON.md`, `DETAILED_ANALYSIS.md`, `FORMAL_CANDIDATE_BRIEFS.md`, and `FORMAL_CANDIDATE_CONFIG_DIFFS.md`.",
        "   For completed formal candidates, the core markdown surfaces must carry Overall Acc / AUC and Final Acc / AUC. The strict report, Stage 1 record, tuning loop, `COMPARISON.md`, `DETAILED_ANALYSIS.md`, and candidate briefs must explicitly expose both Overall and Final deltas versus baseline. The strict report and candidate briefs must also carry `Pred True`, the best-threshold diagnostic, and anchor deltas when available.",
        "3. Winner judgment is only allowed after the above audit is clean and the candidate beats baseline on both Overall Acc and Overall AUC.",
        "4. Failed strict full-train candidates must still be reflected in `STATUS.md`, the strict report, the Stage 1 record, the tuning loop, and the candidate briefs before they count as properly analyzed rerun targets.",
        "",
        "## Current Gate",
        "",
    ]

    if baseline is not None:
        lines.append(
            f"- Baseline gate: `Overall Acc > {baseline['overall_acc']:.2f}` and `Overall AUC > {baseline['overall_auc']:.2f}`."
        )
    if task_status is not None:
        lines.append(f"- Current next action: `{task_status.get('next_action')}`")
        lines.append(f"- Winner found: `{task_status.get('winner_found')}`")
    lines.append(f"- Task-conditioned sync freshness: `{sync_freshness}`")
    lines.append(f"- Latest authoritative task-conditioned remote refresh event: `{authoritative_refresh_event or 'none'}`")
    lines.append(f"- Latest task-conditioned refresh attempt: `{refresh_event or 'none'}`")
    if sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
        lines.append("- Current audit is based on an older synced local snapshot; missing artifacts may still reflect unsynced SSH state rather than true remote absence.")
    elif sync_freshness == "fresh_remote":
        lines.append("- Current audit is backed by a fresh SSH sync.")

    lines.extend(
        [
            "",
            "## Candidate Audit",
            "",
            "| Candidate | Status | Metrics+Qual | Metric Set | Stdout Log | Markdown Coverage | Beats Baseline? | Audit State |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            f"| `{row['label']}` / `{row['model']}` | {row['status']} | "
            f"{'ok' if row['artifacts_ok'] else 'missing'} | "
            f"{'complete' if row['metrics_complete'] else 'incomplete'} | "
            f"{'ok' if row['log_ok'] else 'missing'} | "
            f"{'ok' if row['coverage_ok'] else 'missing'} | "
            f"{row['beats']} | `{row['audit']}` |"
        )

    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(f"- `{row['label']}`: {row['note']}")
        if row["missing_surfaces"]:
            lines.append(
                f"- `{row['label']}` missing markdown surfaces: {', '.join(row['missing_surfaces'])}."
            )
        metrics = row["metrics"]
        analysis = row.get("analysis") or {}
        if metrics is not None:
            lines.append(
                f"- `{row['label']}` synced Overall Acc / AUC: **{metrics['overall_acc']:.2f} / {metrics['overall_auc']:.2f}**."
            )
            lines.append(
                f"- `{row['label']}` synced Final Acc / AUC: **{metrics['final_acc']:.2f} / {metrics['final_auc']:.2f}**."
                if metrics.get("final_acc") is not None and metrics.get("final_auc") is not None
                else f"- `{row['label']}` synced Final Acc / AUC: **{metrics.get('final_acc', '--')} / {metrics.get('final_auc', '--')}**."
            )
        if analysis.get("pred_true") is not None:
            lines.append(f"- `{row['label']}` synced `Pred True`: **{analysis['pred_true']:.2f}%**.")
        if analysis.get("best_acc") is not None or analysis.get("best_threshold") is not None:
            lines.append(
                f"- `{row['label']}` best-threshold diagnostic: best Acc **{analysis.get('best_acc'):.2f}** at threshold **{analysis.get('best_threshold'):.3f}**."
            )
        if analysis.get("delta_acc_vs_baseline") is not None or analysis.get("delta_auc_vs_baseline") is not None:
            lines.append(
                f"- `{row['label']}` delta vs baseline Overall Acc / AUC: **{analysis.get('delta_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_auc_vs_baseline', 0.0):+.2f}**."
            )
        if analysis.get("delta_final_acc_vs_baseline") is not None or analysis.get("delta_final_auc_vs_baseline") is not None:
            lines.append(
                f"- `{row['label']}` delta vs baseline Final Acc / AUC: **{analysis.get('delta_final_acc_vs_baseline', 0.0):+.2f} / {analysis.get('delta_final_auc_vs_baseline', 0.0):+.2f}**."
            )
        if analysis.get("delta_acc_vs_anchor") is not None or analysis.get("delta_auc_vs_anchor") is not None:
            lines.append(
                f"- `{row['label']}` delta vs anchor `v1` Overall Acc / AUC: **{analysis.get('delta_acc_vs_anchor', 0.0):+.2f} / {analysis.get('delta_auc_vs_anchor', 0.0):+.2f}**."
            )

    complete_rows = [row for row in rows if row["audit"] == "recorded"]
    lines.extend(["", "## Current Read", ""])
    if complete_rows:
        lines.append("- Every completed formal candidate currently has the required artifacts and markdown coverage.")
    else:
        lines.append("- No completed formal candidate has yet passed the full artifact-plus-markdown audit.")
    if sync_freshness in {"stale_local_failed_refresh", "stale_local_timed_out_refresh"}:
        lines.append("- Because the latest authoritative refresh is stale, this audit is safe for local consistency checks but not for claiming new remote-state completions that have not synced yet.")
    lines.append("- Keep using this audit together with `STRICT_FULL_TRAIN_REPORT.md` before declaring any new Stage 1 formal winner.")
    return "\n".join(lines) + "\n"


def build_audit_state(
    baseline: dict | None,
    task_status: dict | None,
    rows: list[dict],
    refresh_event: str | None,
    authoritative_refresh_event: str | None,
    sync_freshness: str,
) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["audit"]] = counts.get(row["audit"], 0) + 1
    return {
        "baseline": {
            "overall_acc": baseline["overall_acc"] if baseline is not None else None,
            "overall_auc": baseline["overall_auc"] if baseline is not None else None,
            "final_acc": baseline["final_acc"] if baseline is not None else None,
            "final_auc": baseline["final_auc"] if baseline is not None else None,
        },
        "task_status": {
            "next_action": task_status.get("next_action") if task_status else None,
            "winner_found": bool(task_status.get("winner_found")) if task_status else False,
        },
        "sync_freshness": sync_freshness,
        "latest_refresh_event": refresh_event,
        "latest_successful_refresh_event": authoritative_refresh_event,
        "counts": counts,
        "rows": [
            {
                "model": row["model"],
                "label": row["label"],
                "status": row["status"],
                "artifacts_ok": row["artifacts_ok"],
                "metrics_complete": row["metrics_complete"],
                "log_ok": row["log_ok"],
                "coverage_ok": row["coverage_ok"],
                "missing_surfaces": row["missing_surfaces"],
                "beats_baseline": row["beats"],
                "audit": row["audit"],
                "note": row["note"],
                "metrics": row["metrics"],
                "analysis": row["analysis"],
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_status", default="results/cel_stage1_last_layer/task_conditioned_status.json")
    parser.add_argument("--baseline_metrics", default="results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt")
    parser.add_argument("--anchor_metrics", default="results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v1_qwen3_1.7b.txt")
    parser.add_argument("--metrics_dir", default="results/cel_stage1_last_layer/metrics")
    parser.add_argument("--qual_dir", default="results/cel_stage1_last_layer/qual")
    parser.add_argument("--refresh_log", default="results/cel_stage1_last_layer/task_conditioned_refresh.log")
    parser.add_argument("--out", default="results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md")
    parser.add_argument("--json_out", default="results/cel_stage1_last_layer/formal_experiment_audit.json")
    args = parser.parse_args()

    task_status = load_json(Path(args.task_status))
    baseline = parse_metrics(Path(args.baseline_metrics))
    anchor = parse_metrics(Path(args.anchor_metrics))
    refresh_event = latest_refresh_event(Path(args.refresh_log))
    authoritative_refresh_event = latest_successful_refresh_event(Path(args.refresh_log)) or refresh_event
    sync_freshness = sync_freshness_label(refresh_event)
    task_status_map = round3_status_map(task_status)
    metrics_dir = Path(args.metrics_dir)
    qual_dir = Path(args.qual_dir)
    rows = [
        audit_row(model_name, baseline, anchor, task_status_map, metrics_dir, qual_dir)
        for model_name in FORMAL_CANDIDATE_ORDER
    ]
    Path(args.out).write_text(
        build_report(baseline, task_status, rows, refresh_event, authoritative_refresh_event, sync_freshness),
        encoding="utf-8",
    )
    Path(args.json_out).write_text(
        json.dumps(
            build_audit_state(
                baseline,
                task_status,
                rows,
                refresh_event,
                authoritative_refresh_event,
                sync_freshness,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
