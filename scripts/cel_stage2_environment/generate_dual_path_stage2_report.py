#!/usr/bin/env python3

"""Summarize completed dual-path Stage 2 rounds without running experiments."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cel_stage2_environment/dual_path"
A_MODULE_OVERALL_AUC = 75.38
PAPER_OVERALL_AUC = 76.71
CONTRACT_SUCCESS_MARKER = "Stage 2 tensor contracts passed on"

AUDIT_FIELDS = (
    "audit_pass",
    "identity_ok",
    "warmup_freeze_ok",
    "b_warmup_freeze_ok",
    "a_joint_update_ok",
    "joint_update_ok",
    "selector_fp32_ok",
    "environment_fp32_ok",
    "complement_contract_ok",
)
CONFIG_FIELDS = (
    "env_mode",
    "env_beta",
    "env_hidden_dim",
    "env_num_layers",
    "env_num_heads",
    "env_ffn_dim",
    "env_output_postprocess",
    "env_output_ratio",
    "lambda_r",
    "lambda_m",
    "lambda_cons",
    "beta_start_ratio",
    "consistency_ramp_fraction",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def format_metric(metrics: dict[str, Any] | None, split: str) -> str:
    if not metrics:
        return "pending"
    values = metrics.get(split)
    if not values:
        return "pending"
    return f"{format_number(values.get('acc'))}/{format_number(values.get('auc'))}"


def bool_mark(value: Any) -> str:
    if value is None:
        return "pending"
    return "pass" if value is True else "fail"


def component_update_marks(audit: dict[str, Any] | None, field: str) -> str:
    if not audit:
        return "pending"
    values = audit.get(field)
    if not isinstance(values, dict):
        return "pending"
    return ", ".join(
        f"{component}={bool_mark(changed)}"
        for component, changed in sorted(values.items())
    )


def load_round(round_id: str) -> dict[str, Any]:
    round_root = RESULTS / round_id
    audit_path = round_root / "audit/AUDIT.json"
    audit = read_json(audit_path) if audit_path.is_file() else None
    manifest = audit.get("manifests", {}).get("joint", {}) if audit else {}
    decision_path = round_root / "round_decision.md"
    return {
        "round_id": round_id,
        "root": round_root,
        "audit_path": audit_path,
        "audit": audit,
        "manifest": manifest,
        "config": read_env(round_root / "round_config.env"),
        "decision_path": decision_path,
        "decision": decision_path.read_text(encoding="utf-8").strip() if decision_path.is_file() else None,
    }


def get_evaluation(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    audit = record["audit"]
    if not audit:
        return None
    return audit.get("evaluation_reports", {}).get(split)


def require_reports(records: list[dict[str, Any]], split: str) -> None:
    missing = [record["round_id"] for record in records if get_evaluation(record, split) is None]
    if missing:
        raise FileNotFoundError(f"missing {split} report for: {', '.join(missing)}")


def tensor_contract_status() -> tuple[str, Path]:
    path = RESULTS / "tensor_contracts.final.stdout.log"
    if not path.is_file():
        return "pending", path
    log_text = path.read_text(encoding="utf-8", errors="replace")
    return ("pass" if CONTRACT_SUCCESS_MARKER in log_text else "fail"), path


def closeout_status(records: list[dict[str, Any]], contract_status: str) -> tuple[bool, list[str]]:
    missing = []
    if contract_status != "pass":
        missing.append("tensor-contract closeout")
    for record in records:
        round_id = record["round_id"]
        audit = record["audit"]
        if not audit or audit.get("audit_pass") is not True:
            missing.append(f"{round_id} passing audit")
        if get_evaluation(record, "validation") is None:
            missing.append(f"{round_id} validation report")
        if get_evaluation(record, "test") is None:
            missing.append(f"{round_id} final-test report")
        if not record["decision"]:
            missing.append(f"{round_id} decision record")
    return not missing, missing


def best_mixed_test(records: list[dict[str, Any]]) -> tuple[str, float] | None:
    candidates = []
    for record in records:
        report = get_evaluation(record, "test")
        if not report:
            continue
        auc = report.get("paths", {}).get("mixed", {}).get("overall", {}).get("auc")
        try:
            candidates.append((record["round_id"], float(auc)))
        except (TypeError, ValueError):
            continue
    return max(candidates, key=lambda item: item[1]) if candidates else None


def config_changes(previous: dict[str, Any], current: dict[str, Any]) -> str:
    previous_manifest = previous["manifest"]
    current_manifest = current["manifest"]
    changes = []
    for field in CONFIG_FIELDS:
        before = previous_manifest.get(field)
        after = current_manifest.get(field)
        if before != after:
            changes.append(f"{field}: `{before}` -> `{after}`")

    lr_fields = (
        "BOOTSTRAP_LR",
        "SELECTOR_LR",
        "ENVIRONMENT_LR",
        "CALIBRATOR_LR",
    )
    for field in lr_fields:
        before = previous["config"].get(field)
        after = current["config"].get(field)
        if before != after:
            changes.append(f"{field}: `{before}` -> `{after}`")
    return "; ".join(changes) if changes else "unchanged"


def add_metric_table(lines: list[str], records: list[dict[str, Any]], split: str) -> None:
    lines.extend([
        f"## {split.title()} Metrics",
        "",
        "Metrics use `Acc/AUC`. The primary path is mixed; evidence and reversal are diagnostic paths.",
        "",
        "| Round | Candidate | Mixed Overall | Mixed Final | Evidence Overall | Evidence Final | Reversal Overall | Reversal Final |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        report = get_evaluation(record, split)
        paths = report.get("paths", {}) if report else {}
        candidate_id = record["audit"].get("candidate_id", "pending") if record["audit"] else "pending"
        lines.append(
            f"| `{record['round_id']}` | `{candidate_id}` | "
            f"{format_metric(paths.get('mixed'), 'overall')} | "
            f"{format_metric(paths.get('mixed'), 'final')} | "
            f"{format_metric(paths.get('evidence'), 'overall')} | "
            f"{format_metric(paths.get('evidence'), 'final')} | "
            f"{format_metric(paths.get('non_evidence'), 'overall')} | "
            f"{format_metric(paths.get('non_evidence'), 'final')} |"
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", default=("round1", "round2", "round3"))
    parser.add_argument("--require-validation", action="store_true")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--require-contracts", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "STAGE2_DUAL_PATH_REPORT.md",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the report without creating or replacing an output file",
    )
    args = parser.parse_args()

    records = [load_round(round_id) for round_id in args.rounds]
    if args.require_validation:
        require_reports(records, "validation")
    if args.require_test:
        require_reports(records, "test")
    contract_status, contract_path = tensor_contract_status()
    if args.require_contracts and contract_status != "pass":
        raise FileNotFoundError(
            f"missing or failed tensor-contract closeout: {contract_path}"
        )
    closeout_ready, closeout_missing = closeout_status(records, contract_status)

    lines = [
        "# Stage 2 Dual-path A+B Report",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Task: MathDial ATC Dialogue-KT, Qwen3-1.7B + LoRA.",
        "- Method: complementary `h_r/h_n`, contextual Transformer B, shared `P(h_r)`/`P(h_m)` and reversal `P(h_n)` diagnostics.",
        "- Experimental scope: three configuration rounds, all using model/data seed `1221`; this is not a multi-seed robustness claim.",
        f"- A-module reference Overall AUC: `{A_MODULE_OVERALL_AUC:.2f}`; paper target Overall AUC: `{PAPER_OVERALL_AUC:.2f}`.",
        f"- Tensor-contract closeout: `{contract_status}` (`{contract_path.relative_to(ROOT)}`).",
        "",
        "## Completion State",
        "",
        "| Round | Training/audit | Validation | Final test |",
        "|---|---|---|---|",
    ]
    for record in records:
        audit = record["audit"]
        lines.append(
            f"| `{record['round_id']}` | {bool_mark(audit.get('audit_pass')) if audit else 'pending'} | "
            f"{'available' if get_evaluation(record, 'validation') else 'pending'} | "
            f"{'available' if get_evaluation(record, 'test') else 'pending'} |"
        )

    if any(get_evaluation(record, "validation") for record in records):
        lines.append("")
        add_metric_table(lines, records, "validation")
        lines.extend([
            "## Validation Diagnostics",
            "",
            "| Round | JS | mean `|p_r-p_m|` | `h_nb` RMS | `beta*h_nb` RMS | Complement raw max | Complement relative RMS | Weight complement max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for record in records:
            evaluation = get_evaluation(record, "validation")
            diagnostics = evaluation.get("diagnostics", {}) if evaluation else {}
            lines.append(
                f"| `{record['round_id']}` | {format_number(diagnostics.get('stage2_loss_cons'), 6)} | "
                f"{format_number(diagnostics.get('p_r_p_m_abs_mean'), 6)} | "
                f"{format_number(diagnostics.get('h_nb_rms'), 6)} | "
                f"{format_number(diagnostics.get('beta_h_nb_rms'), 6)} | "
                f"{format_number(diagnostics.get('complement_max_abs_error'), 8)} | "
                f"{format_number(diagnostics.get('complement_relative_rms_error'), 8)} | "
                f"{format_number(diagnostics.get('weight_complement_max_abs_error'), 8)} |"
            )
        lines.append("")

    if any(get_evaluation(record, "test") for record in records):
        add_metric_table(lines, records, "test")

    lines.extend([
        "## Configuration Iterations",
        "",
        "| Round | B / objective configuration | Change from prior round |",
        "|---|---|---|",
    ])
    for index, record in enumerate(records):
        manifest = record["manifest"]
        config = (
            f"B=`{manifest.get('env_mode', 'pending')}` "
            f"({manifest.get('env_hidden_dim', 'pending')}h/"
            f"{manifest.get('env_num_layers', 'pending')}L/"
            f"{manifest.get('env_num_heads', 'pending')}heads/"
            f"ffn={manifest.get('env_ffn_dim', 'pending')}); "
            f"beta=`{manifest.get('env_beta', 'pending')}`; "
            f"ratio=`{manifest.get('env_output_ratio', 'pending')}`; "
            f"lambda_cons=`{manifest.get('lambda_cons', 'pending')}`"
        )
        changes = "initial configuration" if index == 0 else config_changes(records[index - 1], record)
        lines.append(f"| `{record['round_id']}` | {config} | {changes} |")
    lines.append("")

    lines.extend([
        "## Audit State",
        "",
        "| Round | " + " | ".join(AUDIT_FIELDS) + " |",
        "|---|" + "|".join("---" for _ in AUDIT_FIELDS) + "|",
    ])
    for record in records:
        audit = record["audit"] or {}
        cells = [bool_mark(audit.get(field)) for field in AUDIT_FIELDS]
        lines.append(f"| `{record['round_id']}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend([
        "## Parameter Update Detail",
        "",
        "Each strict joint phase must update every trainable component listed below; warmup freeze checks must keep the preceding components unchanged.",
        "",
        "| Round | A joint component updates | Final joint component updates |",
        "|---|---|---|",
    ])
    for record in records:
        audit = record["audit"]
        lines.append(
            f"| `{record['round_id']}` | "
            f"{component_update_marks(audit, 'a_joint_component_updates')} | "
            f"{component_update_marks(audit, 'joint_component_updates')} |"
        )
    lines.append("")

    lines.extend([
        "## Observed Problems and Engineering Adjustments",
        "",
        "- The initial long-batch preflight did not scan the complete training split and missed the actual worst collated geometry (`sample_index=6802`, 4 rows, 2009 tokens, attention risk `16144324`).",
        "- A first Round 1 joint attempt reached an out-of-memory failure near the former high-risk region. The archived failure logs are retained as recovery evidence and are not counted as formal candidate results.",
        "- The formal path now scans the complete split, uses expanded execution below the attention-risk threshold, and uses serial evidence/mixed execution with exact gradient splitting for high-risk batches. The real-Qwen joint and B-only worst-batch preflights then passed.",
        "- Subsequent rounds remain validation-driven: B amplitude, output ratio, consistency weight/ramp, and selector learning rates may change only from validation diagnostics; no test result, control run, or additional seed drives a configuration change.",
        "",
    ])

    lines.extend(["## Per-round Decisions", ""])
    for record in records:
        lines.append(f"### {record['round_id']}")
        lines.append("")
        if record["decision"]:
            lines.append(record["decision"])
        else:
            lines.append("Pending validation-driven decision record.")
        lines.append("")

    lines.extend([
        "## Stage 2 Closeout Gate",
        "",
        "All three rounds require final-test reports, passing audits, a passing tensor-contract closeout, and documented validation-driven decisions. The final interpretation must distinguish configuration observations from multi-seed evidence.",
        "",
        f"- Technical closeout: `{'eligible' if closeout_ready else 'not eligible'}`.",
    ])
    if closeout_ready:
        best = best_mixed_test(records)
        if best is not None:
            best_round, best_auc = best
            lines.extend([
                f"- Best mixed test Overall AUC: `{best_auc:.2f}` from `{best_round}`.",
                f"- Delta vs A-module reference: `{best_auc - A_MODULE_OVERALL_AUC:+.2f}`.",
                f"- Delta vs paper target: `{best_auc - PAPER_OVERALL_AUC:+.2f}`.",
            ])
            if best_auc >= PAPER_OVERALL_AUC:
                lines.append(
                    "- Research conclusion: the best audited Stage 2 candidate reaches the paper Overall AUC target; the current Stage 2 cycle can close with positive target evidence."
                )
            elif best_auc >= A_MODULE_OVERALL_AUC:
                lines.append(
                    "- Research conclusion: the best audited Stage 2 candidate improves on the A-module reference but remains below the paper target; the requested three-round cycle can close, while the performance gap must remain explicit before any Stage 3 claim."
                )
            else:
                lines.append(
                    "- Research conclusion: the best audited Stage 2 candidate does not improve on the A-module reference; the requested three-round cycle can close as a negative configuration study, but the Stage 2 mechanism is not validated as a performance improvement and Stage 3 should not start from this result."
                )
        lines.append(
            "- Scope conclusion: this is a three-round, seed-1221 configuration study, not a multi-seed robustness claim."
        )
    else:
        lines.append("- Missing closeout evidence: " + ", ".join(f"`{item}`" for item in closeout_missing) + ".")

    report_text = "\n".join(lines) + "\n"
    if args.stdout:
        print(report_text, end="")
        return

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
