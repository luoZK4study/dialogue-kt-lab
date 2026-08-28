#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cel_stage2_environment"
PAIRED_BOOTSTRAP_PATH = RESULTS / "PAIRED_BOOTSTRAP.json"
A_MODULE_OVERALL_AUC = 75.38
PAPER_OVERALL_AUC = 76.71


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    audit_paths = sorted(RESULTS.glob("*/AUDIT.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
    paired = (
        json.loads(PAIRED_BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        if PAIRED_BOOTSTRAP_PATH.is_file()
        else None
    )

    lines = [
        "# CEL Stage 2 First-Round Historical Results",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Audited candidates: `{len(reports)}`",
        f"- A-module reference Overall AUC: `{A_MODULE_OVERALL_AUC:.2f}`",
        f"- Paper target Overall AUC: `{PAPER_OVERALL_AUC:.2f}`",
        "- Interpretation: legacy single-path environment-residual pilots; not the current dual-path A+B method.",
        "",
        "| Candidate | Historical direction | Output | Audit | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta AUC vs A module |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for report in reports:
        metrics = report["metrics"]["stage2"]
        # The audit schema keeps the historical key for compatibility.
        delta_auc = report["deltas"]["stage1_v26"]["overall_auc"]
        manifest = report["manifests"]["joint"]
        output_config = manifest.get("env_output_postprocess", "none")
        if output_config != "none":
            output_config += f"@{manifest.get('env_output_ratio', 'n/a')}"
        lines.append(
            f"| `{report['candidate_id']}` | `{manifest['env_mode']}` | `{output_config}` | "
            f"`{report['audit_pass']}` | {metrics['overall_acc']:.2f} | {metrics['overall_auc']:.2f} | "
            f"{metrics['final_acc']:.2f} | {metrics['final_auc']:.2f} | {delta_auc:+.2f} |"
        )

    lines.extend([
        "",
        "## Historical Branch Diagnostics",
        "",
        "The stored diagnostic keys predate the canonical terminology. `gate_mean` is displayed as the mean of `a`; the legacy selected/non-selected fields are displayed as historical `|a|` separation.",
        "",
        "| Candidate | a Mean | Historical selected-|a| gap | Raw B-output L2 | Scaled B-output L2 | B/Input | B/A-evidence Cos | B/Input Cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for report in reports:
        diagnostics = report.get("diagnostics", {})
        selected_gap = None
        if "env_selected_score" in diagnostics and "env_non_selected_score" in diagnostics:
            selected_gap = diagnostics["env_selected_score"] - diagnostics["env_non_selected_score"]
        output_input_ratio = diagnostics.get("env_to_input_l2_ratio")
        if output_input_ratio is None and diagnostics.get("env_input_l2_mean"):
            output_input_ratio = diagnostics.get("env_l2_mean", 0.0) / diagnostics["env_input_l2_mean"]
        lines.append(
            f"| `{report['candidate_id']}` | {fmt(diagnostics.get('gate_mean'))} | "
            f"{fmt(selected_gap)} | {fmt(diagnostics.get('env_raw_l2_mean', diagnostics.get('env_l2_mean')), 2)} | "
            f"{fmt(diagnostics.get('env_scaled_l2_mean'), 2)} | {fmt(output_input_ratio)} | "
            f"{fmt(diagnostics.get('env_rationale_cosine'))} | {fmt(diagnostics.get('env_input_cosine'))} |"
        )

    if paired:
        lines.extend([
            "",
            "## Paired Dialogue-Cluster Bootstrap",
            "",
            f"Resamples / seed: `{paired['bootstrap_samples']} / {paired['seed']}`. "
            "Intervals are paired percentage-point deltas versus the A-module reference.",
            "",
            "| Candidate | Overall AUC delta | 95% CI | P(>0) | Final AUC delta | 95% CI | P(>0) | Non-final AUC delta | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for report in reports:
            candidate = paired.get("candidates", {}).get(report["candidate_id"])
            if not candidate:
                continue
            subsets = candidate["subsets"]
            cells = []
            for subset_name in ("overall", "final", "non_final"):
                subset = subsets[subset_name]
                bootstrap = subset["bootstrap_delta"]["auc"]
                cells.extend([
                    f"{subset['observed_delta']['auc']:+.2f}",
                    f"[{bootstrap['ci95_low']:+.2f}, {bootstrap['ci95_high']:+.2f}]",
                ])
                if subset_name != "non_final":
                    cells.append(f"{bootstrap['probability_positive']:.3f}")
            lines.append(f"| `{report['candidate_id']}` | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Historical Interpretation and Current Scope",
        "",
        "- All recorded first-round candidates remain historical pilots; no Stage 2 winner is declared.",
        "- Their metrics do not test complementary `h_r/h_n`, separate `p_r/p_m`, or prediction consistency.",
        "- The dual-path implementation, tensor contracts, and audit support for `a -> h_r/h_n/h_nb/h_m` are complete.",
        "- The current main B is a 4-layer, width-1024 contextual Transformer; Shuffle, token-wise MLP, and small Transformer variants remain historical pilots.",
        "- The active scope is three validation-driven configuration rounds using only seed `1221`. Controls, ablations, additional seeds, and Stage 3 are explicitly out of scope for this run.",
        "- Evidence-weight saturation and B-output scale remain diagnostics; add a regularizer only if full-module training shows the corresponding collapse.",
        "- Do not start Stage 3 before the target A+B mechanism is validated.",
    ])

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
