#!/usr/bin/env python3

"""Create validation-only Stage 2 iteration decisions and launcher settings.

The script deliberately consumes only a completed validation audit.  It never
reads test results, starts training, or changes a checkpoint.  The resulting
environment file is consumed by the sequential Stage 2 supervisor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "cel_stage2_environment" / "dual_path"

ROUND_CONFIG_KEYS = {
    "STAGE2_ENV_BETA": "ENV_BETA",
    "STAGE2_ENV_OUTPUT_RATIO": "ENV_OUTPUT_RATIO",
    "STAGE2_LAMBDA_CONS": "LAMBDA_CONS",
    "STAGE2_BETA_START_RATIO": "BETA_START_RATIO",
    "STAGE2_CONSISTENCY_RAMP": "CONSISTENCY_RAMP",
    "STAGE2_A_BOOTSTRAP_LR": "BOOTSTRAP_LR",
    "STAGE2_SELECTOR_LR": "SELECTOR_LR",
    "STAGE2_ENVIRONMENT_LR": "ENVIRONMENT_LR",
    "STAGE2_CALIBRATOR_LR": "CALIBRATOR_LR",
}
DEFAULT_CONFIG = {
    "STAGE2_ENV_BETA": 0.10,
    "STAGE2_ENV_OUTPUT_RATIO": 1.0,
    "STAGE2_LAMBDA_CONS": 0.10,
    "STAGE2_BETA_START_RATIO": 0.20,
    "STAGE2_CONSISTENCY_RAMP": 0.25,
    "STAGE2_A_BOOTSTRAP_LR": 0.00020,
    "STAGE2_SELECTOR_LR": 0.00001,
    "STAGE2_ENVIRONMENT_LR": 0.00010,
    "STAGE2_CALIBRATOR_LR": 0.00010,
}
# Metrics are reported in percentage-point AUC units (0-100), so a half-point
# gap is treated as a material validation signal rather than run-to-run noise.
MATERIAL_AUC_GAP = 0.50


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing or non-numeric {name}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite {name}: {number!r}")
    return number


def bounded(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def load_record(round_id: str) -> dict[str, Any]:
    round_root = RESULTS_ROOT / round_id
    audit_path = round_root / "audit" / "AUDIT.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing validation audit for {round_id}: {audit_path}")
    audit = read_json(audit_path)
    if audit.get("audit_pass") is not True:
        raise RuntimeError(f"{round_id} validation audit did not pass")
    validation = audit.get("evaluation_reports", {}).get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{round_id} audit has no validation report")
    if validation.get("evaluation_split") != "validation":
        raise ValueError(f"{round_id} audit has an invalid validation report")

    paths = validation.get("paths", {})
    diagnostics = validation.get("diagnostics", {})
    mixed = paths.get("mixed", {}).get("overall", {})
    evidence = paths.get("evidence", {}).get("overall", {})
    reversal = paths.get("non_evidence", {}).get("overall", {})
    manifest = audit.get("manifests", {}).get("joint", {})
    if not isinstance(manifest, dict):
        raise ValueError(f"{round_id} audit has no joint manifest")

    config = dict(DEFAULT_CONFIG)
    round_config = read_env(round_root / "round_config.env")
    for target_key, source_key in ROUND_CONFIG_KEYS.items():
        if source_key in round_config:
            config[target_key] = finite_number(round_config[source_key], source_key)
    config["STAGE2_ENV_BETA"] = finite_number(
        manifest.get("env_beta", config["STAGE2_ENV_BETA"]), "env_beta"
    )
    config["STAGE2_ENV_OUTPUT_RATIO"] = finite_number(
        manifest.get("env_output_ratio", config["STAGE2_ENV_OUTPUT_RATIO"]),
        "env_output_ratio",
    )
    config["STAGE2_LAMBDA_CONS"] = finite_number(
        manifest.get("lambda_cons", config["STAGE2_LAMBDA_CONS"]), "lambda_cons"
    )
    config["STAGE2_BETA_START_RATIO"] = finite_number(
        manifest.get("beta_start_ratio", config["STAGE2_BETA_START_RATIO"]),
        "beta_start_ratio",
    )
    config["STAGE2_CONSISTENCY_RAMP"] = finite_number(
        manifest.get(
            "consistency_ramp_fraction", config["STAGE2_CONSISTENCY_RAMP"]
        ),
        "consistency_ramp_fraction",
    )

    h_r_rms = finite_number(diagnostics.get("h_r_rms"), "h_r_rms")
    beta_h_nb_rms = finite_number(diagnostics.get("beta_h_nb_rms"), "beta_h_nb_rms")
    if h_r_rms <= 0:
        raise ValueError(f"{round_id} has non-positive h_r_rms")

    return {
        "round_id": round_id,
        "root": round_root,
        "audit": audit,
        "config": config,
        "metrics": {
            "mixed_auc": finite_number(mixed.get("auc"), "mixed overall AUC"),
            "evidence_auc": finite_number(evidence.get("auc"), "evidence overall AUC"),
            "reversal_auc": finite_number(reversal.get("auc"), "reversal overall AUC"),
            "probability_gap": finite_number(
                diagnostics.get("p_r_p_m_abs_mean"), "p_r_p_m_abs_mean"
            ),
            "js": finite_number(diagnostics.get("stage2_loss_cons"), "stage2_loss_cons"),
            "h_r_rms": h_r_rms,
            "h_nb_rms": finite_number(diagnostics.get("h_nb_rms"), "h_nb_rms"),
            "beta_h_nb_rms": beta_h_nb_rms,
            "contribution_ratio": beta_h_nb_rms / h_r_rms,
            "saturation": finite_number(
                diagnostics.get("a_saturation_frac"), "a_saturation_frac"
            ),
            "complement_error": finite_number(
                diagnostics.get("complement_max_abs_error"), "complement_max_abs_error"
            ),
            "complement_relative_rms_error": finite_number(
                diagnostics.get("complement_relative_rms_error"),
                "complement_relative_rms_error",
            ),
            "weight_complement_error": finite_number(
                diagnostics.get("weight_complement_max_abs_error"),
                "weight_complement_max_abs_error",
            ),
        },
    }


def ensure_changed(config: dict[str, float], source: dict[str, float]) -> None:
    """Guarantee that an iteration is a real configuration iteration."""

    if any(abs(config[key] - source[key]) > 1e-12 for key in config):
        return
    config["STAGE2_LAMBDA_CONS"] = bounded(
        source["STAGE2_LAMBDA_CONS"] + 0.05, 0.05, 0.30
    )


def decide_round2(source: dict[str, Any]) -> tuple[dict[str, float], str]:
    config = dict(source["config"])
    metrics = source["metrics"]
    ratio = metrics["contribution_ratio"]
    gap = metrics["probability_gap"]
    js = metrics["js"]
    mixed_vs_evidence_auc = metrics["mixed_auc"] - metrics["evidence_auc"]

    if ratio < 0.04:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 1.5, 0.075, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 1.25, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.15)
        reason = (
            "B contribution was weak relative to h_r, so Round 2 increases the "
            "mixed-path contribution while retaining a consistency safeguard."
        )
    elif mixed_vs_evidence_auc < -MATERIAL_AUC_GAP:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 0.75, 0.05, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 0.90, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.20)
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.40
        )
        reason = (
            "The mixed validation AUC was more than 0.50 percentage points below "
            "the evidence path, so Round 2 reduces B amplitude and strengthens "
            "prediction agreement."
        )
    elif ratio > 0.20 and (gap > 0.035 or js > 0.0010):
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 0.75, 0.05, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 0.90, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.20)
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.40
        )
        reason = (
            "B contribution and path disagreement were both high, so Round 2 "
            "reduces B amplitude and strengthens the shared-path consistency term."
        )
    elif gap > 0.025 or js > 0.0005:
        config["STAGE2_LAMBDA_CONS"] = bounded(
            max(config["STAGE2_LAMBDA_CONS"] * 1.75, 0.15), 0.05, 0.30
        )
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.40
        )
        reason = (
            "The two prediction paths disagreed despite a non-collapsed B output, "
            "so Round 2 strengthens and lengthens consistency optimization."
        )
    else:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 1.25, 0.075, 0.20)
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.15)
        reason = (
            "The initial paths were aligned and B was active, so Round 2 makes a "
            "bounded increase in B contribution while preserving alignment pressure."
        )

    if metrics["saturation"] > 0.25:
        config["STAGE2_A_BOOTSTRAP_LR"] = bounded(
            config["STAGE2_A_BOOTSTRAP_LR"] * 0.5, 0.000025, 0.00020
        )
        config["STAGE2_SELECTOR_LR"] = bounded(
            config["STAGE2_SELECTOR_LR"] * 0.5, 0.0000025, 0.00001
        )
        reason += " Selector saturation also lowers bootstrap and joint selector learning rates."
    ensure_changed(config, source["config"])
    return config, reason


def decide_round3(source: dict[str, Any], reference: dict[str, Any]) -> tuple[dict[str, float], str]:
    config = dict(source["config"])
    metrics = source["metrics"]
    ref_metrics = reference["metrics"]
    ratio = metrics["contribution_ratio"]
    gap = metrics["probability_gap"]
    js = metrics["js"]
    mixed_vs_evidence_auc = metrics["mixed_auc"] - metrics["evidence_auc"]

    if ratio < 0.04:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 1.4, 0.075, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 1.20, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.18)
        reason = (
            "Round 2 still showed a weak B contribution, so Round 3 further raises "
            "only the bounded mixed-path amplitude and consistency weight."
        )
    elif mixed_vs_evidence_auc < -MATERIAL_AUC_GAP:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 0.75, 0.05, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 0.90, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.24)
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.45
        )
        reason = (
            "Round 2 mixed validation AUC remained more than 0.50 percentage points "
            "below evidence, so Round 3 further constrains B amplitude while strengthening "
            "the shared prediction objective."
        )
    elif ratio > 0.20 and (gap > 0.035 or js > 0.0010):
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 0.75, 0.05, 0.20)
        config["STAGE2_ENV_OUTPUT_RATIO"] = bounded(
            config["STAGE2_ENV_OUTPUT_RATIO"] * 0.90, 0.75, 1.50
        )
        config["STAGE2_LAMBDA_CONS"] = max(config["STAGE2_LAMBDA_CONS"], 0.24)
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.45
        )
        reason = (
            "Round 2 retained excessive mixed-path perturbation, so Round 3 trades "
            "some B amplitude for stronger prediction agreement."
        )
    elif metrics["mixed_auc"] + MATERIAL_AUC_GAP < ref_metrics["mixed_auc"]:
        config["STAGE2_ENV_BETA"] = reference["config"]["STAGE2_ENV_BETA"]
        config["STAGE2_ENV_OUTPUT_RATIO"] = reference["config"]["STAGE2_ENV_OUTPUT_RATIO"]
        config["STAGE2_LAMBDA_CONS"] = bounded(
            max(config["STAGE2_LAMBDA_CONS"], reference["config"]["STAGE2_LAMBDA_CONS"]),
            0.05,
            0.30,
        )
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.35
        )
        reason = (
            "Round 2 mixed validation AUC materially regressed from Round 1, so Round 3 "
            "returns to the earlier B amplitude regime while retaining the learned "
            "consistency adjustment."
        )
    elif gap > 0.020 or js > 0.0004:
        config["STAGE2_LAMBDA_CONS"] = bounded(
            max(config["STAGE2_LAMBDA_CONS"] * 1.35, 0.18), 0.05, 0.30
        )
        config["STAGE2_CONSISTENCY_RAMP"] = max(
            config["STAGE2_CONSISTENCY_RAMP"], 0.45
        )
        reason = (
            "Round 2 primary validation was usable but the prediction paths remained "
            "too far apart, so Round 3 focuses on the consistency schedule."
        )
    else:
        config["STAGE2_ENV_BETA"] = bounded(config["STAGE2_ENV_BETA"] * 1.10, 0.05, 0.20)
        config["STAGE2_LAMBDA_CONS"] = bounded(
            max(config["STAGE2_LAMBDA_CONS"], 0.15), 0.05, 0.30
        )
        reason = (
            "Round 2 kept B active and paths aligned, so Round 3 applies a small, "
            "bounded contribution refinement before the configuration is frozen."
        )

    if metrics["saturation"] > 0.25:
        config["STAGE2_A_BOOTSTRAP_LR"] = bounded(
            config["STAGE2_A_BOOTSTRAP_LR"] * 0.5, 0.000025, 0.00020
        )
        config["STAGE2_SELECTOR_LR"] = bounded(
            config["STAGE2_SELECTOR_LR"] * 0.5, 0.0000025, 0.00001
        )
        reason += " Selector saturation also lowers bootstrap and joint selector learning rates."
    ensure_changed(config, source["config"])
    return config, reason


def write_env(path: Path, config: dict[str, float], candidate_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated from validation diagnostics only. Do not edit during a formal run.",
        f"STAGE2_CANDIDATE_ID={candidate_id}",
        "STAGE2_MODEL_SEED=1221",
    ]
    for key in sorted(config):
        lines.append(f"{key}={format_number(config[key], 8)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_lines(record: dict[str, Any]) -> list[str]:
    metrics = record["metrics"]
    return [
        "| Metric | Validation value |",
        "|---|---:|",
        f"| Mixed Overall AUC | {metrics['mixed_auc']:.4f} |",
        f"| Evidence Overall AUC | {metrics['evidence_auc']:.4f} |",
        f"| Reversal Overall AUC | {metrics['reversal_auc']:.4f} |",
        f"| JS | {metrics['js']:.6f} |",
        f"| mean |p_r-p_m| | {metrics['probability_gap']:.6f} |",
        f"| h_nb RMS | {metrics['h_nb_rms']:.6f} |",
        f"| beta*h_nb / h_r RMS | {metrics['contribution_ratio']:.6f} |",
        f"| a saturation fraction | {metrics['saturation']:.6f} |",
        f"| complement raw max error | {metrics['complement_error']:.8f} |",
        f"| complement relative RMS error | {metrics['complement_relative_rms_error']:.8f} |",
        f"| weight complement max error | {metrics['weight_complement_error']:.8f} |",
    ]


def config_lines(config: dict[str, float]) -> list[str]:
    return [f"- `{key}`: `{format_number(value, 8)}`" for key, value in sorted(config.items())]


def write_next_decision(
    source: dict[str, Any],
    target_round: str,
    config: dict[str, float],
    reason: str,
) -> Path:
    target_candidate = f"stage2_dual_{target_round}_contextual_transformer_seed1221"
    target_root = RESULTS_ROOT / target_round
    env_path = target_root / "next_round.env"
    write_env(env_path, config, target_candidate)
    lines = [
        f"# {source['round_id']} Validation-driven Decision",
        "",
        "## Evidence",
        "",
        *metric_lines(source),
        "",
        "- Validation audit pass: `true`.",
        "- This decision uses validation metrics and diagnostics only; no test result or control experiment informed it.",
        "",
        f"## Decision For {target_round}",
        "",
        reason,
        "",
        f"- Candidate: `{target_candidate}`",
        "- Initialization: raw Qwen3-1.7B base, never another candidate checkpoint.",
        "- Model/data seed: `1221`.",
        "- B remains the contextual Transformer; no architecture-reducing ablation is introduced.",
        "",
        "## Exact Configuration",
        "",
        *config_lines(config),
        "",
        f"Generated launcher environment: `{env_path.name}` in `{target_round}`.",
    ]
    path = source["root"] / "round_decision.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def write_final_decision(source: dict[str, Any]) -> Path:
    lines = [
        "# Round 3 Validation-driven Freeze Decision",
        "",
        "## Evidence",
        "",
        *metric_lines(source),
        "",
        "- Validation audit pass: `true`.",
        "- This decision uses validation metrics and diagnostics only; no test result or control experiment informed it.",
        "",
        "## Decision",
        "",
        "Round 3 is the third and final configured candidate. Its configuration is frozen; the next permitted action is final test and final audit for all three completed rounds, not a fourth training run.",
        "",
        "## Frozen Configuration",
        "",
        *config_lines(source["config"]),
    ]
    path = source["root"] / "round_decision.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_round", choices=("round1", "round2", "round3"))
    parser.add_argument("--target-round", choices=("round2", "round3"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize == (args.target_round is not None):
        parser.error("provide exactly one of --target-round or --finalize")

    source = load_record(args.source_round)
    if args.finalize:
        if args.source_round != "round3":
            parser.error("only round3 can be frozen")
        path = write_final_decision(source)
        print(path)
        return

    assert args.target_round is not None
    if args.source_round == "round1" and args.target_round == "round2":
        config, reason = decide_round2(source)
    elif args.source_round == "round2" and args.target_round == "round3":
        reference = load_record("round1")
        config, reason = decide_round3(source, reference)
    else:
        parser.error("round transitions must be round1->round2 or round2->round3")
    path = write_next_decision(source, args.target_round, config, reason)
    print(path)


if __name__ == "__main__":
    main()
